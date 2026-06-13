"""
LangGraph per-file review harness — hub-and-spoke orchestration.

Implements all five Anthropic harness principles
(anthropic.com/engineering/harness-design-long-running-apps):

  1. Loop management    — hard step cap (_MAX_TOTAL_STEPS), deterministic pre-checks,
                          GPT orchestrator decides when to stop
  2. LLM failures       — error categorisation (fatal / transient), each node never
                          raises; transient failures degrade gracefully
  3. Tool call patterns — verify_node wrapped with per-call timeout; git apply uses
                          --check (no side effects); test gate auto-reverts
  4. Context compaction — accumulated lists trimmed before being injected into prompts
                          (_tail / _compact_for_prompt helpers); LangGraph state keeps
                          full history for routing decisions
  5. Verification       — three independent gates (AST → test runner → LLM evaluator);
                          evaluator uses fast_gate model (8B) separate from generator
                          (70B); PlannerContract acceptance criteria wired through

Graph topology — hub-and-spoke (GPT-4o is the routing hub):

    orchestrate (GPT-4o) ──┬──→ analyze_fault (DeepSeek reason) ──┐
         ↑                 ├──→ code_review   (Groq fast_gate)    │
         │                 ├──→ plan          (DeepSeek reason)   │
         └─────────────────├──→ generate      (Qwen generation)   ┤
                           ├──→ verify        (Groq fast_gate)    │
                           ├──→ reflect       (DeepSeek reason)   │
                           └──→ END                               ┘

GPT-4o sees the full pipeline state at every junction and decides the
next worker node. No edges are hardcoded between workers — GPT routes
dynamically based on what has been completed and what the state shows.

Self-evaluation bias mitigation (article 1):
  - Analyzer / Planner / Reflector: role="reason"     (DeepSeek-R1-32B) — deep reasoning
  - Generator:                       role="generation" (Qwen2.5-Coder-32B) — code writing
  - Evaluator:                       role="fast_gate"  (Groq llama-3.1-8b) — independent 8B
  - Orchestrator:                    role="chat"       (GPT-4o) — dynamic hub router

Anti-patterns avoided:
  - Environmental degradation: pre-flight check before graph starts
  - One-shot trap: retry loop with Reflexion
  - Premature victory: eval_score hard threshold (≥3) enforced by evaluator
  - Context anxiety: LangGraph state is structured data, not a prompt conversation
"""
from __future__ import annotations

import concurrent.futures
import contextvars
import logging
import operator
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TOTAL_STEPS = 20        # hard cap — prevents runaway loops regardless of routing
_GRAPH_TIMEOUT_SECONDS = 900  # 15 min max per file (test gate alone can take 300s)
_PROMPT_MAX_REFLECTIONS = 3   # context compaction: send only last N to LLM
_PROMPT_MAX_PATCHES = 3
_PROMPT_MAX_FAILURES = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tail(lst: list, n: int) -> list:
    """Return last n items — used for context compaction in prompts."""
    return lst[-n:] if len(lst) > n else lst


def _compact_for_prompt(items: list, max_n: int, max_chars: int, label: str = "") -> str:
    """Trim a list to last max_n items and max_chars total for safe LLM injection."""
    trimmed = _tail(items, max_n)
    result = "\n".join(f"  [{label}{i+1}] {str(s)[:max_chars // max(len(trimmed), 1)]}"
                       for i, s in enumerate(trimmed))
    if len(items) > max_n:
        result = f"  ... (showing last {max_n} of {len(items)}) ...\n" + result
    return result or "  (none)"


def _safe_len(value: Any) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except Exception:
        return 0


def _node_start_metadata(state: dict) -> dict:
    """Small, non-sensitive metadata shared by all harness node spans."""
    return {
        "file": state.get("file_path"),
        "risk": state.get("risk_level"),
        "attempt": state.get("attempt", 0),
        "max_retries": state.get("max_retries"),
        "total_steps": state.get("total_steps", 0),
        "completed_steps_count": _safe_len(state.get("completed_steps", [])),
        "has_graph_context": bool(state.get("graph_context")),
        "reflections_count": _safe_len(state.get("reflections", [])),
        "patches_tried_count": _safe_len(state.get("patches_tried", [])),
        "failure_reasons_count": _safe_len(state.get("failure_reasons", [])),
    }


def _node_result_metadata(result: Any) -> dict:
    """Summarise a node return value without sending full source or patches."""
    if not isinstance(result, dict):
        return {"result_type": type(result).__name__}

    metadata: dict[str, Any] = {}
    for key in (
        "applies_cleanly",
        "verification_passed",
        "verification_reason",
        "eval_score",
        "orchestrator_decision",
        "attempt",
        "total_steps",
    ):
        if key in result:
            value = result[key]
            if isinstance(value, str):
                value = value[:300]
            metadata[key] = value

    if "fault_desc" in result:
        metadata["fault_desc_preview"] = str(result["fault_desc"])[:300]
    if "patch" in result:
        metadata["patch_size"] = len(result.get("patch") or "")
    if "plan_contract" in result:
        contract = result.get("plan_contract") or {}
        metadata["contract_acceptance_criteria_count"] = _safe_len(
            contract.get("acceptance_criteria", [])
        ) if isinstance(contract, dict) else 0
    if "review_comments" in result:
        metadata["review_comments_count"] = _safe_len(result.get("review_comments", []))
    if "failure_reasons" in result:
        reasons = result.get("failure_reasons") or []
        metadata["failure_reasons_added"] = _safe_len(reasons)
        if reasons:
            metadata["last_failure_reason"] = str(reasons[-1])[:300]
    if "events" in result:
        events = result.get("events") or []
        metadata["events_added"] = _safe_len(events)
        if events:
            metadata["event_types"] = [str(ev[0]) for ev in events[:5] if ev]
    return metadata


def _node_span_input(name: str, state: dict) -> dict:
    """Build a compact, node-specific input dict for the Langfuse span."""
    base = {
        "file": state.get("file_path"),
        "attempt": state.get("attempt", 0),
        "risk": state.get("risk_level"),
    }
    if name == "analyze_fault":
        base["issue"] = (state.get("issue_text") or "")[:300]
        base["reanalyze"] = state.get("orchestrator_decision") == "reanalyze"
    elif name == "code_review":
        base["fault"] = (state.get("fault_desc") or "")[:300]
    elif name == "plan":
        base["fault"] = (state.get("fault_desc") or "")[:300]
    elif name == "generate":
        base["fault"] = (state.get("fault_desc") or "")[:300]
        base["has_plan"] = bool(state.get("plan_context"))
        base["reflection_count"] = _safe_len(state.get("reflections", []))
    elif name == "verify":
        base["patch_size"] = len(state.get("patch") or "")
    elif name == "reflect":
        base["failure_reason"] = (state.get("verification_reason") or "")[:300]
        base["score"] = state.get("eval_score", 0)
    elif name == "orchestrate":
        completed = state.get("completed_steps", [])
        step_counts: dict[str, int] = {}
        for s in completed:
            step_counts[s] = step_counts.get(s, 0) + 1
        base["completed"] = step_counts
        base["score"] = state.get("eval_score", 0)
        base["verification_passed"] = state.get("verification_passed", False)
        base["applies_cleanly"] = state.get("applies_cleanly", False)
    return base


def _node_span_output(name: str, result: dict) -> dict:
    """Build a clear, node-specific output dict for the Langfuse span."""
    out: dict[str, Any] = {"node": name}
    if name == "analyze_fault":
        out["fault"] = (result.get("fault_desc") or "")[:400]
    elif name == "code_review":
        out["comments_count"] = _safe_len(result.get("review_comments", []))
        comments = result.get("review_comments") or []
        if comments:
            out["severities"] = [c.get("severity") for c in comments[:5]]
    elif name == "plan":
        contract = result.get("plan_contract") or {}
        out["root_cause"] = (contract.get("root_cause") or "")[:300]
        out["criteria_count"] = _safe_len(contract.get("acceptance_criteria", []))
        out["files_to_modify"] = contract.get("files_to_modify", [])
    elif name == "generate":
        out["patch_size"] = len(result.get("patch") or "")
        out["attempt"] = result.get("attempt", 0)
    elif name == "verify":
        out["passed"] = result.get("verification_passed", False)
        out["score"] = result.get("eval_score", 0)
        out["reason"] = (result.get("verification_reason") or "")[:300]
        out["applies_cleanly"] = result.get("applies_cleanly", False)
    elif name == "reflect":
        reflections = result.get("reflections") or []
        out["reflection"] = (reflections[0] if reflections else "")[:400]
    elif name == "orchestrate":
        action = result.get("next_action", "done")
        out["next_action"] = action
        out["action_label"] = {
            "analyze_fault": "🔍 analyze_fault",
            "code_review":   "👁 code_review",
            "plan":          "📋 plan",
            "generate":      "⚙️ generate",
            "verify":        "🧪 verify",
            "reflect":       "🔄 reflect",
            "done":          "✅ done",
        }.get(action, action)
        if result.get("orchestrator_decision"):
            out["final_decision"] = result["orchestrator_decision"]
    return out


def _instrument_node(name: str):
    """Wrap a LangGraph node in a Langfuse span with clear input/output."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(state: HarnessState) -> dict:
            from src_bot.observability.langfuse_ctx import (
                langfuse_span,
                langfuse_update_current_span,
            )

            with langfuse_span(
                f"harness.{name}",
                input=_node_span_input(name, state),
                metadata=_node_start_metadata(state),
            ):
                try:
                    result = fn(state)
                    langfuse_update_current_span(
                        output=_node_span_output(name, result) if isinstance(result, dict) else {"node": name},
                        metadata=_node_result_metadata(result),
                    )
                    return result
                except Exception as e:
                    langfuse_update_current_span(
                        metadata={"error_type": type(e).__name__, "error": str(e)[:500]},
                        status_message=str(e)[:500],
                    )
                    raise

        return wrapper

    return decorator


def _is_fatal_llm_error(exc: Exception) -> bool:
    """
    Return True for errors that mean the LLM cannot recover (abort node immediately).
    Return False for transient errors (rate limit, conn reset) where the router's
    _RateLimitedLLM has already exhausted retries — degrade gracefully.

    Fatal = no amount of retrying will help:
      - 401 invalid API key     → configuration error, must fix .env
      - 403 permission denied   → wrong key scope
      - Import / attribute error → code bug, missing dependency
    """
    msg = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    fatal_type_patterns = (
        "importerror", "modulenotfounderror",
        "attributeerror",
        "notimplementederror",
    )
    for pat in fatal_type_patterns:
        if pat in exc_type:
            return True

    fatal_msg_patterns = (
        "valueerror: unknown llm_provider",
        "invalid api key",          # Groq/OpenAI 401
        "invalid_api_key",          # structured error code
        "authentication",           # generic auth failure
        "401",                      # HTTP Unauthorized
        "403",                      # HTTP Forbidden
        "permission denied",
        "model_decommissioned",     # Groq decommissioned model (e.g. deepseek-r1)
        "model decommissioned",
        "is no longer supported",   # Groq deprecation message
        "has been decommissioned",  # alternate Groq phrasing
    )
    for pat in fatal_msg_patterns:
        if pat in msg:
            return True

    return False


def _preflight_check(file_path: str, workdir: Path) -> Optional[str]:
    """
    Verify the environment before starting the graph.
    Returns an error string if the check fails, None if OK.

    Checks (article 2 — session startup checklist):
      - workdir exists and is a git repo
      - file exists in workdir
      - working directory is clean (no uncommitted changes from prior attempts)
    """
    if not workdir.exists():
        return f"workdir does not exist: {workdir}"

    if not (workdir / ".git").exists():
        return f"workdir is not a git repo: {workdir}"

    target = workdir / file_path
    if not target.exists():
        return f"file not found in workdir: {file_path}"

    # Verify git status is clean so patches apply against known-clean state
    import subprocess
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workdir, capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return f"git status failed: {result.stderr.strip()[:200]}"

    # Uncommitted changes (modified tracked files) would confuse git apply --check
    dirty_lines = [l for l in result.stdout.splitlines() if not l.startswith("??")]
    if dirty_lines:
        # Reset to clean state rather than aborting — uncommitted changes from a
        # previous (crashed) harness run should not block the next attempt.
        logger.warning(
            "[harness] Dirty working tree before review — resetting to HEAD: %s",
            dirty_lines[:3],
        )
        subprocess.run(
            ["git", "checkout", "--", "."],
            cwd=workdir, capture_output=True, timeout=15,
        )

    return None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HarnessResult:
    patch: str = ""
    applies_cleanly: bool = False
    eval_score: int = 0
    eval_reason: str = ""
    fault_desc: str = ""
    plan_context: str = ""
    orchestrator_decision: str = ""
    reflexion_attempts: int = 0
    risk_level: str = "medium"
    review_comments: list = field(default_factory=list)  # [{severity, line, message}]
    events: list = field(default_factory=list)  # [(event_type, data), ...]


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class HarnessState(TypedDict):
    # ── Inputs (set once, never overwritten) ──────────────────────────────
    issue_text: str
    file_path: str
    graph_context: str
    workdir: str
    project_id: str
    risk_level: str
    max_retries: int

    # ── Accumulated (append-only via operator.add) ─────────────────────────
    # Full history kept in state for routing decisions.
    # Only last _PROMPT_MAX_* entries are injected into LLM prompts.
    reflections: Annotated[list[str], operator.add]
    patches_tried: Annotated[list[str], operator.add]
    failure_reasons: Annotated[list[str], operator.add]
    events: Annotated[list, operator.add]
    completed_steps: Annotated[list[str], operator.add]  # full call history for orchestrator

    # ── Mutable per iteration ──────────────────────────────────────────────
    fault_desc: str
    file_content: str
    review_comments: list     # structured comments from code_review_node
    plan_context: str
    plan_contract: dict          # PlannerContract serialised as dict (JSON-safe)
    patch: str
    applies_cleanly: bool
    verification_passed: bool
    verification_reason: str
    eval_score: int
    attempt: int                 # attempts within current analysis cycle
    total_steps: int             # global step counter — hard cap _MAX_TOTAL_STEPS
    reanalyze_count: int         # kept for backward compat; not used in hub-and-spoke routing
    next_action: str             # orchestrator's routing decision (set by orchestrate_node)
    orchestrator_decision: str   # "accept_best" | "abort" — only set when next_action=="done"


# ---------------------------------------------------------------------------
# Node: analyze_fault  (reason LLM)
# ---------------------------------------------------------------------------

@_instrument_node("analyze_fault")
def analyze_fault_node(state: HarnessState) -> dict:
    """Reason LLM — WHY is this file faulty and WHERE exactly?

    Context compaction: only last _PROMPT_MAX_REFLECTIONS reflections are
    injected when this is a reanalysis cycle.
    """
    from src_bot.llm.router import get_llm
    from langchain_core.messages import HumanMessage

    file_path = state["file_path"]
    workdir = Path(state["workdir"])

    # Read file from disk
    try:
        file_content = (workdir / file_path).read_text(errors="ignore")
    except Exception as e:
        return {
            "fault_desc": f"Could not read file: {e}",
            "file_content": "",
            "completed_steps": ["analyze_fault"],
            "total_steps": state.get("total_steps", 0) + 1,
            "events": [("phase", {"phase": 2, "file": file_path, "status": "error", "error": str(e)})],
        }

    # Add line numbers — helps LLM reference exact lines (ACI pattern)
    lines = file_content.split("\n")
    numbered = "\n".join(f"{i+1:4d} | {l}" for i, l in enumerate(lines))
    content_for_prompt = numbered[:8000]

    # Context compaction: trim past reflections injected into prompt
    prior_context = ""
    if state["reflections"]:
        trimmed = _tail(state["reflections"], _PROMPT_MAX_REFLECTIONS)
        prior_context = (
            "\n\n## Prior attempts failed — approach from a different angle.\n"
            "Previously identified faults (led to incorrect patches):\n"
            + "\n".join(f"  [{i+1}] {r[:150]}" for i, r in enumerate(trimmed))
            + "\n"
        )

    prompt = (
        f"Analyze this file and identify the specific fault.\n\n"
        f"Issue: {state['issue_text'][:1500]}\n"
        f"File: {file_path}\n"
        f"Call graph context:\n{state['graph_context'][:800]}\n"
        f"{prior_context}\n"
        f"File content:\n```\n{content_for_prompt}\n```\n\n"
        f"Describe the fault in 2-4 sentences: "
        f"what is wrong, on which lines, and what correct behavior should be."
    )

    llm = get_llm(role="reason", temperature=0)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        fault_desc = response.content.strip()
        if not fault_desc:
            fault_desc = "Analysis returned empty response"
    except Exception as e:
        if _is_fatal_llm_error(e):
            logger.error("[harness] Fatal LLM error in analyze_fault: %s", e)
            return {
                "fault_desc": f"FATAL: {e}",
                "file_content": file_content,
                "total_steps": _MAX_TOTAL_STEPS,  # trigger hard cap
                "events": [("phase", {"phase": 2, "file": file_path, "status": "fatal_error"})],
            }
        logger.warning("[harness] Transient error in analyze_fault: %s", e)
        fault_desc = f"Analysis failed (transient): {e}"

    return {
        "fault_desc": fault_desc,
        "file_content": file_content,
        "completed_steps": ["analyze_fault"],
        "total_steps": state.get("total_steps", 0) + 1,
        "events": [("phase", {
            "phase": 2, "file": file_path,
            "fault": fault_desc[:120], "status": "done",
        })],
    }


# ---------------------------------------------------------------------------
# Node: code_review  (fast_gate LLM — structured review comments)
# ---------------------------------------------------------------------------

@_instrument_node("code_review")
def code_review_node(state: HarnessState) -> dict:
    """Fast-gate LLM — generate structured code review comments for the file.

    Runs after analyze_fault so fault_desc is available as context.
    Produces 2-5 comments: bugs, warnings, and refactoring suggestions.
    This is independent from the patch path — review_comments are always
    returned even when the patch never applies cleanly.
    """
    import json, re
    from src_bot.llm.router import get_llm
    from langchain_core.messages import HumanMessage

    file_path = state["file_path"]
    file_content = state.get("file_content", "")
    fault_desc = state.get("fault_desc", "")

    if not file_content:
        return {
            "review_comments": [],
            "completed_steps": ["code_review"],
            "total_steps": state.get("total_steps", 0) + 1,
        }

    # First 100 lines with line numbers — enough for structural review
    lines = file_content.split("\n")
    numbered = "\n".join(f"{i+1:4d} | {l}" for i, l in enumerate(lines[:100]))

    prompt = (
        f"You are a senior code reviewer. Analyze this Python file and provide structured feedback.\n\n"
        f"File: {file_path}\n"
        f"Primary fault already identified: {fault_desc[:250]}\n\n"
        f"File content (first 100 lines):\n```\n{numbered}\n```\n\n"
        f"Return a JSON array of 2-5 review comments covering bugs, warnings, "
        f"and refactoring/improvement opportunities BEYOND the primary fault.\n"
        f'Respond with ONLY this JSON array, no explanation:\n'
        f'[{{"severity":"bug|warning|suggestion","line":<line_number_or_null>,'
        f'"message":"specific actionable feedback"}}]'
    )

    llm = get_llm(role="fast_gate", temperature=0)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = (response.content or "").strip()
        # Extract JSON array (handles markdown code fences)
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            comments = json.loads(match.group())
            clean = []
            for c in comments:
                if not isinstance(c, dict) or "message" not in c:
                    continue
                clean.append({
                    "severity": str(c.get("severity", "suggestion")).lower(),
                    "line": c.get("line"),
                    "message": str(c.get("message", ""))[:300],
                })
            return {
                "review_comments": clean,
                "completed_steps": ["code_review"],
                "total_steps": state.get("total_steps", 0) + 1,
                "events": [("review_comments", {
                    "file": file_path,
                    "comments": clean,
                })],
            }
    except Exception as e:
        logger.warning("[harness] code_review_node error (non-fatal): %s", e)

    return {
        "review_comments": [],
        "completed_steps": ["code_review"],
        "total_steps": state.get("total_steps", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Node: plan  (reason LLM)
# ---------------------------------------------------------------------------

@_instrument_node("plan")
def plan_node(state: HarnessState) -> dict:
    """Reason LLM — build structured PlannerContract (acceptance criteria + approach).

    Sprint contract pattern from article 1: the contract gives the evaluator
    external criteria to check against, reducing self-evaluation bias.
    """
    from api.agent.planner import generate_plan_sync

    file_path = state["file_path"]
    try:
        contract = generate_plan_sync(
            fault_description=state["fault_desc"],
            issue_text=state["issue_text"],
            file_path=file_path,
            graph_context=state["graph_context"],
            mode="bug_fix",
        )
        plan_context = contract.to_prompt_section()
        plan_contract = contract.to_dict()
        root_cause_short = contract.root_cause[:120]
    except Exception as e:
        if _is_fatal_llm_error(e):
            logger.error("[harness] Fatal error in plan_node: %s", e)
            return {
                "plan_context": "",
                "plan_contract": {},
                "total_steps": _MAX_TOTAL_STEPS,
                "events": [("phase", {"phase": 2.5, "file": file_path, "status": "fatal_error"})],
            }
        logger.warning("[harness] plan_node failed (continuing without contract): %s", e)
        plan_context = ""
        plan_contract = {}
        root_cause_short = "(planning failed)"

    return {
        "plan_context": plan_context,
        "plan_contract": plan_contract,
        "completed_steps": ["plan"],
        "total_steps": state.get("total_steps", 0) + 1,
        "events": [("phase", {
            "phase": 2.5, "file": file_path, "status": "planned",
            "root_cause": root_cause_short,
        })],
    }


# ---------------------------------------------------------------------------
# Node: generate  (generation LLM)
# ---------------------------------------------------------------------------

@_instrument_node("generate")
def generate_node(state: HarnessState) -> dict:
    """Generation LLM — write the unified diff patch.

    Context compaction: only the last reflection is injected into the prompt
    (the reflector distils all prior attempts into one actionable critique).
    """
    from run_swebench_v2 import phase3_generate_patch
    from src_bot.llm.router import get_llm

    file_path = state["file_path"]
    attempt = state.get("attempt", 0) + 1
    total_steps = state.get("total_steps", 0) + 1

    # Only the most recent reflection is useful — older ones are already incorporated
    reflection = state["reflections"][-1] if state["reflections"] else ""
    last_patch = state["patches_tried"][-1] if state["patches_tried"] else ""

    llm = get_llm(role="generation", temperature=0)
    try:
        patch = phase3_generate_patch(
            llm,
            state["issue_text"],
            file_path,
            state["fault_desc"],
            state["file_content"],
            reflection,
            last_patch,
            state["plan_context"],
        )
    except Exception as e:
        if _is_fatal_llm_error(e):
            logger.error("[harness] Fatal error in generate_node: %s", e)
            return {
                "patch": "",
                "attempt": attempt,
                "total_steps": _MAX_TOTAL_STEPS,
                "patches_tried": [],
                "events": [("phase", {"phase": 3, "file": file_path, "attempt": attempt, "status": "fatal_error"})],
            }
        logger.warning("[harness] Transient error in generate_node: %s", e)
        patch = ""

    events: list = [("phase", {
        "phase": 3, "file": file_path,
        "attempt": attempt, "max_attempts": state["max_retries"],
        "status": "generated", "patch_size": len(patch),
    })]

    # Frustration detection: shrinking patch = generator confused (article 2 pattern)
    if last_patch and patch and len(patch) < len(last_patch) * 0.6:
        events.append(("phase", {
            "phase": 3, "file": file_path, "attempt": attempt,
            "status": "frustration_detected",
            "message": "Patch shrinking — generator confused, broadening context next attempt",
        }))

    return {
        "patch": patch,
        "attempt": attempt,
        "total_steps": total_steps,
        "patches_tried": [patch] if patch else [],
        "completed_steps": ["generate"],
        "events": events,
    }


# ---------------------------------------------------------------------------
# Node: verify  (AST → test → LLM evaluator, independent of generator)
# ---------------------------------------------------------------------------

@_instrument_node("verify")
def verify_node(state: HarnessState) -> dict:
    """Multi-gate verification pipeline.

    Gate independence (article 1): evaluator uses role='fast_gate' (8B model),
    generator uses role='generation' (70B). Different models, different
    weights — genuine independent evaluation.

    PlannerContract wired through: accept criteria from plan_node are passed
    to the LLM gate so it evaluates against sprint contract terms, not a
    generic rubric.

    Tool call timeout: verify_patch is wrapped in a ThreadPoolExecutor with
    an explicit deadline (_GRAPH_TIMEOUT_SECONDS / 3) to prevent a hung test
    suite from blocking the harness indefinitely.
    """
    from run_swebench_v2 import _verify_patch_applies
    from src_bot.verification.pipeline import verify_patch
    from src_bot.config.config import configs

    file_path = state["file_path"]
    workdir = Path(state["workdir"])
    patch = state["patch"]
    attempt = state.get("attempt", 1)

    if not patch:
        reason = "Empty patch generated"
        return {
            "applies_cleanly": False,
            "verification_passed": False,
            "verification_reason": reason,
            "eval_score": 0,
            "failure_reasons": [reason],
            "completed_steps": ["verify"],
            "total_steps": state.get("total_steps", 0) + 1,
            "events": [("phase", {"phase": 3, "file": file_path, "attempt": attempt, "status": "empty"})],
        }

    # git apply --check — no side effects, safe to call repeatedly
    applies_ok, apply_error = _verify_patch_applies(patch, workdir)
    if not applies_ok:
        reason = f"git apply failed: {apply_error[:150]}"
        return {
            "applies_cleanly": False,
            "verification_passed": False,
            "verification_reason": reason,
            "eval_score": 0,
            "failure_reasons": [reason],
            "completed_steps": ["verify"],
            "total_steps": state.get("total_steps", 0) + 1,
            "events": [("phase", {
                "phase": 3, "file": file_path, "attempt": attempt,
                "status": "apply_failed", "error": apply_error[:100],
            })],
        }

    # Reconstruct PlannerContract from serialised dict for contract-aware evaluation
    contract = None
    if state.get("plan_contract"):
        try:
            from api.agent.planner import PlannerContract
            contract = PlannerContract(**state["plan_contract"])
        except Exception:
            pass  # fall back to generic evaluation

    # Tool call timeout — test gate can run up to EXECUTION_GATE_TIMEOUT seconds
    verify_timeout = min(configs.EXECUTION_GATE_TIMEOUT + 60, _GRAPH_TIMEOUT_SECONDS // 3)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                verify_patch,
                patch, workdir,
                state["issue_text"], state["fault_desc"],
                contract, "bug_fix",
                configs.EXECUTION_GATE_ENABLED,
                configs.EXECUTION_GATE_TEST_CMD,
                configs.EXECUTION_GATE_TIMEOUT,
            )
            verification = future.result(timeout=verify_timeout)
    except concurrent.futures.TimeoutError:
        reason = f"verify_patch timed out after {verify_timeout}s"
        logger.warning("[harness] %s (file=%s)", reason, file_path)
        return {
            "applies_cleanly": True,
            "verification_passed": False,
            "verification_reason": reason,
            "eval_score": 0,
            "failure_reasons": [reason],
            "completed_steps": ["verify"],
            "total_steps": state.get("total_steps", 0) + 1,
            "events": [("phase", {"phase": 3, "file": file_path, "attempt": attempt, "status": "timeout"})],
        }
    except Exception as e:
        reason = f"verify_patch error: {e}"
        logger.warning("[harness] %s", reason)
        return {
            "applies_cleanly": True,
            "verification_passed": False,
            "verification_reason": reason,
            "eval_score": 0,
            "failure_reasons": [str(e)],
            "completed_steps": ["verify"],
            "total_steps": state.get("total_steps", 0) + 1,
            "events": [("phase", {"phase": 3, "file": file_path, "attempt": attempt, "status": "verify_error"})],
        }

    llm_score = verification.llm_score or 0
    llm_reason = verification.llm_reason or ""
    passed = verification.passed
    rejection = verification.rejection_reason or ""

    events: list = [("phase", {
        "phase": 3, "file": file_path, "attempt": attempt,
        "status": "evaluated" if passed else "verification_rejected",
        "eval_score": llm_score, "eval_reason": llm_reason,
        "ast_passed": verification.ast_result.passed,
        "tests_skipped": (
            verification.test_result.skipped
            if verification.test_result else True
        ),
    })]

    return {
        "applies_cleanly": True,
        "verification_passed": passed,
        "verification_reason": rejection or llm_reason,
        "eval_score": llm_score,
        "failure_reasons": [] if passed else [rejection or "verification failed"],
        "completed_steps": ["verify"],
        "total_steps": state.get("total_steps", 0) + 1,
        "events": events,
    }


# ---------------------------------------------------------------------------
# Node: reflect  (reason LLM)
# ---------------------------------------------------------------------------

@_instrument_node("reflect")
def reflect_node(state: HarnessState) -> dict:
    """Reason LLM — self-critique of the failed patch.

    The reflection is a distilled, actionable critique — it replaces ALL prior
    reflections as the active guidance for the next generation attempt.
    Context compaction is applied: the reflector sees last _PROMPT_MAX_REFLECTIONS
    previous reflections, not the full unbounded list.
    """
    from src_bot.reflexion.reflector import generate_reflection
    from src_bot.llm.router import get_llm

    file_path = state["file_path"]
    bug_desc = f"{state['issue_text'][:400]}\n\nFault: {state['fault_desc']}"

    llm = get_llm(role="reason", temperature=0)
    try:
        # Context compaction: reflector receives only last N prior reflections
        prior = _tail(state["reflections"], _PROMPT_MAX_REFLECTIONS)
        reflection = generate_reflection(
            llm,
            bug_desc,
            state["patch"],
            state["verification_reason"],
            state["graph_context"],
            prior,
        )
    except Exception as e:
        logger.warning("[harness] reflect_node error: %s", e)
        reflection = f"Previous patch failed: {state['verification_reason']}"

    return {
        "reflections": [reflection],
        "completed_steps": ["reflect"],
        "total_steps": state.get("total_steps", 0) + 1,
        "events": [("phase", {
            "phase": 3, "file": file_path,
            "attempt": state["attempt"], "status": "reflecting",
        })],
    }


# ---------------------------------------------------------------------------
# Node: orchestrate  ← HARNESS CORE (GPT-4o dynamic routing hub)
# ---------------------------------------------------------------------------

def _resolve_final_decision(state: dict) -> str:
    """Determine accept_best vs abort when the orchestrator decides to finish."""
    if state.get("verification_passed"):
        return "accept_best"
    if state.get("applies_cleanly") and state.get("eval_score", 0) >= 2:
        return "accept_best"
    if state.get("patches_tried"):
        return "accept_best"  # best-effort: ship the last attempt
    return "abort"


def _fallback_action(state: dict, step_counts: dict) -> str:
    """Deterministic routing fallback when GPT is unavailable."""
    if not state.get("fault_desc"):
        return "analyze_fault"
    if not state.get("review_comments") and step_counts.get("code_review", 0) == 0:
        return "code_review"
    if not state.get("plan_contract") and step_counts.get("plan", 0) == 0:
        return "plan"
    if not state.get("patch"):
        return "generate"
    attempt = state.get("attempt", 0)
    max_retries = state.get("max_retries", 3)
    if not state.get("verification_passed") and attempt < max_retries:
        return "reflect" if state.get("patches_tried") else "generate"
    return "done"


_ORCHESTRATE_LABELS = {
    "analyze_fault": "🔍 analyze_fault",
    "code_review":   "👁 code_review",
    "plan":          "📋 plan",
    "generate":      "⚙️ generate",
    "verify":        "🧪 verify",
    "reflect":       "🔄 reflect",
    "done":          "✅ done",
}

_ORCHESTRATE_VALID = frozenset(_ORCHESTRATE_LABELS)


@_instrument_node("orchestrate")
def orchestrate_node(state: HarnessState) -> dict:
    """GPT-4o routing hub — decides next action at every junction.

    Runs first (entry point) and after every worker node. GPT sees the full
    pipeline state and picks one of 7 actions or "done". The prompt is purely
    descriptive — GPT is given the state, not instructions on how to route.

    Self-evaluation bias mitigation (article 1):
      - Deterministic pre-check: verified patch → auto-accept, no LLM call
      - Hard cap guard: total_steps >= _MAX_TOTAL_STEPS → force done
      - GPT sees WHAT has been done (step_counts), not HOW to do it
      - PlannerContract acceptance_criteria included when state has a contract
    """
    from src_bot.llm.router import get_llm
    from langchain_core.messages import HumanMessage

    file_path = state["file_path"]
    completed = state.get("completed_steps", [])
    total_steps = state.get("total_steps", 0)

    # Hard cap — prevent runaway loop
    if total_steps >= _MAX_TOTAL_STEPS:
        logger.warning(
            "[harness] Hard step cap (%d) reached for %s", total_steps, file_path
        )
        final_decision = _resolve_final_decision(state)
        return {
            "next_action": "done",
            "orchestrator_decision": final_decision,
            "events": [("orchestrate", {
                "file": file_path, "next_action": "done",
                "action_label": _ORCHESTRATE_LABELS["done"],
                "reason": "hard_cap", "final_decision": final_decision,
            })],
        }

    # Deterministic pre-check: skip LLM if patch already verified
    if state.get("verification_passed") and state.get("applies_cleanly"):
        logger.info("[harness] Orchestrator auto-accept (verified) for %s", file_path)
        return {
            "next_action": "done",
            "orchestrator_decision": "accept_best",
            "events": [("orchestrate", {
                "file": file_path, "next_action": "done",
                "action_label": _ORCHESTRATE_LABELS["done"],
                "auto": True, "final_decision": "accept_best",
                "message": f"patch verified score={state.get('eval_score', 0)}/5",
            })],
        }

    # Build step counts for GPT context
    step_counts: dict[str, int] = {}
    for s in completed:
        step_counts[s] = step_counts.get(s, 0) + 1

    fault_desc = state.get("fault_desc", "")
    patch = state.get("patch", "")
    plan_contract = state.get("plan_contract") or {}
    attempt = state.get("attempt", 0)
    max_retries = state.get("max_retries", 3)
    eval_score = state.get("eval_score", 0)

    criteria_hint = ""
    if plan_contract.get("acceptance_criteria"):
        criteria_hint = (
            "\nFix contract criteria:\n"
            + "\n".join(f"  - {c}" for c in plan_contract["acceptance_criteria"][:3])
            + "\n"
        )

    prompt = (
        f"You orchestrate a multi-agent code fix pipeline. Decide the NEXT single action.\n\n"
        f"FILE: {file_path}\n"
        f"ISSUE: {state['issue_text'][:300]}\n\n"
        f"PIPELINE STATE:\n"
        f"- Steps completed: {dict(step_counts) or 'none'}\n"
        f"- fault_desc: {'set (' + fault_desc[:100] + ')' if fault_desc else 'NOT YET'}\n"
        f"- review_comments: {len(state.get('review_comments') or [])} comments\n"
        f"- plan_contract: {'set' if plan_contract else 'NOT YET'}\n"
        f"- patch: {'generated (size={:d})'.format(len(patch)) if patch else 'NOT YET'}\n"
        f"- verification: {'✅ PASSED' if state.get('verification_passed') else ('❌ FAILED score=' + str(eval_score) + '/5 — ' + (state.get('verification_reason') or '')[:80] if patch else 'not run')}\n"
        f"- attempts: {attempt}/{max_retries}\n"
        f"- total_steps: {total_steps}/{_MAX_TOTAL_STEPS}\n"
        f"{criteria_hint}\n"
        f"AVAILABLE ACTIONS:\n"
        f"  analyze_fault  — identify exact fault (run {step_counts.get('analyze_fault', 0)}x; max 2x)\n"
        f"  code_review    — structured review comments (run {step_counts.get('code_review', 0)}x)\n"
        f"  plan           — create fix contract (run {step_counts.get('plan', 0)}x)\n"
        f"  generate       — write patch (run {step_counts.get('generate', 0)}x)\n"
        f"  verify         — test patch (run {step_counts.get('verify', 0)}x)\n"
        f"  reflect        — critique failed patch (run {step_counts.get('reflect', 0)}x)\n"
        f"  done           — finish pipeline\n\n"
        f"Reply with EXACTLY one keyword:"
    )

    llm = get_llm(role="chat", temperature=0)  # GPT-4o: orchestrator
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = (response.content or "").strip().lower().split()[0]
        action = raw if raw in _ORCHESTRATE_VALID else _fallback_action(state, step_counts)
    except Exception as e:
        logger.warning("[harness] orchestrate_node LLM error: %s — using fallback", e)
        action = _fallback_action(state, step_counts)

    final_decision = _resolve_final_decision(state) if action == "done" else ""

    logger.info(
        "[harness] orchestrator → %s (file=%s steps=%d attempts=%d)",
        action, file_path, total_steps, attempt,
    )

    return {
        "next_action": action,
        "orchestrator_decision": final_decision,
        "events": [("orchestrate", {
            "file": file_path,
            "next_action": action,
            "action_label": _ORCHESTRATE_LABELS.get(action, action),
            "step_counts": step_counts,
            "total_steps": total_steps,
            **({"final_decision": final_decision} if action == "done" else {}),
        })],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_from_orchestrate(state: HarnessState) -> str:
    """Hub router — translates GPT's next_action into a graph edge."""
    if state.get("total_steps", 0) >= _MAX_TOTAL_STEPS:
        return "done"
    action = state.get("next_action", "done")
    return action if action in _ORCHESTRATE_VALID else "done"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_graph():
    from langgraph.graph import StateGraph, END

    g = StateGraph(HarnessState)

    # Hub node — GPT-4o decides routing at every junction
    g.add_node("orchestrate", orchestrate_node)

    # Worker nodes
    g.add_node("analyze_fault", analyze_fault_node)
    g.add_node("code_review", code_review_node)
    g.add_node("plan", plan_node)
    g.add_node("generate", generate_node)
    g.add_node("verify", verify_node)
    g.add_node("reflect", reflect_node)

    # GPT orchestrator is the entry point and the routing hub
    g.set_entry_point("orchestrate")
    g.add_conditional_edges(
        "orchestrate",
        _route_from_orchestrate,
        {
            "analyze_fault": "analyze_fault",
            "code_review":   "code_review",
            "plan":          "plan",
            "generate":      "generate",
            "verify":        "verify",
            "reflect":       "reflect",
            "done":          END,
        },
    )

    # All worker nodes return to the orchestrate hub after completion
    for _node in ("analyze_fault", "code_review", "plan", "generate", "verify", "reflect"):
        g.add_edge(_node, "orchestrate")

    return g.compile()


_graph: Any = None  # compiled once at first call


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_file_review(
    issue_text: str,
    file_path: str,
    graph_context: str,
    workdir: Path,
    project_id: str,
    risk_level: str = "medium",
    max_retries: int = 3,
) -> HarnessResult:
    """
    Run the LangGraph harness for a single file.

    Called from reviewer.py via asyncio.run_in_executor (synchronous, in a
    thread). The function:
      1. Runs a pre-flight environment check (article 2 — session startup)
      2. Builds the initial LangGraph state (structured handoff artifact)
      3. Invokes the graph with a hard timeout (_GRAPH_TIMEOUT_SECONDS)
      4. Returns a HarnessResult with patch data + SSE events to emit
    """
    _RISK_RETRIES = {"low": 2, "medium": 3, "high": 5}
    resolved_retries = _RISK_RETRIES.get(risk_level, max_retries)

    # ── Pre-flight check (article 2: session startup checklist) ──────────
    err = _preflight_check(file_path, workdir)
    if err:
        logger.error("[harness] Pre-flight failed for %s: %s", file_path, err)
        return HarnessResult(
            fault_desc=f"Pre-flight check failed: {err}",
            events=[("phase", {"phase": 2, "file": file_path, "status": "preflight_failed", "error": err})],
        )

    initial: HarnessState = {
        "issue_text": issue_text,
        "file_path": file_path,
        "graph_context": graph_context,
        "workdir": str(workdir),
        "project_id": project_id,
        "risk_level": risk_level,
        "max_retries": resolved_retries,
        # accumulated (start empty)
        "reflections": [],
        "patches_tried": [],
        "failure_reasons": [],
        "completed_steps": [],
        "events": [("phase", {"phase": 2, "file": file_path, "status": "running"})],
        # mutable
        "fault_desc": "",
        "file_content": "",
        "review_comments": [],
        "plan_context": "",
        "plan_contract": {},
        "patch": "",
        "applies_cleanly": False,
        "verification_passed": False,
        "verification_reason": "",
        "eval_score": 0,
        "attempt": 0,
        "total_steps": 0,
        "reanalyze_count": 0,
        "next_action": "",
        "orchestrator_decision": "",
    }

    # ── Graph execution with hard timeout ──────────────────────────────────
    # Capture the current Python context so Langfuse/OTel span context set by
    # langfuse_span("harness_file_review") in reviewer.py is propagated into the
    # inner thread. Without this, all graph LLM calls appear as disconnected root
    # traces in Langfuse instead of children of the harness span.
    _ctx = contextvars.copy_context()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_ctx.run, _get_graph().invoke, initial)
            try:
                final = future.result(timeout=_GRAPH_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                future.cancel()
                logger.error(
                    "[harness] Graph timed out after %ds for %s",
                    _GRAPH_TIMEOUT_SECONDS, file_path,
                )
                return HarnessResult(
                    fault_desc=f"Harness timed out after {_GRAPH_TIMEOUT_SECONDS}s",
                    events=[("phase", {
                        "phase": 3, "file": file_path, "status": "timeout",
                        "message": f"Harness exceeded {_GRAPH_TIMEOUT_SECONDS}s budget",
                    })],
                )
    except Exception as e:
        logger.error("[harness] Graph execution failed for %s: %s", file_path, e)
        return HarnessResult(
            fault_desc=str(e),
            events=[("phase", {"phase": 3, "file": file_path, "status": "error", "error": str(e)})],
        )

    # ── Extract best patch from final state ───────────────────────────────
    patch = final.get("patch", "")
    applies = final.get("applies_cleanly", False)
    decision = final.get("orchestrator_decision", "")

    # Orchestrator said accept_best but the last attempted patch didn't apply —
    # try using any earlier patch that DID apply (stored in patches_tried).
    if not applies and decision == "accept_best":
        # We don't track which patch applied — best effort: use last tried
        patch = (final.get("patches_tried") or [""])[-1]

    return HarnessResult(
        patch=patch,
        applies_cleanly=applies,
        eval_score=final.get("eval_score", 0),
        eval_reason=final.get("verification_reason", ""),
        fault_desc=final.get("fault_desc", ""),
        plan_context=final.get("plan_context", ""),
        orchestrator_decision=decision,
        reflexion_attempts=final.get("attempt", 0),
        risk_level=risk_level,
        review_comments=final.get("review_comments", []),
        events=final.get("events", []),
    )
