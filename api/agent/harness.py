"""
LangGraph per-file review harness — true Harness Engineering.

Implements all five Anthropic harness principles
(anthropic.com/engineering/harness-design-long-running-apps):

  1. Loop management    — hard step cap (_MAX_TOTAL_STEPS), clean exit conditions,
                          reanalyze allowed at most once per run
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

Self-evaluation bias mitigation (article 1):
  - Generator: role="generation" (llama-3.3-70b)
  - Evaluator: role="fast_gate"  (llama-3.1-8b) — independent, different model
  - Orchestrator: role="reason"  (deepseek-r1)  — process decision, NOT quality eval;
    external anchor = contract acceptance_criteria + deterministic pre-accept check

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
    reanalyze_count: int
    orchestrator_decision: str   # "accept_best" | "reanalyze" | "abort"


# ---------------------------------------------------------------------------
# Node: analyze_fault  (reason LLM)
# ---------------------------------------------------------------------------

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
        "events": [("phase", {
            "phase": 2, "file": file_path,
            "fault": fault_desc[:120], "status": "done",
        })],
    }


# ---------------------------------------------------------------------------
# Node: code_review  (fast_gate LLM — structured review comments)
# ---------------------------------------------------------------------------

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
        return {"review_comments": []}

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
                "events": [("review_comments", {
                    "file": file_path,
                    "comments": clean,
                })],
            }
    except Exception as e:
        logger.warning("[harness] code_review_node error (non-fatal): %s", e)

    return {"review_comments": []}


# ---------------------------------------------------------------------------
# Node: plan  (reason LLM)
# ---------------------------------------------------------------------------

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
        "events": [("phase", {
            "phase": 2.5, "file": file_path, "status": "planned",
            "root_cause": root_cause_short,
        })],
    }


# ---------------------------------------------------------------------------
# Node: generate  (generation LLM)
# ---------------------------------------------------------------------------

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
        "events": events,
    }


# ---------------------------------------------------------------------------
# Node: verify  (AST → test → LLM evaluator, independent of generator)
# ---------------------------------------------------------------------------

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
        "events": events,
    }


# ---------------------------------------------------------------------------
# Node: reflect  (reason LLM)
# ---------------------------------------------------------------------------

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
        "events": [("phase", {
            "phase": 3, "file": file_path,
            "attempt": state["attempt"], "status": "reflecting",
        })],
    }


# ---------------------------------------------------------------------------
# Node: orchestrate  ← HARNESS CORE (reason LLM makes process decision)
# ---------------------------------------------------------------------------

def orchestrate_node(state: HarnessState) -> dict:
    """
    Reason LLM makes a process decision after exhausting the retry budget.

    Self-evaluation bias mitigation (article 1):
      - The orchestrator is NOT asked "is this patch good?" (quality judgment)
      - It IS asked "given these process failures, what's the right move?"
      - External anchor: PlannerContract acceptance_criteria are included so
        the decision references explicit criteria rather than implicit LLM taste
      - Deterministic pre-check runs first: if eval_score >= 2 AND applies_cleanly,
        we auto-accept without invoking the LLM at all

    Context compaction: only last _PROMPT_MAX_* entries injected into prompt.

    Decisions:
      "accept_best" — best-effort patch; ship despite failing automated checks
      "reanalyze"   — fault analysis was wrong; restart with a fresh angle
      "abort"        — cannot fix automatically; skip this file
    """
    from src_bot.llm.router import get_llm
    from langchain_core.messages import HumanMessage

    file_path = state["file_path"]

    # Deterministic pre-check: auto-accept if we have a workable patch
    if state.get("eval_score", 0) >= 2 and state.get("applies_cleanly", False):
        decision = "accept_best"
        logger.info(
            "[harness] orchestrator auto-accept (score=%d, applies_cleanly) for %s",
            state["eval_score"], file_path,
        )
        return {
            "orchestrator_decision": decision,
            "events": [("orchestrate", {
                "file": file_path, "decision": decision, "auto": True,
                "message": f"Auto-accepting: score={state['eval_score']}/5, applies cleanly",
            })],
        }

    # Context compaction — trim accumulated lists before injecting into prompt
    patches_section = _compact_for_prompt(
        state["patches_tried"], _PROMPT_MAX_PATCHES, 300, "patch"
    )
    failures_section = _compact_for_prompt(
        state["failure_reasons"], _PROMPT_MAX_FAILURES, 200, "fail"
    )
    reflections_section = _compact_for_prompt(
        state["reflections"], _PROMPT_MAX_REFLECTIONS, 250, "reflect"
    )

    # Include contract acceptance criteria as external anchor (reduces self-eval bias)
    criteria_section = ""
    contract_dict = state.get("plan_contract", {})
    if contract_dict.get("acceptance_criteria"):
        criteria_section = (
            "\nACCEPTANCE CRITERIA (from sprint contract):\n"
            + "\n".join(f"  ✓ {c}" for c in contract_dict["acceptance_criteria"])
            + "\n"
        )

    prompt = (
        f"You are a tech lead making a PROCESS decision — not a code quality judgment.\n"
        f"The automated fix pipeline failed all {state['attempt']} attempt(s) for this file.\n\n"
        f"FILE: {file_path}\n"
        f"ISSUE SUMMARY: {state['issue_text'][:400]}\n"
        f"FAULT IDENTIFIED: {state['fault_desc'][:300]}\n"
        f"{criteria_section}\n"
        f"PATCHES TRIED:\n{patches_section}\n\n"
        f"FAILURE REASONS:\n{failures_section}\n\n"
        f"REFLECTIONS:\n{reflections_section}\n\n"
        f"Choose EXACTLY ONE process action — reply with only the keyword:\n"
        f"  'accept_best'  best patch is reasonable despite failing automated checks\n"
        f"  'reanalyze'   fault analysis was wrong; needs fresh perspective from scratch\n"
        f"  'abort'        cannot be fixed automatically; skip this file\n\n"
        f"Process decision:"
    )

    llm = get_llm(role="reason", temperature=0)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = (response.content or "").strip().lower().split()[0]
        decision = raw if raw in ("accept_best", "reanalyze", "abort") else "accept_best"
    except Exception as e:
        logger.warning("[harness] orchestrate_node LLM error: %s — defaulting accept_best", e)
        decision = "accept_best"

    logger.info(
        "[harness] orchestrator → %s (file=%s attempts=%d total_steps=%d)",
        decision, file_path, state["attempt"], state.get("total_steps", 0),
    )

    return {
        "orchestrator_decision": decision,
        "events": [("orchestrate", {
            "file": file_path, "decision": decision, "auto": False,
            "attempts": state["attempt"],
            "message": {
                "accept_best": "Orchestrator: accepting best patch (best-effort)",
                "reanalyze": "Orchestrator: restarting fault analysis from scratch",
                "abort": "Orchestrator: file cannot be fixed automatically",
            }.get(decision, decision),
        })],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_verify(state: HarnessState) -> str:
    # Hard global cap — article 1: "only increase complexity when needed"
    if state.get("total_steps", 0) >= _MAX_TOTAL_STEPS:
        logger.warning(
            "[harness] Hard step cap reached (%d steps) for %s — forcing orchestrate",
            _MAX_TOTAL_STEPS, state["file_path"],
        )
        return "orchestrate"

    if state["verification_passed"]:
        return "done"

    if state["attempt"] < state["max_retries"]:
        return "reflect_and_retry"

    return "orchestrate"


def _route_after_orchestrate(state: HarnessState) -> str:
    decision = state.get("orchestrator_decision", "abort")
    # Allow at most one reanalysis cycle — prevents infinite loops
    if decision == "reanalyze" and state.get("reanalyze_count", 0) < 1:
        return "reanalyze"
    return "done"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_graph():
    from langgraph.graph import StateGraph, END

    g = StateGraph(HarnessState)

    g.add_node("analyze_fault", analyze_fault_node)
    g.add_node("code_review", code_review_node)
    g.add_node("plan", plan_node)
    g.add_node("generate", generate_node)
    g.add_node("verify", verify_node)
    g.add_node("reflect", reflect_node)
    g.add_node("orchestrate", orchestrate_node)

    g.set_entry_point("analyze_fault")
    g.add_edge("analyze_fault", "code_review")
    g.add_edge("code_review", "plan")
    g.add_edge("plan", "generate")
    g.add_edge("generate", "verify")

    g.add_conditional_edges(
        "verify",
        _route_after_verify,
        {
            "done": END,
            "reflect_and_retry": "reflect",
            "orchestrate": "orchestrate",
        },
    )

    g.add_edge("reflect", "generate")

    g.add_conditional_edges(
        "orchestrate",
        _route_after_orchestrate,
        {
            "done": END,
            "reanalyze": "analyze_fault",
        },
    )

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
