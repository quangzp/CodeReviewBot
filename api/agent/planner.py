"""
Planner Agent — generates a structured contract before patch generation.

The Planner sits between Phase 2 (fault localization) and Phase 3 (patch
generation). It converts a fault description into a concrete plan that:
  - constrains the generator (must_preserve, approach)
  - gives the evaluator acceptance criteria to check against

This closes the harness gap where the generator "guesses" what to change
and the evaluator uses a generic rubric.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from src_bot.llm.router import get_llm

logger = logging.getLogger(__name__)


@dataclass
class PlannerContract:
    root_cause: str = ""
    approach: str = ""
    must_preserve: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    files_to_modify: list[str] = field(default_factory=list)
    risk_notes: str = ""

    def to_prompt_section(self) -> str:
        """Format for injection into the generator (Phase 3) prompt."""
        parts = [
            "## Pre-Iteration Contract (read before writing the patch)\n",
            f"**Root cause:** {self.root_cause}",
            f"**Approach:** {self.approach}",
        ]
        if self.must_preserve:
            parts.append("**Must preserve (do NOT change these):**")
            for item in self.must_preserve:
                parts.append(f"  - {item}")
        if self.acceptance_criteria:
            parts.append("**Acceptance criteria (your patch MUST satisfy ALL of these):**")
            for criterion in self.acceptance_criteria:
                parts.append(f"  - {criterion}")
        if self.risk_notes:
            parts.append(f"**Risk notes:** {self.risk_notes}")
        return "\n".join(parts)

    def to_evaluator_section(self) -> str:
        """Format for injection into the evaluator prompt — replaces generic rubric."""
        parts = [
            "## Contract to evaluate against\n",
            f"**Root cause:** {self.root_cause}",
            f"**Expected approach:** {self.approach}",
        ]
        if self.must_preserve:
            parts.append("**Must preserve (FAIL if any of these changed):**")
            for item in self.must_preserve:
                parts.append(f"  - {item}")
        if self.acceptance_criteria:
            parts.append("**Acceptance criteria:**")
            for criterion in self.acceptance_criteria:
                parts.append(f"  - {criterion}")
        parts.append(
            "\nScore 5 = satisfies ALL criteria, preserves everything in must_preserve\n"
            "Score 4 = satisfies most criteria, minor deviations\n"
            "Score 3 = partially meets criteria\n"
            "Score 2 = wrong approach, does not address root cause\n"
            "Score 1 = breaks must_preserve constraints or is completely unrelated"
        )
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_BUG_FIX_PROMPT = """\
You are a senior developer planning a targeted bug fix. Analyze the fault and create a structured plan.

Issue: {issue_text}
File: {file_path}
Fault description: {fault_description}
Graph context (callers / callees): {graph_context}

Reply with ONLY valid JSON — no markdown fences, no explanation:
{{
  "root_cause": "What is fundamentally wrong (1 sentence)",
  "approach": "How to fix it — strategy not code (1-2 sentences)",
  "must_preserve": ["function signatures / behaviors that MUST NOT change"],
  "acceptance_criteria": ["the patch MUST ...", "the patch MUST NOT ..."],
  "files_to_modify": ["{file_path}"],
  "risk_notes": "What could go wrong if the fix is too broad (1 sentence)"
}}"""

_REFACTOR_PROMPT = """\
You are a senior developer planning a code refactoring. Analyze the request and create a structured plan.

Refactoring request: {issue_text}
File: {file_path}
Detected code smells: {fault_description}
Graph context (callers / callees): {graph_context}

Reply with ONLY valid JSON — no markdown fences, no explanation:
{{
  "root_cause": "What structural problem needs to be addressed (1 sentence)",
  "approach": "How to refactor — strategy not code (1-2 sentences)",
  "must_preserve": ["all public APIs", "all function signatures", "behavior of callers"],
  "acceptance_criteria": ["behavior is identical before and after", "..."],
  "files_to_modify": ["{file_path}"],
  "risk_notes": "What could break if the refactoring is too aggressive (1 sentence)"
}}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_plan(
    fault_description: str,
    issue_text: str,
    file_path: str,
    file_content: str = "",
    graph_context: str = "",
    mode: str = "bug_fix",
) -> PlannerContract:
    """Generate a structured plan/contract before patch generation.

    Uses role='generation' (70b model) for best reasoning quality.
    Fallback: if LLM fails or JSON is malformed, returns a minimal contract
    with root_cause=fault_description so the pipeline never blocks.

    Args:
        fault_description: Phase 2 fault localization output (or smell context for refactor).
        issue_text: Original user issue / bug description.
        file_path: Target file path.
        file_content: Not used in prompt (kept for future ACI context window).
        graph_context: Callers/callees from Neo4j.
        mode: "bug_fix" or "refactor".

    Returns:
        PlannerContract with structured plan.
    """
    try:
        from langchain_core.messages import HumanMessage

        template = _REFACTOR_PROMPT if mode == "refactor" else _BUG_FIX_PROMPT
        prompt = template.format(
            issue_text=issue_text[:1500],
            file_path=file_path,
            fault_description=fault_description[:800],
            graph_context=graph_context[:600],
        )

        llm = get_llm(role="reason", temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown fences if the model added them
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return PlannerContract(
                root_cause=str(data.get("root_cause", fault_description[:200])),
                approach=str(data.get("approach", "")),
                must_preserve=list(data.get("must_preserve", [])),
                acceptance_criteria=list(data.get("acceptance_criteria", [])),
                files_to_modify=list(data.get("files_to_modify", [file_path])),
                risk_notes=str(data.get("risk_notes", "")),
            )

    except Exception as e:
        logger.warning("[planner] Failed to generate plan: %s", e)

    # Minimal fallback — never block the pipeline
    return PlannerContract(
        root_cause=fault_description[:200],
        approach="Fix the identified fault directly with a minimal change.",
        files_to_modify=[file_path],
    )


def generate_plan_sync(
    fault_description: str,
    issue_text: str,
    file_path: str,
    graph_context: str = "",
    mode: str = "bug_fix",
) -> PlannerContract:
    """Thin wrapper for use with loop.run_in_executor (no file_content needed)."""
    return generate_plan(
        fault_description=fault_description,
        issue_text=issue_text,
        file_path=file_path,
        graph_context=graph_context,
        mode=mode,
    )
