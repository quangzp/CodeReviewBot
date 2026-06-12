"""
Patch quality evaluator — the "Evaluator" in Planner → Generator → Evaluator.

Harness Engineering principle: don't rely on a single model pass.
After the Generator produces a patch, the Evaluator independently checks:

  1. Does the patch actually address the stated fault?
  2. Is the change minimal and focused (not over-engineering)?
  3. Are there obvious regressions introduced?

Uses llama-3.1-8b-instant (fast, free) so it adds minimal latency.
Returns a score 1-5 and a reason. Patches scoring < 3 are rejected
and trigger another reflexion attempt with the evaluator's feedback.
"""
from __future__ import annotations

from src_bot.config.config import configs
from src_bot.llm.router import get_llm

EVALUATOR_PROMPT = """You are a code patch quality evaluator. A developer wrote a patch to fix a bug.
Your job is to score this patch from 1 to 5.

BUG DESCRIPTION:
REPLACE_BUG

PATCH:
REPLACE_PATCH

Score the patch:
5 = Directly fixes the stated bug, minimal change, no obvious regressions
4 = Fixes the bug but slightly over-engineered or minor style issues
3 = Partially addresses the bug, may miss edge cases
2 = Wrong approach, does not fix the root cause
1 = Completely wrong, introduces regressions or is unrelated to the bug

Reply with ONLY this JSON (no explanation):
{"score": <1-5>, "reason": "<one sentence>"}"""

REFACTOR_EVALUATOR_PROMPT = """You are a code refactoring quality evaluator. Score the refactoring patch from 1 to 5.

REFACTORING REQUEST:
REPLACE_BUG

PATCH:
REPLACE_PATCH

Score using ALL five criteria:
1. Behavior preservation (most important): Does the patch preserve ALL existing behavior?
2. Readability improvement: Does the change improve code clarity?
3. Complexity reduction: Does it reduce cyclomatic complexity or nesting depth?
4. Backward compatibility: Are function signatures, return types, and public APIs unchanged?
5. Minimality: Is the change focused on what was requested, not over-engineered?

Scoring:
5 = Preserves ALL behavior, clearly improves readability/structure, minimal change, APIs unchanged
4 = Good refactoring, minor style issues, or slightly broader than requested
3 = Partially addresses the request, may have minor behavior drift or miss some scope
2 = Changes behavior unexpectedly, breaks an API signature, or does not address the request
1 = Breaks functionality, introduces bugs, or is completely unrelated

Reply with ONLY this JSON (no explanation):
{"score": <1-5>, "reason": "<one sentence>"}"""


CONTRACT_EVALUATOR_PROMPT = """You are a code patch quality evaluator checking against a specific plan.

BUG/REFACTOR DESCRIPTION:
REPLACE_BUG

PATCH:
REPLACE_PATCH

REPLACE_CONTRACT

Reply with ONLY this JSON (no explanation):
{"score": <1-5>, "reason": "<one sentence>"}"""


def evaluate_patch_with_contract(
    bug_description: str,
    patch: str,
    contract: "PlannerContract",  # type: ignore[name-defined]
    mode: str = "bug_fix",
    llm_role: str = "fast_gate",
) -> tuple[int, str]:
    """Evaluate patch compliance against a specific PlannerContract.

    llm_role: which model evaluates. Default 'fast_gate' (8B) is independent
    from the 'generation' (70B) model that wrote the patch — this separation
    is the key anti-self-evaluation-bias mechanism (Anthropic article 1).

    Falls back to generic evaluate_patch() if contract evaluation fails.
    """
    if not patch or not patch.strip():
        return 1, "Empty patch"

    try:
        from langchain_core.messages import HumanMessage
        import json, re

        prompt = (
            CONTRACT_EVALUATOR_PROMPT
            .replace("REPLACE_BUG", bug_description[:500])
            .replace("REPLACE_PATCH", patch[:1500])
            .replace("REPLACE_CONTRACT", contract.to_evaluator_section())
        )
        llm = get_llm(role=llm_role, temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = int(data.get("score", 3))
            reason = data.get("reason", "")
            return max(1, min(5, score)), reason
    except Exception:
        pass

    # Fallback to generic evaluator
    return evaluate_patch(bug_description, patch, mode, llm_role=llm_role)


def evaluate_patch(
    bug_description: str,
    patch: str,
    mode: str = "bug_fix",
    llm_role: str = "fast_gate",
) -> tuple[int, str]:
    """Evaluate patch quality using a fast, independent LLM.

    Args:
        mode:     "bug_fix" or "refactor" — selects the scoring rubric.
        llm_role: model role for evaluation. Default 'fast_gate' (8B) is
                  independent from 'generation' (70B) that wrote the patch.

    Returns (score 1-5, reason).
    score >= 3 = acceptable, < 3 = reject and retry.
    Falls back to score=3 if evaluation fails (fail open).
    """
    if not patch or not patch.strip():
        return 1, "Empty patch"

    try:
        from langchain_core.messages import HumanMessage
        import json, re
        template = REFACTOR_EVALUATOR_PROMPT if mode == "refactor" else EVALUATOR_PROMPT
        prompt = (
            template
            .replace("REPLACE_BUG", bug_description[:500])
            .replace("REPLACE_PATCH", patch[:1500])
        )
        llm = get_llm(role=llm_role, temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = int(data.get("score", 3))
            reason = data.get("reason", "")
            return max(1, min(5, score)), reason
    except Exception:
        pass

    return 3, "Evaluation unavailable (fail open)"
