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

from groq import Groq
from src_bot.config.config import configs

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


def evaluate_patch(bug_description: str, patch: str) -> tuple[int, str]:
    """
    Evaluate patch quality using a fast LLM.

    Returns (score 1-5, reason).
    score >= 3 = acceptable, < 3 = reject and retry.
    Falls back to score=3 if evaluation fails (fail open).
    """
    if not patch or not patch.strip():
        return 1, "Empty patch"

    try:
        client = Groq(api_key=configs.GROQ_API_KEY)
        prompt = (
            EVALUATOR_PROMPT
            .replace("REPLACE_BUG", bug_description[:500])
            .replace("REPLACE_PATCH", patch[:1500])
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
        )
        import json, re
        raw = response.choices[0].message.content.strip()
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = int(data.get("score", 3))
            reason = data.get("reason", "")
            return max(1, min(5, score)), reason
    except Exception:
        pass

    return 3, "Evaluation unavailable (fail open)"
