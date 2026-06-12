from __future__ import annotations

"""
Self-Reflection model (M_sr) — generates verbal feedback after a failed patch attempt.

The reflection is stored in episodic memory and provided to the Actor
on the next retry, enabling the agent to learn from its mistakes
within a single task (no weight updates needed).

Based on: "Reflexion: Language Agents with Verbal Reinforcement Learning"
(Shinn et al., 2023)
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel


REFLECTION_PROMPT = """You are a debugging assistant analyzing why a code patch FAILED.

ORIGINAL BUG:
{bug_description}

PATCH THAT WAS APPLIED:
```diff
{failed_patch}
```

TEST OUTPUT (why it failed):
{test_output}

GRAPH CONTEXT (callers and dependencies):
{graph_context}

PREVIOUS REFLECTIONS:
{previous_reflections}

Analyze the failure and write a concise self-reflection:
1. What specifically went wrong with this patch?
2. Which callers or dependencies were broken (use graph context)?
3. What concrete change should be made in the next attempt?

Be specific. Reference exact function names and line numbers.
Keep your reflection under 200 words.
"""


def generate_reflection(
    llm: BaseChatModel,
    bug_description: str,
    failed_patch: str,
    test_output: str,
    graph_context: str,
    previous_reflections: list[str] | None = None,
) -> str:
    """
    Generate a verbal self-reflection after a failed patch attempt.

    This reflection is stored in memory and provided to the patch generator
    on the next retry, acting as a 'semantic gradient' that guides the
    agent toward a correct fix.

    Args:
        llm: The language model for self-reflection.
        bug_description: The original bug description.
        failed_patch: The unified diff that failed.
        test_output: Combined stdout+stderr from the failed test run.
        graph_context: Graph relationships for understanding blast radius.
        previous_reflections: Reflections from earlier attempts (episodic memory).

    Returns:
        A verbal self-reflection string.
    """
    prev_str = ""
    if previous_reflections:
        for i, ref in enumerate(previous_reflections, 1):
            prev_str += f"\n--- Reflection from attempt {i} ---\n{ref}\n"
    else:
        prev_str = "This is the first attempt."

    filled = REFLECTION_PROMPT.format(
        bug_description=bug_description,
        failed_patch=failed_patch,
        test_output=test_output[:2000],
        graph_context=graph_context or "No graph context available.",
        previous_reflections=prev_str,
    )

    response = llm.invoke(filled)
    return response.content
