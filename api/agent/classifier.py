"""
Intent classifier — lightweight pre-check before the main agent runs.

Runs FIRST on every user message using a small, fast model (llama-3.1-8b-instant).
Categorises intent into one of three buckets:

  IN_SCOPE   — add project, review PR, fix bug, list/status queries
  OUT_SCOPE  — anything unrelated (general coding questions, chat, etc.)
  CLARIFY    — in-scope but missing required info (which project? which bug?)

Only IN_SCOPE messages reach the main agent LLM, saving tokens and making
the refusal boundary much harder to bypass via prompt injection.
"""
from __future__ import annotations

from enum import Enum

from src_bot.config.config import configs
from src_bot.llm.router import get_llm


class Intent(str, Enum):
    IN_SCOPE  = "IN_SCOPE"
    OUT_SCOPE = "OUT_SCOPE"
    CLARIFY   = "CLARIFY"


CLASSIFIER_PROMPT_TEMPLATE = """You are an intent classifier for a code review bot.

The bot has EXACTLY these capabilities:
1. Add / onboard a GitHub project (user provides a GitHub URL or repo name)
2. Review a pull request (user provides a PR URL or PR number + repo)
3. Fix a bug in a project (user describes a bug + repo name)
4. Refactor code in a project (user asks to improve, clean up, restructure, or refactor code)
5. List projects, check project status, reindex a project
6. Show recent reviews or check review status
7. Explore an onboarded project: explain repository structure, frameworks, libraries,
   entrypoints, architecture, data/request flow, or how modules fit together

Classify the user message into ONE of:
- IN_SCOPE   : clearly relates to one of the capabilities above
- OUT_SCOPE  : general programming questions, explanations, small talk, anything else
- CLARIFY    : relates to the capabilities but is missing required info AND cannot be inferred from context

IMPORTANT: If recent conversation provides context (e.g. a project was just added or a review
was just started), short follow-up questions like "Is it done?", "What's the status?",
"How's it going?", "Done yet?", "Is it ready?" are IN_SCOPE — the agent can infer the subject
from history. Only mark CLARIFY if context genuinely does NOT provide the missing information.

Reply with ONLY a JSON object, nothing else:
- In scope: {"intent": "IN_SCOPE"}
- Out of scope: {"intent": "OUT_SCOPE"}
- Needs clarification: {"intent": "CLARIFY", "missing": "what info is needed"}

Examples:
"Add github.com/django/django"                          -> IN_SCOPE
"Review PR https://github.com/x/y/pull/1"              -> IN_SCOPE
"Fix the null pointer bug in elastalert2"               -> IN_SCOPE
"Refactor kibana_discover.py in elastalert-jertel"      -> IN_SCOPE
"Clean up the auth module in my project"               -> IN_SCOPE
"Improve code quality in nicholasgibson2/elastalert"   -> IN_SCOPE
"Explain the structure of owner/repo"                  -> IN_SCOPE
"Project owner/repo uses which frameworks?"            -> IN_SCOPE
"Luồng request trong owner/repo chạy như thế nào?"     -> IN_SCOPE
"Các thư viện chính của project này là gì?"            -> IN_SCOPE
"What projects do I have?"                             -> IN_SCOPE
"Reindex elastalert2"                                  -> IN_SCOPE
"What is GraphRAG?"                                    -> OUT_SCOPE unless asking how THIS repo uses it
"How do I write a binary search?"                      -> OUT_SCOPE
"Tell me a joke"                                       -> OUT_SCOPE
"Fix the bug"  (no prior context)      -> CLARIFY (missing: repo name and bug description)
"Review the PR" (no prior context)     -> CLARIFY (missing: PR URL or PR number and repo name)
"Refactor the code" (no prior context) -> CLARIFY (missing: repo name and what to refactor)
"Explain the architecture" (no prior context) -> CLARIFY (missing: repo name)
"Is it done?" (after adding a project) -> IN_SCOPE  (subject clear from context)
"How's it going?" (during indexing)    -> IN_SCOPE  (subject clear from context)
"Done yet?" (after review started)     -> IN_SCOPE  (subject clear from context)

REPLACE_CONTEXT
User message: REPLACE_MESSAGE"""


OUT_SCOPE_REPLY = (
    "I can only help with:\n"
    "- **Adding** a GitHub project\n"
    "- **Reviewing** a pull request\n"
    "- **Fixing** a bug in an indexed project\n"
    "- **Refactoring** code in an indexed project\n"
    "- **Exploring** an indexed project's structure, libraries, frameworks, and flow\n"
    "- **Checking** project or review status\n\n"
    "What would you like to do?"
)


def classify(message: str, history: list[dict] | None = None) -> tuple[Intent, str]:
    """
    Classify user intent using llama-3.1-8b-instant (fast, free).

    Accepts optional history (last N turns) so short follow-ups like
    "Is it done?" are resolved as IN_SCOPE when context makes the
    subject obvious.

    Returns (intent, clarification_hint).
    clarification_hint is non-empty only when intent == CLARIFY.
    """
    try:
        from langchain_core.messages import HumanMessage

        # Build a short context block from the last 3 turns so the classifier
        # can resolve ambiguous follow-ups without needing the full history.
        context_block = ""
        if history:
            recent = history[-6:]  # last 3 user+assistant pairs at most
            lines = ["Recent conversation:"]
            for turn in recent:
                role = turn.get("role", "user")
                content = str(turn.get("content", ""))[:200]
                lines.append(f"[{role}]: {content}")
            lines.append("")
            context_block = "\n".join(lines) + "\n"

        prompt = (
            CLASSIFIER_PROMPT_TEMPLATE
            .replace("REPLACE_CONTEXT", context_block)
            .replace("REPLACE_MESSAGE", message[:500])
        )
        llm = get_llm(role="fast_gate", temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Parse JSON response
        import json, re
        # Extract JSON even if the model adds extra text
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            intent_str = data.get("intent", "IN_SCOPE").upper()
            intent = Intent(intent_str) if intent_str in Intent._value2member_map_ else Intent.IN_SCOPE
            missing = data.get("missing", "")
            return intent, missing

    except Exception:
        # If classifier fails for any reason, let it through (fail open)
        pass

    return Intent.IN_SCOPE, ""
