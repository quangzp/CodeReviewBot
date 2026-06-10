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
from groq import Groq

from src_bot.config.config import configs


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

Classify the user message into ONE of:
- IN_SCOPE   : clearly relates to one of the 6 capabilities above
- OUT_SCOPE  : general programming questions, explanations, small talk, anything else
- CLARIFY    : relates to the capabilities but is missing required info

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
"What projects do I have?"                             -> IN_SCOPE
"Reindex elastalert2"                                  -> IN_SCOPE
"What is GraphRAG?"                                    -> OUT_SCOPE
"How do I write a binary search?"                      -> OUT_SCOPE
"Tell me a joke"                                       -> OUT_SCOPE
"Fix the bug"                              -> CLARIFY (missing: repo name and bug description)
"Review the PR"                            -> CLARIFY (missing: PR URL or PR number and repo name)
"Refactor the code"                        -> CLARIFY (missing: repo name and what to refactor)

User message: REPLACE_MESSAGE"""


OUT_SCOPE_REPLY = (
    "I can only help with:\n"
    "- **Adding** a GitHub project\n"
    "- **Reviewing** a pull request\n"
    "- **Fixing** a bug in an indexed project\n"
    "- **Refactoring** code in an indexed project\n"
    "- **Checking** project or review status\n\n"
    "What would you like to do?"
)


def classify(message: str) -> tuple[Intent, str]:
    """
    Classify user intent using llama-3.1-8b-instant (fast, free).

    Returns (intent, clarification_hint).
    clarification_hint is non-empty only when intent == CLARIFY.
    """
    try:
        client = Groq(api_key=configs.GROQ_API_KEY)
        prompt = CLASSIFIER_PROMPT_TEMPLATE.replace("REPLACE_MESSAGE", message[:500])
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # always use the small fast model
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60,
        )
        raw = response.choices[0].message.content.strip()

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
