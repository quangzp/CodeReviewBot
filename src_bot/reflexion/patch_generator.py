from __future__ import annotations

"""
Patch Generator — Actor (M_a) in the Reflexion framework.

Generates unified diffs to fix bugs identified during code review.
Uses graph context (callers, callees) to produce patches that don't
break downstream dependencies.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

PATCH_SYSTEM_PROMPT = """You are a senior Python developer fixing a bug in a codebase.

GRAPH CONTEXT (callers and dependencies of the buggy code):
{graph_context}

BUG DESCRIPTION (from review):
{bug_description}

ORIGINAL FILE CONTENT:
{file_content}

PREVIOUS FAILED ATTEMPTS AND REFLECTIONS:
{reflections}

INSTRUCTIONS:
1. Analyze the bug using the graph context to understand which functions call this code.
2. Generate a MINIMAL fix — change only what is necessary.
3. Ensure your fix does not break any of the callers listed in the graph context.
4. Return ONLY a unified diff (patch format). No explanation needed.

Format your response as:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-old line
+new line
 context line
```
"""

FEW_SHOT_EXAMPLES = """
Example 1:
Bug: datetime.now() returns wrong timezone in user profile
Graph context: UserProfile.save() calls get_current_time() in utils/time.py
Fix:
```diff
--- a/utils/time.py
+++ b/utils/time.py
@@ -5,7 +5,7 @@
 from datetime import datetime, timezone

 def get_current_time():
-    return datetime.now()
+    return datetime.now(tz=timezone.utc)
```

Example 2:
Bug: Missing null check causes AttributeError when user has no email
Graph context: send_notification() calls user.get_email(), which is called by OrderService.confirm()
Fix:
```diff
--- a/services/notification.py
+++ b/services/notification.py
@@ -12,6 +12,8 @@
 def send_notification(user, message):
     email = user.get_email()
+    if email is None:
+        return False
     send_email(email, message)
+    return True
```
"""


def generate_patch(
    llm: BaseChatModel,
    bug_description: str,
    file_content: str,
    graph_context: str,
    reflections: list[str] | None = None,
) -> str:
    """
    Generate a unified diff patch to fix the identified bug.

    Args:
        llm: The language model to use for patch generation.
        bug_description: Description of the bug from the review step.
        file_content: The original file content to patch.
        graph_context: Graph relationships (callers, callees) for context.
        reflections: Previous reflection memories from failed attempts.

    Returns:
        A unified diff string.
    """
    reflections_str = ""
    if reflections:
        for i, ref in enumerate(reflections, 1):
            reflections_str += f"\n--- Attempt {i} reflection ---\n{ref}\n"
    else:
        reflections_str = "No previous attempts."

    prompt_text = FEW_SHOT_EXAMPLES + "\n\n" + PATCH_SYSTEM_PROMPT

    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm

    response = chain.invoke({
        "graph_context": graph_context or "No graph context available.",
        "bug_description": bug_description,
        "file_content": file_content,
        "reflections": reflections_str,
    })

    return _extract_diff(response.content)


def _extract_diff(response_text: str) -> str:
    """Extract the unified diff block from the LLM response."""
    if "```diff" in response_text:
        start = response_text.index("```diff") + len("```diff")
        end = response_text.index("```", start)
        return response_text[start:end].strip()
    elif "--- a/" in response_text:
        # Try to extract raw diff
        lines = response_text.split("\n")
        diff_lines = []
        in_diff = False
        for line in lines:
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                in_diff = True
            if in_diff:
                diff_lines.append(line)
        return "\n".join(diff_lines)
    else:
        return response_text
