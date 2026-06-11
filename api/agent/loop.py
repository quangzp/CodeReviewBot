"""
Agent loop — manages a conversational turn.

Flow:
  1. Receive user message + history
  2. LLM sees tool catalog, decides which (if any) to call
  3. Execute tool(s) — emit tool_call + tool_result events
  4. LLM produces final natural-language reply (streamed)
  5. Emit done event

The output is a stream of events the chat UI renders:
  - thinking          → "typing" indicator
  - tool_call         → "🛠 calling add_project(...)"
  - tool_result       → rich render (project card, diff viewer, etc.)
  - message_chunk     → token-by-token text
  - done              → end of turn
  - error             → fatal error
"""
from __future__ import annotations

import json
import asyncio
import re
from typing import AsyncGenerator, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src_bot.config.config import configs
from src_bot.llm.router import get_llm
from api.agent.tools import build_tool_registry, TOOL_DESCRIPTIONS

READ_ONLY_TOOLS = {"list_projects", "project_status", "recent_reviews"}


# ============================================================================
# System prompt
# ============================================================================
SYSTEM_PROMPT = """You are CodeReviewBot — a focused assistant with ONE job: help engineers manage projects, review pull requests, fix bugs, and refactor code in GitHub repositories.

You have access to these tools:

{tool_catalog}

# Strict scope — what you DO

1. **Add / list / reindex projects** — onboard a GitHub repo so it can be reviewed
2. **Review a pull request** — run the full GraphRAG + Reflexion review pipeline on a PR
3. **Fix a bug** — locate and patch a bug described in natural language
4. **Refactor code** — improve code structure/readability without changing behavior
5. **Check review / project status** — report progress on ongoing tasks

That is ALL. You do not answer general programming questions, explain concepts, write code
for the user, discuss architecture, or engage in any conversation outside the above scope.

# Strict scope — what you DO NOT do

- Do NOT answer questions like "What is GraphRAG?", "How does Python work?", "What is a bug?"
- Do NOT write code, explain algorithms, or give advice unrelated to managing projects/reviews
- Do NOT engage in small talk, greetings beyond a single line, or off-topic chat
- Do NOT speculate about reviews that haven't run yet

When a user asks something outside scope, respond with EXACTLY this and nothing more:
"I'm only able to help with adding projects, reviewing pull requests, fixing bugs, and refactoring code. What would you like to do?"

# How to call tools

Output tool calls in this EXACT format (one tool per response):

<tool_call>
{{"name": "tool_name", "args": {{"arg1": "value", "arg2": "value"}}}}
</tool_call>

After a tool runs, write 1-2 short sentences interpreting the result. Don't restate the data — the UI shows it visually.

# IMPORTANT: After a tool returns a successful result (status: success), write ONE short
# confirmation sentence and stop. Do NOT call the same tool again.

# Examples

User: "Add github.com/jertel/elastalert2"
You: <tool_call>{{"name": "add_project", "args": {{"repo_url": "https://github.com/jertel/elastalert2"}}}}</tool_call>

User: "Review PR https://github.com/jertel/elastalert2/pull/1763"
You: <tool_call>{{"name": "review_pr", "args": {{"pr_url": "https://github.com/jertel/elastalert2/pull/1763"}}}}</tool_call>

User: "Fix the timezone bug in parse_deadline() in elastalert2"
You: <tool_call>{{"name": "fix_bug", "args": {{"repo_name": "jertel/elastalert2", "bug_description": "parse_deadline() returns wrong timezone, should return UTC"}}}}</tool_call>

User: "Refactor the URL builder in nicholasgibson2/elastalert-jertel to reduce duplication"
You: <tool_call>{{"name": "refactor_code", "args": {{"repo_name": "nicholasgibson2/elastalert-jertel", "refactor_description": "Reduce duplication in the URL builder functions"}}}}</tool_call>

User: "Fix the bug" (ambiguous)
You: Which project and what bug? Please share the repo name and a short description.

User: "What is GraphRAG?"
You: I'm only able to help with adding projects, reviewing pull requests, fixing bugs, and refactoring code. What would you like to do?

User: "What projects do I have?"
You: <tool_call>{{"name": "list_projects", "args": {{}}}}</tool_call>
"""


def _format_tool_catalog() -> str:
    """Format tools for the system prompt."""
    lines = []
    for t in TOOL_DESCRIPTIONS:
        lines.append(f"## {t['name']}")
        lines.append(t["description"])
        params = t.get("parameters", {}).get("properties", {})
        if params:
            lines.append("Arguments:")
            required = set(t.get("parameters", {}).get("required", []))
            for arg_name, arg_schema in params.items():
                req = " (required)" if arg_name in required else " (optional)"
                lines.append(f"  - {arg_name}{req}: {arg_schema.get('description', '')}")
        lines.append("")
    return "\n".join(lines)


# ============================================================================
# Tool call parsing
# ============================================================================
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def _parse_tool_call(text: str) -> Optional[dict]:
    """Extract first tool_call from text, or None."""
    match = _TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


# ============================================================================
# Agent turn — runs one user → assistant exchange
# ============================================================================
async def run_agent_turn(
    user_message: str,
    history: list[dict],
    max_steps: int = 4,
) -> AsyncGenerator[dict, None]:
    """
    Run one agent turn. Yields events as JSON-serializable dicts.

    Args:
        user_message: current user input
        history: previous messages [{"role": "user"|"assistant", "content": "..."}]
        max_steps: max tool-call iterations before forcing a final answer

    Yields:
        {"type": "thinking"}
        {"type": "tool_call", "name": ..., "args": ...}
        {"type": "tool_result", "name": ..., "summary": ..., "render": ...}
        {"type": "message", "text": ...}
        {"type": "done"}
        {"type": "error", "message": ...}
    """
    try:
        # ----------------------------------------------------------------
        # 0. Intent classification — runs BEFORE the main agent LLM
        #    Uses llama-3.1-8b-instant (fast, free) to gate out-of-scope
        #    messages without wasting tokens on the main model.
        # ----------------------------------------------------------------
        from api.agent.classifier import classify, Intent, OUT_SCOPE_REPLY
        loop = asyncio.get_event_loop()

        yield {"type": "thinking", "step": 0}
        intent, missing = await loop.run_in_executor(None, classify, user_message)

        if intent == Intent.OUT_SCOPE:
            yield {"type": "message", "text": OUT_SCOPE_REPLY}
            yield {"type": "done"}
            return

        if intent == Intent.CLARIFY:
            clarify_msg = f"I need a bit more info to help with that. {missing}."
            yield {"type": "message", "text": clarify_msg}
            yield {"type": "done"}
            return

        # IN_SCOPE — proceed to main agent
        llm = get_llm(role="chat", temperature=0.2)
        tools = build_tool_registry()

        # Build the conversation
        system = SYSTEM_PROMPT.format(tool_catalog=_format_tool_catalog())
        convo = [{"role": "system", "content": system}]
        for h in history[-10:]:  # last 10 turns
            role = h.get("role", "user")
            convo.append({"role": role, "content": str(h.get("content", ""))})
        convo.append({"role": "user", "content": user_message})

        for step in range(max_steps):
            yield {"type": "thinking", "step": step + 1}

            # Format the chat for the LLM
            messages = _convo_to_messages(convo)
            response = await loop.run_in_executor(None, lambda: llm.invoke(messages))
            text = response.content if hasattr(response, "content") else str(response)

            tool_call = _parse_tool_call(text)

            if not tool_call:
                # No tool — this is the final answer
                clean = _TOOL_CALL_RE.sub("", text).strip()
                if clean:
                    yield {"type": "message", "text": clean}
                yield {"type": "done"}
                return

            # Execute the tool
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {}) or {}

            yield {"type": "tool_call", "name": tool_name, "args": tool_args}

            if tool_name not in tools:
                # Give the LLM a chance to self-correct with the available tool list
                convo.append({"role": "assistant", "content": text})
                convo.append({
                    "role": "user",
                    "content": (
                        f"[ERROR] Unknown tool '{tool_name}'. "
                        f"Available tools: {', '.join(tools.keys())}. "
                        "Please try again with a valid tool name, or answer directly without a tool."
                    ),
                })
                yield {
                    "type": "tool_result",
                    "name": tool_name,
                    "summary": f"Unknown tool: {tool_name}",
                    "render": {
                        "kind": "error",
                        "message": f"Unknown tool '{tool_name}'. Available: {', '.join(tools.keys())}",
                    },
                }
                continue

            try:
                result = await tools[tool_name](**tool_args)
            except Exception as e:
                result = {
                    "summary": f"Tool failed: {type(e).__name__}: {e}",
                    "render": {"kind": "error", "message": str(e)},
                }

            yield {
                "type": "tool_result",
                "name": tool_name,
                "summary": result.get("summary", ""),
                "render": result.get("render"),
            }

            # Early exit for read-only tools — they don't set render.status=="success"
            # but small models will re-call them if we don't force a final reply.
            if tool_name in READ_ONLY_TOOLS:
                convo.append({"role": "assistant", "content": text})
                convo.append({
                    "role": "user",
                    "content": (
                        f"[tool_result from {tool_name}]\n{result.get('summary', '')}\n\n"
                        "The data above is already shown to the user. "
                        "Write one short sentence describing what was found. No tool calls."
                    ),
                })
                final = await loop.run_in_executor(
                    None, lambda: llm.invoke(_convo_to_messages(convo))
                )
                final_text = final.content if hasattr(final, "content") else str(final)
                clean = _TOOL_CALL_RE.sub("", final_text).strip()
                if clean:
                    yield {"type": "message", "text": clean}
                yield {"type": "done"}
                return

            # Early exit: if the tool succeeded (fix_bug / refactor_code), stop the loop
            # immediately — small models tend to re-call the tool instead of writing a reply.
            if result.get("render", {}).get("status") == "success":
                convo.append({"role": "assistant", "content": text})
                convo.append({
                    "role": "user",
                    "content": (
                        f"[tool_result from {tool_name}]\n{result.get('summary', '')}\n\n"
                        "Write one short sentence confirming success to the user. No tool calls."
                    ),
                })
                final = await loop.run_in_executor(
                    None, lambda: llm.invoke(_convo_to_messages(convo))
                )
                final_text = final.content if hasattr(final, "content") else str(final)
                clean = _TOOL_CALL_RE.sub("", final_text).strip()
                if clean:
                    yield {"type": "message", "text": clean}
                yield {"type": "done"}
                return

            # Feed the tool result back to the LLM so it can interpret + respond
            convo.append({"role": "assistant", "content": text})
            convo.append({
                "role": "user",
                "content": (
                    f"[tool_result from {tool_name}]\n"
                    f"{result.get('summary', '')}\n\n"
                    f"Now write a brief, natural-language reply to the user about what just happened. "
                    f"Don't call another tool unless absolutely necessary."
                ),
            })

        # Hit max steps without a final answer
        yield {
            "type": "message",
            "text": (
                "I've reached the maximum number of steps for this turn. "
                "Please send another message to continue."
            ),
        }
        yield {"type": "done"}

    except Exception as e:
        import traceback
        yield {
            "type": "error",
            "message": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-500:],
        }


def _convo_to_messages(convo: list[dict]):
    """Convert conversation dicts to LangChain message objects."""
    messages = []
    for msg in convo:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages
