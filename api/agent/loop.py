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
import logging
import re
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src_bot.config.config import configs
from src_bot.llm.router import get_llm
from src_bot.observability.langfuse_ctx import (
    langfuse_flush,
    langfuse_span,
    langfuse_trace,
    langfuse_update_current_span,
)
from api.agent.tools import build_tool_registry, TOOL_DESCRIPTIONS

READ_ONLY_TOOLS = {"list_projects", "project_status", "recent_reviews", "explore_project"}


# ============================================================================
# System prompt
# ============================================================================
SYSTEM_PROMPT = """You are CodeReviewBot — a focused assistant with ONE job: help engineers understand and manage projects, review pull requests, fix bugs, and refactor code in GitHub repositories.

You have access to these tools:

{tool_catalog}

# Strict scope — what you DO

1. **Add / list / reindex projects** — onboard a GitHub repo so it can be reviewed
2. **Review a pull request** — run the full GraphRAG + Reflexion review pipeline on a PR
3. **Apply review patches** — push verified patches from a completed review as a fix-branch PR
4. **Fix a bug** — locate and patch a bug described in natural language
5. **Refactor code** — improve code structure/readability without changing behavior
6. **Explore a project** — explain an indexed repo's structure, libraries, frameworks,
   entrypoints, architecture, and code/request flow using repo context
7. **Check review / project status** — report progress on ongoing tasks

That is ALL. You may explain architecture only for a specific onboarded project. You do not
answer general programming questions, explain concepts detached from a repo, write code for
the user, or engage in any conversation outside the above scope.

# Strict scope — what you DO NOT do

- Do NOT answer questions like "What is GraphRAG?", "How does Python work?", "What is a bug?"
- Do NOT explain generic frameworks/concepts unless the user asks how a specific indexed repo uses them
- Do NOT write code, explain algorithms, or give advice unrelated to managing projects/reviews
- Do NOT engage in small talk, greetings beyond a single line, or off-topic chat
- Do NOT speculate about reviews that haven't run yet

When a user asks something outside scope, respond with EXACTLY this and nothing more:
"I'm only able to help with adding projects, reviewing pull requests, fixing bugs, refactoring code, and exploring indexed projects. What would you like to do?"

# How to call tools

Output tool calls in this EXACT format (one tool per response):

<tool_call>
{{"name": "tool_name", "args": {{"arg1": "value", "arg2": "value"}}}}
</tool_call>

After a tool runs, write 1-2 short sentences interpreting the result. Don't restate the data — the UI shows it visually.

# IMPORTANT: After a tool returns a successful result (status: success), write ONE short
# confirmation sentence and stop. Do NOT call the same tool again.

# Post-review fix flow (MANDATORY — always follow after review_pr)

After `review_pr` returns successfully, you MUST:
1. Call `get_review_detail` with the same PR URL to show the patches and faults visually.
2. After the detail renders, ask: "Would you like me to push these as a fix-branch pull request on GitHub?"
3. Wait for explicit user confirmation before acting.
4. Once the user confirms (e.g. "yes", "go ahead", "apply", "tạo PR", "push it"):
   call `apply_review_fixes` with the same PR URL.

NEVER call `apply_review_fixes` without explicit user confirmation — it creates a branch and opens a PR in the user's repository.

If the user says "apply the fixes", "push the patches", "create fix PR", or similar at any point after a review, call `apply_review_fixes` immediately without asking again.

# Showing review results

When the user asks any of these, call `get_review_detail` with the PR URL:
- "is it done", "show me", "show the results", "what did you find"
- "xem kết quả", "done chưa", "show patch", "hiển thị"
- Any question about what a completed review found

# Examples

User: "Add github.com/jertel/elastalert2"
You: <tool_call>{{"name": "add_project", "args": {{"repo_url": "https://github.com/jertel/elastalert2"}}}}</tool_call>

User: "Review PR https://github.com/jertel/elastalert2/pull/1763"
You: <tool_call>{{"name": "review_pr", "args": {{"pr_url": "https://github.com/jertel/elastalert2/pull/1763"}}}}</tool_call>

[After review_pr succeeds — ALWAYS call get_review_detail next]
You: <tool_call>{{"name": "get_review_detail", "args": {{"pr_url": "https://github.com/jertel/elastalert2/pull/1763"}}}}</tool_call>

[After get_review_detail shows results]
You: "3 file(s) have verified patches ready. Would you like me to push them as a fix-branch pull request on GitHub?"

User: "Yes, go ahead"
You: <tool_call>{{"name": "apply_review_fixes", "args": {{"pr_url": "https://github.com/jertel/elastalert2/pull/1763"}}}}</tool_call>

User: "Is it done? Show me the results — https://github.com/jertel/elastalert2/pull/1763"
You: <tool_call>{{"name": "get_review_detail", "args": {{"pr_url": "https://github.com/jertel/elastalert2/pull/1763"}}}}</tool_call>

User: "Fix the timezone bug in parse_deadline() in elastalert2"
You: <tool_call>{{"name": "fix_bug", "args": {{"repo_name": "jertel/elastalert2", "bug_description": "parse_deadline() returns wrong timezone, should return UTC"}}}}</tool_call>

User: "Refactor the URL builder in nicholasgibson2/elastalert-jertel to reduce duplication"
You: <tool_call>{{"name": "refactor_code", "args": {{"repo_name": "nicholasgibson2/elastalert-jertel", "refactor_description": "Reduce duplication in the URL builder functions"}}}}</tool_call>

User: "Giải thích cấu trúc project nicholasgibson2/elastalert-jertel"
You: <tool_call>{{"name": "explore_project", "args": {{"repo_name": "nicholasgibson2/elastalert-jertel", "question": "Giải thích cấu trúc project"}}}}</tool_call>

User: "Project jertel/elastalert2 dùng framework và thư viện gì?"
You: <tool_call>{{"name": "explore_project", "args": {{"repo_name": "jertel/elastalert2", "question": "Project dùng framework và thư viện gì?"}}}}</tool_call>

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


def _compact_tool_args(args: dict) -> dict:
    """Trim tool args before sending them to observability metadata."""
    compact: dict = {}
    for key, value in (args or {}).items():
        text = str(value)
        compact[key] = text[:300] + ("..." if len(text) > 300 else "")
    return compact


# ============================================================================
# Agent turn — runs one user → assistant exchange
# ============================================================================
async def run_agent_turn(
    user_message: str,
    history: list[dict],
    max_steps: int = 4,
    user_login: str = "",
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
    _lf_trace_cm = None
    try:
        # Open a root Langfuse trace for this chat turn. The helper is a no-op
        # when Langfuse is disabled, so the control flow stays identical.
        try:
            import uuid

            _trace_id = f"chat-{uuid.uuid4().hex[:12]}"
            _lf_trace_cm = langfuse_trace(
                trace_id=_trace_id,
                session_id=user_login or None,
                user_id=user_login or "unknown",
                tags=["agent_chat"],
                metadata={
                    "message_preview": user_message[:120],
                    "history_turns": len(history or []),
                },
            )
            _lf_trace_cm.__enter__()
        except Exception as e:
            logger.debug("Langfuse trace setup skipped: %s", e)

        # ----------------------------------------------------------------
        # 0. Intent classification — runs BEFORE the main agent LLM
        #    Uses llama-3.1-8b-instant (fast, free) to gate out-of-scope
        #    messages without wasting tokens on the main model.
        # ----------------------------------------------------------------
        from api.agent.classifier import classify, Intent, OUT_SCOPE_REPLY
        loop = asyncio.get_event_loop()

        yield {"type": "thinking", "step": 0}
        with langfuse_span(
            "chat.classify_intent",
            metadata={
                "message_preview": user_message[:120],
                "history_turns": len(history or []),
            },
        ):
            intent, missing = await loop.run_in_executor(
                None, lambda: classify(user_message, history=history)
            )
            langfuse_update_current_span(
                metadata={"intent": str(intent), "missing": missing[:200] if missing else ""}
            )

        if intent == Intent.OUT_SCOPE:
            langfuse_update_current_span(output={"reply": OUT_SCOPE_REPLY})
            yield {"type": "message", "text": OUT_SCOPE_REPLY}
            yield {"type": "done"}
            return

        if intent == Intent.CLARIFY:
            clarify_msg = f"I need a bit more info to help with that. {missing}."
            langfuse_update_current_span(output={"reply": clarify_msg})
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
            with langfuse_span(
                "chat.agent_llm",
                metadata={"step": step + 1, "messages": len(messages)},
            ):
                response = await loop.run_in_executor(None, lambda: llm.invoke(messages))
                text = response.content if hasattr(response, "content") else str(response)

                tool_call = _parse_tool_call(text)
                langfuse_update_current_span(
                    metadata={
                        "has_tool_call": bool(tool_call),
                        "tool_name": (tool_call or {}).get("name", ""),
                    },
                    output={"response_preview": text[:500]},
                )

            if not tool_call:
                # No tool — this is the final answer
                clean = _TOOL_CALL_RE.sub("", text).strip()
                if clean:
                    with langfuse_span(
                        "chat.final_response",
                        metadata={"reason": "no_tool", "step": step + 1},
                    ):
                        langfuse_update_current_span(output={"text": clean[:1000]})
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

            with langfuse_span(
                "chat.tool_call",
                metadata={"name": tool_name, "args": _compact_tool_args(tool_args)},
            ):
                try:
                    result = await tools[tool_name](**tool_args, user_login=user_login)
                except Exception as e:
                    result = {
                        "summary": f"Tool failed: {type(e).__name__}: {e}",
                        "render": {"kind": "error", "message": str(e)},
                    }
                render = result.get("render") or {}
                langfuse_update_current_span(
                    metadata={
                        "render_kind": render.get("kind", ""),
                        "status": render.get("status", ""),
                    },
                    output={"summary": result.get("summary", "")[:1000]},
                )

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
                with langfuse_span(
                    "chat.final_response",
                    metadata={"reason": "read_only_tool", "tool_name": tool_name},
                ):
                    final = await loop.run_in_executor(
                        None, lambda: llm.invoke(_convo_to_messages(convo))
                    )
                    final_text = final.content if hasattr(final, "content") else str(final)
                    langfuse_update_current_span(output={"text": final_text[:1000]})
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
                with langfuse_span(
                    "chat.final_response",
                    metadata={"reason": "success_tool", "tool_name": tool_name},
                ):
                    final = await loop.run_in_executor(
                        None, lambda: llm.invoke(_convo_to_messages(convo))
                    )
                    final_text = final.content if hasattr(final, "content") else str(final)
                    langfuse_update_current_span(output={"text": final_text[:1000]})
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
    finally:
        if _lf_trace_cm is not None:
            try:
                _lf_trace_cm.__exit__(None, None, None)
            except Exception:
                pass
        langfuse_flush()


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
