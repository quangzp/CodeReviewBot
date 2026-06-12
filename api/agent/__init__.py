"""
Agent layer — turns the backend into a conversational tool-using agent.

The user chats naturally; the LLM picks which backend capability to invoke.
Each tool wraps an existing API operation and returns structured data that
the chat UI can render specially (e.g. a diff is rendered with a diff viewer
instead of as text).
"""
from api.agent.tools import build_tool_registry, TOOL_DESCRIPTIONS
from api.agent.loop import run_agent_turn

__all__ = ["build_tool_registry", "TOOL_DESCRIPTIONS", "run_agent_turn"]
