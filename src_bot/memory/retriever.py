"""
Memory retriever — fetches institutional memory before each review
and formats it as a context string to inject into the LLM prompt.

Delegates to Graphiti (replacing legacy MemoryStore).
"""
from __future__ import annotations

import asyncio


def build_memory_context(
    *,
    author_login: str,
    repo_name: str,
    files_touched: list[str] | None = None,
    max_patterns: int = 5,
) -> str:
    """
    Build a formatted context string for the LLM prompt.
    Synchronous wrapper around Graphiti's async build_memory_context.

    Returns an empty string if no relevant memory exists yet.
    Note: callers inside an async context should use
    graphiti_store.build_memory_context() directly (async).
    """
    from src_bot.memory.memory_neo4j import build_memory_context as _async_build

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an async context — caller should use the async version
            return ""
        return loop.run_until_complete(_async_build(author_login, repo_name))
    except Exception:
        return ""
