"""
DEAD CODE — do not use or import.

graphiti_core is not installed in this project. Active memory layer is
src_bot/memory/memory_neo4j.py (wrapper around store.py).
This file is kept only for historical reference; delete when no longer needed.

Original description:
Graphiti-backed institutional memory store.

Replaces the custom store.py with the real graphiti-core library,
using Groq (free) for LLM and sentence-transformers (free, local) for embeddings.

Key concepts:
  - Each completed PR review is ingested as a Graphiti "episode"
  - Graphiti automatically extracts entities (Developer, BugPattern) and
    relationships (TENDS_TO) with full bi-temporal tracking
  - We query the graph to build developer context before each review
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from graphiti_core import Graphiti
from graphiti_core.llm_client.groq_client import GroqClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EpisodeType

from src_bot.memory.embedder import SentenceTransformerEmbedder
from src_bot.config.config import configs


def _make_graphiti() -> Graphiti:
    """Build a Graphiti instance using Groq + local sentence-transformers."""
    llm_config = LLMConfig(
        api_key=configs.GROQ_API_KEY,
        model=getattr(configs, "LLM_MODEL", "llama-3.3-70b-versatile"),
        small_model="llama-3.1-8b-instant",   # fast model for simpler extractions
    )
    llm_client = GroqClient(config=llm_config)
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

    return Graphiti(
        uri=configs.APP_NEO4J_URL,
        user=configs.APP_NEO4J_USER,
        password=configs.APP_NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder,
    )


# Module-level singleton — created on first use
_graphiti: Optional[Graphiti] = None


async def get_graphiti() -> Graphiti:
    global _graphiti
    if _graphiti is None:
        _graphiti = _make_graphiti()
        await _graphiti.build_indices_and_constraints()
    return _graphiti


# ---------------------------------------------------------------------------
# Public API — same interface as the old store.py so existing callers work
# ---------------------------------------------------------------------------

async def record_review(
    developer_login: str,
    repo_name: str,
    pr_number: int,
    pr_url: str,
    summary: str,
    bug_patterns: list[str],
    reviewed_at: Optional[str] = None,
) -> None:
    """
    Ingest a completed review as a Graphiti episode.

    Graphiti will automatically:
    - Extract Developer and BugPattern entities
    - Create/update TENDS_TO edges with confidence tracking
    - Handle bi-temporal versioning (valid_at / invalid_at)
    """
    g = await get_graphiti()

    # Format the episode as natural language — Graphiti's LLM extracts entities from this
    pattern_list = ", ".join(bug_patterns) if bug_patterns else "no specific patterns"
    episode_text = (
        f"Developer {developer_login} submitted pull request #{pr_number} "
        f"in repository {repo_name} ({pr_url}). "
        f"Code review summary: {summary}. "
        f"Bug patterns found: {pattern_list}."
    )

    await g.add_episode(
        name=f"review:{repo_name}#{pr_number}",
        episode_body=episode_text,
        source=EpisodeType.text,
        source_description="Automated code review by CodeReviewBot",
        reference_time=datetime.now(timezone.utc),
        group_id=developer_login,   # scopes the graph to this developer
    )


async def build_memory_context(
    developer_login: str,
    repo_name: str,
) -> str:
    """
    Search the Graphiti graph for facts about this developer and return
    a formatted string to inject into the LLM review prompt.
    """
    g = await get_graphiti()

    try:
        # Search for developer-specific patterns
        dev_results = await g.search(
            query=f"bug patterns for developer {developer_login}",
            group_ids=[developer_login],
            num_results=10,
        )

        # Search for repo-specific patterns
        repo_results = await g.search(
            query=f"common bugs in {repo_name}",
            num_results=5,
        )

        if not dev_results and not repo_results:
            return ""

        lines = [f"## Developer Memory: {developer_login}"]

        if dev_results:
            lines.append("\n### Known patterns for this developer:")
            for fact in dev_results:
                lines.append(f"- {fact.fact}")

        if repo_results:
            lines.append(f"\n### Known issues in {repo_name}:")
            for fact in repo_results:
                lines.append(f"- {fact.fact}")

        return "\n".join(lines)

    except Exception as e:
        # Memory is optional — never block a review because of it
        return f"(Memory unavailable: {e})"


async def get_developer_profile(developer_login: str) -> dict:
    """
    Return a structured profile for the /api/developers/:login endpoint.
    """
    g = await get_graphiti()

    try:
        results = await g.search(
            query=f"developer {developer_login} bug patterns tendencies",
            group_ids=[developer_login],
            num_results=20,
        )

        facts = [r.fact for r in results]
        return {
            "developer": developer_login,
            "facts": facts,
            "fact_count": len(facts),
        }
    except Exception:
        return {"developer": developer_login, "facts": [], "fact_count": 0}


async def close():
    """Close the Graphiti connection (call on app shutdown)."""
    global _graphiti
    if _graphiti is not None:
        await _graphiti.close()
        _graphiti = None
