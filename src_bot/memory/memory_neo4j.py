"""
Async wrapper around MemoryStore (store.py) for use by pipeline callers.

Provides the same public API as graphiti_store.py so all callers only need to
change their import line. Uses Neo4j directly with no external dependencies.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from src_bot.config.config import configs
from src_bot.memory.store import MemoryStore
from src_bot.memory.schema import ReviewFact

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(
            configs.APP_NEO4J_URL,
            auth=(configs.APP_NEO4J_USER, configs.APP_NEO4J_PASSWORD),
        )
        MemoryStore(_driver).ensure_indexes()
    return _driver


def _get_store() -> MemoryStore:
    return MemoryStore(_get_driver())


async def record_review(
    developer_login: str,
    repo_name: str,
    pr_number: int,
    pr_url: str,
    summary: str,
    bug_patterns: list[str],
    reviewed_at: Optional[str] = None,
) -> None:
    store = _get_store()
    fact = ReviewFact(
        developer_login=developer_login,
        repo_name=repo_name,
        pr_number=pr_number,
        pr_url=pr_url,
        summary=summary,
        bug_patterns=bug_patterns,
        files_touched=[],
        reviewed_at=datetime.now(timezone.utc),
    )
    review_id = f"review-{uuid.uuid4().hex[:12]}"
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, store.record_review, fact, review_id)


async def build_memory_context(developer_login: str, repo_name: str) -> str:
    store = _get_store()
    loop = asyncio.get_event_loop()
    profile = await loop.run_in_executor(None, store.get_developer_profile, developer_login)
    if not profile or not profile.patterns:
        return ""
    lines = [f"## Developer Memory: {developer_login}"]
    for p in profile.patterns[:10]:
        conf_pct = int(p.confidence * 100)
        lines.append(
            f"- [{p.pattern.name}] confidence={conf_pct}%, "
            f"seen {p.evidence_count}x"
        )
    return "\n".join(lines)


async def get_developer_profile(developer_login: str) -> dict:
    store = _get_store()
    loop = asyncio.get_event_loop()
    profile = await loop.run_in_executor(None, store.get_developer_profile, developer_login)
    if not profile:
        return {"developer": developer_login, "facts": [], "fact_count": 0}
    facts = [
        f"{p.pattern.name}: confidence={int(p.confidence * 100)}%, evidence={p.evidence_count}"
        for p in profile.patterns
    ]
    return {
        "developer": developer_login,
        "facts": facts,
        "fact_count": len(facts),
        "patterns": [
            {
                "id": p.pattern.id,
                "name": p.pattern.name,
                "confidence": p.confidence,
                "evidence_count": p.evidence_count,
            }
            for p in profile.patterns
        ],
    }


async def list_developers_from_memory(limit: int = 100) -> list[dict]:
    store = _get_store()
    loop = asyncio.get_event_loop()
    devs = await loop.run_in_executor(None, store.list_developers, limit)
    return [
        {
            "login": d.login,
            "pr_count": d.pr_count,
            "last_seen_at": d.last_seen_at.isoformat() if d.last_seen_at else None,
        }
        for d in devs
    ]


async def close() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
