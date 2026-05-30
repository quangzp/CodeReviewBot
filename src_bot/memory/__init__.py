"""
Graphiti-style memory layer for the code review bot.

This module gives the bot institutional memory:
- Developer profiles (what bug patterns they tend to make)
- Module hotspots (which files have had recent bugs)
- Active conventions (team decisions still in effect)

Built on the existing Neo4j instance (no new infrastructure needed).
Uses a bi-temporal model: every fact has valid_from / valid_to so we can
track when a developer's habits change over time.
"""
from src_bot.memory.schema import (
    Developer,
    BugPattern,
    ModuleStats,
    DeveloperProfile,
)
from src_bot.memory.store import MemoryStore
from src_bot.memory.retriever import build_memory_context
from src_bot.memory.extractor import extract_facts_from_review

__all__ = [
    "Developer",
    "BugPattern",
    "ModuleStats",
    "DeveloperProfile",
    "MemoryStore",
    "build_memory_context",
    "extract_facts_from_review",
]
