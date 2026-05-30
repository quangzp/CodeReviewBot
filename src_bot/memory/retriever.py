"""
Memory retriever — fetches institutional memory before each review
and formats it as a context string to inject into the LLM prompt.

This is what makes the bot "remember" the team.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src_bot.memory.store import MemoryStore


def build_memory_context(
    store: MemoryStore,
    *,
    author_login: str,
    repo_name: str,
    files_touched: list[str],
    max_patterns: int = 5,
) -> str:
    """
    Build a formatted context string for the LLM prompt.

    Includes:
    - Developer profile (bug patterns they tend to make)
    - Module hotspots (files with recent bug history)

    Returns an empty string if no relevant memory exists yet (first review,
    new developer, fresh repo).
    """
    sections = []

    # 1. Developer profile
    profile = store.get_developer_profile(author_login)
    if profile and profile.patterns:
        active_patterns = [p for p in profile.patterns if p.is_active][:max_patterns]
        if active_patterns:
            dev_section = [
                f"## Developer Profile: @{author_login}",
                f"Total PRs reviewed by this bot: {profile.developer.pr_count}",
                f"Bug patterns this developer tends to make:",
            ]
            for p in active_patterns:
                dev_section.append(
                    f"  - **{p.pattern.name}** "
                    f"(confidence {p.confidence:.0%}, seen in {p.evidence_count} PRs)"
                )
            sections.append("\n".join(dev_section))

    # 2. Module hotspots
    project_id = repo_name.replace("/", "_")
    hotspots = []
    for path in files_touched[:10]:
        stats = store.get_module_hotspot(path, project_id, months=6)
        if stats.recent_bug_count > 0:
            hotspots.append((path, stats))

    if hotspots:
        hotspot_section = ["## Module Hotspots"]
        hotspot_section.append("Files with recent bug history (last 6 months):")
        for path, stats in sorted(hotspots, key=lambda x: x[1].recent_bug_count, reverse=True):
            hotspot_section.append(
                f"  - `{path}`: {stats.recent_bug_count} recent bugs "
                f"(total: {stats.total_bug_count})"
            )
        sections.append("\n".join(hotspot_section))

    if not sections:
        return ""

    header = "# 📚 Institutional Memory (what the bot has learned about your team)"
    body = "\n\n".join(sections)
    footer = (
        "\n**Use this context to give a more targeted review.** "
        "Reference patterns explicitly when you find a relevant issue."
    )
    return f"{header}\n\n{body}\n{footer}"
