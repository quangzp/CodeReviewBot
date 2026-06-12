"""
Git history enrichment for Neo4j graph nodes.

Computes per-file signals used in risk scoring:
  - bug_frequency:      # commits with fix/bug/hotfix keywords in last N days
  - contributor_churn:  # distinct authors in last N days
  - change_velocity:    # total commits in last N days
  - last_modifier:      author of most recent commit on this file

These signals are written back to Neo4j nodes as properties,
enabling risk-aware graph traversal.

Adapted from AIDevOpsAssistant's git_enrichment.py, but writes to
Neo4j instead of SQLite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from git import Repo
from git.exc import InvalidGitRepositoryError

BUG_KEYWORDS = {"fix", "bug", "hotfix", "patch", "repair", "resolve", "revert"}
LOOKBACK_DAYS = 90


@dataclass
class FileGitStats:
    """Git history signals for a single file."""

    file_path: str
    bug_frequency: int = 0
    contributor_churn: int = 0
    change_velocity: int = 0
    last_modifier: str = ""
    all_authors: list[str] = field(default_factory=list)


def _is_bug_commit(message: str) -> bool:
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in BUG_KEYWORDS)


def compute_file_stats(
    repo_path: str | Path,
    file_paths: list[str],
    lookback_days: int = LOOKBACK_DAYS,
) -> dict[str, FileGitStats]:
    """
    Query git log for each file and compute history signals.

    Args:
        repo_path: Path to the git repository root.
        file_paths: List of file paths to analyse (absolute or relative).
        lookback_days: How many days back to look.

    Returns:
        Dict mapping file_path -> FileGitStats.
    """
    try:
        repo = Repo(str(repo_path), search_parent_directories=True)
    except InvalidGitRepositoryError:
        return {}

    cutoff = datetime.now() - timedelta(days=lookback_days)
    results: dict[str, FileGitStats] = {}
    repo_root = Path(repo.working_tree_dir)

    for rel_path in file_paths:
        stats = FileGitStats(file_path=rel_path)
        all_authors: list[str] = []
        last_modifier = ""
        last_commit_date = None

        # Normalize path for git
        abs_path = Path(rel_path)
        if abs_path.is_absolute():
            try:
                git_path = abs_path.relative_to(repo_root).as_posix()
            except ValueError:
                git_path = Path(rel_path).as_posix()
        else:
            git_path = Path(rel_path).as_posix()

        try:
            commits = list(repo.iter_commits(paths=git_path))
        except Exception:
            results[rel_path] = stats
            continue

        for commit in commits:
            commit_date = datetime.fromtimestamp(commit.committed_date)
            author = commit.author.name or commit.author.email or "unknown"

            if last_commit_date is None or commit_date > last_commit_date:
                last_modifier = author
                last_commit_date = commit_date

            if commit_date < cutoff:
                continue

            all_authors.append(author)
            stats.change_velocity += 1
            if _is_bug_commit(commit.message):
                stats.bug_frequency += 1

        stats.contributor_churn = len(set(all_authors))
        stats.last_modifier = last_modifier
        stats.all_authors = list(set(all_authors))
        results[rel_path] = stats

    return results


def is_new_contributor(
    repo_path: str | Path,
    file_paths: list[str],
    pr_author: str,
) -> bool:
    """Return True if pr_author has never committed to ANY of the given files."""
    try:
        repo = Repo(str(repo_path), search_parent_directories=True)
    except InvalidGitRepositoryError:
        return True

    for rel_path in file_paths:
        try:
            commits = list(repo.iter_commits(paths=rel_path))
        except Exception:
            continue
        for commit in commits:
            author = commit.author.name or commit.author.email or ""
            if author.lower() == pr_author.lower():
                return False
    return True


def enrich_neo4j_nodes(
    neo4j_service,
    repo_path: str | Path,
    file_paths: list[str],
    lookback_days: int = LOOKBACK_DAYS,
) -> dict[str, FileGitStats]:
    """
    Compute git stats for files and write them as properties on Neo4j nodes.

    This enriches MethodNode/ClassNode/EndpointNode with:
      - bug_frequency
      - contributor_churn
      - change_velocity
      - last_modifier
    """
    stats_map = compute_file_stats(repo_path, file_paths, lookback_days)

    if not stats_map:
        return stats_map

    query = """
    MATCH (n)
    WHERE n.file_path = $file_path
    SET n.bug_frequency = $bug_frequency,
        n.contributor_churn = $contributor_churn,
        n.change_velocity = $change_velocity,
        n.last_modifier = $last_modifier
    """

    with neo4j_service.db.driver.session() as session:
        for file_path, stats in stats_map.items():
            session.run(query, {
                "file_path": file_path,
                "bug_frequency": stats.bug_frequency,
                "contributor_churn": stats.contributor_churn,
                "change_velocity": stats.change_velocity,
                "last_modifier": stats.last_modifier,
            })

    return stats_map
