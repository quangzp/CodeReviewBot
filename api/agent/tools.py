"""
Agent tools — each capability the LLM can invoke.

Tools are async functions that take typed arguments and return a dict
with two keys:
  - "summary": short text the LLM uses to reason about the result
  - "render":  structured payload the chat UI renders inline
               (e.g. {"kind": "diff", "patch": "..."} → diff viewer)

This separation lets the agent reason in text while the UI shows rich content.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Awaitable

from api.database import (
    list_projects,
    get_project,
    get_project_by_repo_name,
    create_project,
    create_review,
    list_reviews,
    get_review,
)
from api.project_indexer import index_project, parse_repo_url
from api.reviewer import run_pr_review, parse_pr_url
from api.agent.fix_bug import fix_bug_in_project
from api.agent.refactor import refactor_code_in_project


# ============================================================================
# Tool catalog — LLM sees these descriptions to decide which to call
# ============================================================================
TOOL_DESCRIPTIONS = [
    {
        "name": "add_project",
        "description": (
            "Onboard a new GitHub repository for code review. Clones the repo and "
            "builds the Neo4j knowledge graph. Takes 30s-5min depending on repo size. "
            "Use this when the user wants to add/upload/onboard/track a new project."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_url": {
                    "type": "string",
                    "description": "Full GitHub URL, e.g. https://github.com/owner/repo",
                },
            },
            "required": ["repo_url"],
        },
    },
    {
        "name": "list_projects",
        "description": (
            "List all onboarded projects with their indexing status and stats. "
            "Use when the user asks what projects exist, what's available, or wants an overview."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "project_status",
        "description": (
            "Get detailed status of one project: indexing progress, node/edge counts, last commit. "
            "Use when the user asks about a specific project's status or details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_name": {
                    "type": "string",
                    "description": "Repo in 'owner/name' form, e.g. 'jertel/elastalert2'",
                },
            },
            "required": ["repo_name"],
        },
    },
    {
        "name": "review_pr",
        "description": (
            "Review a GitHub pull request: detect bugs, suggest patches. "
            "Requires the project to be onboarded already. "
            "Use when the user asks to review a PR or wants feedback on a PR."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pr_url": {
                    "type": "string",
                    "description": "Full GitHub PR URL, e.g. https://github.com/owner/repo/pull/123",
                },
            },
            "required": ["pr_url"],
        },
    },
    {
        "name": "fix_bug",
        "description": (
            "Fix a bug in a project WITHOUT needing a PR. The user describes the bug in natural "
            "language and the agent finds the relevant code, generates a patch, and shows it. "
            "Use when the user wants to fix something specific without going through GitHub."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_name": {
                    "type": "string",
                    "description": "Repo in 'owner/name' form. The project must be already indexed.",
                },
                "bug_description": {
                    "type": "string",
                    "description": (
                        "Natural-language description of the bug. Be as specific as possible "
                        "(file/function names help). Example: 'parse_deadline() in util.py "
                        "returns wrong timezone, should be UTC'"
                    ),
                },
            },
            "required": ["repo_name", "bug_description"],
        },
    },
    {
        "name": "refactor_code",
        "description": (
            "Refactor code in a project: improve structure, reduce duplication, extract methods, "
            "add type hints, or improve readability — WITHOUT changing behavior. "
            "Use when the user wants to clean up or improve code quality, not fix a bug."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_name": {
                    "type": "string",
                    "description": "Repo in 'owner/name' form. The project must be already indexed.",
                },
                "refactor_description": {
                    "type": "string",
                    "description": (
                        "Natural-language description of what to refactor and how. "
                        "Example: 'Extract the URL-building logic in kibana_discover.py into a helper function'"
                    ),
                },
                "file_path": {
                    "type": "string",
                    "description": (
                        "Optional. Target file path relative to repo root "
                        "(e.g. 'elastalert/kibana_discover.py'). "
                        "If omitted, the bot will locate the relevant file automatically."
                    ),
                },
            },
            "required": ["repo_name", "refactor_description"],
        },
    },
    {
        "name": "recent_reviews",
        "description": (
            "List recent reviews, optionally filtered by project. "
            "Use when the user asks about past reviews, history, or what was reviewed recently."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_name": {
                    "type": "string",
                    "description": "Optional. Filter to one project.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10)",
                },
            },
        },
    },
]


# ============================================================================
# Tool implementations
# ============================================================================
async def tool_add_project(repo_url: str, **kwargs) -> dict:
    """Onboard a new project."""
    try:
        repo_name, clean_url = parse_repo_url(repo_url)
    except ValueError as e:
        return {
            "summary": f"Invalid GitHub URL: {e}",
            "render": {"kind": "error", "message": str(e)},
        }

    existing = await get_project_by_repo_name(repo_name)
    if existing:
        return {
            "summary": (
                f"Project '{repo_name}' already exists (status: {existing.status}, "
                f"{existing.node_count} nodes, {existing.file_count} files)."
            ),
            "render": {
                "kind": "project",
                "project": existing.model_dump(mode="json"),
                "already_exists": True,
            },
        }

    record = await create_project(repo_name, clean_url)

    # Kick off indexing in background. The chat UI will poll the project status.
    queue: asyncio.Queue = asyncio.Queue()
    asyncio.create_task(index_project(record, queue))
    # Drain the queue in background so it doesn't fill memory
    async def _drain():
        while True:
            item = await queue.get()
            if item is None:
                break
    asyncio.create_task(_drain())

    return {
        "summary": (
            f"Started onboarding '{repo_name}'. Cloning and building the knowledge graph "
            f"in the background. This usually takes 30s-5min depending on repo size."
        ),
        "render": {
            "kind": "project",
            "project": record.model_dump(mode="json"),
            "indexing": True,
        },
    }


async def tool_list_projects(**kwargs) -> dict:
    """List all projects."""
    projects = await list_projects()
    if not projects:
        return {
            "summary": "No projects onboarded yet.",
            "render": {"kind": "project_list", "projects": []},
        }
    return {
        "summary": (
            f"Found {len(projects)} project(s): "
            + ", ".join(f"{p.repo_name} ({p.status})" for p in projects[:5])
            + ("..." if len(projects) > 5 else "")
        ),
        "render": {
            "kind": "project_list",
            "projects": [p.model_dump(mode="json") for p in projects],
        },
    }


async def tool_project_status(repo_name: str, **kwargs) -> dict:
    """Get status of a project."""
    project = await get_project_by_repo_name(repo_name)
    if not project:
        return {
            "summary": f"Project '{repo_name}' is not onboarded.",
            "render": {"kind": "error", "message": f"'{repo_name}' is not onboarded yet."},
        }
    return {
        "summary": (
            f"'{repo_name}' — status: {project.status}, "
            f"{project.node_count} nodes, {project.edge_count} edges, "
            f"{project.file_count} files."
        ),
        "render": {"kind": "project", "project": project.model_dump(mode="json")},
    }


async def tool_review_pr(pr_url: str, **kwargs) -> dict:
    """Submit a PR for review."""
    try:
        repo_name, pr_number = parse_pr_url(pr_url)
    except ValueError as e:
        return {
            "summary": f"Invalid PR URL: {e}",
            "render": {"kind": "error", "message": str(e)},
        }

    project = await get_project_by_repo_name(repo_name)
    if not project:
        # Auto-onboard the project so the user doesn't have to add it manually
        add_result = await tool_add_project(repo_url=f"https://github.com/{repo_name}")
        if add_result.get("render", {}).get("kind") == "error":
            return add_result
        return {
            "summary": (
                f"Project '{repo_name}' wasn't indexed yet — started indexing it now. "
                f"Indexing usually takes 1-5 minutes. "
                f"Once status shows 'indexed', say 'review the PR again' and I'll run it."
            ),
            "render": {
                "kind": "auto_onboard",
                "project": add_result.get("render", {}).get("project"),
                "pr_url": pr_url,
                "message": (
                    f"Indexing '{repo_name}' in the background. "
                    f"Retry the review when indexing is complete."
                ),
            },
        }
    if project.status != "indexed":
        return {
            "summary": (
                f"Project '{repo_name}' is not ready (status: {project.status}). "
                f"Wait for indexing to complete."
            ),
            "render": {
                "kind": "error",
                "message": f"'{repo_name}' status is {project.status}. Wait for indexing.",
            },
        }

    # Check workspace exists on disk — DB says "indexed" but clone may be missing after reboot
    from api.project_indexer import get_project_dir
    if not get_project_dir(project.id).exists():
        return {
            "summary": (
                f"The workspace for '{repo_name}' is missing — the server was likely restarted "
                f"and the clone was cleared. Please reindex the project first: "
                f"'Reindex {repo_name}', then retry the review."
            ),
            "render": {
                "kind": "error",
                "message": f"Workspace for '{repo_name}' is missing after server restart.",
                "suggestion": f"Reindex {repo_name} first, then retry.",
            },
        }

    record = await create_review(pr_url, repo_name, pr_number, project_id=project.id)

    # Kick off review in background, store events in the agent's review_streams
    # so the chat UI can poll /api/reviews/:id and /api/reviews/:id/stream
    queue: asyncio.Queue = asyncio.Queue()
    # Register the queue in the main event store so the existing SSE endpoint works
    from api.main import _event_queues
    _event_queues[record.id] = queue
    asyncio.create_task(run_pr_review(record, queue))

    return {
        "summary": (
            f"Started reviewing {repo_name}#{pr_number}. "
            f"The pipeline (file localization → fault localization → patch generation + reflexion) "
            f"is running in the background. Open the review for live progress."
        ),
        "render": {
            "kind": "review",
            "review_id": record.id,
            "pr_url": pr_url,
            "repo_name": repo_name,
            "pr_number": pr_number,
        },
    }


async def tool_fix_bug(repo_name: str, bug_description: str, user_login: str = "", **kwargs) -> dict:
    """Fix a bug without a PR — chat-driven fix."""
    project = await get_project_by_repo_name(repo_name)
    if not project:
        add_result = await tool_add_project(repo_url=f"https://github.com/{repo_name}")
        if add_result.get("render", {}).get("kind") == "error":
            return add_result
        return {
            "summary": (
                f"Project '{repo_name}' wasn't indexed yet — started indexing it now. "
                f"Once indexing completes (1-5 min), say 'fix the bug again' and I'll run it."
            ),
            "render": {
                "kind": "auto_onboard",
                "project": add_result.get("render", {}).get("project"),
                "message": f"Indexing '{repo_name}'. Retry fix_bug when status is 'indexed'.",
            },
        }
    if project.status != "indexed":
        return {
            "summary": f"Project '{repo_name}' is not ready (status: {project.status}).",
            "render": {"kind": "error", "message": f"Project status is {project.status}."},
        }

    result = await fix_bug_in_project(project, bug_description, user_login=user_login)
    return result


async def tool_refactor_code(
    repo_name: str,
    refactor_description: str,
    file_path: str = "",
    user_login: str = "",
    **kwargs,
) -> dict:
    """Refactor code in a project without a PR."""
    project = await get_project_by_repo_name(repo_name)
    if not project:
        add_result = await tool_add_project(repo_url=f"https://github.com/{repo_name}")
        if add_result.get("render", {}).get("kind") == "error":
            return add_result
        return {
            "summary": (
                f"Project '{repo_name}' wasn't indexed yet — started indexing it now. "
                f"Once indexing completes (1-5 min), say 'refactor again' and I'll run it."
            ),
            "render": {
                "kind": "auto_onboard",
                "project": add_result.get("render", {}).get("project"),
                "message": f"Indexing '{repo_name}'. Retry refactor when status is 'indexed'.",
            },
        }
    if project.status != "indexed":
        return {
            "summary": f"Project '{repo_name}' is not ready (status: {project.status}).",
            "render": {"kind": "error", "message": f"Project status is {project.status}."},
        }

    result = await refactor_code_in_project(
        project, refactor_description, file_path, user_login=user_login
    )
    return result


async def tool_recent_reviews(repo_name: str = "", limit: int = 10, **kwargs) -> dict:
    """List recent reviews."""
    project_id = None
    if repo_name:
        project = await get_project_by_repo_name(repo_name)
        if project:
            project_id = project.id
    reviews = await list_reviews(project_id=project_id)
    reviews = reviews[:limit]

    if not reviews:
        return {
            "summary": "No reviews found.",
            "render": {"kind": "review_list", "reviews": []},
        }

    return {
        "summary": (
            f"Found {len(reviews)} review(s)"
            + (f" for {repo_name}" if repo_name else "")
            + ". Most recent: "
            + ", ".join(f"#{r.pr_number} ({r.status})" for r in reviews[:3])
        ),
        "render": {
            "kind": "review_list",
            "reviews": [r.model_dump(mode="json") for r in reviews],
        },
    }


# ============================================================================
# Registry
# ============================================================================
def build_tool_registry() -> dict[str, Callable[..., Awaitable[dict]]]:
    """Return name → async function map."""
    return {
        "add_project": tool_add_project,
        "list_projects": tool_list_projects,
        "project_status": tool_project_status,
        "review_pr": tool_review_pr,
        "fix_bug": tool_fix_bug,
        "refactor_code": tool_refactor_code,
        "recent_reviews": tool_recent_reviews,
    }
