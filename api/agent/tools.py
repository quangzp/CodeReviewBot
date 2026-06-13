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
        "name": "explore_project",
        "description": (
            "Explain an onboarded project's structure, frameworks, libraries, entrypoints, "
            "architecture, or code/request flow. Use this when the user wants to understand "
            "how a specific repo is organized or how it works."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_name": {
                    "type": "string",
                    "description": "Repo in 'owner/name' form. The project must be already indexed.",
                },
                "question": {
                    "type": "string",
                    "description": (
                        "The user's project-understanding question, e.g. 'explain the structure', "
                        "'which frameworks are used?', or 'how does the request flow work?'"
                    ),
                },
            },
            "required": ["repo_name", "question"],
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
    {
        "name": "apply_review_fixes",
        "description": (
            "Apply the patches from a completed PR review as a new fix-branch pull request on GitHub. "
            "Only patches that were verified to apply cleanly are used. Creates a 'bot/fix-prN' branch, "
            "commits the patches, and opens a PR. Requires GITHUB_TOKEN with repo write access. "
            "Use when the user wants to actually push the bot's suggested fixes to GitHub — "
            "ALWAYS ask for confirmation before calling this tool (it writes to the user's repo)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pr_url": {
                    "type": "string",
                    "description": "Full GitHub PR URL of the review whose patches to apply, e.g. https://github.com/owner/repo/pull/123",
                },
            },
            "required": ["pr_url"],
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
            f"is running in the background. Open the review for live progress. "
            f"Once complete, I can push the verified patches as a fix-branch PR — just say 'apply the fixes'."
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


def _read_repo_file(repo_dir, rel_path: str, max_chars: int = 4000) -> str:
    """Read a small text file from a repo clone for project exploration."""
    try:
        path = repo_dir / rel_path
        if not path.exists() or not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text) > max_chars:
            return text[:max_chars] + "\n... (truncated)"
        return text
    except Exception:
        return ""


def _repo_tree(repo_dir, max_depth: int = 2, max_entries: int = 120) -> str:
    """Return a compact directory tree, skipping generated/vendor folders."""
    excluded = {
        ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build",
        "data", ".next", "coverage",
    }
    lines: list[str] = []

    def walk(path, prefix: str, depth: int):
        if len(lines) >= max_entries or depth > max_depth:
            return
        try:
            children = sorted(
                [p for p in path.iterdir() if p.name not in excluded and not p.name.startswith(".")],
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except Exception:
            return
        for child in children:
            if len(lines) >= max_entries:
                break
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{prefix}{child.name}{suffix}")
            if child.is_dir():
                walk(child, prefix + "  ", depth + 1)

    walk(repo_dir, "", 1)
    if len(lines) >= max_entries:
        lines.append("... (truncated)")
    return "\n".join(lines)


def _detect_entrypoints(repo_dir) -> list[str]:
    candidates = [
        "api/main.py", "main.py", "app.py", "manage.py", "server.py",
        "src/main.py", "src/app.py", "frontend/src/main.tsx",
        "frontend/src/main.ts", "frontend/src/App.tsx",
        "package.json", "frontend/package.json", "docker-compose.yml",
        "Dockerfile", "pyproject.toml",
    ]
    return [p for p in candidates if (repo_dir / p).exists()]


def _dependency_context(repo_dir) -> str:
    parts: list[str] = []
    for rel in [
        "requirements.txt",
        "requirement.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "package.json",
        "frontend/package.json",
        "docker-compose.yml",
        "Dockerfile",
    ]:
        text = _read_repo_file(repo_dir, rel, max_chars=2500)
        if text:
            parts.append(f"## {rel}\n{text}")
    return "\n\n".join(parts)


def _graph_overview(project_id: str) -> str:
    """Summarize indexed Neo4j graph shape. Fail closed if Neo4j is unavailable."""
    try:
        from neo4j import GraphDatabase
        from src_bot.config.config import configs

        driver = GraphDatabase.driver(
            configs.APP_NEO4J_URL,
            auth=(configs.APP_NEO4J_USER, configs.APP_NEO4J_PASSWORD),
        )
        try:
            with driver.session() as session:
                modules = session.run(
                    """
                    MATCH (m:Module {project_id: $pid})
                    OPTIONAL MATCH (m)-[:DEFINES]->(n:CodeNode {project_id: $pid})
                    RETURN m.file_path AS file, count(n) AS definitions
                    ORDER BY definitions DESC, file ASC
                    LIMIT 12
                    """,
                    pid=project_id,
                ).data()
                callers = session.run(
                    """
                    MATCH (n:CodeNode {project_id: $pid})
                    OPTIONAL MATCH (n)-[:CALLS]->(c:CodeNode {project_id: $pid})
                    WITH n, count(c) AS outgoing
                    WHERE outgoing > 0
                    RETURN n.file_path AS file, n.qualified_name AS name, outgoing
                    ORDER BY outgoing DESC
                    LIMIT 10
                    """,
                    pid=project_id,
                ).data()
        finally:
            driver.close()

        lines = ["Top files by indexed definitions:"]
        lines.extend(f"- {r['file']}: {r['definitions']}" for r in modules)
        lines.append("\nTop call-heavy functions/classes:")
        lines.extend(
            f"- {r['name']} ({r['file']}): calls {r['outgoing']} node(s)"
            for r in callers
        )
        return "\n".join(lines)
    except Exception as e:
        return f"Graph overview unavailable: {type(e).__name__}: {str(e)[:120]}"


async def tool_explore_project(repo_name: str, question: str, **kwargs) -> dict:
    """Explain structure/frameworks/flow of an indexed project."""
    project = await get_project_by_repo_name(repo_name)
    if not project:
        return {
            "summary": f"Project '{repo_name}' is not onboarded.",
            "render": {
                "kind": "error",
                "message": f"'{repo_name}' is not onboarded yet.",
                "suggestion": f"Add https://github.com/{repo_name} first, then ask again.",
            },
        }
    if project.status != "indexed":
        return {
            "summary": f"Project '{repo_name}' is not ready (status: {project.status}).",
            "render": {
                "kind": "error",
                "message": f"Project status is {project.status}. Wait for indexing to finish.",
            },
        }

    from api.project_indexer import get_project_dir
    from langchain_core.messages import HumanMessage
    from src_bot.llm.router import get_llm

    repo_dir = get_project_dir(project.id)
    if not repo_dir.exists():
        return {
            "summary": f"Workspace for '{repo_name}' is missing.",
            "render": {
                "kind": "error",
                "message": f"Workspace for '{repo_name}' is missing after server restart.",
                "suggestion": f"Reindex {repo_name} first, then ask again.",
            },
        }

    summary_text = _read_repo_file(repo_dir, "PROJECT_SUMMARY.txt", max_chars=1500)
    readme = ""
    for candidate in ("README.md", "README.rst", "README.txt"):
        readme = _read_repo_file(repo_dir, candidate, max_chars=4500)
        if readme:
            readme = f"## {candidate}\n{readme}"
            break

    entrypoints = _detect_entrypoints(repo_dir)
    tree = _repo_tree(repo_dir)
    dependencies = _dependency_context(repo_dir)
    graph = _graph_overview(project.id)

    context = (
        f"Repository: {project.repo_name}\n"
        f"Status: {project.status}\n"
        f"Indexed files: {project.file_count}\n"
        f"Graph nodes: {project.node_count}, edges: {project.edge_count}\n"
        f"Last commit: {project.last_commit_sha or 'unknown'}\n\n"
        f"PROJECT_SUMMARY:\n{summary_text or '(none)'}\n\n"
        f"DIRECTORY TREE:\n{tree or '(empty)'}\n\n"
        f"ENTRYPOINT CANDIDATES:\n" + "\n".join(f"- {p}" for p in entrypoints) + "\n\n"
        f"DEPENDENCY / CONFIG FILES:\n{dependencies or '(none found)'}\n\n"
        f"GRAPH OVERVIEW:\n{graph}\n\n"
        f"README:\n{readme or '(none found)'}"
    )

    prompt = (
        "You are a senior engineer helping a teammate understand an indexed repository.\n"
        "Answer only from the repository context below. If something is not visible, say so.\n"
        "Match the user's language. Prefer concise sections and cite file paths when useful.\n"
        "Do not suggest code changes.\n\n"
        f"User question: {question}\n\n"
        f"Repository context:\n{context[:18000]}"
    )

    try:
        loop = asyncio.get_event_loop()
        llm = get_llm(role="chat", temperature=0)
        response = await loop.run_in_executor(
            None, lambda: llm.invoke([HumanMessage(content=prompt)])
        )
        answer = (response.content if hasattr(response, "content") else str(response)).strip()
    except Exception as e:
        answer = (
            f"Could not synthesize a full answer because the LLM call failed: {e}\n\n"
            f"Available context:\n- Entry points: {', '.join(entrypoints) or 'none detected'}\n"
            f"- Indexed files: {project.file_count}\n- Graph nodes: {project.node_count}\n\n"
            f"Directory tree:\n{tree[:2000]}"
        )

    return {
        "summary": f"Explored {repo_name}: {question[:120]}",
        "render": {
            "kind": "project_exploration",
            "repo_name": repo_name,
            "question": question,
            "answer": answer,
            "entrypoints": entrypoints,
            "tree": tree,
        },
    }


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


async def tool_apply_review_fixes(pr_url: str, **kwargs) -> dict:
    """Apply patches from the latest completed review of a PR as a fix-branch PR on GitHub."""
    import os
    from api.reviewer import parse_pr_url, _checkout_pr_base, _cleanup_workdir

    try:
        repo_name, pr_number = parse_pr_url(pr_url)
    except ValueError as e:
        return {"summary": f"Invalid PR URL: {e}", "render": {"kind": "error", "message": str(e)}}

    # Find the most recent completed review for this PR
    reviews = await list_reviews()
    completed = [r for r in reviews if r.pr_url == pr_url and r.status == "completed"]
    if not completed:
        return {
            "summary": (
                f"No completed review found for PR #{pr_number}. "
                f"Run `review_pr` first, then call this when the review finishes."
            ),
            "render": {"kind": "error", "message": "No completed review found for this PR."},
        }

    record = completed[0]  # list_reviews returns newest first

    clean_patches = [fr for fr in record.file_reviews if fr.applies_cleanly and fr.patch]
    if not clean_patches:
        return {
            "summary": (
                f"Review #{pr_number} has no patches that apply cleanly. "
                f"The harness may need another attempt, or the bugs require manual fixes."
            ),
            "render": {
                "kind": "error",
                "message": f"No applicable patches in review {record.id[:8]}.",
            },
        }

    github_token = os.getenv("GITHUB_TOKEN", "")
    if not github_token:
        return {
            "summary": (
                "GITHUB_TOKEN is not set — I can't push to GitHub. "
                "Add a token with 'repo' scope to .env and restart the server."
            ),
            "render": {"kind": "error", "message": "GITHUB_TOKEN not set in .env."},
        }

    try:
        from github import Github
        from api.github_app import create_fix_branch_pr
        from api.project_indexer import get_project_dir

        gh = Github(github_token)
        repo = gh.get_repo(repo_name)
        pr = repo.get_pull(pr_number)

        # Ensure project clone is available on disk
        project_id = record.project_id
        if not project_id or not get_project_dir(project_id).exists():
            return {
                "summary": (
                    f"The project workspace for '{repo_name}' is missing (server may have restarted). "
                    f"Reindex the project first: 'Reindex {repo_name}', then retry."
                ),
                "render": {
                    "kind": "error",
                    "message": "Project workspace missing — reindex required before fix branch can be created.",
                },
            }

        workdir_id = f"fixbranch_{record.id}"
        workdir = _checkout_pr_base(project_id, pr.base.sha, workdir_id, base_ref=pr.base.ref)

        loop = asyncio.get_event_loop()
        fix_pr_url = await loop.run_in_executor(
            None, create_fix_branch_pr,
            repo, pr, record.file_reviews, workdir, github_token,
        )
        _cleanup_workdir(workdir_id)

        if fix_pr_url:
            return {
                "summary": (
                    f"Fix branch PR created: {fix_pr_url}. "
                    f"Applied {len(clean_patches)} patch(es) to branch 'bot/fix-pr{pr_number}'."
                ),
                "render": {
                    "kind": "fix_pr",
                    "fix_pr_url": fix_pr_url,
                    "patches_applied": len(clean_patches),
                    "files": [fr.file_path for fr in clean_patches],
                },
            }
        else:
            return {
                "summary": (
                    "Could not push the fix branch. "
                    "Verify that GITHUB_TOKEN has 'repo' write scope."
                ),
                "render": {"kind": "error", "message": "Fix branch push failed — check token permissions."},
            }

    except Exception as e:
        return {
            "summary": f"Error creating fix branch: {e}",
            "render": {"kind": "error", "message": str(e)[:300]},
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
        "explore_project": tool_explore_project,
        "recent_reviews": tool_recent_reviews,
        "apply_review_fixes": tool_apply_review_fixes,
    }
