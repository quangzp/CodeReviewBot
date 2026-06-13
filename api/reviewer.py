"""
Per-PR reviewer.

Key principle: this code NEVER ingests the repo into Neo4j. The project's
graph must already exist (built once during project onboarding). This module
only queries the existing graph + reads file contents at the PR's base commit.

Flow:
  1. Verify the project exists and is indexed (caller already did this)
  2. Fetch PR data from GitHub (title, description, changed files, base SHA)
  3. Check out the PR's base commit into a per-PR working directory
     (does NOT mutate the project's main clone)
  4. For each changed file: query the existing graph + run 3-phase pipeline
  5. Memory: read profile before, write facts after
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import re
import sys
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_bot.config.config import configs
from src_bot.llm.router import get_llm
from src_bot.observability.langfuse_ctx import langfuse_span, langfuse_update_current_span
from api.models import ReviewRecord, FileReviewResult
from api.database import update_review, get_project
from api.project_indexer import get_project_dir

# Per-PR working directories — /tmp is fine here since these are short-lived
# (created at review start, deleted when review finishes)
PR_WORKDIRS = Path("/tmp/codereviewbot_pr_workdirs")


def parse_pr_url(pr_url: str) -> tuple[str, int]:
    """Parse 'https://github.com/owner/repo/pull/123' → ('owner/repo', 123)"""
    match = re.match(r'https://github\.com/([^/]+/[^/]+)/pull/(\d+)', pr_url.strip())
    if not match:
        raise ValueError(f"Invalid GitHub PR URL: {pr_url}")
    return match.group(1), int(match.group(2))


def _checkout_pr_base(
    project_id: str,
    base_sha: str,
    review_id: str,
    base_ref: str = "",
) -> Path:
    """
    Create a per-PR working directory by copying the project's clone and
    checking out the PR's base commit there. Doesn't disturb the project's
    main clone (which stays on HEAD of default branch).

    base_ref: branch name on the base repo (e.g. "main"). Used to fetch
    the commit from GitHub when the local shallow clone doesn't have it.

    Returns the path to the per-PR working directory.
    """
    PR_WORKDIRS.mkdir(parents=True, exist_ok=True)
    workdir = PR_WORKDIRS / review_id

    # Clean up any leftover
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)

    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        raise RuntimeError(
            f"Project clone not found at {project_dir}. "
            f"The project may need to be re-indexed."
        )

    # Get the GitHub URL from the local project clone so we can fetch directly
    # from GitHub when the shallow clone doesn't have the base commit.
    github_url_result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=project_dir, capture_output=True, text=True, timeout=10,
    )
    github_url = github_url_result.stdout.strip() if github_url_result.returncode == 0 else None

    # Use a lightweight worktree clone (fast, shares git objects)
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(project_dir), str(workdir)],
        capture_output=True, text=True, timeout=120, check=True,
    )

    # Try fast path: base_sha already available in shared object store
    result = subprocess.run(
        ["git", "checkout", base_sha],
        cwd=workdir, capture_output=True, text=True, timeout=60,
    )
    if result.returncode == 0:
        return workdir

    # base_sha not available locally (shallow clone with --depth 1 during indexing).
    # Fetch the base branch directly from GitHub, then checkout the exact SHA.
    if github_url:
        # Prefer fetching by branch name — more reliable than bare SHA fetch
        fetch_ref = base_ref if base_ref else base_sha
        subprocess.run(
            ["git", "fetch", "--depth=50", github_url, fetch_ref],
            cwd=workdir, capture_output=True, text=True, timeout=300,
        )
        result = subprocess.run(
            ["git", "checkout", base_sha],
            cwd=workdir, capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            return workdir

        # If exact SHA still not found after branch fetch, try HEAD of that branch
        # (close enough for review purposes when base is very recent)
        fetch_head = subprocess.run(
            ["git", "rev-parse", "FETCH_HEAD"],
            cwd=workdir, capture_output=True, text=True, timeout=10,
        )
        if fetch_head.returncode == 0:
            subprocess.run(
                ["git", "checkout", fetch_head.stdout.strip()],
                cwd=workdir, capture_output=True, text=True, timeout=60, check=True,
            )
            logger.warning(
                "Could not checkout exact base SHA %s; using FETCH_HEAD instead. "
                "Review diff context may be slightly off.",
                base_sha[:7],
            )
            return workdir

    # Last resort: fall back to local HEAD (project was indexed at HEAD)
    subprocess.run(
        ["git", "checkout", "HEAD"],
        cwd=workdir, capture_output=True, text=True, timeout=60, check=True,
    )
    logger.warning(
        "Could not checkout base SHA %s; falling back to HEAD of local clone. "
        "Review diff context may be slightly off.",
        base_sha[:7],
    )
    return workdir


def _cleanup_workdir(review_id: str):
    """Remove the per-PR working directory."""
    workdir = PR_WORKDIRS / review_id
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)


def _synthesize_dependency_fix(
    file_reviews: list,
    workdir: Path,
) -> "Optional[FileReviewResult]":
    """
    Detect 'No module named X' faults appearing across 2+ files and generate
    a requirements.txt patch. The per-file harness cannot fix missing dependencies
    by patching Python source files — this step handles that class of fault.
    """
    import difflib

    _missing_pat = re.compile(
        r"no module named ['\"]?([\w][\w\.\-]*)['\"]?", re.IGNORECASE
    )

    module_counts: dict = {}
    for fr in file_reviews:
        for m in _missing_pat.finditer(fr.phase2_fault or ""):
            top = m.group(1).split(".")[0].lower()
            module_counts[top] = module_counts.get(top, 0) + 1

    if not module_counts:
        return None

    # Require 2+ files to agree on the same missing module (noise filter)
    confirmed = {mod for mod, cnt in module_counts.items() if cnt >= 2}
    if not confirmed:
        return None

    # Find requirements.txt (check common locations)
    dep_candidates = [
        "requirements.txt", "requirements-dev.txt",
        "requirements/base.txt", "requirements/common.txt",
    ]
    dep_path = dep_filename = None
    for df in dep_candidates:
        c = workdir / df
        if c.exists():
            dep_path, dep_filename = c, df
            break

    if dep_path is None:
        return None

    original_content = dep_path.read_text(errors="ignore")

    # Map common import names → PyPI install names
    _PYPI: dict = {
        "langgraph": "langgraph",
        "langchain": "langchain",
        "langchain_core": "langchain-core",
        "langchain_groq": "langchain-groq",
        "langchain_openai": "langchain-openai",
        "langchain_ollama": "langchain-ollama",
        "langchain_community": "langchain-community",
        "sklearn": "scikit-learn",
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "bs4": "beautifulsoup4",
        "yaml": "PyYAML",
        "dotenv": "python-dotenv",
        "jwt": "PyJWT",
        "aiohttp": "aiohttp",
        "httpx": "httpx",
        "pydantic": "pydantic",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "sqlalchemy": "SQLAlchemy",
        "openai": "openai",
        "anthropic": "anthropic",
        "neo4j": "neo4j",
        "weaviate": "weaviate-client",
    }

    already = original_content.lower()
    to_add = []
    for mod in sorted(confirmed):
        pkg = _PYPI.get(mod, mod.replace("_", "-"))
        if pkg.lower() not in already and mod not in already:
            to_add.append(pkg)

    if not to_add:
        return None

    # Build new requirements.txt content
    sep = "\n" if original_content.endswith("\n") else "\n\n"
    new_content = (
        original_content + sep
        + "# Added by CodeReviewBot — missing dependencies detected\n"
        + "\n".join(to_add) + "\n"
    )

    patch = "".join(difflib.unified_diff(
        original_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{dep_filename}",
        tofile=f"b/{dep_filename}",
    ))
    if not patch:
        return None

    # Verify the patch applies cleanly against the current workdir state
    try:
        r = subprocess.run(
            ["git", "apply", "--check", "--whitespace=fix"],
            input=patch, cwd=workdir, capture_output=True, text=True, timeout=10,
        )
        applies = r.returncode == 0
    except Exception:
        applies = False

    fr = FileReviewResult(file_path=dep_filename)
    fr.phase1_found = True
    fr.phase2_fault = (
        f"Missing dependencies: {', '.join(sorted(confirmed))}. "
        f"Imported across {sum(module_counts[m] for m in confirmed)} file(s) "
        f"but not listed in {dep_filename}."
    )
    fr.patch = patch
    fr.applies_cleanly = applies
    fr.eval_score = 5 if applies else 3
    fr.eval_reason = (
        f"Adds {', '.join(to_add)} to {dep_filename}." if applies
        else "Dependency patch generated but git apply check failed."
    )
    fr.risk_level = "low"
    fr.review_comments = [{
        "severity": "bug",
        "line": None,
        "message": (
            f"Package(s) {', '.join(to_add)} are imported by the code "
            f"but missing from {dep_filename}. "
            f"The project will fail at runtime when installed from this file."
        ),
    }]
    return fr


async def run_pr_review(
    record: ReviewRecord,
    event_queue: asyncio.Queue,
):
    """
    Per-PR review pipeline.

    Assumes: record.project_id is set to a project with status='indexed'.
    Caller (POST /api/reviews) is responsible for that check.
    """
    from github import Github
    from neo4j import GraphDatabase

    async def emit(event_type: str, data: dict):
        await event_queue.put({"type": event_type, **data})

    loop = asyncio.get_event_loop()
    workdir: Optional[Path] = None
    neo4j_driver = None

    # Open a Langfuse root span (= trace) for this review (v3 API).
    # langfuse_trace() is a no-op when LANGFUSE_ENABLED=false.
    from src_bot.observability.langfuse_ctx import langfuse_trace
    _lf_trace = langfuse_trace(
        trace_id=record.id,
        session_id=record.project_id,
        user_id=record.author_login or "unknown",
        tags=["pr_review"],
        metadata={"pr_url": record.pr_url, "repo": record.repo_name},
    )
    _lf_trace.__enter__()

    try:
        record.status = "running"
        await update_review(record.id, record)
        await emit("status", {"message": "Starting review..."})

        # ----------------------------------------------------------------
        # 0. Verify project exists and is indexed
        # ----------------------------------------------------------------
        if not record.project_id:
            raise RuntimeError("Review has no project_id — this shouldn't happen")

        project = await get_project(record.project_id)
        if not project:
            raise RuntimeError(f"Project {record.project_id} not found")
        if project.status != "indexed":
            raise RuntimeError(
                f"Project '{project.repo_name}' is not ready (status: {project.status}). "
                f"Run a re-index first."
            )

        # Check the clone actually exists on disk — DB says "indexed" but /tmp is wiped on reboot
        project_dir = get_project_dir(record.project_id)
        if not project_dir.exists():
            raise RuntimeError(
                f"Workspace for '{project.repo_name}' is missing (server was restarted and /tmp was cleared). "
                f"Please reindex the project first, then retry the review."
            )

        await emit("status", {
            "message": f"Using indexed project {project.repo_name} "
                       f"({project.node_count} nodes, {project.file_count} files)"
        })

        # ----------------------------------------------------------------
        # 1. Fetch PR from GitHub
        # ----------------------------------------------------------------
        repo_name, pr_number = parse_pr_url(record.pr_url)
        await emit("status", {"message": f"Fetching PR #{pr_number} from {repo_name}..."})

        from api.github_app import get_token_for_repo
        github_token = get_token_for_repo(repo_name)

        if github_token:
            gh = Github(github_token)
            await emit("status", {"message": "Authenticated to GitHub"})
        else:
            # Anonymous mode — works for public repos, 60 req/hour limit
            gh = Github()
            await emit("status", {
                "message": (
                    "⚠️ No GITHUB_TOKEN set — using anonymous GitHub API "
                    "(60 req/hour limit; only public repos work)"
                ),
            })

        try:
            repo = gh.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
        except Exception as e:
            # Translate common errors to actionable messages
            err_str = str(e)
            if "401" in err_str or "Bad credentials" in err_str:
                raise RuntimeError(
                    "GitHub rejected the token (401 Bad credentials). "
                    "Either your GITHUB_TOKEN is invalid, or remove it from .env "
                    "to use anonymous mode for public repos."
                )
            if "403" in err_str and ("rate limit" in err_str.lower() or "API rate" in err_str):
                raise RuntimeError(
                    "GitHub API rate limit exceeded. "
                    "Set a valid GITHUB_TOKEN in .env to get 5,000 req/hour "
                    "instead of 60 req/hour anonymous."
                )
            if "404" in err_str:
                raise RuntimeError(
                    f"Could not access {repo_name}#{pr_number}. "
                    "If the repo is private, set GITHUB_TOKEN in .env. "
                    "If it's public, double-check the URL."
                )
            raise

        author_login = pr.user.login if pr.user else "unknown"
        record.author_login = author_login

        # user_request: PR title + description (what the user wants)
        # user_commit:  the actual diff (what they pushed)
        user_request_text = f"PR #{pr_number}: {pr.title}\n\n{pr.body or ''}"

        changed_files = [
            f for f in pr.get_files()
            if f.filename.endswith('.py') and f.status != 'removed'
        ]

        if not changed_files:
            await emit("warning", {"message": "No Python files modified in this PR"})
            record.status = "completed"
            record.total_patches = 0
            await update_review(record.id, record)
            await emit("done", {"total_patches": 0, "total_files": 0})
            return

        changed_paths = [f.filename for f in changed_files]
        await emit("status", {"message": f"Found {len(changed_files)} Python files modified"})

        # ----------------------------------------------------------------
        # 2. Check out PR base commit (separate working dir, no ingest!)
        # ----------------------------------------------------------------
        await emit("status", {"message": f"Checking out base commit {pr.base.sha[:7]}..."})
        workdir = await loop.run_in_executor(
            None,
            lambda: _checkout_pr_base(project.id, pr.base.sha, record.id, base_ref=pr.base.ref),
        )

        # ----------------------------------------------------------------
        # 3. Set up LLMs + Neo4j (reads only)
        # ----------------------------------------------------------------
        # llm       : phase1 file localization + memory extraction
        # llm_reason: memory fact extraction (reason LLM — used after review)
        # The harness (api/agent/harness.py) creates its own role-specific LLMs
        # internally so per-file routing stays self-contained.
        llm = get_llm(role="generation", temperature=0)
        llm_reason = get_llm(role="reason", temperature=0)  # memory extraction only
        neo4j_driver = GraphDatabase.driver(
            configs.APP_NEO4J_URL,
            auth=(configs.APP_NEO4J_USER, configs.APP_NEO4J_PASSWORD),
        )

        # Import pipeline helpers (they only QUERY the graph, no ingestion)
        from run_swebench_v2 import (
            phase1_localize_file, _get_graph_context_for_file,
        )
        from api.agent.harness import run_file_review
        from src_bot.memory.memory_neo4j import build_memory_context, record_review as graphiti_record_review
        from src_bot.memory.extractor import extract_facts_from_review

        # ----------------------------------------------------------------
        # 4. MEMORY READ — what does Graphiti already know about this dev?
        # ----------------------------------------------------------------
        memory_context = await build_memory_context(author_login, repo_name)
        if memory_context:
            await emit("memory", {
                "message": f"Loaded memory profile for @{author_login}",
                "context": memory_context[:500],
            })
            issue_text = f"{memory_context}\n\n---\n\n{user_request_text}"
        else:
            await emit("memory", {
                "message": f"No prior memory for @{author_login} — first review",
            })
            issue_text = user_request_text

        # Inject project summary into issue_text for Planner context
        _summary_file = get_project_dir(project.id) / "PROJECT_SUMMARY.txt"
        if _summary_file.exists():
            _proj_ctx = _summary_file.read_text(encoding="utf-8")
            issue_text = f"[Repository context]\n{_proj_ctx}\n---\n{issue_text}"

        # ----------------------------------------------------------------
        # 5. Per-file review (queries existing graph only)
        # ----------------------------------------------------------------
        project_id = project.id  # the namespace inside Neo4j
        file_reviews: list[FileReviewResult] = []
        patches_generated = 0

        with neo4j_driver.session() as session:
            for i, changed_file in enumerate(changed_files):
                file_path = changed_file.filename
                await emit("file_start", {
                    "file": file_path, "index": i, "total": len(changed_files),
                })

                file_result = FileReviewResult(file_path=file_path)

                # Phase 1: confirm the file (PR tells us which file; we sanity-check)
                await emit("phase", {"phase": 1, "file": file_path, "status": "running"})
                try:
                    with langfuse_span("phase1_file_localization", metadata={"file": file_path}):
                        _lf_ctx = contextvars.copy_context()
                        found_file = await loop.run_in_executor(
                            None,
                            lambda: _lf_ctx.run(
                                phase1_localize_file,
                                llm, issue_text, repo_name, workdir, session, project_id,
                            ),
                        )
                        langfuse_update_current_span(
                            metadata={
                                "file": file_path,
                                "found_file": found_file,
                                "status": "done",
                            }
                        )
                except Exception as _p1_err:
                    # LLM auth failure (401) or connectivity error — fall back to
                    # using the file path from GitHub directly.  Do NOT crash the
                    # whole review; the harness will surface LLM errors per-file.
                    logger.warning(
                        "[phase1] LLM call failed for %s (%s) — using file path directly",
                        file_path, _p1_err,
                    )
                    found_file = file_path if (workdir / file_path).exists() else None
                actual_file = file_path if (workdir / file_path).exists() else found_file
                file_result.phase1_found = bool(actual_file)
                await emit("phase", {
                    "phase": 1, "file": file_path,
                    "found": actual_file, "status": "done",
                })
                if not actual_file:
                    file_reviews.append(file_result)
                    continue

                # ── Risk Scoring (CHID) + Security Escalation ────────────────
                try:
                    from src_bot.risk.scorer import RiskScorer, RiskInput
                    from src_bot.risk.git_enrichment import compute_file_stats, is_new_contributor as _is_new_contributor
                    _git_stats = compute_file_stats(workdir, [actual_file])
                    _fgs = _git_stats.get(actual_file)
                    _is_new = _is_new_contributor(workdir, [actual_file], author_login)
                    _risk_result = RiskScorer().score(RiskInput(
                        blast_radius_size=len(changed_files),
                        pr_size_lines=getattr(changed_file, "changes", 0),
                        bug_frequency=_fgs.bug_frequency if _fgs else 0,
                        contributor_churn=_fgs.contributor_churn if _fgs else 0,
                        is_new_contributor=_is_new,
                    ))
                    file_result.risk_level = _risk_result.level
                except Exception as e:
                    logger.warning("[risk] Risk scoring failed for %s: %s", actual_file, e)

                _SECURITY_PATTERNS = {
                    "auth", "password", "token", "secret", "encrypt",
                    "permission", "role", "admin", "credential", "oauth",
                    "jwt", "session", "csrf", "injection", "sql",
                }
                if any(p in actual_file.lower() for p in _SECURITY_PATTERNS):
                    file_result.risk_level = "high"

                _RISK_TOP_K = {"low": 10, "medium": 15, "high": 25}
                _top_k = _RISK_TOP_K.get(file_result.risk_level, 15)
                # ── End Risk Scoring ──────────────────────────────────────────

                # Graph context (depth scales with risk) — uses session
                graph_context = _get_graph_context_for_file(
                    session, actual_file, project_id, top_k=_top_k
                )

                # ── LangGraph Harness: analyze_fault → plan → generate → verify
                #    → reflect (retry) → orchestrate (dynamic routing by reason LLM) ──
                with langfuse_span("harness_file_review", metadata={"file": actual_file, "risk": file_result.risk_level}):
                    _lf_ctx = contextvars.copy_context()
                    harness_result = await loop.run_in_executor(
                        None,
                        lambda: _lf_ctx.run(
                            run_file_review,
                            issue_text, actual_file, graph_context, workdir,
                            project_id, file_result.risk_level, configs.REFLEXION_MAX_RETRIES,
                        ),
                    )
                    langfuse_update_current_span(
                        metadata={
                            "file": actual_file,
                            "risk": file_result.risk_level,
                            "applies_cleanly": harness_result.applies_cleanly,
                            "eval_score": harness_result.eval_score,
                            "orchestrator_decision": harness_result.orchestrator_decision,
                            "reflexion_attempts": harness_result.reflexion_attempts,
                            "review_comments_count": len(harness_result.review_comments or []),
                            "events_count": len(harness_result.events or []),
                            "patch_size": len(harness_result.patch or ""),
                        },
                        output={
                            "fault": harness_result.fault_desc[:300],
                            "eval_reason": harness_result.eval_reason[:300],
                        },
                    )

                # Replay SSE events collected inside the harness
                for ev_type, ev_data in harness_result.events:
                    await emit(ev_type, ev_data)

                # Map harness result → FileReviewResult
                file_result.phase2_fault = harness_result.fault_desc[:200]
                file_result.patch = harness_result.patch
                file_result.applies_cleanly = harness_result.applies_cleanly
                file_result.eval_score = harness_result.eval_score
                file_result.eval_reason = harness_result.eval_reason
                file_result.reflexion_attempts = harness_result.reflexion_attempts
                file_result.review_comments = harness_result.review_comments
                if harness_result.patch:
                    patches_generated += 1
                # ── End Harness ────────────────────────────────────────────────

                file_reviews.append(file_result)

        # ── Dependency fix synthesis (cross-file pattern detection) ──────────
        # When multiple files share the same "No module named X" fault, the
        # per-file harness can't fix it (you don't patch .py files for a missing
        # pip package). This step detects that pattern and patches requirements.txt.
        if workdir:
            dep_fix = _synthesize_dependency_fix(file_reviews, workdir)
            if dep_fix:
                file_reviews.append(dep_fix)
                if dep_fix.patch and dep_fix.applies_cleanly:
                    patches_generated += 1
                await emit("file_start", {
                    "file": dep_fix.file_path,
                    "index": len(changed_files),
                    "total": len(changed_files) + 1,
                })
                await emit("phase", {
                    "phase": 2, "file": dep_fix.file_path,
                    "fault": dep_fix.phase2_fault[:120], "status": "done",
                })
                await emit("phase", {
                    "phase": 3, "file": dep_fix.file_path,
                    "attempt": 1, "status": "generated",
                    "patch_size": len(dep_fix.patch),
                })
                logger.info(
                    "[reviewer] Synthesized dependency fix for %s: %s",
                    dep_fix.file_path, dep_fix.phase2_fault[:80],
                )

        # ── Harness: Friction Summary SSE (repo-harness TRACE_SPEC) ──────────
        _scores = [fr.eval_score for fr in file_reviews if fr.eval_score is not None]
        _friction = []
        for _fr in file_reviews:
            if _fr.eval_score is not None and _fr.eval_score < 3:
                _friction.append(
                    f"{_fr.file_path}: quality={_fr.eval_score}/5 ({_fr.eval_reason[:80]})"
                )
            if _fr.reflexion_attempts > 2:
                _friction.append(
                    f"{_fr.file_path}: needed {_fr.reflexion_attempts} reflexion attempts"
                )
            if not _fr.applies_cleanly and _fr.patch:
                _friction.append(f"{_fr.file_path}: patch never applied cleanly")
        await emit("friction_summary", {
            "files_reviewed": len(file_reviews),
            "patches_generated": patches_generated,
            "avg_eval_score": round(sum(_scores) / len(_scores), 2) if _scores else None,
            "friction": _friction,
            "quality": "good" if not _friction else "needs_attention",
        })
        # ── End Friction Summary ────────────────────────────────────────────

        # ----------------------------------------------------------------
        # 6a. META-REVIEW — overall PR assessment (configurable LLM)
        # ----------------------------------------------------------------
        if configs.META_REVIEW_ENABLED:
            try:
                import json, re as _re
                from langchain_core.messages import HumanMessage as _HM

                # Determine meta-review LLM (OpenAI GPT-4 if configured, else reason LLM)
                _meta_provider = configs.META_REVIEW_LLM_PROVIDER or None
                _meta_model = configs.META_REVIEW_LLM_MODEL or None
                if _meta_provider:
                    _meta_llm = get_llm(provider=_meta_provider, model=_meta_model, temperature=0)
                else:
                    _meta_llm = get_llm(role="reason", temperature=0)

                # Build concise summary of all file reviews for the meta-review prompt
                _review_blocks = []
                for _fr in file_reviews[:8]:  # cap at 8 files to stay within token budget
                    if not _fr.phase2_fault and not _fr.review_comments:
                        continue
                    _comments = "\n".join(
                        f"  [{c.get('severity','?')}] L{c.get('line','?')}: {c.get('message','')}"
                        for c in (_fr.review_comments or [])[:4]
                    )
                    _review_blocks.append(
                        f"### {_fr.file_path} (risk={_fr.risk_level}, score={_fr.eval_score}/5)\n"
                        f"Fault: {_fr.phase2_fault[:120]}\n"
                        f"Patch: {'✓ applies' if _fr.applies_cleanly else '✗ did not apply'}\n"
                        + (_comments if _comments else "  (no extra comments)")
                    )

                _meta_prompt = (
                    f"You are a senior software architect. Review this automated PR analysis and provide an overall assessment.\n\n"
                    f"PR: {issue_text[:400]}\n\n"
                    f"FILE ANALYSIS RESULTS:\n{'---'.join(_review_blocks)}\n\n"
                    f"Respond with ONLY this JSON (no explanation):\n"
                    f'{{"summary":"overall PR quality in 2-3 sentences",'
                    f'"priority_issues":["top issue 1","top issue 2","top issue 3"],'
                    f'"refactoring_opportunities":["suggestion 1","suggestion 2"],'
                    f'"overall_score":<1-5>}}'
                )

                with langfuse_span("meta_review", metadata={"files": len(file_reviews)}):
                    _lf_ctx = contextvars.copy_context()
                    _meta_raw = await loop.run_in_executor(
                        None,
                        lambda: _lf_ctx.run(
                            _meta_llm.invoke,
                            [_HM(content=_meta_prompt)],
                        ),
                    )
                    _meta_text = (_meta_raw.content or "").strip()
                    # Strip DeepSeek thinking block if present
                    _meta_text = _re.sub(r'<think>.*?</think>', '', _meta_text, flags=_re.DOTALL).strip()
                    _meta_match = _re.search(r'\{.*\}', _meta_text, _re.DOTALL)
                    if _meta_match:
                        meta_review = json.loads(_meta_match.group())
                        record.meta_review = meta_review
                        langfuse_update_current_span(
                            metadata={
                                "status": "parsed",
                                "overall_score": meta_review.get("overall_score"),
                                "priority_issues_count": len(meta_review.get("priority_issues") or []),
                                "refactoring_opportunities_count": len(
                                    meta_review.get("refactoring_opportunities") or []
                                ),
                            },
                            output={
                                "summary": str(meta_review.get("summary", ""))[:500],
                            },
                        )
                        await emit("meta_review", meta_review)
            except Exception as _mr_err:
                logger.warning("[meta_review] Failed (non-fatal): %s", _mr_err)

        # ----------------------------------------------------------------
        # 6b. MEMORY WRITE — learn from this review
        # ----------------------------------------------------------------
        await emit("status", {"message": "Learning from this review..."})

        review_output = "\n\n".join([
            f"### {fr.file_path}\nFault: {fr.phase2_fault}\n"
            f"Patch (applies: {fr.applies_cleanly}):\n```diff\n{fr.patch[:1500]}\n```"
            for fr in file_reviews if fr.patch or fr.phase2_fault
        ])

        try:
            fact = await loop.run_in_executor(
                None,
                lambda: extract_facts_from_review(
                    llm_reason,
                    repo_name=repo_name,
                    pr_number=pr_number,
                    pr_url=record.pr_url,
                    author_login=author_login,
                    files_touched=changed_paths,
                    issue_text=f"{pr.title}\n\n{pr.body or ''}",
                    review_output=review_output,
                ),
            )
            # Write to Graphiti — it extracts entities/relationships automatically
            await graphiti_record_review(
                developer_login=author_login,
                repo_name=repo_name,
                pr_number=pr_number,
                pr_url=record.pr_url,
                summary=fact.summary,
                bug_patterns=fact.bug_patterns,
            )
            await emit("memory", {
                "message": f"Learned: {fact.summary}",
                "patterns": fact.bug_patterns,
            })
        except Exception as e:
            logger.warning("[memory] Failed to write memory: %s", e)
            await emit("memory", {
                "message": f"Memory write failed (non-fatal): {str(e)[:120]}",
            })

        # ----------------------------------------------------------------
        # 7. GitHub integration — post comment + create fix branch (opt-in)
        # ----------------------------------------------------------------
        if configs.GITHUB_POST_REVIEW_COMMENTS or configs.GITHUB_AUTO_FIX_PR:
            from api.github_app import (
                format_pr_review_comment, post_pr_review_comment,
                create_fix_branch_pr,
            )

        if configs.GITHUB_POST_REVIEW_COMMENTS and github_token:
            await emit("status", {"message": "Posting review comment to GitHub PR..."})
            comment_body = format_pr_review_comment(
                file_reviews, pr_number, repo_name, patches_generated,
            )
            ok = await loop.run_in_executor(
                None, post_pr_review_comment, repo, pr, comment_body,
            )
            await emit("github_comment", {
                "posted": ok,
                "message": "Review comment posted to GitHub PR" if ok
                           else "Could not post review comment (check token permissions)",
            })

        if configs.GITHUB_AUTO_FIX_PR and github_token and workdir:
            await emit("status", {"message": "Creating fix branch on GitHub..."})
            fix_pr_url = await loop.run_in_executor(
                None, create_fix_branch_pr,
                repo, pr, file_reviews, workdir, github_token,
            )
            if fix_pr_url:
                await emit("github_fix_pr", {
                    "url": fix_pr_url,
                    "message": f"Fix branch PR created: {fix_pr_url}",
                })
            else:
                await emit("github_fix_pr", {
                    "url": None,
                    "message": "Could not create fix branch (no clean patches or token lacks push access)",
                })

        # ----------------------------------------------------------------
        # 8. Save & finalize
        # ----------------------------------------------------------------
        record.file_reviews = file_reviews
        record.total_patches = patches_generated
        record.status = "completed"
        await update_review(record.id, record)

        await emit("done", {
            "total_patches": patches_generated,
            "total_files": len(changed_files),
        })

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        record.status = "failed"
        record.error = error_msg
        await update_review(record.id, record)
        await emit("error", {"message": error_msg, "traceback": traceback.format_exc()[-500:]})
    finally:
        if neo4j_driver:
            neo4j_driver.close()
        # Clean up per-PR working directory
        try:
            _cleanup_workdir(record.id)
        except Exception:
            pass
        try:
            _lf_trace.__exit__(None, None, None)
        except Exception:
            pass
        await event_queue.put(None)  # close SSE
