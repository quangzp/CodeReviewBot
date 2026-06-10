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
import re
import sys
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_bot.config.config import configs
from src_bot.llm.router import get_llm
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


def _checkout_pr_base(project_id: str, base_sha: str, review_id: str) -> Path:
    """
    Create a per-PR working directory by copying the project's clone and
    checking out the PR's base commit there. Doesn't disturb the project's
    main clone (which stays on HEAD of default branch).

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

    # Use a lightweight worktree clone (fast, shares git objects)
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(project_dir), str(workdir)],
        capture_output=True, text=True, timeout=120, check=True,
    )

    # Fetch the PR base commit if we don't have it
    subprocess.run(
        ["git", "fetch", "--depth=1", "origin", base_sha],
        cwd=workdir, capture_output=True, text=True, timeout=180,
    )

    # Check out the base commit
    result = subprocess.run(
        ["git", "checkout", base_sha],
        cwd=workdir, capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        # Fallback: try via the source clone
        subprocess.run(
            ["git", "fetch", "--depth=1", str(project_dir), base_sha],
            cwd=workdir, capture_output=True, text=True, timeout=120,
        )
        subprocess.run(
            ["git", "checkout", base_sha],
            cwd=workdir, capture_output=True, text=True, timeout=60, check=True,
        )

    return workdir


def _cleanup_workdir(review_id: str):
    """Remove the per-PR working directory."""
    workdir = PR_WORKDIRS / review_id
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)


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
            lambda: _checkout_pr_base(project.id, pr.base.sha, record.id),
        )

        # ----------------------------------------------------------------
        # 3. Set up LLM + Neo4j (reads only)
        # ----------------------------------------------------------------
        llm = get_llm(provider=configs.LLM_PROVIDER, model=configs.LLM_MODEL, temperature=0)
        neo4j_driver = GraphDatabase.driver(
            configs.APP_NEO4J_URL,
            auth=(configs.APP_NEO4J_USER, configs.APP_NEO4J_PASSWORD),
        )

        # Import pipeline helpers (they only QUERY the graph, no ingestion)
        from run_swebench_v2 import (
            phase1_localize_file, phase2_localize_fault, phase3_generate_patch,
            _verify_patch_applies, _get_graph_context_for_file,
        )
        from src_bot.reflexion.reflector import generate_reflection
        from src_bot.memory.graphiti_store import build_memory_context, record_review as graphiti_record_review
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
                found_file = await loop.run_in_executor(
                    None, phase1_localize_file,
                    llm, issue_text, repo_name, workdir, session, project_id,
                )
                actual_file = file_path if (workdir / file_path).exists() else found_file
                file_result.phase1_found = bool(actual_file)
                await emit("phase", {
                    "phase": 1, "file": file_path,
                    "found": actual_file, "status": "done",
                })
                if not actual_file:
                    file_reviews.append(file_result)
                    continue

                # Phase 2: fault localization (uses existing graph)
                await emit("phase", {"phase": 2, "file": file_path, "status": "running"})
                fault_desc, file_content = await loop.run_in_executor(
                    None, phase2_localize_fault,
                    llm, issue_text, actual_file, workdir, session, project_id,
                )
                file_result.phase2_fault = fault_desc[:200]
                await emit("phase", {
                    "phase": 2, "file": file_path,
                    "fault": fault_desc[:100], "status": "done",
                })
                if not file_content:
                    file_reviews.append(file_result)
                    continue

                # ── Harness: Risk Scoring (CHID) + Content Escalation + Context Budget ──
                try:
                    from src_bot.risk.scorer import RiskScorer, RiskInput
                    _risk_result = RiskScorer().score(RiskInput(
                        blast_radius_size=len(changed_files),
                        pr_size_lines=getattr(changed_file, "changes", 0),
                    ))
                    file_result.risk_level = _risk_result.level
                except Exception:
                    pass

                # Content-based escalation: security-sensitive files → always high
                _SECURITY_PATTERNS = {
                    "auth", "password", "token", "secret", "encrypt",
                    "permission", "role", "admin", "credential", "oauth",
                    "jwt", "session", "csrf", "injection", "sql",
                }
                if any(p in actual_file.lower() for p in _SECURITY_PATTERNS):
                    file_result.risk_level = "high"

                # Map risk level → context depth + retry budget
                _RISK_TOP_K = {"low": 10, "medium": 15, "high": 25}
                _RISK_RETRIES = {"low": 2, "medium": 3, "high": 5}
                _top_k = _RISK_TOP_K.get(file_result.risk_level, 15)
                max_retries = _RISK_RETRIES.get(
                    file_result.risk_level, configs.REFLEXION_MAX_RETRIES
                )
                # ── End Risk Scoring ─────────────────────────────────────────────

                # Graph context for Reflexion (depth scales with risk)
                graph_context = _get_graph_context_for_file(
                    session, actual_file, project_id, top_k=_top_k
                )

                # Phase 3: patch generation + Reflexion loop
                reflections: list[str] = []
                last_patch = ""
                last_error = ""

                for attempt in range(1, max_retries + 1):
                    await emit("phase", {
                        "phase": 3, "file": file_path,
                        "attempt": attempt, "max_attempts": max_retries,
                        "status": "running",
                    })

                    if attempt == 1:
                        patch = await loop.run_in_executor(
                            None, phase3_generate_patch,
                            llm, issue_text, actual_file, fault_desc, file_content,
                        )
                    else:
                        try:
                            reflection = await loop.run_in_executor(
                                None, generate_reflection,
                                llm,
                                f"{issue_text[:500]}\n\nFault: {fault_desc}",
                                last_patch, last_error, graph_context, reflections,
                            )
                            reflections.append(reflection)
                        except Exception:
                            reflection = f"Previous patch failed: {last_error}"
                            reflections.append(reflection)

                        patch = await loop.run_in_executor(
                            None, phase3_generate_patch,
                            llm, issue_text, actual_file, fault_desc, file_content,
                            reflection, last_patch,
                        )

                    if not patch:
                        last_error = "Empty patch generated"
                        await emit("phase", {
                            "phase": 3, "file": file_path,
                            "attempt": attempt, "status": "empty",
                        })
                        continue

                    applies_ok, apply_error = _verify_patch_applies(patch, workdir)
                    if applies_ok:
                        # ── Harness: Evaluator ──────────────────────────────
                        # Independent LLM scores patch quality before accepting.
                        eval_score, eval_reason = 3, "not evaluated"
                        try:
                            from api.agent.evaluator import evaluate_patch
                            eval_score, eval_reason = await loop.run_in_executor(
                                None, evaluate_patch,
                                f"{issue_text[:300]}\nFault: {fault_desc}", patch,
                            )
                            await emit("phase", {
                                "phase": 3, "file": file_path,
                                "attempt": attempt, "status": "evaluated",
                                "eval_score": eval_score, "eval_reason": eval_reason,
                            })
                        except Exception:
                            pass

                        if eval_score < 3 and attempt < max_retries:
                            # Evaluator rejected — treat like a failed attempt
                            last_patch = patch
                            last_error = (
                                f"[EVALUATOR REJECTION score={eval_score}/5]\n"
                                f"{eval_reason}\n\n"
                                f"The patch applied cleanly but failed quality review. "
                                f"Focus on improving logic, not syntax."
                            )
                            await emit("phase", {
                                "phase": 3, "file": file_path,
                                "attempt": attempt, "status": "evaluator_rejected",
                                "eval_score": eval_score, "eval_reason": eval_reason,
                            })
                            continue
                        # ── End Evaluator ────────────────────────────────────

                        file_result.patch = patch
                        file_result.applies_cleanly = True
                        file_result.reflexion_attempts = attempt
                        file_result.eval_score = eval_score
                        file_result.eval_reason = eval_reason
                        patches_generated += 1
                        await emit("phase", {
                            "phase": 3, "file": file_path,
                            "attempt": attempt, "status": "success",
                            "patch_size": len(patch),
                            "eval_score": eval_score,
                        })
                        break
                    else:
                        # ── Harness: Frustration detection ───────────────────
                        if last_patch and len(patch) < len(last_patch) * 0.6:
                            await emit("phase", {
                                "phase": 3, "file": file_path,
                                "attempt": attempt, "status": "frustration_detected",
                                "message": "Patch shrinking — agent confused, widening context",
                            })
                        # ── End Frustration detection ─────────────────────────
                        last_patch = patch
                        last_error = f"git apply failed: {apply_error}"
                        await emit("phase", {
                            "phase": 3, "file": file_path,
                            "attempt": attempt, "status": "apply_failed",
                            "error": apply_error[:100],
                        })

                file_reviews.append(file_result)

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
        # 6. MEMORY WRITE — learn from this review
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
                    llm,
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
            print(f"  [memory] Failed to write memory: {e}")

        # ----------------------------------------------------------------
        # 7. Save & finalize
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
        await event_queue.put(None)  # close SSE
