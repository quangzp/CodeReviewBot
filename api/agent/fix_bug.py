"""
Fix a bug WITHOUT a pull request.

Flow:
  1. Project is already indexed (graph in Neo4j)
  2. User describes the bug in natural language
  3. BM25 search the graph for candidate files
  4. Run the same 3-phase pipeline as PR review:
       Phase 1: file localization (which file?)
       Phase 2: fault localization (which function?)
       Phase 3: patch generation (with reflexion retries)
  5. Return the patch as a structured render payload

The result includes:
  - the patch (unified diff)
  - whether it applies cleanly
  - the located file and function
  - graph context (what calls this, what it calls)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src_bot.config.config import configs
from src_bot.llm.router import get_llm
from api.models import ProjectRecord
from api.project_indexer import get_project_dir


async def fix_bug_in_project(
    project: ProjectRecord,
    bug_description: str,
) -> dict:
    """
    Standalone bug fix using only the bug description + indexed project graph.
    Returns a dict with 'summary' and 'render' suitable for chat display.
    """
    from neo4j import GraphDatabase
    from run_swebench_v2 import (
        phase1_localize_file,
        phase2_localize_fault,
        phase3_generate_patch,
        _verify_patch_applies,
        _normalize_patch,
        _get_graph_context_for_file,
    )
    from src_bot.reflexion.reflector import generate_reflection

    loop = asyncio.get_event_loop()
    project_id = project.id
    repo_dir = get_project_dir(project_id)

    if not repo_dir.exists():
        return {
            "summary": f"Project clone for '{project.repo_name}' is missing.",
            "render": {
                "kind": "error",
                "message": "Project clone is missing on disk. Re-index the project.",
            },
        }

    # Build the LLM + Neo4j driver
    llm = get_llm(role="generation", temperature=0)
    driver = GraphDatabase.driver(
        configs.APP_NEO4J_URL,
        auth=(configs.APP_NEO4J_USER, configs.APP_NEO4J_PASSWORD),
    )

    try:
        # ── Harness: Memory READ ─────────────────────────────────────────────
        try:
            from src_bot.memory.memory_neo4j import build_memory_context
            _owner = project.repo_name.split("/")[0]
            _memory_ctx = await build_memory_context(_owner, project.repo_name)
            if _memory_ctx:
                bug_description = f"{_memory_ctx}\n\n---\n\n{bug_description}"
        except Exception as e:
            print(f"  [memory] Memory read failed: {e}")
        # ── End Memory READ ──────────────────────────────────────────────────

        with driver.session() as session:
            # --------------------------------------------------------------
            # Phase 1 — find the file
            # --------------------------------------------------------------
            file_path = await loop.run_in_executor(
                None,
                phase1_localize_file,
                llm, bug_description, project.repo_name, repo_dir, session, project_id,
            )

            if not file_path:
                return {
                    "summary": (
                        "Could not locate a relevant file from the bug description. "
                        "Try being more specific (mention file or function names)."
                    ),
                    "render": {
                        "kind": "fix_attempt",
                        "status": "no_file",
                        "bug_description": bug_description,
                    },
                }

            # --------------------------------------------------------------
            # Phase 2 — find the function/line
            # --------------------------------------------------------------
            fault_desc, file_content = await loop.run_in_executor(
                None,
                phase2_localize_fault,
                llm, bug_description, file_path, repo_dir, session, project_id,
            )

            if not file_content:
                return {
                    "summary": f"Located file {file_path} but could not read its contents.",
                    "render": {
                        "kind": "fix_attempt",
                        "status": "no_content",
                        "file_path": file_path,
                        "bug_description": bug_description,
                    },
                }

            graph_context = _get_graph_context_for_file(session, file_path, project_id)

            # --------------------------------------------------------------
            # Phase 3 — generate patch with reflexion
            # --------------------------------------------------------------
            max_retries = configs.REFLEXION_MAX_RETRIES
            reflections: list[str] = []
            last_patch = ""
            last_error = ""
            final_patch = ""
            final_eval_score = 3
            final_eval_reason = "not evaluated"
            attempts = 0
            applies = False

            for attempt in range(1, max_retries + 1):
                attempts = attempt
                if attempt == 1:
                    patch = await loop.run_in_executor(
                        None,
                        phase3_generate_patch,
                        llm, bug_description, file_path, fault_desc, file_content,
                    )
                else:
                    try:
                        reflection = await loop.run_in_executor(
                            None,
                            generate_reflection,
                            llm,
                            f"{bug_description}\n\nFault: {fault_desc}",
                            last_patch, last_error, graph_context, reflections,
                        )
                        reflections.append(reflection)
                    except Exception:
                        reflections.append(f"Previous failed: {last_error}")

                    patch = await loop.run_in_executor(
                        None,
                        phase3_generate_patch,
                        llm, bug_description, file_path, fault_desc, file_content,
                        reflections[-1] if reflections else "", last_patch,
                    )

                if not patch:
                    last_error = "Empty patch"
                    continue

                patch = _normalize_patch(patch, repo_dir)
                applies_ok, apply_error = _verify_patch_applies(patch, repo_dir)
                if applies_ok:
                    # ── Harness: Evaluator ──────────────────────────────────
                    eval_score, eval_reason = 3, "not evaluated"
                    try:
                        from api.agent.evaluator import evaluate_patch
                        eval_score, eval_reason = await loop.run_in_executor(
                            None, evaluate_patch,
                            f"{bug_description[:300]}\nFault: {fault_desc}", patch,
                        )
                    except Exception:
                        pass

                    if eval_score < 3 and attempt < max_retries:
                        last_patch = patch
                        last_error = (
                            f"[EVALUATOR REJECTION score={eval_score}/5]\n"
                            f"{eval_reason}\n\n"
                            f"The patch applied cleanly but failed quality review. "
                            f"Focus on improving logic, not syntax."
                        )
                        continue
                    # ── End Evaluator ────────────────────────────────────────

                    final_patch = patch
                    final_eval_score = eval_score
                    final_eval_reason = eval_reason
                    applies = True
                    break
                else:
                    # ── Harness: Frustration detection ───────────────────────
                    if last_patch and len(patch) < len(last_patch) * 0.6:
                        apply_error += "\nHint: Patch is shrinking — avoid removing context lines."
                    # ── End Frustration detection ────────────────────────────
                    last_patch = patch
                    last_error = f"[APPLY FAILED]\n{apply_error}"

            if not final_patch:
                return {
                    "summary": (
                        f"Located bug in {file_path} ({fault_desc[:80]}) "
                        f"but couldn't generate a patch that applies cleanly "
                        f"after {attempts} attempts."
                    ),
                    "render": {
                        "kind": "fix_attempt",
                        "status": "patch_failed",
                        "file_path": file_path,
                        "fault_description": fault_desc,
                        "attempts": attempts,
                        "last_error": last_error,
                    },
                }

            # ── Harness: Memory WRITE ────────────────────────────────────────
            try:
                from src_bot.memory.extractor import extract_facts_from_review
                from src_bot.memory.memory_neo4j import record_review as graphiti_record_review
                _owner = project.repo_name.split("/")[0]
                fact = await loop.run_in_executor(
                    None,
                    lambda: extract_facts_from_review(
                        llm,
                        repo_name=project.repo_name,
                        pr_number=0,
                        pr_url=f"chat-fix/{project.id}",
                        author_login=_owner,
                        files_touched=[file_path],
                        issue_text=bug_description,
                        review_output=(
                            f"Fixed: {fault_desc}\n"
                            f"Patch (eval_score={final_eval_score}/5):\n"
                            f"```diff\n{final_patch[:1000]}\n```"
                        ),
                    ),
                )
                await graphiti_record_review(
                    developer_login=_owner,
                    repo_name=project.repo_name,
                    pr_number=0,
                    pr_url=f"chat-fix/{project.id}",
                    summary=fact.summary,
                    bug_patterns=fact.bug_patterns,
                )
            except Exception as e:
                print(f"  [memory] Chat fix memory write failed: {e}")
            # ── End Memory WRITE ─────────────────────────────────────────────

            return {
                "summary": (
                    f"Found and fixed the bug in {file_path}: {fault_desc[:100]}. "
                    f"Patch applies cleanly, eval_score={final_eval_score}/5, "
                    f"{attempts} attempt{'s' if attempts > 1 else ''}."
                ),
                "render": {
                    "kind": "fix_attempt",
                    "status": "success",
                    "file_path": file_path,
                    "fault_description": fault_desc,
                    "patch": final_patch,
                    "attempts": attempts,
                    "applies_cleanly": True,
                    "eval_score": final_eval_score,
                    "eval_reason": final_eval_reason,
                    "repo_name": project.repo_name,
                },
            }
    finally:
        driver.close()
