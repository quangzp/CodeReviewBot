"""
Refactor code WITHOUT a pull request.

Flow:
  1. Project is already indexed (graph in Neo4j)
  2. User describes what to refactor in natural language
  3. Phase 1: Locate the target file (BM25 + LLM), skipped if file_path given
  4. Phase 3: Generate a refactoring patch (no Phase 2 — no "fault" to find)
  5. Reflexion loop: retry with self-reflection if patch doesn't apply
  6. Evaluator: quality gate using refactor rubric

Returns the same render payload format as fix_bug for uniform chat UI rendering.
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


async def refactor_code_in_project(
    project: ProjectRecord,
    refactor_description: str,
    file_path: str = "",
) -> dict:
    """
    Standalone refactoring using only the description + indexed project graph.

    Args:
        project: The indexed project record.
        refactor_description: Natural language description of what to refactor.
        file_path: Optional. If provided, skip Phase 1 file localization.

    Returns a dict with 'summary' and 'render' suitable for chat display.
    """
    from neo4j import GraphDatabase
    from run_swebench_v2 import (
        phase1_localize_file,
        phase3_refactor_patch,
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

    llm = get_llm(provider=configs.LLM_PROVIDER, model=configs.LLM_MODEL, temperature=0)
    driver = GraphDatabase.driver(
        configs.APP_NEO4J_URL,
        auth=(configs.APP_NEO4J_USER, configs.APP_NEO4J_PASSWORD),
    )

    try:
        with driver.session() as session:
            # ------------------------------------------------------------------
            # Phase 1 — find the file (skip if user provided file_path)
            # ------------------------------------------------------------------
            if not file_path:
                file_path = await loop.run_in_executor(
                    None,
                    phase1_localize_file,
                    llm, refactor_description, project.repo_name, repo_dir, session, project_id,
                )

            if not file_path:
                return {
                    "summary": (
                        "Could not locate a relevant file from the refactoring description. "
                        "Try specifying the file path directly."
                    ),
                    "render": {
                        "kind": "fix_attempt",
                        "status": "no_file",
                        "bug_description": refactor_description,
                    },
                }

            # ------------------------------------------------------------------
            # Read file content directly (no Phase 2 fault localization)
            # ------------------------------------------------------------------
            target = repo_dir / file_path
            if not target.exists():
                return {
                    "summary": f"File '{file_path}' not found in the project.",
                    "render": {
                        "kind": "fix_attempt",
                        "status": "no_content",
                        "file_path": file_path,
                        "bug_description": refactor_description,
                    },
                }

            file_content = target.read_text(errors="ignore")
            graph_context = _get_graph_context_for_file(session, file_path, project_id)

            # ------------------------------------------------------------------
            # Reflexion loop — Phase 3 with refactor prompt
            # ------------------------------------------------------------------
            max_retries = configs.REFLEXION_MAX_RETRIES
            reflections: list[str] = []
            last_patch = ""
            last_error = ""
            final_patch = ""
            final_eval_score = 3
            final_eval_reason = "not evaluated"
            attempts = 0

            for attempt in range(1, max_retries + 1):
                attempts = attempt
                if attempt == 1:
                    patch = await loop.run_in_executor(
                        None,
                        phase3_refactor_patch,
                        llm, refactor_description, file_path, file_content,
                    )
                else:
                    try:
                        reflection = await loop.run_in_executor(
                            None,
                            generate_reflection,
                            llm,
                            refactor_description,
                            last_patch, last_error, graph_context, reflections,
                        )
                        reflections.append(reflection)
                    except Exception:
                        reflections.append(f"Previous failed: {last_error}")

                    patch = await loop.run_in_executor(
                        None,
                        phase3_refactor_patch,
                        llm, refactor_description, file_path, file_content,
                        reflections[-1] if reflections else "", last_patch,
                    )

                if not patch:
                    last_error = "Empty patch"
                    continue

                patch = _normalize_patch(patch, repo_dir)
                applies_ok, apply_error = _verify_patch_applies(patch, repo_dir)
                if applies_ok:
                    # Evaluator with refactor rubric
                    eval_score, eval_reason = 3, "not evaluated"
                    try:
                        from api.agent.evaluator import evaluate_patch
                        eval_score, eval_reason = await loop.run_in_executor(
                            None, evaluate_patch,
                            refactor_description[:300], patch, "refactor",
                        )
                    except Exception:
                        pass

                    if eval_score < 3 and attempt < max_retries:
                        last_patch = patch
                        last_error = (
                            f"[EVALUATOR REJECTION score={eval_score}/5]\n"
                            f"{eval_reason}\n\n"
                            "The patch applied cleanly but failed quality review. "
                            "Ensure behavior is fully preserved and the refactoring is complete."
                        )
                        continue

                    final_patch = patch
                    final_eval_score = eval_score
                    final_eval_reason = eval_reason
                    break
                else:
                    if last_patch and len(patch) < len(last_patch) * 0.6:
                        apply_error += "\nHint: Patch is shrinking — avoid removing context lines."
                    last_patch = patch
                    last_error = f"[APPLY FAILED]\n{apply_error}"

            if not final_patch:
                return {
                    "summary": (
                        f"Located '{file_path}' but couldn't generate a refactoring patch "
                        f"that applies cleanly after {attempts} attempts."
                    ),
                    "render": {
                        "kind": "fix_attempt",
                        "status": "patch_failed",
                        "file_path": file_path,
                        "fault_description": refactor_description[:200],
                        "attempts": attempts,
                        "last_error": last_error,
                    },
                }

            return {
                "summary": (
                    f"Refactored {file_path} as requested. "
                    f"Patch applies cleanly, eval_score={final_eval_score}/5, "
                    f"{attempts} attempt{'s' if attempts > 1 else ''}."
                ),
                "render": {
                    "kind": "fix_attempt",
                    "status": "success",
                    "file_path": file_path,
                    "fault_description": refactor_description[:200],
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
