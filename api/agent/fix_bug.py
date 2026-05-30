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
    llm = get_llm(provider=configs.LLM_PROVIDER, model=configs.LLM_MODEL, temperature=0)
    driver = GraphDatabase.driver(
        configs.APP_NEO4J_URL,
        auth=(configs.APP_NEO4J_USER, configs.APP_NEO4J_PASSWORD),
    )

    try:
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

                applies_ok, apply_error = _verify_patch_applies(patch, repo_dir)
                if applies_ok:
                    final_patch = patch
                    applies = True
                    break
                else:
                    last_patch = patch
                    last_error = f"git apply failed: {apply_error}"

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

            return {
                "summary": (
                    f"Found and fixed the bug in {file_path}: {fault_desc[:100]}. "
                    f"Generated a patch ({len(final_patch)} chars, applies cleanly, "
                    f"{attempts} attempt{'s' if attempts > 1 else ''})."
                ),
                "render": {
                    "kind": "fix_attempt",
                    "status": "success",
                    "file_path": file_path,
                    "fault_description": fault_desc,
                    "patch": final_patch,
                    "attempts": attempts,
                    "applies_cleanly": True,
                    "repo_name": project.repo_name,
                },
            }
    finally:
        driver.close()
