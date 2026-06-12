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
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src_bot.config.config import configs
from src_bot.llm.router import get_llm
from api.models import ProjectRecord
from api.project_indexer import get_project_dir


async def detect_refactoring_targets(
    project: ProjectRecord,
    file_path: str = "",
) -> list[dict]:
    """
    Detect code smells in a project using the Neo4j graph.
    Returns a list of smell dicts suitable for chat display.
    """
    from neo4j import GraphDatabase
    from api.agent.smell_detector import detect_all_smells

    driver = GraphDatabase.driver(
        configs.APP_NEO4J_URL,
        auth=(configs.APP_NEO4J_USER, configs.APP_NEO4J_PASSWORD),
    )
    try:
        with driver.session() as session:
            smells = detect_all_smells(session, project.id, file_path)
            return [
                {
                    "smell_type": s.smell_type,
                    "file_path": s.file_path,
                    "name": s.name,
                    "severity": s.severity,
                    "description": s.description,
                    "metric_value": s.metric_value,
                }
                for s in smells
            ]
    finally:
        driver.close()


async def refactor_code_in_project(
    project: ProjectRecord,
    refactor_description: str,
    file_path: str = "",
    user_login: str = "",
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

    # Set Langfuse trace context before creating the LLM
    _langfuse_active = False
    try:
        from src_bot.observability.langfuse_ctx import set_trace_context, clear_trace_context, LangfuseTraceContext
        _repo_owner = project.repo_name.split("/")[0]
        set_trace_context(LangfuseTraceContext(
            trace_id=f"refactor-{project_id}",
            session_id=project_id,
            user_id=_repo_owner,
            tags=["chat_refactor"],
            metadata={"repo": project.repo_name},
        ))
        _langfuse_active = True
    except Exception:
        pass

    llm = get_llm(role="generation", temperature=0)
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

            # Detect code smells to enrich the refactor prompt
            smell_context = ""
            try:
                from api.agent.smell_detector import detect_all_smells
                smells = detect_all_smells(session, project_id, file_path)
                if smells:
                    smell_lines = ["## Detected code smells in this file:"]
                    for s in smells[:5]:
                        smell_lines.append(
                            f"  - [{s.severity.upper()}] {s.smell_type}: {s.description}"
                        )
                    smell_context = "\n".join(smell_lines)
            except Exception:
                pass

            if smell_context:
                refactor_description = (
                    f"{smell_context}\n\n---\n\nRefactoring request: {refactor_description}"
                )

            # ------------------------------------------------------------------
            # Phase 2.5 — Planner: generate structured contract
            # ------------------------------------------------------------------
            from api.agent.planner import generate_plan_sync
            contract = await loop.run_in_executor(
                None,
                generate_plan_sync,
                smell_context or refactor_description[:500],
                refactor_description,
                file_path,
                graph_context,
                "refactor",
            )

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
                    # ── Harness: Multi-gate Verification ─────────────────────
                    from src_bot.verification.pipeline import verify_patch
                    verification = await loop.run_in_executor(
                        None,
                        verify_patch,
                        patch,
                        repo_dir,
                        refactor_description,
                        "",
                        contract,
                        "refactor",
                        configs.EXECUTION_GATE_ENABLED,
                        configs.EXECUTION_GATE_TEST_CMD,
                        configs.EXECUTION_GATE_TIMEOUT,
                    )
                    eval_score = verification.llm_score
                    eval_reason = verification.llm_reason

                    if not verification.passed and attempt < max_retries:
                        last_patch = patch
                        last_error = verification.rejection_reason
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

            # Memory write — record refactoring in Graphiti
            try:
                from src_bot.memory.extractor import extract_facts_from_review
                from src_bot.memory.memory_neo4j import (
                    record_review as graphiti_record_review,
                    record_topic_interest,
                )
                _author = user_login or project.repo_name.split("/")[0]
                fact = await loop.run_in_executor(
                    None,
                    lambda: extract_facts_from_review(
                        llm,
                        repo_name=project.repo_name,
                        pr_number=0,
                        pr_url=f"chat-refactor/{project.id}",
                        author_login=_author,
                        files_touched=[file_path],
                        issue_text=refactor_description,
                        review_output=(
                            f"Refactored: {file_path}\n"
                            f"Patch (eval_score={final_eval_score}/5):\n"
                            f"```diff\n{final_patch[:1000]}\n```"
                        ),
                    ),
                )
                await graphiti_record_review(
                    developer_login=_author,
                    repo_name=project.repo_name,
                    pr_number=0,
                    pr_url=f"chat-refactor/{project.id}",
                    summary=fact.summary,
                    bug_patterns=fact.bug_patterns,
                )
                await record_topic_interest(
                    _author,
                    f"module:{file_path}",
                    file_path.rsplit("/", 1)[-1],
                    "module",
                )
                for pat in fact.bug_patterns:
                    await record_topic_interest(
                        _author,
                        f"pattern:{pat}",
                        pat.replace("-", " ").title(),
                        "pattern",
                    )
            except Exception as e:
                logger.warning("[memory] Refactor memory write failed: %s", e)

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
        if _langfuse_active:
            try:
                clear_trace_context()
            except Exception:
                pass
