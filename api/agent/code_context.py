"""
Shared CodeContext — encapsulates the result of Phase 1 + Phase 2.

Both fix_bug and refactor pipelines need the same foundation:
  1. Which file?       (Phase 1: BM25 + LLM)
  2. What's in it?     (Phase 2: graph context + file content)
  3. What calls it?    (graph expansion)
  4. What does memory say? (Graphiti developer context)

This module builds that context ONCE and passes it to whichever lens
(bug fix / refactor / review) needs it.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from api.models import ProjectRecord
from api.project_indexer import get_project_dir
from src_bot.config.config import configs
from src_bot.llm.router import get_llm


@dataclass
class CodeContext:
    """Immutable context about a specific code location."""
    project: ProjectRecord
    repo_dir: Path
    file_path: str
    file_content: str
    fault_description: str      # Phase 2 output (or refactor target description)
    graph_context: str          # callers/callees from Neo4j
    memory_context: str         # Graphiti developer patterns
    line_count: int = 0

    @property
    def is_valid(self) -> bool:
        return bool(self.file_path and self.file_content)


async def build_code_context(
    project: ProjectRecord,
    description: str,
    file_path: str = "",
    mode: str = "bug_fix",      # "bug_fix" | "refactor" | "review"
    developer_login: str = "",
) -> CodeContext:
    """
    Build a CodeContext by running Phase 1 (file localization) and
    Phase 2 (fault localization), plus graph and memory context.

    Args:
        project: The indexed project record.
        description: Natural language description (bug report or refactor request).
        file_path: Optional. Skip Phase 1 if provided.
        mode: "bug_fix" runs Phase 2 fault localization; "refactor" skips it.
        developer_login: Optional. Used for memory context lookup.

    Returns:
        A CodeContext. Check .is_valid before using.
    """
    from neo4j import GraphDatabase
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
    from run_swebench_v2 import (
        phase1_localize_file,
        phase2_localize_fault,
        _get_graph_context_for_file,
    )

    loop = asyncio.get_event_loop()
    repo_dir = get_project_dir(project.id)

    _empty = CodeContext(
        project=project, repo_dir=repo_dir,
        file_path="", file_content="",
        fault_description="", graph_context="", memory_context="",
    )

    if not repo_dir.exists():
        return _empty

    llm = get_llm(provider=configs.LLM_PROVIDER, model=configs.LLM_MODEL, temperature=0)
    driver = GraphDatabase.driver(
        configs.APP_NEO4J_URL,
        auth=(configs.APP_NEO4J_USER, configs.APP_NEO4J_PASSWORD),
    )

    try:
        # Memory context (non-blocking — failure returns empty string)
        memory_context = ""
        try:
            from src_bot.memory.memory_neo4j import build_memory_context
            owner = developer_login or project.repo_name.split("/")[0]
            memory_context = await build_memory_context(owner, project.repo_name)
        except Exception:
            pass

        enriched_description = description
        if memory_context:
            enriched_description = f"{memory_context}\n\n---\n\n{description}"

        with driver.session() as session:
            # Phase 1: file localization (skip if caller provided file_path)
            if not file_path:
                file_path = await loop.run_in_executor(
                    None,
                    phase1_localize_file,
                    llm, enriched_description, project.repo_name,
                    repo_dir, session, project.id,
                )

            if not file_path:
                return CodeContext(
                    project=project, repo_dir=repo_dir,
                    file_path="", file_content="",
                    fault_description="", graph_context="",
                    memory_context=memory_context,
                )

            target = repo_dir / file_path
            if not target.exists():
                return CodeContext(
                    project=project, repo_dir=repo_dir,
                    file_path=file_path, file_content="",
                    fault_description="", graph_context="",
                    memory_context=memory_context,
                )

            file_content = target.read_text(errors="ignore")

            # Phase 2: fault localization (bug_fix mode only)
            fault_description = ""
            if mode == "bug_fix":
                fault_description, _ = await loop.run_in_executor(
                    None,
                    phase2_localize_fault,
                    llm, enriched_description, file_path,
                    repo_dir, session, project.id,
                )
            elif mode == "refactor":
                fault_description = description

            # Graph context (callers + blast radius)
            graph_context = _get_graph_context_for_file(
                session, file_path, project.id
            )

            return CodeContext(
                project=project,
                repo_dir=repo_dir,
                file_path=file_path,
                file_content=file_content,
                fault_description=fault_description,
                graph_context=graph_context,
                memory_context=memory_context,
                line_count=len(file_content.splitlines()),
            )
    finally:
        driver.close()
