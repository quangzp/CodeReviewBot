"""
Project indexer — clones a GitHub repo and builds the Neo4j knowledge graph.

This runs ONCE per project (during onboarding) and then incrementally on
push events. The PR review path NEVER triggers ingestion — it only queries
the already-built graph.

Emits SSE events to a queue for live progress UI.
"""
from __future__ import annotations

import asyncio
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_bot.config.config import configs
from api.models import ProjectRecord
from api.database import update_project

# Where we keep cloned repos — use data/ so it survives server restarts
# /tmp is wiped on reboot; data/ lives next to the SQLite database
PROJECTS_DIR = Path(__file__).parent.parent / "data" / "projects"


def parse_repo_url(repo_url: str) -> tuple[str, str]:
    """
    Parse 'https://github.com/owner/repo[.git]' → ('owner/repo', clean_url).
    """
    url = repo_url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    match = re.match(r"https://github\.com/([^/]+/[^/]+)", url)
    if not match:
        raise ValueError(f"Invalid GitHub repo URL: {repo_url}")
    return match.group(1), match.group(0)


def get_project_dir(project_id: str) -> Path:
    """Persistent clone location for a project."""
    return PROJECTS_DIR / project_id


async def detect_default_branch(repo_dir: Path) -> str:
    """Find the default branch (main/master) of a cloned repo."""
    loop = asyncio.get_event_loop()

    def _detect():
        try:
            result = subprocess.run(
                ["git", "remote", "show", "origin"],
                cwd=repo_dir, capture_output=True, text=True, timeout=15,
            )
            for line in result.stdout.splitlines():
                if "HEAD branch:" in line:
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return "main"

    return await loop.run_in_executor(None, _detect)


async def get_head_commit(repo_dir: Path) -> str:
    loop = asyncio.get_event_loop()

    def _get():
        try:
            r = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_dir, capture_output=True, text=True, timeout=10,
            )
            return r.stdout.strip()
        except Exception:
            return ""

    return await loop.run_in_executor(None, _get)


async def index_project(
    record: ProjectRecord,
    event_queue: asyncio.Queue,
):
    """
    Run the full indexing pipeline for a project.

    Phases:
      1. Clone (or pull if already exists)
      2. Detect default branch
      3. Parse all source files via AST → write CodeNodes to Neo4j
      4. Build CALLS/IMPORTS/DEFINES edges
      5. Mark project as 'indexed'

    Failures update record.status='failed' and record.error.
    Always emits a final 'done' or 'error' event and closes the queue.
    """
    from neo4j import GraphDatabase
    from swebench.neo4j_ingest import ingest_repo_to_neo4j

    loop = asyncio.get_event_loop()

    async def emit(event_type: str, **data):
        await event_queue.put({"type": event_type, **data})

    try:
        # --------------------------------------------------------------
        # Phase 1: Clone / update
        # --------------------------------------------------------------
        record.status = "cloning"
        record.progress_pct = 5
        await update_project(record.id, record)
        await emit("status", message=f"Cloning {record.repo_name}...", progress=5)

        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        repo_dir = get_project_dir(record.id)

        def _clone():
            if repo_dir.exists():
                # Pull latest
                subprocess.run(
                    ["git", "fetch", "origin"],
                    cwd=repo_dir, capture_output=True, text=True, timeout=600,
                )
                subprocess.run(
                    ["git", "reset", "--hard", "origin/HEAD"],
                    cwd=repo_dir, capture_output=True, text=True, timeout=60,
                )
            else:
                subprocess.run(
                    ["git", "clone", "--depth", "1", record.repo_url, str(repo_dir)],
                    capture_output=True, text=True, timeout=600,
                )

        await loop.run_in_executor(None, _clone)

        if not repo_dir.exists() or not any(repo_dir.iterdir()):
            raise RuntimeError(f"Failed to clone {record.repo_url}")

        record.default_branch = await detect_default_branch(repo_dir)
        record.last_commit_sha = await get_head_commit(repo_dir)

        await emit("status",
                   message=f"Cloned @ {record.last_commit_sha[:7]} (branch: {record.default_branch})",
                   progress=20)

        # --------------------------------------------------------------
        # Phase 2: Count files (for progress UI)
        # --------------------------------------------------------------
        record.status = "indexing"
        record.progress_pct = 25
        await update_project(record.id, record)

        def _count_files():
            return sum(
                1 for _ in repo_dir.rglob("*.py")
                if "__pycache__" not in str(_)
                and "/.git/" not in str(_)
                and not _.name.startswith(".")
            )

        file_count = await loop.run_in_executor(None, _count_files)
        record.file_count = file_count
        await update_project(record.id, record)
        await emit("status",
                   message=f"Found {file_count} Python files to index",
                   progress=30, file_count=file_count)

        # --------------------------------------------------------------
        # Phase 3: Ingest into Neo4j
        # --------------------------------------------------------------
        await emit("status", message="Building knowledge graph in Neo4j...", progress=35)

        def _ingest():
            return ingest_repo_to_neo4j(
                repo_dir,
                record.id,                            # project_id
                configs.APP_NEO4J_URL,
                configs.APP_NEO4J_USER,
                configs.APP_NEO4J_PASSWORD,
                max_files=10000,                      # cap for very large repos
            )

        stats = await loop.run_in_executor(None, _ingest)

        # ingest_repo_to_neo4j returns {files, nodes, relationships}
        # We also query the DB to include Module nodes (which the ingester counts separately)
        if stats:
            ingested_files = stats.get("files", 0)
            code_nodes = stats.get("nodes", 0)
            call_rels = stats.get("relationships", 0)
            # Total nodes = Module nodes (one per file) + CodeNode entities
            # Total edges = CALLS + DEFINES (one DEFINES per code node)
            record.node_count = ingested_files + code_nodes
            record.edge_count = call_rels + code_nodes  # DEFINES = one per code node
            record.file_count = ingested_files
        else:
            record.node_count = 0
            record.edge_count = 0

        record.progress_pct = 90

        await emit("status",
                   message=f"Built graph: {record.node_count} nodes, {record.edge_count} edges from {record.file_count} files",
                   progress=90,
                   node_count=record.node_count,
                   edge_count=record.edge_count,
                   file_count=record.file_count)

        # --------------------------------------------------------------
        # Phase 4: Generate + store embeddings for vector search
        # --------------------------------------------------------------
        if configs.EMBEDDING_ENABLED:
            await emit("status", message="Generating embeddings for vector search...", progress=92)

            def _embed():
                from neo4j import GraphDatabase as _GDB
                from swebench.neo4j_ingest import store_embeddings
                drv = _GDB.driver(
                    configs.APP_NEO4J_URL,
                    auth=(configs.APP_NEO4J_USER, configs.APP_NEO4J_PASSWORD),
                )
                try:
                    with drv.session() as sess:
                        return store_embeddings(sess, record.id, configs.EMBEDDING_MODEL)
                finally:
                    drv.close()

            try:
                embedded = await loop.run_in_executor(None, _embed)
                await emit("status",
                           message=f"Embeddings ready: {embedded} code nodes indexed for vector search",
                           progress=95)
            except Exception as emb_err:
                await emit("status",
                           message=f"Vector embeddings skipped (non-fatal): {emb_err}",
                           progress=95)

        record.progress_pct = 95

        # Write project summary for Planner context injection
        def _write_summary():
            summary_path = repo_dir / "PROJECT_SUMMARY.txt"
            summary_path.write_text(
                f"Repository: {record.repo_name}\n"
                f"Language: Python\n"
                f"Files: {record.file_count} Python source files\n"
                f"Knowledge graph: {record.node_count} nodes, {record.edge_count} edges\n"
                f"Indexed: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n",
                encoding="utf-8",
            )
        await loop.run_in_executor(None, _write_summary)

        # --------------------------------------------------------------
        # Phase 4: Mark as indexed
        # --------------------------------------------------------------
        record.status = "indexed"
        record.progress_pct = 100
        record.last_indexed_at = datetime.now(timezone.utc).isoformat()
        record.error = None
        await update_project(record.id, record)

        await emit("done",
                   message="Project indexed and ready to receive PRs",
                   node_count=record.node_count,
                   edge_count=record.edge_count,
                   file_count=record.file_count)

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        record.status = "failed"
        record.error = error_msg
        await update_project(record.id, record)
        await emit("error", message=error_msg, traceback=traceback.format_exc()[-500:])
    finally:
        await event_queue.put(None)  # close SSE


async def reindex_files(
    record: ProjectRecord,
    changed_files: list[str],
    event_queue: Optional[asyncio.Queue] = None,
):
    """
    Incremental re-index: re-parse only the changed files (used by push webhook).
    Lighter than a full re-index — runs in seconds.
    """
    from neo4j import GraphDatabase
    from swebench.neo4j_ingest import ingest_repo_to_neo4j

    loop = asyncio.get_event_loop()
    repo_dir = get_project_dir(record.id)

    if not repo_dir.exists():
        # Repo not cloned yet — fall back to full index
        queue = event_queue or asyncio.Queue()
        await index_project(record, queue)
        return

    # Pull latest from origin
    def _pull():
        subprocess.run(["git", "fetch", "origin"], cwd=repo_dir,
                       capture_output=True, timeout=300)
        subprocess.run(["git", "reset", "--hard", "origin/HEAD"], cwd=repo_dir,
                       capture_output=True, timeout=60)

    await loop.run_in_executor(None, _pull)

    # Re-ingest the whole project for now (incremental upsert is supported
    # since nodes are merged by ast_hash/path). Future improvement: only
    # touch the changed files.
    def _ingest():
        return ingest_repo_to_neo4j(
            repo_dir,
            record.id,
            configs.APP_NEO4J_URL,
            configs.APP_NEO4J_USER,
            configs.APP_NEO4J_PASSWORD,
            max_files=10000,
        )

    stats = await loop.run_in_executor(None, _ingest)

    if stats:
        ingested_files = stats.get("files", 0)
        code_nodes = stats.get("nodes", 0)
        call_rels = stats.get("relationships", 0)
        record.node_count = ingested_files + code_nodes
        record.edge_count = call_rels + code_nodes
        record.file_count = ingested_files
    record.last_commit_sha = await get_head_commit(repo_dir)
    record.last_indexed_at = datetime.now(timezone.utc).isoformat()
    record.status = "indexed"
    await update_project(record.id, record)

    if event_queue:
        await event_queue.put({"type": "done", "message": "Incremental re-index complete"})
        await event_queue.put(None)
