"""
SWE-bench repo ingestion — one-time Neo4j population per repo.

SWE-bench Lite tasks come from ~12 Python repos (django, scikit-learn, etc.).
We ingest each repo once and cache the graph, then reuse it across all tasks
from that repo.

Usage:
    python -m swebench.ingest_repo --repo-path /path/to/django --project-id django

Or programmatically:
    from swebench.ingest_repo import ingest_repo
    ingest_repo("/path/to/django", "django")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Cache file to track which repos have been ingested
CACHE_FILE = Path(__file__).parent / ".ingestion_cache.json"


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _compute_repo_hash(repo_path: Path) -> str:
    """Quick hash based on the HEAD commit to detect repo changes."""
    head_file = repo_path / ".git" / "HEAD"
    if head_file.exists():
        ref = head_file.read_text().strip()
        if ref.startswith("ref:"):
            ref_path = repo_path / ".git" / ref.split(": ")[1]
            if ref_path.exists():
                return ref_path.read_text().strip()[:12]
        return ref[:12]
    return "unknown"


def is_repo_ingested(repo_path: str | Path, project_id: str) -> bool:
    """Check if a repo has already been ingested (cached)."""
    cache = _load_cache()
    repo_path = Path(repo_path)
    current_hash = _compute_repo_hash(repo_path)
    cached = cache.get(project_id, {})
    return cached.get("hash") == current_hash


def ingest_repo(
    repo_path: str | Path,
    project_id: str,
    force: bool = False,
) -> None:
    """
    Ingest a repo into Neo4j (code property graph) and Weaviate (vectors).

    This is a one-time operation per repo. Results are cached so subsequent
    SWE-bench tasks from the same repo skip ingestion.

    Args:
        repo_path: Path to the cloned repository.
        project_id: Unique ID for this repo (e.g., "django", "scikit-learn").
        force: If True, re-ingest even if cache says it's already done.
    """
    repo_path = Path(repo_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"Repo not found: {repo_path}")

    if not force and is_repo_ingested(repo_path, project_id):
        print(f"Repo '{project_id}' already ingested (cached). Use --force to re-ingest.")
        return

    print(f"Ingesting repo '{project_id}' from {repo_path}...")

    # Step 1: Find all Python files
    py_files = list(repo_path.rglob("*.py"))
    py_files = [f for f in py_files if ".venv" not in str(f) and "node_modules" not in str(f)]
    print(f"  Found {len(py_files)} Python files")

    # Step 2: Parse and import into Neo4j
    _ingest_to_neo4j(repo_path, py_files, project_id)

    # Step 3: Generate embeddings and store in Neo4j vector index
    _store_embeddings(project_id)

    # Step 4: Update cache
    cache = _load_cache()
    cache[project_id] = {
        "hash": _compute_repo_hash(repo_path),
        "path": str(repo_path),
        "file_count": len(py_files),
    }
    _save_cache(cache)
    print(f"  Ingestion complete for '{project_id}'")


def _ingest_to_neo4j(repo_path: Path, py_files: list[Path], project_id: str):
    """Parse Python files with tree-sitter and create Neo4j graph nodes."""
    from src_bot.neo4jdb.neo4j_db import Neo4jDB

    print("  Importing to Neo4j...")
    db = Neo4jDB()

    # Clear existing nodes for this project
    with db.driver.session() as session:
        session.run(
            "MATCH (n) WHERE n.project_id = $pid DETACH DELETE n",
            {"pid": project_id},
        )

    # Import files as nodes
    # NOTE: This is a simplified ingestion. For production, use tree-sitter
    # to parse AST and create MethodNode, ClassNode, EndpointNode, etc.
    # Use swebench/neo4j_ingest.py for full AST parsing.
    import_count = 0
    with db.driver.session() as session:
        for py_file in py_files:
            try:
                content = py_file.read_text(errors="replace")
                rel_path = str(py_file.relative_to(repo_path))
                ast_hash = hashlib.md5(content.encode()).hexdigest()

                # Create a basic file node
                session.run(
                    """
                    MERGE (n:MethodNode {ast_hash: $hash})
                    SET n.name = $name,
                        n.file_path = $path,
                        n.content = $content,
                        n.project_id = $pid
                    """,
                    {
                        "hash": ast_hash,
                        "name": py_file.stem,
                        "path": rel_path,
                        "content": content[:5000],  # cap for large files
                        "pid": project_id,
                    },
                )
                import_count += 1
            except Exception as e:
                print(f"  Warning: skipped {py_file}: {e}")

    db.close()
    print(f"  Imported {import_count} nodes to Neo4j")


def _store_embeddings(project_id: str):
    """Generate embeddings for Neo4j CodeNodes and store them for vector search."""
    from src_bot.neo4jdb.neo4j_db import Neo4jDB
    from src_bot.config.config import configs
    from swebench.neo4j_ingest import store_embeddings

    print("  Generating embeddings and storing in Neo4j vector index...")
    db = Neo4jDB()
    try:
        with db.driver.session() as session:
            count = store_embeddings(session, project_id, configs.EMBEDDING_MODEL)
        print(f"  Stored embeddings for {count} nodes")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Ingest a repo into Neo4j (code graph + vector embeddings)")
    parser.add_argument("--repo-path", required=True, help="Path to cloned repo")
    parser.add_argument("--project-id", required=True, help="Unique project ID")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if cached")
    args = parser.parse_args()

    ingest_repo(args.repo_path, args.project_id, force=args.force)


if __name__ == "__main__":
    main()
