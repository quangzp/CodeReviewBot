"""
Re-ingest all indexed projects into Neo4j.

Run this after updating neo4j_ingest.py with new fields (F007: line_count,
method_count, INHERITS, IMPORTS) to populate existing nodes with the new data.

Usage:
    python3 scripts/reingest_projects.py
    python3 scripts/reingest_projects.py --project nicholasgibson2_elastalert-jertel
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("APP_NEO4J_PASSWORD", "CodeReview2024!")


async def reingest_all(project_filter: str | None = None):
    from api.database import list_projects
    from api.project_indexer import get_project_dir
    from swebench.neo4j_ingest import ingest_repo_to_neo4j
    from src_bot.config.config import configs

    projects = await list_projects()
    if not projects:
        print("No projects found in database.")
        return

    if project_filter:
        projects = [p for p in projects if project_filter in p.id or project_filter in p.repo_name]
        if not projects:
            print(f"No project matching '{project_filter}'")
            return

    print(f"Found {len(projects)} project(s) to re-ingest:")
    for p in projects:
        print(f"  • {p.id} ({p.repo_name})")
    print()

    total_start = time.time()
    results = []

    for project in projects:
        repo_dir = get_project_dir(project.id)
        if not repo_dir.exists():
            print(f"[SKIP] {project.id}: repo dir not found at {repo_dir}")
            results.append((project.id, "skipped", "repo dir missing"))
            continue

        print(f"[START] Re-ingesting {project.repo_name} ({project.id})")
        start = time.time()
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                ingest_repo_to_neo4j,
                repo_dir,
                project.id,
                configs.APP_NEO4J_URL,
                configs.APP_NEO4J_USER,
                configs.APP_NEO4J_PASSWORD,
            )
            nodes = result.get("nodes", 0)
            rels = result.get("relationships", 0)
            files = result.get("files", 0)
            elapsed = time.time() - start
            print(f"  [OK] {files} files, {nodes} nodes, {rels} rels — {elapsed:.1f}s")
            results.append((project.id, "ok", f"{nodes} nodes, {rels} rels"))
        except Exception as e:
            elapsed = time.time() - start
            print(f"  [FAIL] {e} — {elapsed:.1f}s")
            results.append((project.id, "failed", str(e)))

    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'─'*60}")
    print(f"Re-ingest complete in {total_elapsed:.1f}s")
    for pid, status, detail in results:
        icon = "✓" if status == "ok" else ("⚠" if status == "skipped" else "✗")
        print(f"  {icon} {pid}: {detail}")

    # Verify Neo4j data
    print("\nVerifying Neo4j data quality...")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            configs.APP_NEO4J_URL,
            auth=(configs.APP_NEO4J_USER, configs.APP_NEO4J_PASSWORD),
        )
        with driver.session() as s:
            row = s.run(
                "MATCH (n:CodeNode) "
                "RETURN count(n) as total, "
                "count(n.line_count) as has_line, "
                "count(n.method_count) as has_method"
            ).single()
            inh = s.run("MATCH ()-[r:INHERITS]->() RETURN count(r) as cnt").single()
            imp = s.run("MATCH ()-[r:IMPORTS]->() RETURN count(r) as cnt").single()
        driver.close()
        print(f"  CodeNode total   : {row['total']}")
        print(f"  Has line_count   : {row['has_line']} / {row['total']}")
        print(f"  Has method_count : {row['has_method']} / {row['total']}")
        print(f"  INHERITS edges   : {inh['cnt']}")
        print(f"  IMPORTS edges    : {imp['cnt']}")
    except Exception as e:
        print(f"  Could not verify: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-ingest all projects into Neo4j")
    parser.add_argument("--project", help="Filter by project id or repo name")
    args = parser.parse_args()

    asyncio.run(reingest_all(args.project))
