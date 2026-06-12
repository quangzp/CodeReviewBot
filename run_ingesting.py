#!/usr/bin/env python3
"""
Pre-ingest SWE-bench repository snapshots into Neo4j.

This script uses the same repo checkout, Neo4j project_id, and ingestion cache
keys as run_swebench_v2.py. Run it before prediction so run_swebench_v2.py can
skip ingestion for snapshots that are already cached as ingested.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neo4j import GraphDatabase

from run_swebench_v2 import (
    CACHED_TASKS_PATH,
    INGESTION_CACHE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    checkout_commit,
    clone_repo,
    load_ingestion_cache,
    make_ingestion_cache_key,
    make_project_id,
    save_ingestion_cache,
)
from swebench.neo4j_ingest import ingest_repo_to_neo4j


DEFAULT_MAX_FILES = int(os.getenv("SWEBENCH_INGEST_MAX_FILES", "1500"))


def load_tasks() -> list[dict]:
    cache_path = Path(CACHED_TASKS_PATH)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print("SWE-bench Lite task cache not found; downloading from HuggingFace...")
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "datasets is required to download SWE-bench Lite. "
            "Install it or run run_swebench_v2.py once to create the cache."
        ) from exc

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    tasks = [dict(row) for row in ds]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(tasks), encoding="utf-8")
    print(f"Downloaded and cached {len(tasks)} tasks at {cache_path}")
    return tasks


def filter_tasks(tasks: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.instance_id:
        wanted = set(args.instance_id)
        tasks = [task for task in tasks if task["instance_id"] in wanted]

    if args.repo:
        repos = set(args.repo)
        tasks = [task for task in tasks if task["repo"] in repos]

    if args.offset:
        tasks = tasks[args.offset:]

    if args.limit is not None:
        tasks = tasks[: args.limit]

    return tasks


def unique_snapshots(tasks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    snapshots: list[dict] = []
    for task in tasks:
        cache_key = make_ingestion_cache_key(task["repo"], task["base_commit"])
        if cache_key in seen:
            continue
        seen.add(cache_key)
        snapshots.append(task)
    return snapshots


def check_neo4j() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    finally:
        driver.close()


def ingest_snapshot(task: dict, cache: dict, max_files: int, force: bool) -> str:
    repo_name = task["repo"]
    base_commit = task["base_commit"]
    cache_key = make_ingestion_cache_key(repo_name, base_commit)
    project_id = make_project_id(repo_name, base_commit)

    if not force and cache.get(cache_key, {}).get("ingested"):
        print(f"  SKIP {cache_key} ({project_id})")
        return "skipped"

    repo_dir = clone_repo(repo_name)
    checkout_commit(repo_dir, base_commit)

    print(f"  INGEST {cache_key} ({project_id})")
    try:
        stats = ingest_repo_to_neo4j(
            repo_dir=repo_dir,
            project_id=project_id,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            max_files=max_files,
        )
        cache[cache_key] = {
            "ingested": True,
            "repo": repo_name,
            "base_commit": base_commit,
            "project_id": project_id,
            "max_files": max_files,
            **stats,
        }
        save_ingestion_cache(cache)
        return "ingested"
    except Exception as exc:
        cache[cache_key] = {
            "ingested": False,
            "repo": repo_name,
            "base_commit": base_commit,
            "project_id": project_id,
            "max_files": max_files,
            "error": str(exc),
        }
        save_ingestion_cache(cache)
        print(f"  FAILED {cache_key}: {exc}")
        return "failed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-ingest SWE-bench Lite base_commit snapshots into Neo4j"
    )
    parser.add_argument("--limit", "-n", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--instance-id", action="append", default=None)
    parser.add_argument("--repo", action="append", default=None)
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--force", action="store_true", help="Re-ingest even if cached")
    args = parser.parse_args()

    tasks = filter_tasks(load_tasks(), args)
    snapshots = unique_snapshots(tasks)

    print(f"Tasks selected: {len(tasks)}")
    print(f"Unique repo@base_commit snapshots: {len(snapshots)}")
    print(f"Neo4j: {NEO4J_URI}")
    print(f"Ingestion cache: {INGESTION_CACHE}")
    print(f"Max files per snapshot: {args.max_files}")

    check_neo4j()
    print("Neo4j: connected")

    cache = load_ingestion_cache()
    counts: Counter[str] = Counter()
    start = time.time()

    for index, task in enumerate(snapshots, 1):
        t0 = time.time()
        print(
            f"\n[{index}/{len(snapshots)}] "
            f"{task['repo']} @ {task['base_commit'][:12]}"
        )
        status = ingest_snapshot(
            task=task,
            cache=cache,
            max_files=args.max_files,
            force=args.force,
        )
        counts[status] += 1
        print(f"  ({time.time() - t0:.1f}s)")

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"DONE in {elapsed / 60:.1f}min")
    for status, count in counts.most_common():
        print(f"{status}: {count}")


if __name__ == "__main__":
    main()
