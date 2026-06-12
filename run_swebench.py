#!/usr/bin/env python3
"""
SWE-bench Lite runner — Neo4j GraphRAG + Reflexion pipeline.

Pipeline per task:
  1. Clone repo (cached) + checkout base commit
  2. Ingest repo into Neo4j (cached per repo, skipped if already done)
  3. BM25 search (Phase 1) → seed nodes
  4. Graph expansion (Phase 2) → callers/callees context
  5. LLM generates unified diff patch with graph context
  6. Save prediction

Usage:
    python3 run_swebench.py --limit 5 --output predictions.jsonl
    python3 run_swebench.py --output predictions.jsonl          # all 300
    python3 run_swebench.py --instance-id astropy__astropy-12907 --output p.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neo4j import GraphDatabase
from src_bot.llm.router import get_llm
from src_bot.config.config import configs
from swebench.neo4j_ingest import (
    ingest_repo_to_neo4j,
    bm25_search,
    graph_expand_simple,
    _sanitize_lucene,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHED_TASKS_PATH = "/tmp/swebench_lite_tasks.json"
REPOS_DIR = Path("/tmp/swebench_repos")
INGESTION_CACHE = Path("/tmp/swebench_ingestion_cache.json")

NEO4J_URI = configs.APP_NEO4J_URL
NEO4J_USER = configs.APP_NEO4J_USER
NEO4J_PASSWORD = configs.APP_NEO4J_PASSWORD

SWEBENCH_REPOS = {
    "django/django": "https://github.com/django/django.git",
    "scikit-learn/scikit-learn": "https://github.com/scikit-learn/scikit-learn.git",
    "sympy/sympy": "https://github.com/sympy/sympy.git",
    "matplotlib/matplotlib": "https://github.com/matplotlib/matplotlib.git",
    "pytest-dev/pytest": "https://github.com/pytest-dev/pytest.git",
    "pallets/flask": "https://github.com/pallets/flask.git",
    "psf/requests": "https://github.com/psf/requests.git",
    "astropy/astropy": "https://github.com/astropy/astropy.git",
    "pydata/xarray": "https://github.com/pydata/xarray.git",
    "sphinx-doc/sphinx": "https://github.com/sphinx-doc/sphinx.git",
    "mwaskom/seaborn": "https://github.com/mwaskom/seaborn.git",
    "pylint-dev/pylint": "https://github.com/pylint-dev/pylint.git",
}

# ---------------------------------------------------------------------------
# Ingestion cache
# ---------------------------------------------------------------------------
def load_ingestion_cache() -> dict:
    if INGESTION_CACHE.exists():
        return json.loads(INGESTION_CACHE.read_text())
    return {}

def save_ingestion_cache(cache: dict):
    INGESTION_CACHE.write_text(json.dumps(cache, indent=2))

def is_ingested(project_id: str, cache: dict) -> bool:
    return cache.get(project_id, {}).get("ingested", False)

def mark_ingested(project_id: str, cache: dict, stats: dict):
    cache[project_id] = {"ingested": True, **stats}
    save_ingestion_cache(cache)

# ---------------------------------------------------------------------------
# Repo management
# ---------------------------------------------------------------------------
def clone_repo(repo_name: str) -> Path:
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    safe = repo_name.replace("/", "__")
    repo_dir = REPOS_DIR / safe
    if not repo_dir.exists():
        url = SWEBENCH_REPOS.get(repo_name, f"https://github.com/{repo_name}.git")
        print(f"  Cloning {repo_name}...")
        subprocess.run(["git", "clone", "--depth", "1", url, str(repo_dir)],
                       capture_output=True, text=True, timeout=600)
    return repo_dir

def checkout_commit(repo_dir: Path, commit: str):
    subprocess.run(["git", "fetch", "--unshallow"], cwd=repo_dir,
                   capture_output=True, timeout=600)
    subprocess.run(["git", "fetch", "origin", commit], cwd=repo_dir,
                   capture_output=True, timeout=120)
    subprocess.run(["git", "checkout", commit], cwd=repo_dir,
                   capture_output=True, timeout=60)

# ---------------------------------------------------------------------------
# Neo4j GraphRAG retrieval
# ---------------------------------------------------------------------------
def retrieve_graph_context(
    session,
    issue_text: str,
    project_id: str,
    top_k: int = 15,
) -> str:
    """
    Two-phase retrieval:
      Phase 1: BM25 search → seed nodes
      Phase 2: Graph expansion → callers/callees

    Returns formatted context string for LLM.
    """
    # Phase 1: BM25
    seeds = bm25_search(session, issue_text[:1000], project_id, top_k=top_k)
    if not seeds:
        return ""

    seed_ids = [s["node_id"] for s in seeds if s.get("node_id")]

    # Phase 2: Graph expansion
    expanded = []
    if seed_ids:
        expanded = graph_expand_simple(session, seed_ids, project_id, max_results=30)

    # Format context
    parts = ["## GraphRAG Context (from Neo4j code property graph)\n"]

    # Seed nodes (most relevant)
    parts.append("### Top relevant code nodes (BM25 search):")
    for node in seeds[:8]:
        parts.append(f"\n**{node.get('type','?')}: {node.get('qualified_name', node.get('name','?'))}**")
        parts.append(f"File: `{node.get('file_path','?')}`")
        if node.get("docstring"):
            parts.append(f"Docstring: {node['docstring'][:200]}")
        if node.get("content"):
            parts.append(f"```python\n{node['content'][:800]}\n```")

    # Expanded context (callers/callees)
    if expanded:
        parts.append("\n### Related code (callers & callees from graph):")
        seen = {s.get("qualified_name") for s in seeds}
        for node in expanded[:10]:
            qname = node.get("qualified_name", node.get("name", ""))
            if qname in seen:
                continue
            seen.add(qname)
            parts.append(f"\n**{node.get('type','?')}: {qname}**")
            parts.append(f"File: `{node.get('file_path','?')}`")
            calls = node.get("calls", [])
            called_by = node.get("called_by", [])
            if calls:
                parts.append(f"Calls: {', '.join(calls[:5])}")
            if called_by:
                parts.append(f"Called by: {', '.join(called_by[:5])}")
            if node.get("content"):
                parts.append(f"```python\n{node['content'][:500]}\n```")

    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Patch generation with graph context
# ---------------------------------------------------------------------------
def generate_patch_with_context(
    llm,
    issue_text: str,
    graph_context: str,
) -> str:
    prompt = f"""You are a senior Python developer fixing a real GitHub issue.

## Issue Description
{issue_text[:3000]}

## Code Context (from knowledge graph — relevant functions, callers, callees)
{graph_context[:12000] if graph_context else "No graph context available."}

## Instructions
1. Analyze the issue using the code context above.
2. Identify the exact root cause and which file/function needs changing.
3. Generate a MINIMAL unified diff patch — change only what is necessary.
4. Make sure the fix doesn't break callers listed in the context.

Return ONLY a unified diff in ```diff ... ``` format. No explanation.

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-old line
+new line
 context line
```"""

    try:
        time.sleep(2)  # Groq rate limit
        response = llm.invoke(prompt)
        return _extract_diff(response.content)
    except Exception as e:
        print(f"  LLM error: {e}")
        return ""

def _extract_diff(text: str) -> str:
    if "```diff" in text:
        s = text.index("```diff") + 7
        e = text.index("```", s)
        return text[s:e].strip()
    elif "--- a/" in text:
        lines, dl, ind = text.split("\n"), [], False
        for line in lines:
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                ind = True
            if ind:
                dl.append(line)
        return "\n".join(dl)
    return text

# ---------------------------------------------------------------------------
# Main task runner
# ---------------------------------------------------------------------------
def run_single_task(
    task: dict,
    llm,
    neo4j_driver,
    ingestion_cache: dict,
) -> dict:
    instance_id = task["instance_id"]
    repo_name = task["repo"]
    base_commit = task["base_commit"]
    issue_text = task["problem_statement"]
    hints = task.get("hints_text", "")
    if hints:
        issue_text += f"\n\nHints:\n{hints}"

    project_id = repo_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"[{instance_id}]  repo={repo_name}")

    # Step 1: Clone + checkout
    repo_dir = clone_repo(repo_name)
    checkout_commit(repo_dir, base_commit)

    # Step 2: Ingest repo into Neo4j (once per repo, cached)
    if not is_ingested(project_id, ingestion_cache):
        print(f"  Ingesting {repo_name} into Neo4j...")
        try:
            stats = ingest_repo_to_neo4j(
                repo_dir, project_id,
                NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
                max_files=300,
            )
            mark_ingested(project_id, ingestion_cache, stats)
        except Exception as e:
            print(f"  Ingestion failed: {e} — falling back to LLM-only")
    else:
        print(f"  Neo4j: already ingested (cached)")

    # Step 3: Retrieve graph context + generate patch
    graph_context = ""
    try:
        with neo4j_driver.session() as session:
            graph_context = retrieve_graph_context(
                session, issue_text, project_id, top_k=15
            )
        seeds_found = "✓" if graph_context else "✗"
        print(f"  Graph context: {seeds_found} ({len(graph_context)} chars)")
    except Exception as e:
        print(f"  Graph retrieval failed: {e}")

    # Step 4: Generate patch
    patch = generate_patch_with_context(llm, issue_text, graph_context)
    print(f"  Patch: {'✓ ' + str(len(patch)) + ' chars' if patch else '✗ none'}")

    return {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": f"neo4j-graphrag-{configs.LLM_MODEL}",
    }

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SWE-bench Lite — Neo4j GraphRAG runner")
    parser.add_argument("--output", "-o", default="predictions.jsonl")
    parser.add_argument("--limit", "-n", type=int, default=None)
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip tasks already in output file")
    args = parser.parse_args()

    # Load tasks from cache
    if not os.path.exists(CACHED_TASKS_PATH):
        print("ERROR: Run this first to cache the dataset:")
        print("  python3 -c \"from datasets import load_dataset; import json; "
              "ds=load_dataset('princeton-nlp/SWE-bench_Lite',split='test'); "
              f"json.dump([dict(t) for t in ds], open('{CACHED_TASKS_PATH}','w'))\"")
        sys.exit(1)

    tasks = json.loads(Path(CACHED_TASKS_PATH).read_text())
    print(f"Loaded {len(tasks)} tasks")

    # Filter
    if args.instance_id:
        tasks = [t for t in tasks if t["instance_id"] == args.instance_id]
        if not tasks:
            print(f"Task '{args.instance_id}' not found"); sys.exit(1)
    if args.limit:
        tasks = tasks[:args.limit]

    # Resume: skip already-done tasks
    done_ids = set()
    existing_preds = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    done_ids.add(p["instance_id"])
                    existing_preds.append(p)
        print(f"Resuming: {len(done_ids)} tasks already done, {len(tasks)-len(done_ids)} remaining")
        tasks = [t for t in tasks if t["instance_id"] not in done_ids]

    print(f"Running {len(tasks)} tasks...")

    # Init LLM + Neo4j
    llm = get_llm(provider=configs.LLM_PROVIDER, model=configs.LLM_MODEL, temperature=0)
    print(f"LLM: {configs.LLM_PROVIDER}/{configs.LLM_MODEL}")

    try:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()
        print(f"Neo4j: connected ({NEO4J_URI})")
    except Exception as e:
        print(f"Neo4j connection failed: {e}")
        print("Check your .env: APP_NEO4J_URL, APP_NEO4J_USER, APP_NEO4J_PASSWORD")
        sys.exit(1)

    ingestion_cache = load_ingestion_cache()
    predictions = list(existing_preds)
    start_time = time.time()
    errors = 0

    for i, task in enumerate(tasks):
        task_start = time.time()
        print(f"\n[{i+1+len(done_ids)}/{len(tasks)+len(done_ids)}]", end=" ")

        try:
            pred = run_single_task(task, llm, neo4j_driver, ingestion_cache)
            predictions.append(pred)
            if not pred["model_patch"]:
                errors += 1
        except Exception as e:
            print(f"  TASK ERROR: {e}")
            predictions.append({
                "instance_id": task["instance_id"],
                "model_patch": "",
                "model_name_or_path": f"neo4j-graphrag-{configs.LLM_MODEL}",
            })
            errors += 1
            time.sleep(5)

        task_time = time.time() - task_start
        print(f"  ({task_time:.1f}s)")

        # Save after every task
        with open(args.output, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

        # Progress report every 10 tasks
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            patched = sum(1 for p in predictions if p["model_patch"])
            rate = (i + 1) / elapsed * 60
            eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
            print(f"\n  ── Progress: {patched}/{i+1+len(done_ids)} patched | "
                  f"{errors} errors | {rate:.0f} tasks/min | ETA {eta:.0f}min ──")

    neo4j_driver.close()

    total_time = time.time() - start_time
    patched = sum(1 for p in predictions if p["model_patch"])
    total = len(predictions)
    print(f"\n{'='*60}")
    print(f"DONE: {total} tasks in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Patches generated: {patched}/{total} ({100*patched/total:.1f}%)")
    print(f"Errors/empty: {errors}")
    print(f"Output: {args.output}")
    print(f"\nNext step — evaluate with SWE-bench harness:")
    print(f"  python3 -m swebench.harness.run_evaluation \\")
    print(f"    --predictions_path {args.output} \\")
    print(f"    --max_workers 4 --run_id graphrag_run1")


if __name__ == "__main__":
    main()
