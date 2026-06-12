"""
SWE-bench Lite runner — runs the GraphRAG + Reflexion pipeline on SWE-bench tasks.

Usage:
    # Run on a subset (for development)
    python -m swebench.runner --limit 20 --output predictions.jsonl

    # Run all 300 tasks
    python -m swebench.runner --output predictions.jsonl

    # Run a single task
    python -m swebench.runner --instance-id django__django-16379 --output predictions.jsonl

After generating predictions, evaluate with:
    python -m swebench.evaluate --predictions predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Fix mutex deadlock with tokenizers on macOS Python 3.9
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset

from src_bot.llm.router import get_llm
from src_bot.config.config import configs


# SWE-bench Lite repos and their GitHub URLs
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


def clone_or_update_repo(repo_name: str, base_dir: Path) -> Path:
    """Clone repo if not present, or pull latest if it exists."""
    safe_name = repo_name.replace("/", "__")
    repo_dir = base_dir / safe_name

    if repo_dir.exists():
        print(f"  Repo '{repo_name}' already cloned at {repo_dir}")
        return repo_dir

    url = SWEBENCH_REPOS.get(repo_name)
    if not url:
        # Try to construct URL
        url = f"https://github.com/{repo_name}.git"

    print(f"  Cloning {repo_name}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(repo_dir)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    return repo_dir


def checkout_commit(repo_dir: Path, base_commit: str):
    """Checkout a specific commit for SWE-bench task."""
    # Need to unshallow first if we did --depth 1
    subprocess.run(
        ["git", "fetch", "--unshallow"],
        cwd=repo_dir,
        capture_output=True,
        timeout=600,
    )
    subprocess.run(
        ["git", "fetch", "origin", base_commit],
        cwd=repo_dir,
        capture_output=True,
        timeout=120,
    )
    subprocess.run(
        ["git", "checkout", base_commit],
        cwd=repo_dir,
        capture_output=True,
        timeout=60,
    )


def run_single_task(
    task: dict,
    repos_dir: Path,
) -> dict:
    """
    Run the GraphRAG + Reflexion pipeline on a single SWE-bench task.

    Args:
        task: SWE-bench task dict with keys: instance_id, repo, base_commit,
              problem_statement, hints_text, test_patch, patch, etc.
        repos_dir: Directory where repos are cloned.

    Returns:
        Prediction dict with: instance_id, model_patch, model_name_or_path.
    """
    instance_id = task["instance_id"]
    repo_name = task["repo"]
    base_commit = task["base_commit"]
    problem_statement = task["problem_statement"]
    hints = task.get("hints_text", "")

    print(f"\n{'='*60}")
    print(f"Task: {instance_id}")
    print(f"Repo: {repo_name}")
    print(f"{'='*60}")

    # Step 1: Clone repo and checkout base commit
    repo_dir = clone_or_update_repo(repo_name, repos_dir)
    checkout_commit(repo_dir, base_commit)

    # Step 2: Try to ingest into Neo4j + Weaviate (skip if DBs not available)
    graph_available = False
    try:
        from swebench.ingest_repo import ingest_repo, is_repo_ingested
        project_id = repo_name.replace("/", "_")
        if not is_repo_ingested(repo_dir, project_id):
            ingest_repo(repo_dir, project_id)
        graph_available = True
    except ImportError:
        print(f"  Graph ingestion skipped (ingest_repo not available)")
        print(f"  Running in LLM-only mode (no GraphRAG context)")
    except Exception as e:
        print(f"  Graph ingestion skipped (Neo4j/Weaviate not available): {e}")
        print(f"  Running in LLM-only mode (no GraphRAG context)")

    # Step 3: Run the pipeline
    # Combine problem statement + hints as the issue text
    issue_text = problem_statement
    if hints:
        issue_text += f"\n\nHints:\n{hints}"

    if graph_available:
        # Full GraphRAG pipeline
        model_patch = _run_with_graphrag(issue_text, repo_dir)
    else:
        # LLM-only mode (no graph context)
        model_patch = _run_llm_only(issue_text, repo_dir)

    print(f"  Patch: {'Generated' if model_patch else 'None'}")

    return {
        "instance_id": instance_id,
        "model_patch": model_patch,
        "model_name_or_path": f"graphrag-reflexion-{os.getenv('LLM_MODEL', 'unknown')}",
    }


def run_swebench(
    output_path: str,
    limit: int | None = None,
    instance_id: str | None = None,
    repos_dir: str | None = None,
):
    """
    Run the pipeline on SWE-bench Lite tasks and save predictions.

    Args:
        output_path: Path to save predictions.jsonl.
        limit: Max number of tasks to run (None = all).
        instance_id: Run only this specific task.
        repos_dir: Directory for cloned repos (default: /tmp/swebench_repos).
    """
    repos_dir = Path(repos_dir or "/tmp/swebench_repos")
    repos_dir.mkdir(parents=True, exist_ok=True)

    # Load SWE-bench Lite dataset
    print("Loading SWE-bench Lite dataset...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    print(f"Total tasks: {len(ds)}")

    # Filter tasks
    tasks = list(ds)
    if instance_id:
        tasks = [t for t in tasks if t["instance_id"] == instance_id]
        if not tasks:
            print(f"Task '{instance_id}' not found in dataset")
            sys.exit(1)
    if limit:
        tasks = tasks[:limit]

    print(f"Running {len(tasks)} tasks...")

    # Run tasks and collect predictions
    predictions = []
    start_time = time.time()

    for i, task in enumerate(tasks):
        task_start = time.time()
        print(f"\n[{i+1}/{len(tasks)}] ", end="")

        prediction = run_single_task(task, repos_dir)
        predictions.append(prediction)

        task_time = time.time() - task_start
        print(f"  Time: {task_time:.1f}s")

        # Save after each task (in case of crash)
        _save_predictions(predictions, output_path)

    total_time = time.time() - start_time
    patched = sum(1 for p in predictions if p["model_patch"])
    print(f"\n{'='*60}")
    print(f"DONE: {len(predictions)} tasks in {total_time:.0f}s")
    print(f"Patches generated: {patched}/{len(predictions)}")
    print(f"Predictions saved to: {output_path}")


def _run_with_graphrag(issue_text: str, repo_dir: Path) -> str:
    """Stub — superseded by run_swebench_v2.py which uses the full pipeline."""
    raise NotImplementedError(
        "_run_with_graphrag is deprecated. Use run_swebench_v2.py instead."
    )


def _run_llm_only(issue_text: str, repo_dir: Path) -> str:
    """
    LLM-only mode — no Neo4j/Weaviate required.

    Reads relevant files from the repo, sends them with the issue
    to the LLM, and asks for a unified diff patch.
    """
    import glob
    import time as _time

    llm = get_llm(
        provider=configs.LLM_PROVIDER,
        model=configs.LLM_MODEL,
        temperature=0,
    )

    # Gather repo context: find Python files likely related to the issue
    # Use keywords from the issue to find relevant files
    keywords = _extract_keywords(issue_text)
    relevant_files = _find_relevant_files(repo_dir, keywords, max_files=10)

    file_context = ""
    for fpath in relevant_files:
        try:
            content = fpath.read_text(errors="ignore")
            # Cap each file at 200 lines to stay within context window
            lines = content.split("\n")
            if len(lines) > 200:
                content = "\n".join(lines[:200]) + "\n... (truncated)"
            rel = fpath.relative_to(repo_dir)
            file_context += f"\n\n### File: {rel}\n```python\n{content}\n```"
        except Exception:
            continue

    prompt = f"""You are a senior Python developer. Fix the following bug by generating a unified diff patch.

ISSUE DESCRIPTION:
{issue_text[:3000]}

RELEVANT SOURCE FILES:
{file_context[:12000]}

INSTRUCTIONS:
1. Analyze the issue carefully.
2. Identify the root cause in the source files above.
3. Generate a MINIMAL fix — change only what is necessary.
4. Return ONLY a unified diff patch. No explanation.

Format:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-old line
+new line
 context line
```
"""

    try:
        # Groq rate limit: 30 req/min — add small delay
        _time.sleep(2)
        response = llm.invoke(prompt)
        patch = response.content

        # Extract diff block
        if "```diff" in patch:
            start = patch.index("```diff") + len("```diff")
            end = patch.index("```", start)
            return patch[start:end].strip()
        elif "--- a/" in patch:
            lines = patch.split("\n")
            diff_lines = []
            in_diff = False
            for line in lines:
                if line.startswith("--- a/") or line.startswith("+++ b/"):
                    in_diff = True
                if in_diff:
                    diff_lines.append(line)
            return "\n".join(diff_lines)
        return patch
    except Exception as e:
        print(f"  LLM error: {e}")
        return ""


def _extract_keywords(issue_text: str) -> list[str]:
    """Extract likely-relevant keywords from issue text for file search."""
    import re

    # Look for Python identifiers, class names, function names, file paths
    words = set()

    # Find anything that looks like a Python path (e.g., django/db/models/query.py)
    paths = re.findall(r'[\w/]+\.py', issue_text)
    for p in paths:
        words.add(p)
        # Also add the filename stem
        words.add(Path(p).stem)

    # Find CamelCase class names
    classes = re.findall(r'\b[A-Z][a-zA-Z]+\b', issue_text)
    words.update(classes[:10])

    # Find function/method names (word followed by parentheses)
    funcs = re.findall(r'\b(\w+)\s*\(', issue_text)
    words.update(funcs[:10])

    # Find identifiers with underscores (likely function/variable names)
    underscored = re.findall(r'\b\w+_\w+\b', issue_text)
    words.update(underscored[:10])

    return list(words)[:20]


def _find_relevant_files(repo_dir: Path, keywords: list[str], max_files: int = 10) -> list[Path]:
    """Find Python files in repo that match keywords from the issue using grep."""
    relevant = set()

    # Primary strategy: grep for keywords in file content (most accurate)
    for kw in keywords[:10]:
        if len(kw) < 3:
            continue
        try:
            grep_result = subprocess.run(
                ["grep", "-rl", kw, "--include=*.py", str(repo_dir)],
                capture_output=True, text=True, timeout=30,
            )
            for line in grep_result.stdout.strip().split("\n"):
                line = line.strip()
                if line and "__pycache__" not in line and ".git/" not in line:
                    relevant.add(Path(line))
        except Exception:
            continue

    # Score files: files matching more keywords rank higher
    scored = {}
    for fpath in relevant:
        rel_path = str(fpath.relative_to(repo_dir))
        score = 0
        for kw in keywords:
            if kw.lower() in rel_path.lower():
                score += 5  # keyword in path
        # Count how many keywords grep found in this file
        try:
            content = fpath.read_text(errors="ignore")[:5000]
            for kw in keywords[:10]:
                if kw in content:
                    score += 3
        except Exception:
            pass
        # Prefer non-test files
        if "/test" in rel_path or "test_" in rel_path:
            score -= 2
        scored[fpath] = score

    result = sorted(scored.keys(), key=lambda f: -scored[f])
    return result[:max_files]


def _save_predictions(predictions: list[dict], output_path: str):
    """Save predictions in JSONL format (SWE-bench expected format)."""
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run GraphRAG + Reflexion on SWE-bench Lite"
    )
    parser.add_argument(
        "--output", "-o",
        default="predictions.jsonl",
        help="Output predictions file (default: predictions.jsonl)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Max tasks to run (default: all)",
    )
    parser.add_argument(
        "--instance-id",
        default=None,
        help="Run only this specific task instance",
    )
    parser.add_argument(
        "--repos-dir",
        default="/tmp/swebench_repos",
        help="Directory for cloned repos",
    )
    args = parser.parse_args()

    run_swebench(
        output_path=args.output,
        limit=args.limit,
        instance_id=args.instance_id,
        repos_dir=args.repos_dir,
    )


if __name__ == "__main__":
    main()
