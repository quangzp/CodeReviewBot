"""
SWE-bench evaluation wrapper — runs the official harness on your predictions.

Requires Docker to be running.

Usage:
    # Evaluate predictions
    python -m swebench.evaluate --predictions predictions.jsonl

    # Evaluate with custom run ID
    python -m swebench.evaluate --predictions predictions.jsonl --run-id my_graphrag_v1

Prerequisites:
    pip install swebench
    docker info  # verify Docker is running
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def check_docker():
    """Verify Docker is running."""
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        print("ERROR: Docker is not running. Start Docker Desktop first.")
        print("The SWE-bench evaluation harness requires Docker to run tests.")
        sys.exit(1)
    print("Docker is running.")


def check_predictions(predictions_path: str) -> int:
    """Validate predictions file format and return count."""
    path = Path(predictions_path)
    if not path.exists():
        print(f"ERROR: Predictions file not found: {predictions_path}")
        sys.exit(1)

    count = 0
    with_patch = 0
    with open(path) as f:
        for line in f:
            pred = json.loads(line)
            assert "instance_id" in pred, "Missing instance_id"
            assert "model_patch" in pred, "Missing model_patch"
            count += 1
            if pred["model_patch"]:
                with_patch += 1

    print(f"Predictions: {count} tasks, {with_patch} with patches")
    return count


def run_evaluation(
    predictions_path: str,
    run_id: str = "graphrag_reflexion",
    max_workers: int = 4,
    dataset: str = "princeton-nlp/SWE-bench_Lite",
):
    """
    Run the official SWE-bench evaluation harness.

    This applies each prediction patch in an isolated Docker container,
    runs the task's test suite, and reports pass@1 results.
    """
    check_docker()
    check_predictions(predictions_path)

    print(f"\nStarting SWE-bench evaluation...")
    print(f"  Dataset: {dataset}")
    print(f"  Run ID: {run_id}")
    print(f"  Max workers: {max_workers}")
    print(f"  This may take a while (minutes to hours)...\n")

    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset,
        "--predictions_path", predictions_path,
        "--max_workers", str(max_workers),
        "--run_id", run_id,
    ]

    try:
        result = subprocess.run(
            cmd,
            timeout=3600 * 6,  # 6 hour timeout for full run
        )
        if result.returncode == 0:
            print("\nEvaluation complete!")
            _print_results(run_id)
        else:
            print(f"\nEvaluation exited with code {result.returncode}")
    except subprocess.TimeoutExpired:
        print("\nEvaluation timed out after 6 hours")
    except FileNotFoundError:
        print("\nERROR: swebench package not found. Install with:")
        print("  pip install swebench")


def _print_results(run_id: str):
    """Try to print evaluation results summary."""
    results_dir = Path(f"logs/{run_id}")
    if not results_dir.exists():
        results_dir = Path(run_id)

    # Look for results file
    for pattern in ["results.json", "*.results.json", "report.json"]:
        results_files = list(results_dir.rglob(pattern)) if results_dir.exists() else []
        if results_files:
            with open(results_files[0]) as f:
                results = json.load(f)
            total = results.get("total", 0)
            resolved = results.get("resolved", 0)
            rate = (resolved / total * 100) if total > 0 else 0
            print(f"\n{'='*60}")
            print(f"RESULTS: {resolved}/{total} resolved ({rate:.1f}%)")
            print(f"{'='*60}")
            return

    print("Results file not found. Check the logs directory for details.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against SWE-bench Lite"
    )
    parser.add_argument(
        "--predictions", "-p",
        required=True,
        help="Path to predictions.jsonl",
    )
    parser.add_argument(
        "--run-id",
        default="graphrag_reflexion",
        help="Run ID for this evaluation",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel Docker containers",
    )
    args = parser.parse_args()

    run_evaluation(
        predictions_path=args.predictions,
        run_id=args.run_id,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
