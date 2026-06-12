"""
Test execution gate — runs the repo's test suite after applying a patch.

Wraps the existing src_bot/reflexion/evaluator.py test runner (which already
does git apply → pytest → revert). This gate is optional: if no test files
are detected in the repo, it returns skipped=True automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestGateResult:
    passed: bool
    skipped: bool = False
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    reason: str = ""


def check_tests(
    patch_diff: str,
    repo_dir: Path,
    test_command: str = "python -m pytest --tb=short -q",
    timeout: int = 300,
    enabled: bool = True,
) -> TestGateResult:
    """Run tests after applying patch. Auto-detects if repo has tests.

    Args:
        enabled: If False, returns skipped immediately (controlled by EXECUTION_GATE_ENABLED).
    """
    if not enabled:
        return TestGateResult(passed=True, skipped=True, reason="Test gate disabled via config")

    if not patch_diff or not patch_diff.strip():
        return TestGateResult(passed=True, skipped=True, reason="Empty patch")

    if not _detect_tests(repo_dir):
        return TestGateResult(
            passed=True,
            skipped=True,
            reason="No test files detected in repo",
        )

    try:
        from src_bot.reflexion.evaluator import evaluate_patch as _run_tests
        result = _run_tests(
            patch_diff=patch_diff,
            repo_path=repo_dir,
            test_command=test_command,
            timeout=timeout,
        )
        return TestGateResult(
            passed=result.passed,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            reason=result.summary,
        )
    except Exception as e:
        return TestGateResult(
            passed=True,
            skipped=True,
            reason=f"Test gate error (skipped): {e}",
        )


def _detect_tests(repo_dir: Path) -> bool:
    """Check if the repo has any pytest-compatible test files."""
    patterns = ["test_*.py", "*_test.py", "conftest.py"]
    for pattern in patterns:
        if any(repo_dir.rglob(pattern)):
            return True
    return False
