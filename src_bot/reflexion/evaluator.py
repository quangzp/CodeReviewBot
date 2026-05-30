from __future__ import annotations

"""
Evaluator (M_e) — runs tests to verify if a patch fixes the bug.

For GitHub PRs: applies patch locally and runs pytest.
For SWE-bench: applies patch in Docker container and runs the task's test suite.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalResult:
    """Result of running tests after applying a patch."""

    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    test_command: str

    @property
    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"Tests {status} (exit code {self.exit_code})"


def evaluate_patch(
    patch_diff: str,
    repo_path: str | Path,
    test_command: str = "python -m pytest --tb=short -q",
    timeout: int = 300,
) -> EvalResult:
    """
    Apply a patch to a repo and run tests.

    Args:
        patch_diff: Unified diff string to apply.
        repo_path: Path to the git repository root.
        test_command: Test command to run (default: pytest).
        timeout: Max seconds to wait for tests.

    Returns:
        EvalResult with test output and pass/fail status.
    """
    repo_path = Path(repo_path)

    # Write patch to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False
    ) as f:
        f.write(patch_diff)
        patch_file = f.name

    try:
        # Apply patch
        apply_result = subprocess.run(
            ["git", "apply", "--check", patch_file],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if apply_result.returncode != 0:
            return EvalResult(
                passed=False,
                exit_code=apply_result.returncode,
                stdout="",
                stderr=f"Patch apply failed:\n{apply_result.stderr}",
                test_command="git apply --check",
            )

        # Actually apply the patch
        subprocess.run(
            ["git", "apply", patch_file],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Run tests
        test_result = subprocess.run(
            test_command.split(),
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return EvalResult(
            passed=test_result.returncode == 0,
            exit_code=test_result.returncode,
            stdout=test_result.stdout[-3000:],  # cap output for LLM context
            stderr=test_result.stderr[-3000:],
            test_command=test_command,
        )

    except subprocess.TimeoutExpired:
        return EvalResult(
            passed=False,
            exit_code=-1,
            stdout="",
            stderr=f"Test execution timed out after {timeout}s",
            test_command=test_command,
        )
    except Exception as e:
        return EvalResult(
            passed=False,
            exit_code=-1,
            stdout="",
            stderr=f"Evaluation error: {str(e)}",
            test_command=test_command,
        )
    finally:
        # Revert the patch so repo stays clean
        subprocess.run(
            ["git", "checkout", "."],
            cwd=repo_path,
            capture_output=True,
            timeout=30,
        )
        Path(patch_file).unlink(missing_ok=True)
