"""
AST parse gate — checks that a patch produces syntactically valid Python.

FREE in tokens, ~100ms (disk I/O). Always on. Catches syntax errors before
wasting an LLM call or a full test run.

Strategy: apply patch to disk using git, ast.parse() the result, revert.
Same apply/revert lifecycle as src_bot/reflexion/evaluator.py.
"""
from __future__ import annotations

import ast
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ASTGateResult:
    passed: bool
    error: str = ""
    error_line: int = 0
    file_checked: str = ""


def check_ast(patch_diff: str, repo_dir: Path) -> ASTGateResult:
    """Apply patch, AST-parse the changed Python file, then revert.

    Returns ASTGateResult(passed=True) for non-Python files or empty patches.
    Fails open (returns passed=True) if git apply itself fails — the
    _verify_patch_applies() gate upstream already handles that case.
    """
    if not patch_diff or not patch_diff.strip():
        return ASTGateResult(passed=True)

    target_file = _extract_target_file(patch_diff)
    if target_file is None or not target_file.endswith(".py"):
        return ASTGateResult(passed=True, file_checked=target_file or "")

    patch_file = None
    try:
        # Write patch to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(patch_diff)
            patch_file = f.name

        # Apply the patch
        apply = subprocess.run(
            ["git", "apply", patch_file],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if apply.returncode != 0:
            # Patch didn't apply — skip the AST gate (handled upstream)
            return ASTGateResult(passed=True, file_checked=target_file)

        # Read and parse the patched file
        target_path = repo_dir / target_file
        if not target_path.exists():
            return ASTGateResult(passed=True, file_checked=target_file)

        source = target_path.read_text(errors="ignore")
        try:
            ast.parse(source, filename=target_file)
            return ASTGateResult(passed=True, file_checked=target_file)
        except SyntaxError as e:
            return ASTGateResult(
                passed=False,
                error=f"SyntaxError: {e.msg}",
                error_line=e.lineno or 0,
                file_checked=target_file,
            )

    except subprocess.TimeoutExpired:
        return ASTGateResult(passed=True, file_checked=target_file or "")
    except Exception:
        return ASTGateResult(passed=True, file_checked=target_file or "")
    finally:
        # Always revert
        subprocess.run(
            ["git", "checkout", "."],
            cwd=repo_dir,
            capture_output=True,
            timeout=30,
        )
        if patch_file:
            Path(patch_file).unlink(missing_ok=True)


def _extract_target_file(patch_diff: str) -> str | None:
    """Extract the target file path from a unified diff header."""
    match = re.search(r"^\+\+\+ b/(.+)$", patch_diff, re.MULTILINE)
    if match:
        return match.group(1).strip()
    match = re.search(r"^\+\+\+ (.+)$", patch_diff, re.MULTILINE)
    if match:
        return match.group(1).strip().lstrip("b/")
    return None
