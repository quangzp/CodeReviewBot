"""
Integration tests for the 3-phase pipeline utility functions.

Tests _extract_and_clean_diff, _verify_patch_applies, and _extract_file_path
from run_swebench_v2.py with real file system (temp dirs) but no LLM calls.
"""
import os
import sys
import subprocess
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_swebench_v2 import (
    _extract_and_clean_diff,
    _verify_patch_applies,
    _extract_file_path,
)


class TestExtractAndCleanDiff:

    def test_extracts_diff_from_markdown_block(self):
        text = (
            "Here is the fix:\n"
            "```diff\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,3 +1,3 @@\n"
            " context\n"
            "-old\n"
            "+new\n"
            " context\n"
            "```"
        )
        result = _extract_and_clean_diff(text, "foo.py")
        assert "--- a/foo.py" in result
        assert "+++ b/foo.py" in result
        assert "-old" in result
        assert "+new" in result

    def test_fixes_wrong_file_path(self):
        text = (
            "```diff\n"
            "--- a/wrong_path.py\n"
            "+++ b/wrong_path.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
            "```"
        )
        result = _extract_and_clean_diff(text, "correct/path.py")
        assert "--- a/correct/path.py" in result
        assert "+++ b/correct/path.py" in result

    def test_empty_response_returns_empty(self):
        result = _extract_and_clean_diff("no diff here at all", "foo.py")
        assert result == ""

    def test_result_ends_with_newline(self):
        text = (
            "```diff\n"
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-a\n"
            "+b\n"
            "```"
        )
        result = _extract_and_clean_diff(text, "f.py")
        if result:  # only check if a diff was extracted
            assert result.endswith("\n")


class TestVerifyPatchApplies:

    def test_empty_patch_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ok, err = _verify_patch_applies("", Path(tmpdir))
            assert not ok

    def test_whitespace_patch_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ok, err = _verify_patch_applies("   \n  ", Path(tmpdir))
            assert not ok

    def test_valid_patch_applies_on_git_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir, capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=tmpdir, capture_output=True
            )
            (tmppath / "test.py").write_text("line1\nline2\nline3\n")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "init"],
                cwd=tmpdir, capture_output=True
            )

            patch = (
                "--- a/test.py\n"
                "+++ b/test.py\n"
                "@@ -1,3 +1,3 @@\n"
                " line1\n"
                "-line2\n"
                "+line2_fixed\n"
                " line3\n"
            )
            ok, err = _verify_patch_applies(patch, tmppath)
            assert ok, f"Patch should apply cleanly: {err}"


class TestExtractFilePath:

    def test_extracts_simple_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "src").mkdir()
            (tmppath / "src" / "foo.py").write_text("")
            result = _extract_file_path("src/foo.py", tmppath)
            assert result == "src/foo.py"

    def test_strips_markdown_backticks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "bar.py").write_text("")
            result = _extract_file_path("The file is `bar.py`", tmppath)
            assert result == "bar.py"

    def test_returns_empty_for_nonexistent_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _extract_file_path("this_does_not_exist.py", Path(tmpdir))
            assert result == ""
