"""Unit tests for the CodeContext dataclass."""
import pytest
from pathlib import Path

from api.agent.code_context import CodeContext


class TestCodeContextIsValid:

    def _make(self, file_path="", file_content=""):
        return CodeContext(
            project=None,
            repo_dir=Path("/tmp"),
            file_path=file_path,
            file_content=file_content,
            fault_description="",
            graph_context="",
            memory_context="",
        )

    def test_valid_when_both_file_path_and_content(self):
        ctx = self._make(file_path="src/foo.py", file_content="def foo(): pass")
        assert ctx.is_valid is True

    def test_invalid_when_no_file_path(self):
        ctx = self._make(file_path="", file_content="some content")
        assert ctx.is_valid is False

    def test_invalid_when_no_file_content(self):
        ctx = self._make(file_path="src/foo.py", file_content="")
        assert ctx.is_valid is False

    def test_invalid_when_both_empty(self):
        ctx = self._make()
        assert ctx.is_valid is False


class TestCodeContextLineCount:

    def test_line_count_set(self):
        ctx = CodeContext(
            project=None,
            repo_dir=Path("/tmp"),
            file_path="f.py",
            file_content="line1\nline2\nline3",
            fault_description="",
            graph_context="",
            memory_context="",
            line_count=3,
        )
        assert ctx.line_count == 3

    def test_line_count_default_zero(self):
        ctx = CodeContext(
            project=None,
            repo_dir=Path("/tmp"),
            file_path="f.py",
            file_content="content",
            fault_description="",
            graph_context="",
            memory_context="",
        )
        assert ctx.line_count == 0
