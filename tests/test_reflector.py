"""Unit tests for src_bot/reflexion/reflector.py — mocks the LLM."""
from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_llm(response_text: str = "Reflection: the patch was wrong because X.") -> MagicMock:
    llm = MagicMock()
    resp = MagicMock()
    resp.content = response_text
    llm.invoke.return_value = resp
    return llm


class TestGenerateReflectionReturn:

    def test_returns_string(self):
        from src_bot.reflexion.reflector import generate_reflection
        result = generate_reflection(
            llm=_make_llm("Fixed reflection text."),
            bug_description="Off-by-one error",
            failed_patch="--- a/foo.py\n+++ b/foo.py\n-    if x = 0:",
            test_output="AssertionError: expected True",
            graph_context="foo() called by bar()",
        )
        assert isinstance(result, str)

    def test_returns_llm_content(self):
        from src_bot.reflexion.reflector import generate_reflection
        llm = _make_llm("Specific reflection output.")
        result = generate_reflection(
            llm=llm,
            bug_description="bug",
            failed_patch="patch",
            test_output="error",
            graph_context="ctx",
        )
        assert result == "Specific reflection output."

    def test_calls_llm_once(self):
        from src_bot.reflexion.reflector import generate_reflection
        llm = _make_llm()
        generate_reflection(
            llm=llm,
            bug_description="bug",
            failed_patch="patch",
            test_output="error",
            graph_context="ctx",
        )
        assert llm.invoke.call_count == 1


class TestGenerateReflectionPromptContent:

    def test_bug_description_in_prompt(self):
        from src_bot.reflexion.reflector import generate_reflection
        llm = _make_llm()
        generate_reflection(
            llm=llm,
            bug_description="UNIQUE_BUG_MARKER_XYZ",
            failed_patch="patch",
            test_output="error",
            graph_context="ctx",
        )
        prompt = llm.invoke.call_args[0][0]
        assert "UNIQUE_BUG_MARKER_XYZ" in prompt

    def test_test_output_in_prompt(self):
        from src_bot.reflexion.reflector import generate_reflection
        llm = _make_llm()
        generate_reflection(
            llm=llm,
            bug_description="bug",
            failed_patch="patch",
            test_output="UNIQUE_ERROR_MARKER_ABC",
            graph_context="ctx",
        )
        prompt = llm.invoke.call_args[0][0]
        assert "UNIQUE_ERROR_MARKER_ABC" in prompt

    def test_previous_reflections_included_in_prompt(self):
        from src_bot.reflexion.reflector import generate_reflection
        llm = _make_llm()
        generate_reflection(
            llm=llm,
            bug_description="bug",
            failed_patch="patch",
            test_output="error",
            graph_context="ctx",
            previous_reflections=["PREV_REFLECTION_MARKER_999"],
        )
        prompt = llm.invoke.call_args[0][0]
        assert "PREV_REFLECTION_MARKER_999" in prompt

    def test_no_previous_reflections_shows_first_attempt(self):
        from src_bot.reflexion.reflector import generate_reflection
        llm = _make_llm()
        generate_reflection(
            llm=llm,
            bug_description="bug",
            failed_patch="patch",
            test_output="error",
            graph_context="ctx",
            previous_reflections=None,
        )
        prompt = llm.invoke.call_args[0][0]
        assert "first attempt" in prompt.lower()

    def test_empty_graph_context_uses_fallback(self):
        from src_bot.reflexion.reflector import generate_reflection
        llm = _make_llm()
        generate_reflection(
            llm=llm,
            bug_description="bug",
            failed_patch="patch",
            test_output="error",
            graph_context="",
        )
        prompt = llm.invoke.call_args[0][0]
        assert "No graph context available" in prompt

    def test_test_output_truncated_at_2000_chars(self):
        from src_bot.reflexion.reflector import generate_reflection
        llm = _make_llm()
        long_output = "E" * 5000
        generate_reflection(
            llm=llm,
            bug_description="bug",
            failed_patch="patch",
            test_output=long_output,
            graph_context="ctx",
        )
        prompt = llm.invoke.call_args[0][0]
        # The full 5000 char string should not appear verbatim
        assert "E" * 2001 not in prompt

    def test_multiple_previous_reflections_all_included(self):
        from src_bot.reflexion.reflector import generate_reflection
        llm = _make_llm()
        refs = ["REFL_A", "REFL_B", "REFL_C"]
        generate_reflection(
            llm=llm,
            bug_description="bug",
            failed_patch="patch",
            test_output="error",
            graph_context="ctx",
            previous_reflections=refs,
        )
        prompt = llm.invoke.call_args[0][0]
        for ref in refs:
            assert ref in prompt
