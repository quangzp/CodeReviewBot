"""Unit tests for the intent classifier."""
import pytest
from unittest.mock import patch, MagicMock

from api.agent.classifier import classify, Intent


def _make_llm_mock(content: str):
    """Return a mock LLM whose .invoke() returns a message with the given content."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = content
    mock_llm.invoke.return_value = mock_response
    return mock_llm


class TestClassifyFunction:

    @patch("api.agent.classifier.get_llm")
    def test_in_scope_add_project(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock('{"intent": "IN_SCOPE"}')
        intent, missing = classify("Add github.com/django/django")
        assert intent == Intent.IN_SCOPE
        assert missing == ""

    @patch("api.agent.classifier.get_llm")
    def test_out_scope(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock('{"intent": "OUT_SCOPE"}')
        intent, missing = classify("What is the weather today?")
        assert intent == Intent.OUT_SCOPE

    @patch("api.agent.classifier.get_llm")
    def test_clarify(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock(
            '{"intent": "CLARIFY", "missing": "which repo?"}'
        )
        intent, missing = classify("Fix the bug")
        assert intent == Intent.CLARIFY
        assert "repo" in missing

    @patch("api.agent.classifier.get_llm")
    def test_fail_open_on_connection_error(self, mock_get_llm):
        mock_get_llm.side_effect = Exception("Connection refused")
        intent, missing = classify("anything")
        assert intent == Intent.IN_SCOPE

    @patch("api.agent.classifier.get_llm")
    def test_fail_open_on_bad_json(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock("not json at all")
        intent, missing = classify("Add a project")
        assert intent == Intent.IN_SCOPE

    @patch("api.agent.classifier.get_llm")
    def test_review_pr_in_scope(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock('{"intent": "IN_SCOPE"}')
        intent, _ = classify("Review PR https://github.com/org/repo/pull/42")
        assert intent == Intent.IN_SCOPE
