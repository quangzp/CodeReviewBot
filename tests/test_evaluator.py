"""Unit tests for the patch quality evaluator."""
import pytest
from unittest.mock import patch, MagicMock

from api.agent.evaluator import evaluate_patch


def _make_llm_mock(content: str):
    """Return a mock LLM whose .invoke() returns a message with the given content."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = content
    mock_llm.invoke.return_value = mock_response
    return mock_llm


class TestEvaluatePatch:

    def test_empty_patch_returns_1(self):
        score, reason = evaluate_patch("some bug", "")
        assert score == 1
        assert "Empty" in reason

    def test_whitespace_only_patch_returns_1(self):
        score, reason = evaluate_patch("some bug", "   \n  ")
        assert score == 1

    @patch("api.agent.evaluator.get_llm")
    def test_good_patch_score_5(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock('{"score": 5, "reason": "Perfect fix"}')
        score, reason = evaluate_patch(
            "null pointer bug",
            "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-bad\n+good\n"
        )
        assert score == 5
        assert "Perfect" in reason

    @patch("api.agent.evaluator.get_llm")
    def test_refactor_mode_dispatches_correctly(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock('{"score": 4, "reason": "Good refactor"}')
        score, reason = evaluate_patch(
            "extract method",
            "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-x\n+y\n",
            mode="refactor"
        )
        assert score == 4
        assert "Good" in reason

    @patch("api.agent.evaluator.get_llm")
    def test_fail_open_returns_3(self, mock_get_llm):
        mock_get_llm.side_effect = Exception("API down")
        score, reason = evaluate_patch("bug", "some patch content here")
        assert score == 3
        assert "unavailable" in reason.lower() or "fail" in reason.lower()

    @patch("api.agent.evaluator.get_llm")
    def test_score_clamped_to_max_5(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock('{"score": 99, "reason": "overflow"}')
        score, _ = evaluate_patch("bug", "some valid patch")
        assert score == 5

    @patch("api.agent.evaluator.get_llm")
    def test_score_clamped_to_min_1(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock('{"score": -5, "reason": "terrible"}')
        score, _ = evaluate_patch("bug", "some patch")
        assert score == 1
