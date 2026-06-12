"""
Unit tests for src_bot/memory/store.py.

Strategy: mock the Neo4j Driver — no real database needed.
The mock intercepts all session.run() calls so we can assert
query count and keyword presence without executing Cypher.
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_mock_driver():
    """Return (driver, session) pair where session is shared across context manager uses."""
    session = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


def _make_review_fact(patterns=None, files=None):
    from src_bot.memory.schema import ReviewFact
    return ReviewFact(
        developer_login="dev-user",
        repo_name="owner/repo",
        pr_number=42,
        pr_url="https://github.com/owner/repo/pull/42",
        files_touched=files or ["src/main.py", "src/utils.py"],
        bug_patterns=patterns or ["missing-null-check", "unchecked-return"],
        summary="Fixed null handling in main module",
        reviewed_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# write path
# ---------------------------------------------------------------------------

class TestMemoryStoreRecordReview:

    def test_runs_queries_for_developer_review_module_pattern(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        store = MemoryStore(driver)
        store.record_review(_make_review_fact(), review_id="rev-001")
        # At minimum: 1 dev upsert + 1 review create + N module + M pattern queries
        assert session.run.call_count >= 4

    def test_developer_login_appears_in_queries(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        store = MemoryStore(driver)
        store.record_review(_make_review_fact(), review_id="rev-002")
        all_calls_text = " ".join(str(c) for c in session.run.call_args_list)
        assert "dev-user" in all_calls_text

    def test_review_id_appears_in_queries(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        store = MemoryStore(driver)
        store.record_review(_make_review_fact(), review_id="UNIQUE-REVIEW-ID-XYZ")
        all_calls_text = " ".join(str(c) for c in session.run.call_args_list)
        assert "UNIQUE-REVIEW-ID-XYZ" in all_calls_text

    def test_no_patterns_still_creates_developer_and_review(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        store = MemoryStore(driver)
        store.record_review(_make_review_fact(patterns=[]), review_id="rev-003")
        assert session.run.call_count >= 2

    def test_multiple_files_touched_runs_multiple_module_queries(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        store = MemoryStore(driver)
        fact = _make_review_fact(files=["a.py", "b.py", "c.py"], patterns=[])
        store.record_review(fact, review_id="rev-004")
        # 3 separate module merges (one per file)
        call_text = " ".join(str(c) for c in session.run.call_args_list)
        assert "a.py" in call_text
        assert "b.py" in call_text
        assert "c.py" in call_text


class TestMemoryStoreEnsureIndexes:

    def test_creates_constraints_and_indexes(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        store = MemoryStore(driver)
        store.ensure_indexes()
        assert session.run.call_count >= 4  # Developer, BugPattern, Review, Topic, Module

    def test_developer_constraint_query_executed(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        store = MemoryStore(driver)
        store.ensure_indexes()
        call_text = " ".join(str(c) for c in session.run.call_args_list)
        assert "Developer" in call_text


# ---------------------------------------------------------------------------
# read path
# ---------------------------------------------------------------------------

class TestMemoryStoreGetDeveloperProfile:

    def test_returns_none_for_missing_developer(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        mock_result = MagicMock()
        mock_result.single.return_value = None
        session.run.return_value = mock_result
        store = MemoryStore(driver)
        assert store.get_developer_profile("nobody") is None

    def test_returns_none_when_d_key_missing(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        mock_result = MagicMock()
        mock_result.single.return_value = {"d": None, "patterns_data": []}
        session.run.return_value = mock_result
        store = MemoryStore(driver)
        assert store.get_developer_profile("nobody") is None


class TestMemoryStoreGetModuleHotspot:

    def test_returns_default_stats_for_missing_module(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        mock_result = MagicMock()
        mock_result.single.return_value = None
        session.run.return_value = mock_result
        store = MemoryStore(driver)
        stats = store.get_module_hotspot("src/main.py", "owner_repo")
        assert stats.path == "src/main.py"
        assert stats.project_id == "owner_repo"
        assert stats.recent_bug_count == 0
        assert stats.total_bug_count == 0

    def test_returns_default_when_m_key_missing(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        mock_result = MagicMock()
        mock_result.single.return_value = {"m": None, "recent_bugs": 0}
        session.run.return_value = mock_result
        store = MemoryStore(driver)
        stats = store.get_module_hotspot("x.py", "proj")
        assert stats.recent_bug_count == 0


class TestMemoryStoreListDevelopers:

    def test_returns_empty_list_when_no_developers(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        session.run.return_value = iter([])
        store = MemoryStore(driver)
        assert store.list_developers() == []

    def test_passes_limit_to_query(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        session.run.return_value = iter([])
        store = MemoryStore(driver)
        store.list_developers(limit=5)
        call_text = str(session.run.call_args)
        assert "5" in call_text


class TestMemoryStoreTopics:

    def test_record_topic_interest_runs_multiple_queries(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        store = MemoryStore(driver)
        store.record_topic_interest(
            developer_login="dev-user",
            topic_id="module:src/main.py",
            topic_name="src/main.py",
            category="module",
        )
        assert session.run.call_count >= 2

    def test_get_developer_topics_returns_list(self):
        from src_bot.memory.store import MemoryStore
        driver, session = _make_mock_driver()
        session.run.return_value = iter([])
        store = MemoryStore(driver)
        result = store.get_developer_topics("dev-user")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

class TestMemoryStoreConstants:

    def test_pattern_threshold_is_positive(self):
        from src_bot.memory.store import PATTERN_THRESHOLD
        assert PATTERN_THRESHOLD > 0

    def test_confidence_per_evidence_in_range(self):
        from src_bot.memory.store import CONFIDENCE_PER_EVIDENCE
        assert 0.0 < CONFIDENCE_PER_EVIDENCE <= 1.0
