"""Unit tests for src_bot/risk/scorer.py — no external dependencies needed."""
from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src_bot.risk.scorer import RiskInput, RiskScorer


class TestRiskScorerBasics:

    @pytest.fixture
    def scorer(self):
        return RiskScorer()

    def test_zero_input_gives_low_risk(self, scorer):
        result = scorer.score(RiskInput())
        assert result.level == "low"
        assert result.score == 0.0

    def test_all_max_signals_give_high_risk(self, scorer):
        inp = RiskInput(
            blast_radius_size=9999,
            coverage_gap_ratio=1.0,
            bug_frequency=9999,
            contributor_churn=9999,
            is_new_contributor=True,
            pr_size_lines=9999,
        )
        result = scorer.score(inp)
        assert result.level == "high"
        assert result.score == 1.0

    def test_new_contributor_flag_increases_score(self, scorer):
        base = scorer.score(RiskInput(is_new_contributor=False))
        with_new = scorer.score(RiskInput(is_new_contributor=True))
        assert with_new.score > base.score

    def test_score_capped_at_one(self, scorer):
        inp = RiskInput(
            blast_radius_size=99999,
            coverage_gap_ratio=999.0,
            bug_frequency=99999,
            contributor_churn=99999,
            is_new_contributor=True,
            pr_size_lines=99999,
        )
        result = scorer.score(inp)
        assert result.score <= 1.0

    def test_coverage_gap_capped_at_one(self, scorer):
        r1 = scorer.score(RiskInput(coverage_gap_ratio=1.0))
        r2 = scorer.score(RiskInput(coverage_gap_ratio=999.0))
        assert r1.score == r2.score


class TestRiskScorerBreakdown:

    @pytest.fixture
    def scorer(self):
        return RiskScorer()

    def test_breakdown_contains_all_six_signals(self, scorer):
        result = scorer.score(RiskInput())
        names = [s.name for s in result.breakdown]
        assert "blast_radius_size" in names
        assert "coverage_gap_ratio" in names
        assert "bug_frequency" in names
        assert "contributor_churn" in names
        assert "new_contributor" in names
        assert "pr_size_lines" in names

    def test_contributions_sum_to_score(self, scorer):
        inp = RiskInput(blast_radius_size=5, bug_frequency=3, pr_size_lines=100)
        result = scorer.score(inp)
        total = sum(s.contribution for s in result.breakdown)
        assert abs(round(total, 3) - result.score) < 0.01

    def test_to_dict_has_required_keys(self, scorer):
        d = scorer.score(RiskInput(pr_size_lines=50)).to_dict()
        assert set(d.keys()) >= {"score", "level", "breakdown"}
        assert isinstance(d["breakdown"], list)
        assert len(d["breakdown"]) == 6

    def test_explanation_contains_level_and_score(self, scorer):
        result = scorer.score(RiskInput(blast_radius_size=5))
        text = result.explanation()
        assert result.level.upper() in text
        assert "Risk" in text


class TestRiskScorerThresholds:

    @pytest.fixture
    def scorer(self):
        return RiskScorer()

    def test_level_low_when_score_below_threshold(self, scorer):
        result = scorer.score(RiskInput())
        assert result.level == "low"

    def test_level_high_when_score_above_threshold(self, scorer):
        inp = RiskInput(
            blast_radius_size=20,
            coverage_gap_ratio=1.0,
            bug_frequency=10,
            contributor_churn=10,
            is_new_contributor=True,
            pr_size_lines=500,
        )
        result = scorer.score(inp)
        assert result.level == "high"

    def test_level_is_one_of_three_values(self, scorer):
        for inp in [
            RiskInput(),
            RiskInput(pr_size_lines=250),
            RiskInput(blast_radius_size=9999),
        ]:
            result = scorer.score(inp)
            assert result.level in ("low", "medium", "high")


class TestRiskScorerConfig:

    def test_missing_config_file_uses_builtin_defaults(self, tmp_path):
        scorer = RiskScorer(config_path=tmp_path / "nonexistent.yaml")
        result = scorer.score(RiskInput())
        assert result is not None
        assert result.level == "low"
        assert len(result.breakdown) == 6

    def test_default_config_path_accepted(self):
        scorer = RiskScorer()
        assert scorer.weights is not None
        assert scorer.caps is not None
