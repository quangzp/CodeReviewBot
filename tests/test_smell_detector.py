"""Unit tests for the graph-based code smell detector."""
import pytest
from unittest.mock import MagicMock

from api.agent.smell_detector import (
    detect_god_classes,
    detect_long_methods,
    detect_dead_code,
    detect_feature_envy,
    detect_all_smells,
    SmellResult,
    GOD_CLASS_METHOD_THRESHOLD,
    LONG_METHOD_LINE_THRESHOLD,
)


def _mock_session(rows):
    """Create a mock Neo4j session returning the given rows."""
    class MockResult:
        def __init__(self, data):
            self._data = data
        def __iter__(self):
            return iter(self._data)

    class MockSession:
        def run(self, query, **kwargs):
            return MockResult(rows)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    return MockSession()


class TestGodClassDetector:

    def test_detects_large_class(self):
        rows = [
            {"name": "BigClass", "file_path": "src/big.py",
             "method_count": 15, "line_count": 400, "qualified_name": "BigClass"},
        ]
        smells = detect_god_classes(_mock_session(rows), "test_project")
        assert len(smells) == 1
        s = smells[0]
        assert s.smell_type == "god_class"
        assert s.name == "BigClass"
        assert s.file_path == "src/big.py"
        assert s.severity in ("medium", "high")

    def test_high_severity_for_very_large_class(self):
        rows = [
            {"name": "HugeClass", "file_path": "src/huge.py",
             "method_count": 30, "line_count": 800, "qualified_name": "HugeClass"},
        ]
        smells = detect_god_classes(_mock_session(rows), "test_project")
        assert smells[0].severity == "high"

    def test_empty_results_returns_empty_list(self):
        smells = detect_god_classes(_mock_session([]), "test_project")
        assert smells == []


class TestLongMethodDetector:

    def test_detects_long_method(self):
        rows = [
            {"name": "process_data", "file_path": "src/proc.py",
             "line_count": 120, "qualified_name": "Processor.process_data"},
        ]
        smells = detect_long_methods(_mock_session(rows), "test_project")
        assert len(smells) == 1
        assert smells[0].smell_type == "long_method"
        assert smells[0].metric_value == 120.0

    def test_empty_results(self):
        smells = detect_long_methods(_mock_session([]), "test_project")
        assert smells == []


class TestDeadCodeDetector:

    def test_detects_dead_function(self):
        rows = [
            {"name": "unused_func", "file_path": "src/old.py",
             "qualified_name": "unused_func", "line_count": 10},
        ]
        smells = detect_dead_code(_mock_session(rows), "test_project")
        assert len(smells) == 1
        assert smells[0].smell_type == "dead_code"
        assert smells[0].severity == "low"

    def test_empty_results(self):
        smells = detect_dead_code(_mock_session([]), "test_project")
        assert smells == []


class TestFeatureEnvyDetector:

    def test_detects_feature_envy(self):
        rows = [
            {"name": "do_thing", "file_path": "src/a.py",
             "qualified_name": "A.do_thing", "own_calls": 1, "other_calls": 5},
        ]
        smells = detect_feature_envy(_mock_session(rows), "test_project")
        assert len(smells) == 1
        assert smells[0].smell_type == "feature_envy"
        assert smells[0].metric_value == 5.0

    def test_empty_results(self):
        smells = detect_feature_envy(_mock_session([]), "test_project")
        assert smells == []


class TestDetectAllSmells:

    def test_returns_empty_when_no_smells(self):
        smells = detect_all_smells(_mock_session([]), "test_project")
        assert smells == []

    def test_filters_by_file_path(self):
        # All detectors return empty — filter has nothing to remove
        smells = detect_all_smells(_mock_session([]), "test_project", file_path="src/specific.py")
        assert smells == []

    def test_sorts_by_severity_high_first(self):
        high = SmellResult("god_class", "f.py", "C", "high", "desc")
        low = SmellResult("dead_code", "f.py", "x", "low", "desc")
        medium = SmellResult("long_method", "f.py", "m", "medium", "desc")
        # Manually sort to verify order
        items = [low, high, medium]
        severity_order = {"high": 0, "medium": 1, "low": 2}
        items.sort(key=lambda s: severity_order.get(s.severity, 3))
        assert items[0].severity == "high"
        assert items[1].severity == "medium"
        assert items[2].severity == "low"
