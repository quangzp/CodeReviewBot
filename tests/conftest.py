"""Shared test fixtures for the CodeReviewBot test suite."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("APP_NEO4J_URL", "bolt://localhost:7687")
os.environ.setdefault("APP_NEO4J_USER", "neo4j")
os.environ.setdefault("APP_NEO4J_PASSWORD", "test_password")
os.environ.setdefault("GROQ_API_KEY", "test_key_not_real")
os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("LLM_MODEL", "llama-3.3-70b-versatile")


@pytest.fixture
def mock_neo4j_session():
    """A mock Neo4j session that returns empty results by default."""
    class MockResult:
        def __init__(self, data=None):
            self._data = data or []

        def __iter__(self):
            return iter(self._data)

        def single(self):
            return self._data[0] if self._data else None

    class MockSession:
        def __init__(self):
            self._result_data = []

        def set_result(self, data):
            self._result_data = data

        def run(self, query, **kwargs):
            return MockResult(self._result_data)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    return MockSession()


@pytest.fixture
def sample_project():
    """A sample ProjectRecord for testing."""
    from api.models import ProjectRecord
    return ProjectRecord(
        id="test_org_test_repo",
        repo_name="test_org/test_repo",
        repo_url="https://github.com/test_org/test_repo",
        status="indexed",
        node_count=100,
        edge_count=50,
        file_count=20,
        created_at="2025-01-01T00:00:00",
        updated_at="2025-01-01T00:00:00",
    )
