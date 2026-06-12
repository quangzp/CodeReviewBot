"""
Integration tests for api/reviewer.py.

Strategy:
  - Mock GitHub API (PyGithub) — no real network calls
  - Mock Neo4j driver — no real database
  - Mock LLM — deterministic responses
  - Use real temp git repo for _checkout_pr_base / ast_gate
  - Test the full run_pr_review() async flow end-to-end
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("APP_NEO4J_URL", "bolt://localhost:7687")
os.environ.setdefault("APP_NEO4J_USER", "neo4j")
os.environ.setdefault("APP_NEO4J_PASSWORD", "test_password")
os.environ.setdefault("GROQ_API_KEY", "test_key_not_real")
os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("EXECUTION_GATE_ENABLED", "false")
os.environ.setdefault("LANGFUSE_ENABLED", "false")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_git_repo(base_dir: Path, filename: str = "utils.py", content: str = "def add(a, b):\n    return a + b\n") -> Path:
    """Create a minimal git repo with one Python file and return its path."""
    repo = base_dir / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)
    (repo / filename).write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)
    return repo


def _get_head_sha(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True, text=True
    )
    return result.stdout.strip()


def _make_mock_github_file(filename: str = "utils.py", changes: int = 5) -> MagicMock:
    f = MagicMock()
    f.filename = filename
    f.status = "modified"
    f.changes = changes
    f.patch = "@@ -1,1 +1,1 @@\n-def add(a, b):\n+def add(a, b):  # fixed\n"
    return f


def _make_mock_pr(repo_path: Path, filename: str = "utils.py") -> MagicMock:
    sha = _get_head_sha(repo_path)
    pr = MagicMock()
    pr.user = MagicMock(login="dev-user")
    pr.title = "Fix add function"
    pr.body = "Fixes a bug in the add function"
    pr.base = MagicMock(sha=sha)
    pr.number = 42
    pr.get_files.return_value = [_make_mock_github_file(filename)]
    return pr


# ---------------------------------------------------------------------------
# parse_pr_url
# ---------------------------------------------------------------------------

class TestParsePrUrl:

    def test_valid_url(self):
        from api.reviewer import parse_pr_url
        repo, num = parse_pr_url("https://github.com/owner/repo/pull/123")
        assert repo == "owner/repo"
        assert num == 123

    def test_trailing_slash(self):
        from api.reviewer import parse_pr_url
        repo, num = parse_pr_url("https://github.com/owner/repo/pull/42 ")
        assert repo == "owner/repo"
        assert num == 42

    def test_invalid_url_raises(self):
        from api.reviewer import parse_pr_url
        with pytest.raises(ValueError):
            parse_pr_url("not-a-github-url")

    def test_non_pr_url_raises(self):
        from api.reviewer import parse_pr_url
        with pytest.raises(ValueError):
            parse_pr_url("https://github.com/owner/repo/issues/5")


# ---------------------------------------------------------------------------
# run_pr_review — end-to-end with mocks
# ---------------------------------------------------------------------------

class TestRunPrReview:

    @pytest.fixture
    def git_repo(self, tmp_path):
        return _make_git_repo(tmp_path)

    def _make_review_record(self, repo_name: str = "owner/repo", pr_url: str = "https://github.com/owner/repo/pull/42"):
        from api.models import ReviewRecord
        return ReviewRecord(
            id="test-review-001",
            project_id="owner_repo",
            repo_name=repo_name,
            pr_url=pr_url,
            pr_number=42,
            status="pending",
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T00:00:00",
        )

    def _make_mock_project(self, repo_dir: Path, repo_name: str = "owner/repo"):
        from api.models import ProjectRecord
        p = ProjectRecord(
            id="owner_repo",
            repo_name=repo_name,
            repo_url=f"https://github.com/{repo_name}",
            status="indexed",
            node_count=10,
            file_count=1,
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T00:00:00",
        )
        return p

    @pytest.mark.asyncio
    async def test_review_completes_with_no_python_files(self, git_repo):
        """If no Python files in PR, review completes with total_patches=0."""
        record = self._make_review_record()
        project = self._make_mock_project(git_repo)
        event_queue = asyncio.Queue()
        pr_mock = MagicMock()
        pr_mock.user = MagicMock(login="dev-user")
        pr_mock.title = "JS only PR"
        pr_mock.body = ""
        pr_mock.base = MagicMock(sha=_get_head_sha(git_repo))
        # No Python files
        pr_mock.get_files.return_value = [
            MagicMock(filename="app.js", status="modified"),
        ]

        with (
            patch("api.reviewer.get_project", AsyncMock(return_value=project)),
            patch("api.reviewer.update_review", AsyncMock()),
            patch("api.reviewer.get_project_dir", return_value=git_repo),
            patch("api.github_app.get_token_for_repo", return_value=None),
            patch("github.Github") as mock_gh_cls,
        ):
            mock_gh_cls.return_value.get_repo.return_value.get_pull.return_value = pr_mock

            from api.reviewer import run_pr_review
            await run_pr_review(record, event_queue)

        events = []
        while not event_queue.empty():
            item = event_queue.get_nowait()
            if item is not None:
                events.append(item)

        types = [e.get("type") for e in events]
        assert "done" in types
        done_event = next(e for e in events if e.get("type") == "done")
        assert done_event["total_patches"] == 0

    @pytest.mark.asyncio
    async def test_review_fails_gracefully_when_project_not_indexed(self, git_repo):
        """If project status != indexed, review fails with error event."""
        from api.models import ProjectRecord
        record = self._make_review_record()
        event_queue = asyncio.Queue()

        unindexed = ProjectRecord(
            id="owner_repo",
            repo_name="owner/repo",
            repo_url="https://github.com/owner/repo",
            status="pending",  # not indexed
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T00:00:00",
        )

        with (
            patch("api.reviewer.get_project", AsyncMock(return_value=unindexed)),
            patch("api.reviewer.update_review", AsyncMock()),
        ):
            from api.reviewer import run_pr_review
            await run_pr_review(record, event_queue)

        events = []
        while not event_queue.empty():
            item = event_queue.get_nowait()
            if item is not None:
                events.append(item)

        types = [e.get("type") for e in events]
        assert "error" in types

    @pytest.mark.asyncio
    async def test_review_fails_gracefully_when_project_missing_on_disk(self, tmp_path):
        """If project clone missing on disk, review fails with error event."""
        record = self._make_review_record()
        event_queue = asyncio.Queue()
        project = self._make_mock_project(tmp_path)

        missing_dir = tmp_path / "nonexistent"

        with (
            patch("api.reviewer.get_project", AsyncMock(return_value=project)),
            patch("api.reviewer.update_review", AsyncMock()),
            patch("api.reviewer.get_project_dir", return_value=missing_dir),
        ):
            from api.reviewer import run_pr_review
            await run_pr_review(record, event_queue)

        events = []
        while not event_queue.empty():
            item = event_queue.get_nowait()
            if item is not None:
                events.append(item)

        types = [e.get("type") for e in events]
        assert "error" in types

    @pytest.mark.asyncio
    async def test_review_full_pipeline_with_mocked_llm(self, git_repo):
        """Full pipeline: Python file changed, LLM mocked, patch generated."""
        record = self._make_review_record()
        project = self._make_mock_project(git_repo)
        event_queue = asyncio.Queue()
        pr_mock = _make_mock_pr(git_repo)

        mock_llm_response = MagicMock()
        mock_llm_response.content = (
            "```diff\n"
            "--- a/utils.py\n+++ b/utils.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-def add(a, b):\n+def add(a, b):  # fixed\n"
            "    return a + b\n```"
        )

        with (
            patch("api.reviewer.get_project", AsyncMock(return_value=project)),
            patch("api.reviewer.update_review", AsyncMock()),
            patch("api.reviewer.get_project_dir", return_value=git_repo),
            patch("api.github_app.get_token_for_repo", return_value=None),
            patch("github.Github") as mock_gh_cls,
            patch("api.reviewer.get_llm") as mock_get_llm,
            patch("neo4j.GraphDatabase.driver") as mock_driver,
            patch("run_swebench_v2.phase1_localize_file", return_value="utils.py"),
            patch("run_swebench_v2.phase2_localize_fault", return_value=("off-by-one in add()", "def add(a, b):\n    return a + b\n")),
            patch("run_swebench_v2.phase3_generate_patch", return_value="--- a/utils.py\n+++ b/utils.py\n@@ -1,2 +1,2 @@\n-def add(a, b):\n+def add(a, b):  # fixed\n return a + b\n"),
            patch("run_swebench_v2._verify_patch_applies", return_value=(False, "patch mismatch")),
            patch("api.agent.planner.generate_plan_sync") as mock_planner,
            patch("src_bot.memory.memory_neo4j.build_memory_context", return_value=""),
            patch("src_bot.memory.memory_neo4j.record_review", new_callable=AsyncMock),
            patch("src_bot.memory.extractor.extract_facts_from_review") as mock_extract,
        ):
            mock_gh_cls.return_value.get_repo.return_value.get_pull.return_value = pr_mock
            mock_get_llm.return_value = MagicMock(invoke=MagicMock(return_value=mock_llm_response))
            mock_driver.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_driver.return_value.session.return_value.__enter__ = MagicMock(return_value=MagicMock(run=MagicMock(return_value=[])))
            mock_driver.return_value.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_driver.return_value.close = MagicMock()

            from unittest.mock import MagicMock as MM
            mock_contract = MM()
            mock_contract.to_prompt_section.return_value = "## Plan\n..."
            mock_planner.return_value = mock_contract

            mock_extract.return_value = MagicMock(summary="Fixed add()", bug_patterns=[])

            from api.reviewer import run_pr_review
            await run_pr_review(record, event_queue)

        events = []
        while not event_queue.empty():
            item = event_queue.get_nowait()
            if item is not None:
                events.append(item)

        types = [e.get("type") for e in events]
        error_events = [e for e in events if e.get("type") == "error"]
        # Should always emit done (even if patch didn't apply)
        assert "done" in types, f"Expected 'done' event, got types={types}, errors={error_events}"
        assert "error" not in types
