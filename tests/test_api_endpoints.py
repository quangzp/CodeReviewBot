"""
API endpoint tests — uses httpx.AsyncClient with ASGITransport (compatible with httpx 0.24+).

Tests verify routing, auth behaviour, and input validation.
Side-effectful operations are mocked so no external services are needed.
"""
import os
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

# Force auth disabled
os.environ.pop("GITHUB_OAUTH_CLIENT_ID", None)


@pytest.fixture(scope="module")
def app():
    """Build the FastAPI app once per module with DB init patched."""
    with patch("api.database.init_db", new_callable=AsyncMock):
        with patch("src_bot.memory.memory_neo4j.close", new_callable=AsyncMock):
            import importlib
            import api.main as main_mod
            importlib.reload(main_mod)
            return main_mod.app


def _run(coro):
    """Run a coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def aclient(app):
    """Yield a sync-callable wrapper around httpx.AsyncClient + ASGITransport."""
    import httpx

    class SyncClient:
        def __init__(self, a):
            self._app = a

        def _req(self, method, url, **kwargs):
            async def _inner():
                transport = httpx.ASGITransport(app=self._app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://testserver"
                ) as c:
                    return await getattr(c, method)(url, **kwargs)
            return _run(_inner())

        def get(self, url, **kw): return self._req("get", url, **kw)
        def post(self, url, **kw): return self._req("post", url, **kw)
        def delete(self, url, **kw): return self._req("delete", url, **kw)
        def patch(self, url, **kw): return self._req("patch", url, **kw)

    return SyncClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_responds(self, aclient):
        """Health endpoint must always respond — even if services are down."""
        r = aclient.get("/health")
        assert r.status_code in (200, 503)
        data = r.json()
        assert "status" in data
        assert "neo4j" in data

    def test_health_status_values(self, aclient):
        r = aclient.get("/health")
        assert r.json()["status"] in ("ok", "degraded")


# ---------------------------------------------------------------------------
# Auth — no GITHUB_OAUTH_CLIENT_ID → auth disabled → "dev" user
# ---------------------------------------------------------------------------
class TestAuth:
    def test_me_returns_dev_user(self, aclient):
        r = aclient.get("/auth/me")
        assert r.status_code == 200
        data = r.json()
        assert data["auth_enabled"] is False
        assert data["user"]["login"] == "dev"

    def test_logout_200(self, aclient):
        r = aclient.post("/auth/logout")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------
class TestProjects:
    def test_list_empty(self, aclient):
        with patch("api.main.list_projects", new_callable=AsyncMock, return_value=[]):
            r = aclient.get("/api/projects")
        assert r.status_code == 200
        assert r.json() == []

    def test_create_bad_url(self, aclient):
        r = aclient.post("/api/projects", json={"repo_url": "not-a-github-url"})
        assert r.status_code == 400

    def test_create_duplicate(self, aclient):
        mock_existing = MagicMock()
        mock_existing.status = "indexed"
        with patch("api.main.get_project_by_repo_name", new_callable=AsyncMock, return_value=mock_existing):
            r = aclient.post("/api/projects", json={"repo_url": "https://github.com/owner/repo"})
        assert r.status_code == 409

    def test_get_not_found(self, aclient):
        with patch("api.main.get_project", new_callable=AsyncMock, return_value=None):
            r = aclient.get("/api/projects/nonexistent_id")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------
class TestReviews:
    def test_submit_bad_url(self, aclient):
        r = aclient.post("/api/reviews", json={"pr_url": "not-a-pr-url"})
        assert r.status_code == 400

    def test_submit_project_not_onboarded(self, aclient):
        with patch("api.main.get_project_by_repo_name", new_callable=AsyncMock, return_value=None):
            r = aclient.post(
                "/api/reviews",
                json={"pr_url": "https://github.com/owner/repo/pull/1"},
            )
        assert r.status_code == 400
        assert "not been onboarded" in r.json()["detail"]

    def test_submit_project_not_indexed(self, aclient):
        mock_project = MagicMock()
        mock_project.status = "indexing"
        with patch("api.main.get_project_by_repo_name", new_callable=AsyncMock, return_value=mock_project):
            r = aclient.post(
                "/api/reviews",
                json={"pr_url": "https://github.com/owner/repo/pull/1"},
            )
        assert r.status_code == 400
        assert "not ready" in r.json()["detail"]

    def test_get_not_found(self, aclient):
        with patch("api.main.get_review", new_callable=AsyncMock, return_value=None):
            r = aclient.get("/api/reviews/nonexistent")
        assert r.status_code == 404

    def test_list_empty(self, aclient):
        with patch("api.main.list_reviews", new_callable=AsyncMock, return_value=[]):
            r = aclient.get("/api/reviews")
        assert r.status_code == 200
        assert r.json() == []


# ---------------------------------------------------------------------------
# Chat sessions
# ---------------------------------------------------------------------------
class TestChatSessions:
    def test_list_empty(self, aclient):
        with patch("api.main.list_chat_sessions", new_callable=AsyncMock, return_value=[]):
            r = aclient.get("/api/chat/sessions")
        assert r.status_code == 200

    def test_get_not_found(self, aclient):
        with patch("api.main.get_chat_session", new_callable=AsyncMock, return_value=None):
            r = aclient.get("/api/chat/sessions/nonexistent")
        assert r.status_code == 404

    def test_delete_not_found(self, aclient):
        with patch("api.main.delete_chat_session", new_callable=AsyncMock, return_value=False):
            r = aclient.delete("/api/chat/sessions/nonexistent")
        assert r.status_code == 404
