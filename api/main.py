import sys
import os
import asyncio
import json
import secrets
import logging

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    ReviewRequest,
    ReviewRecord,
    ProjectCreateRequest,
    ProjectRecord,
)
from api.database import (
    init_db,
    create_review,
    get_review,
    list_reviews,
    update_review,
    create_project,
    update_project,
    get_project,
    get_project_by_repo_name,
    list_projects,
    delete_project,
)
from api.reviewer import run_pr_review, parse_pr_url
from api.project_indexer import index_project, reindex_files, parse_repo_url
from api.github_app import (
    verify_webhook_signature,
    get_oauth_authorize_url,
    exchange_oauth_code,
)

# In-memory store for SSE event queues — keyed by review_id OR project_id
_event_queues: dict[str, asyncio.Queue] = {}
_project_event_queues: dict[str, asyncio.Queue] = {}

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
# Simple session store: {session_token: user_dict}
# In production replace with Redis or a proper DB table.
_sessions: dict[str, dict] = {}

# Pending OAuth states to prevent CSRF: {state: True}
_oauth_states: dict[str, bool] = {}


def _is_auth_enabled() -> bool:
    """Auth is enabled only if GITHUB_OAUTH_CLIENT_ID is configured."""
    return bool(os.getenv("GITHUB_OAUTH_CLIENT_ID", ""))


async def get_current_user(request: Request) -> Optional[dict]:
    """
    Dependency: extract the current user from the session cookie.
    Returns None if auth is disabled (dev mode).
    """
    if not _is_auth_enabled():
        return {"login": "dev", "name": "Developer (auth disabled)"}

    session_token = request.cookies.get("session")
    if not session_token:
        return None
    return _sessions.get(session_token)


async def require_user(user: Optional[dict] = Depends(get_current_user)) -> dict:
    """Dependency: raises 401 if not logged in."""
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# ---------------------------------------------------------------------------
# App startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    from src_bot.memory.memory_neo4j import close as memory_close
    await memory_close()


app = FastAPI(title="GraphRAG Code Review Bot", lifespan=lifespan)

# CORS — allow React dev server and same-origin prod
_allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]
if os.getenv("APP_URL"):
    _allowed_origins.append(os.getenv("APP_URL"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check — lightweight, no auth required
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """
    Check liveness of all required services.
    Returns 200 if everything is reachable, 503 with detail if not.
    """
    from src_bot.config.config import configs
    status: dict = {"neo4j": "ok", "weaviate": "ok", "llm_provider": configs.LLM_PROVIDER}

    # Check Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            configs.APP_NEO4J_URL,
            auth=(configs.APP_NEO4J_USER, configs.APP_NEO4J_PASSWORD),
        )
        driver.verify_connectivity()
        driver.close()
    except Exception as e:
        status["neo4j"] = f"error: {e}"

    # Check Weaviate (optional — skip if not configured)
    try:
        import httpx
        r = httpx.get("http://localhost:8080/v1/.well-known/ready", timeout=2.0)
        if r.status_code != 200:
            status["weaviate"] = f"error: HTTP {r.status_code}"
    except Exception as e:
        status["weaviate"] = f"unreachable: {e}"

    failed = [k for k, v in status.items() if isinstance(v, str) and v.startswith("error")]
    if failed:
        from fastapi.responses import JSONResponse as _JSONResponse
        return _JSONResponse(status_code=503, content={"status": "degraded", **status})

    return {"status": "ok", **status}


# ---------------------------------------------------------------------------
# Auth: GitHub OAuth flow
# ---------------------------------------------------------------------------
@app.get("/auth/login")
async def login():
    """Redirect the browser to GitHub OAuth."""
    state = secrets.token_urlsafe(16)
    _oauth_states[state] = True
    url = get_oauth_authorize_url(state)
    return RedirectResponse(url)


@app.get("/auth/callback")
async def oauth_callback(code: str, state: str):
    """GitHub redirects here after the user approves the OAuth app."""
    if state not in _oauth_states:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")
    del _oauth_states[state]

    user = exchange_oauth_code(code)

    session_token = secrets.token_urlsafe(32)
    _sessions[session_token] = {
        "login": user.get("login"),
        "name": user.get("name") or user.get("login"),
        "avatar_url": user.get("avatar_url"),
        "access_token": user.get("access_token"),
    }

    frontend_url = os.getenv("APP_URL", "http://localhost:3000")
    response = RedirectResponse(url=f"{frontend_url}/")
    response.set_cookie(
        "session",
        session_token,
        httponly=True,
        samesite="lax",
        secure=os.getenv("APP_URL", "").startswith("https"),
        max_age=60 * 60 * 24 * 7,  # 7 days
    )
    return response


@app.get("/auth/me")
async def me(user: Optional[dict] = Depends(get_current_user)):
    """Return current user info (or null if not logged in)."""
    if user is None:
        return JSONResponse({"user": None, "auth_enabled": True})
    return JSONResponse({
        "user": {
            "login": user.get("login"),
            "name": user.get("name"),
            "avatar_url": user.get("avatar_url"),
        },
        "auth_enabled": _is_auth_enabled(),
    })


@app.post("/auth/logout")
async def logout(request: Request):
    session_token = request.cookies.get("session")
    if session_token:
        _sessions.pop(session_token, None)
    response = JSONResponse({"status": "logged out"})
    response.delete_cookie("session")
    return response


# ---------------------------------------------------------------------------
# REST: Projects — onboard a repo (clone + index into Neo4j knowledge graph)
# ---------------------------------------------------------------------------
@app.post("/api/projects", response_model=ProjectRecord)
async def create_project_endpoint(
    req: ProjectCreateRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_user),
):
    """Onboard a new project — kicks off background indexing."""
    try:
        repo_name, clean_url = parse_repo_url(req.repo_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check if it already exists
    existing = await get_project_by_repo_name(repo_name)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Project '{repo_name}' already exists (status: {existing.status})"
        )

    record = await create_project(
        repo_name=repo_name,
        repo_url=clean_url,
        default_branch=req.default_branch or "main",
    )

    queue: asyncio.Queue = asyncio.Queue()
    _project_event_queues[record.id] = queue
    background_tasks.add_task(index_project, record, queue)

    return record


@app.get("/api/projects")
async def list_projects_endpoint(user: dict = Depends(require_user)):
    return await list_projects()


@app.get("/api/projects/{project_id}", response_model=ProjectRecord)
async def get_project_endpoint(project_id: str, user: dict = Depends(require_user)):
    record = await get_project(project_id)
    if not record:
        raise HTTPException(status_code=404, detail="Project not found")
    return record


@app.post("/api/projects/{project_id}/reindex", response_model=ProjectRecord)
async def reindex_project_endpoint(
    project_id: str,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_user),
):
    """Trigger a full re-index of an existing project."""
    record = await get_project(project_id)
    if not record:
        raise HTTPException(status_code=404, detail="Project not found")

    record.status = "pending"
    record.progress_pct = 0
    record.error = None
    await update_project(project_id, record)

    queue: asyncio.Queue = asyncio.Queue()
    _project_event_queues[project_id] = queue
    background_tasks.add_task(index_project, record, queue)

    return record


@app.delete("/api/projects/{project_id}")
async def delete_project_endpoint(project_id: str, user: dict = Depends(require_user)):
    record = await get_project(project_id)
    if not record:
        raise HTTPException(status_code=404, detail="Project not found")
    await delete_project(project_id)
    return {"status": "deleted", "id": project_id}


@app.get("/api/projects/{project_id}/stream")
async def stream_project_indexing(project_id: str, user: dict = Depends(require_user)):
    """SSE stream of indexing progress."""
    record = await get_project(project_id)
    if not record:
        raise HTTPException(status_code=404, detail="Project not found")

    if record.status in ("indexed", "failed"):
        async def already_done():
            yield f"data: {json.dumps({'type': 'done', 'status': record.status})}\n\n"
        return StreamingResponse(already_done(), media_type="text/event-stream")

    queue = _project_event_queues.get(project_id)
    if not queue:
        raise HTTPException(status_code=404, detail="No active indexing stream")

    async def event_generator():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
        _project_event_queues.pop(project_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/projects/{project_id}/reviews")
async def list_reviews_for_project(project_id: str, user: dict = Depends(require_user)):
    """List all reviews for a specific project."""
    record = await get_project(project_id)
    if not record:
        raise HTTPException(status_code=404, detail="Project not found")
    return await list_reviews(project_id=project_id)


# ---------------------------------------------------------------------------
# REST: Submit a PR for review
# ---------------------------------------------------------------------------
@app.post("/api/reviews", response_model=ReviewRecord)
async def submit_review(
    req: ReviewRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_user),
):
    try:
        repo_name, pr_number = parse_pr_url(req.pr_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Verify the project has been indexed
    project = await get_project_by_repo_name(repo_name)
    if not project:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Repo '{repo_name}' has not been onboarded yet. "
                f"Add it as a project first (POST /api/projects)."
            ),
        )
    if project.status != "indexed":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Project '{repo_name}' is not ready (status: {project.status}). "
                f"Wait for indexing to complete or trigger a re-index."
            ),
        )

    record = await create_review(req.pr_url, repo_name, pr_number, project_id=project.id)

    queue: asyncio.Queue = asyncio.Queue()
    _event_queues[record.id] = queue

    background_tasks.add_task(run_pr_review, record, queue)
    return record


# ---------------------------------------------------------------------------
# REST: List all reviews
# ---------------------------------------------------------------------------
@app.get("/api/reviews")
async def get_reviews(user: dict = Depends(require_user)):
    return await list_reviews()


# ---------------------------------------------------------------------------
# REST: Get a single review
# ---------------------------------------------------------------------------
@app.get("/api/reviews/{review_id}", response_model=ReviewRecord)
async def get_review_by_id(review_id: str, user: dict = Depends(require_user)):
    record = await get_review(review_id)
    if not record:
        raise HTTPException(status_code=404, detail="Review not found")
    return record


# ---------------------------------------------------------------------------
# SSE: Real-time progress stream
# ---------------------------------------------------------------------------
@app.get("/api/reviews/{review_id}/stream")
async def stream_review(review_id: str, user: dict = Depends(require_user)):
    record = await get_review(review_id)
    if not record:
        raise HTTPException(status_code=404, detail="Review not found")

    if record.status in ("completed", "failed"):
        async def already_done():
            yield f"data: {json.dumps({'type': 'done', 'status': record.status})}\n\n"
        return StreamingResponse(already_done(), media_type="text/event-stream")

    queue = _event_queues.get(review_id)
    if not queue:
        raise HTTPException(status_code=404, detail="Stream not available")

    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
        _event_queues.pop(review_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Memory: developer profiles & learned patterns (read from Neo4j MemoryStore)
# ---------------------------------------------------------------------------

@app.get("/api/developers")
async def list_developers(user: dict = Depends(require_user)):
    """List all developers the bot has seen."""
    from src_bot.memory.memory_neo4j import list_developers_from_memory
    try:
        return await list_developers_from_memory(limit=100)
    except Exception:
        return []


@app.get("/api/developers/{login}")
async def get_developer(login: str, user: dict = Depends(require_user)):
    """Get full developer profile from memory."""
    from src_bot.memory.memory_neo4j import get_developer_profile
    try:
        profile = await get_developer_profile(login)
        if not profile or profile.get("fact_count", 0) == 0:
            raise HTTPException(status_code=404, detail="Developer not found")
        return profile
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="Memory store unavailable")


# ---------------------------------------------------------------------------
# Chat agent — natural-language entry point
# ---------------------------------------------------------------------------
from pydantic import BaseModel as _BaseModel
from typing import List as _List, Optional as _Opt

from api.models import (
    ChatSession,
    ChatSessionSummary,
    CreateSessionRequest,
    RenameSessionRequest,
)
from api.database import (
    create_chat_session,
    list_chat_sessions,
    get_chat_session,
    append_chat_message,
    rename_chat_session,
    delete_chat_session,
)


class ChatMessage(_BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(_BaseModel):
    message: str
    session_id: _Opt[str] = None             # if absent, a new session is created
    history: _List[ChatMessage] = []         # legacy — ignored when session_id is set


# ---------------------------------------------------------------------------
# Sessions CRUD
# ---------------------------------------------------------------------------
@app.post("/api/chat/sessions", response_model=ChatSession)
async def create_session_endpoint(
    req: CreateSessionRequest,
    user: dict = Depends(require_user),
):
    return await create_chat_session(user["login"], req.title)


@app.get("/api/chat/sessions")
async def list_sessions_endpoint(user: dict = Depends(require_user)):
    return await list_chat_sessions(user["login"])


@app.get("/api/chat/sessions/{session_id}", response_model=ChatSession)
async def get_session_endpoint(session_id: str, user: dict = Depends(require_user)):
    s = await get_chat_session(session_id, user["login"])
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s


@app.patch("/api/chat/sessions/{session_id}")
async def rename_session_endpoint(
    session_id: str,
    req: RenameSessionRequest,
    user: dict = Depends(require_user),
):
    ok = await rename_chat_session(session_id, user["login"], req.title)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ok", "title": req.title}


@app.delete("/api/chat/sessions/{session_id}")
async def delete_session_endpoint(session_id: str, user: dict = Depends(require_user)):
    ok = await delete_chat_session(session_id, user["login"])
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Streaming chat — now persists to a session
# ---------------------------------------------------------------------------
def _auto_title_from_message(text: str) -> str:
    """Generate a short title from the first user message (~6 words)."""
    words = text.strip().split()
    title = " ".join(words[:8])
    if len(text) > len(title):
        title += "..."
    return title[:80] or "New chat"


@app.post("/api/chat")
async def chat_endpoint(
    req: ChatRequest,
    user: dict = Depends(require_user),
):
    """
    Streaming chat endpoint. Persists messages to the given session
    (creates one if session_id is absent).

    Emits SSE events: session, thinking, tool_call, tool_result, message, done, error.
    The 'session' event carries the session ID (useful when one is auto-created).
    """
    from api.agent import run_agent_turn

    user_login = user["login"]

    # Resolve / create session
    session = None
    if req.session_id:
        session = await get_chat_session(req.session_id, user_login)
    if not session:
        title = _auto_title_from_message(req.message)
        session = await create_chat_session(user_login, title)
    elif session.title in ("New chat", "") and req.message:
        # Auto-rename first-message session
        new_title = _auto_title_from_message(req.message)
        await rename_chat_session(session.id, user_login, new_title)
        session.title = new_title

    # Build history from persisted messages (last 20, text-only)
    history = []
    for m in session.messages[-20:]:
        history.append({"role": m.role, "content": m.content})

    # Persist the new user message
    await append_chat_message(session.id, "user", req.message)

    async def event_generator():
        # Emit session info first so the client knows which session is active
        yield f"data: {json.dumps({'type': 'session', 'id': session.id, 'title': session.title})}\n\n"

        assistant_text_buf: list[str] = []

        async for event in run_agent_turn(
            user_message=req.message,
            history=history,
            user_login=user_login,
        ):
            # Persist as we go
            etype = event.get("type")
            if etype == "tool_result":
                # Store each tool result as its own message (role=assistant, tool_name set)
                await append_chat_message(
                    session.id,
                    role="assistant",
                    content=event.get("summary", ""),
                    tool_name=event.get("name"),
                    render_data=event.get("render"),
                )
            elif etype == "message":
                assistant_text_buf.append(event.get("text", ""))
            elif etype == "done":
                if assistant_text_buf:
                    await append_chat_message(
                        session.id,
                        role="assistant",
                        content="\n\n".join(assistant_text_buf),
                    )
                    assistant_text_buf.clear()

            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Webhook: GitHub events
#
# Handles two event kinds:
#   1. pull_request (opened/reopened/synchronize) → review the PR
#   2. push (to default branch)                    → incremental re-index
# ---------------------------------------------------------------------------
@app.post("/webhook")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    payload_bytes = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")

    if not verify_webhook_signature(payload_bytes, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload = json.loads(payload_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event_type = request.headers.get("X-GitHub-Event", "")

    # ---------- push event → incremental re-index --------------------
    if event_type == "push":
        # Only re-index pushes to the default branch
        ref = payload.get("ref", "")  # e.g. "refs/heads/main"
        repo_full_name = payload.get("repository", {}).get("full_name", "")
        default_branch = payload.get("repository", {}).get("default_branch", "main")

        if ref != f"refs/heads/{default_branch}":
            return {"status": "ignored", "reason": f"push to non-default branch {ref}"}

        # Find the project
        project = await get_project_by_repo_name(repo_full_name)
        if not project:
            return {"status": "ignored", "reason": f"project {repo_full_name} not onboarded"}
        if project.status != "indexed":
            return {"status": "ignored", "reason": f"project status={project.status}"}

        # Collect changed files from commits (for logging/future per-file optimization)
        changed: set[str] = set()
        for commit in payload.get("commits", []):
            for f in commit.get("added", []):
                changed.add(f)
            for f in commit.get("modified", []):
                changed.add(f)

        # Kick off incremental re-index in background
        background_tasks.add_task(reindex_files, project, sorted(changed))
        return {
            "status": "reindexing",
            "project": repo_full_name,
            "changed_files": len(changed),
        }

    # ---------- pull_request event → review ---------------------------
    if event_type == "pull_request":
        action = payload.get("action", "")
        if action not in ("opened", "reopened", "synchronize"):
            return {"status": "ignored", "action": action}

        pr_url = payload.get("pull_request", {}).get("html_url", "")
        repo_full_name = payload.get("repository", {}).get("full_name", "")
        if not pr_url or not repo_full_name:
            return {"status": "ignored", "reason": "missing pr url or repo"}

        # Verify project is onboarded + indexed
        project = await get_project_by_repo_name(repo_full_name)
        if not project:
            return {
                "status": "ignored",
                "reason": f"project {repo_full_name} not onboarded — add it as a project first",
            }
        if project.status != "indexed":
            return {
                "status": "ignored",
                "reason": f"project status={project.status} — wait for indexing to complete",
            }

        try:
            repo_name, pr_number = parse_pr_url(pr_url)
            record = await create_review(pr_url, repo_name, pr_number, project_id=project.id)
            queue: asyncio.Queue = asyncio.Queue()
            _event_queues[record.id] = queue
            background_tasks.add_task(run_pr_review, record, queue)
            return {"status": "ok", "review_id": record.id}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    return {"status": "ignored", "event": event_type}
