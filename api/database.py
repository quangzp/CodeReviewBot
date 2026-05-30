import json
import uuid
import aiosqlite
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from api.models import (
    ReviewRecord,
    ReviewSummary,
    ProjectRecord,
    ProjectSummary,
    ChatSession,
    ChatSessionSummary,
    ChatMessageRecord,
)

DB_PATH = Path(__file__).parent.parent / "data" / "reviews.db"


# ============================================================================
# Schema initialization
# ============================================================================
async def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                repo_name TEXT NOT NULL UNIQUE,
                repo_url TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                result_json TEXT NOT NULL DEFAULT '{}'
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                pr_url TEXT NOT NULL,
                repo_name TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                result_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
            )
        """)
        # Migration: add project_id column to old reviews table if missing
        async with db.execute("PRAGMA table_info(reviews)") as cursor:
            cols = [row[1] async for row in cursor]
        if "project_id" not in cols:
            await db.execute("ALTER TABLE reviews ADD COLUMN project_id TEXT")

        await db.execute("CREATE INDEX IF NOT EXISTS idx_reviews_project ON reviews(project_id)")

        # Chat sessions (ChatGPT-style persistent conversations)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                user_login TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_name TEXT,
                render_data TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON chat_sessions(user_login, updated_at DESC)")

        await db.commit()


# ============================================================================
# Project CRUD
# ============================================================================
def _make_project_id(repo_name: str) -> str:
    """e.g. 'facebook/react' → 'facebook_react'"""
    return repo_name.replace("/", "_")


async def create_project(repo_name: str, repo_url: str, default_branch: str = "main") -> ProjectRecord:
    project_id = _make_project_id(repo_name)
    now = datetime.now(timezone.utc).isoformat()
    record = ProjectRecord(
        id=project_id,
        repo_name=repo_name,
        repo_url=repo_url,
        default_branch=default_branch,
        status="pending",
        progress_pct=0,
        created_at=now,
        updated_at=now,
    )
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO projects (id, repo_name, repo_url, status, created_at, updated_at, result_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (project_id, repo_name, repo_url, "pending", now, now, record.model_dump_json()),
        )
        await db.commit()
    return record


async def update_project(project_id: str, record: ProjectRecord):
    record.updated_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE projects SET status=?, updated_at=?, result_json=? WHERE id=?",
            (record.status, record.updated_at, record.model_dump_json(), project_id),
        )
        await db.commit()


async def get_project(project_id: str) -> Optional[ProjectRecord]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT result_json FROM projects WHERE id=?", (project_id,)) as cursor:
            row = await cursor.fetchone()
    if row:
        return ProjectRecord.model_validate_json(row[0])
    return None


async def get_project_by_repo_name(repo_name: str) -> Optional[ProjectRecord]:
    project_id = _make_project_id(repo_name)
    return await get_project(project_id)


async def list_projects() -> List[ProjectSummary]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT result_json FROM projects ORDER BY updated_at DESC LIMIT 100"
        ) as cursor:
            rows = await cursor.fetchall()
    result = []
    for row in rows:
        p = ProjectRecord.model_validate_json(row[0])
        result.append(ProjectSummary(
            id=p.id,
            repo_name=p.repo_name,
            repo_url=p.repo_url,
            status=p.status,
            progress_pct=p.progress_pct,
            node_count=p.node_count,
            file_count=p.file_count,
            last_indexed_at=p.last_indexed_at,
            created_at=p.created_at,
        ))
    return result


async def delete_project(project_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM projects WHERE id=?", (project_id,))
        # Reviews keep their data but lose the project link (ON DELETE SET NULL handles this)
        await db.commit()


# ============================================================================
# Review CRUD
# ============================================================================
async def create_review(
    pr_url: str,
    repo_name: str,
    pr_number: int,
    project_id: Optional[str] = None,
) -> ReviewRecord:
    review_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    record = ReviewRecord(
        id=review_id,
        project_id=project_id,
        pr_url=pr_url,
        repo_name=repo_name,
        pr_number=pr_number,
        status="pending",
        created_at=now,
        updated_at=now,
    )
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO reviews (id, project_id, pr_url, repo_name, pr_number, status, created_at, updated_at, result_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (review_id, project_id, pr_url, repo_name, pr_number, "pending", now, now, record.model_dump_json()),
        )
        await db.commit()
    return record


async def update_review(review_id: str, record: ReviewRecord):
    record.updated_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE reviews SET status=?, updated_at=?, result_json=? WHERE id=?",
            (record.status, record.updated_at, record.model_dump_json(), review_id),
        )
        await db.commit()


async def get_review(review_id: str) -> Optional[ReviewRecord]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT result_json FROM reviews WHERE id=?", (review_id,)) as cursor:
            row = await cursor.fetchone()
    if row:
        return ReviewRecord.model_validate_json(row[0])
    return None


async def list_reviews(project_id: Optional[str] = None) -> List[ReviewSummary]:
    """List reviews, optionally filtered to a single project."""
    async with aiosqlite.connect(DB_PATH) as db:
        if project_id:
            query = "SELECT result_json FROM reviews WHERE project_id=? ORDER BY created_at DESC LIMIT 100"
            params = (project_id,)
        else:
            query = "SELECT result_json FROM reviews ORDER BY created_at DESC LIMIT 100"
            params = ()
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

    result = []
    for row in rows:
        r = ReviewRecord.model_validate_json(row[0])
        result.append(ReviewSummary(
            id=r.id,
            project_id=r.project_id,
            pr_url=r.pr_url,
            repo_name=r.repo_name,
            pr_number=r.pr_number,
            status=r.status,
            created_at=r.created_at,
            total_patches=r.total_patches,
            total_files=len(r.file_reviews),
            author_login=r.author_login,
        ))
    return result


# ============================================================================
# Chat sessions CRUD
# ============================================================================
async def create_chat_session(user_login: str, title: Optional[str] = None) -> ChatSession:
    session_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()
    title = title or "New chat"
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO chat_sessions (id, title, user_login, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, title, user_login, now, now),
        )
        await db.commit()
    return ChatSession(
        id=session_id, title=title, user_login=user_login,
        created_at=now, updated_at=now, messages=[],
    )


async def list_chat_sessions(user_login: str, limit: int = 100) -> List[ChatSessionSummary]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """
            SELECT s.id, s.title, s.user_login, s.created_at, s.updated_at,
                   (SELECT COUNT(*) FROM chat_messages WHERE session_id = s.id) AS msg_count,
                   (SELECT content FROM chat_messages WHERE session_id = s.id ORDER BY created_at DESC LIMIT 1) AS last_msg
            FROM chat_sessions s
            WHERE s.user_login = ?
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            (user_login, limit),
        ) as cursor:
            rows = await cursor.fetchall()

    result = []
    for row in rows:
        result.append(ChatSessionSummary(
            id=row[0], title=row[1], user_login=row[2],
            created_at=row[3], updated_at=row[4],
            message_count=row[5] or 0,
            last_message_preview=(row[6] or "")[:120],
        ))
    return result


async def get_chat_session(session_id: str, user_login: str) -> Optional[ChatSession]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, title, user_login, created_at, updated_at FROM chat_sessions WHERE id = ? AND user_login = ?",
            (session_id, user_login),
        ) as cursor:
            srow = await cursor.fetchone()
        if not srow:
            return None
        async with db.execute(
            "SELECT id, session_id, role, content, tool_name, render_data, created_at "
            "FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ) as cursor:
            mrows = await cursor.fetchall()

    messages = []
    for m in mrows:
        render_data = None
        if m[5]:
            try:
                render_data = json.loads(m[5])
            except Exception:
                pass
        messages.append(ChatMessageRecord(
            id=m[0], session_id=m[1], role=m[2], content=m[3],
            tool_name=m[4], render_data=render_data, created_at=m[6],
        ))

    return ChatSession(
        id=srow[0], title=srow[1], user_login=srow[2],
        created_at=srow[3], updated_at=srow[4], messages=messages,
    )


async def append_chat_message(
    session_id: str,
    role: str,
    content: str,
    tool_name: Optional[str] = None,
    render_data: Optional[dict] = None,
) -> ChatMessageRecord:
    msg_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()
    render_json = json.dumps(render_data) if render_data is not None else None
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO chat_messages (id, session_id, role, content, tool_name, render_data, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (msg_id, session_id, role, content, tool_name, render_json, now),
        )
        # Bump the session's updated_at
        await db.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        await db.commit()
    return ChatMessageRecord(
        id=msg_id, session_id=session_id, role=role, content=content,
        tool_name=tool_name, render_data=render_data, created_at=now,
    )


async def rename_chat_session(session_id: str, user_login: str, new_title: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "UPDATE chat_sessions SET title = ? WHERE id = ? AND user_login = ?",
            (new_title[:200], session_id, user_login),
        )
        await db.commit()
        return cursor.rowcount > 0


async def delete_chat_session(session_id: str, user_login: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(
            "DELETE FROM chat_sessions WHERE id = ? AND user_login = ?",
            (session_id, user_login),
        )
        # Manually cascade in case PRAGMA is off
        await db.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        await db.commit()
        return cursor.rowcount > 0
