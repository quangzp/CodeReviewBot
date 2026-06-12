from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ============================================================================
# Projects — onboarded codebases with pre-built knowledge graphs
# ============================================================================
class ProjectCreateRequest(BaseModel):
    repo_url: str  # e.g. "https://github.com/owner/repo"
    default_branch: Optional[str] = None  # if None, detected from GitHub


class ProjectRecord(BaseModel):
    id: str                            # repo_name with / → _ (e.g. "owner_repo")
    repo_name: str                     # "owner/repo"
    repo_url: str
    default_branch: str = "main"
    status: str = "pending"            # pending | cloning | indexing | indexed | failed
    progress_pct: int = 0              # 0-100
    error: Optional[str] = None

    # Stats (filled in once indexing completes)
    node_count: int = 0
    edge_count: int = 0
    file_count: int = 0
    last_commit_sha: Optional[str] = None
    last_indexed_at: Optional[str] = None

    created_at: str
    updated_at: str


class ProjectSummary(BaseModel):
    """Lightweight version for list views."""
    id: str
    repo_name: str
    repo_url: str
    status: str
    progress_pct: int
    node_count: int
    file_count: int
    last_indexed_at: Optional[str] = None
    created_at: str


# ============================================================================
# Reviews — per-PR analyses that USE a project's existing graph
# ============================================================================
class ReviewRequest(BaseModel):
    pr_url: str  # e.g. "https://github.com/owner/repo/pull/123"


class FileReviewResult(BaseModel):
    file_path: str
    phase1_found: bool = False
    phase2_fault: str = ""
    patch: str = ""
    applies_cleanly: bool = False
    reflexion_attempts: int = 1
    eval_score: Optional[int] = None   # LLM quality score 1-5 (api/agent/evaluator.py)
    eval_reason: str = ""              # Evaluator feedback text
    risk_level: str = "unknown"        # low | medium | high | unknown
    # Structured code review comments from code_review_node (fast_gate LLM)
    review_comments: List[dict] = Field(default_factory=list)
    # Planner contract (Phase B)
    plan_contract: Optional[dict] = None
    # Verification gate results (Phase C)
    ast_passed: Optional[bool] = None
    tests_passed: Optional[bool] = None
    tests_skipped: bool = True


class ReviewRecord(BaseModel):
    id: str
    project_id: Optional[str] = None     # FK → projects.id (the graph used for review)
    pr_url: str
    repo_name: str
    pr_number: int
    status: str                           # pending | running | completed | failed
    created_at: str
    updated_at: str
    file_reviews: List[FileReviewResult] = Field(default_factory=list)
    total_patches: int = 0
    error: Optional[str] = None
    author_login: Optional[str] = None    # captured for memory layer
    meta_review: Optional[dict] = None   # overall PR assessment by meta-review LLM


class ReviewSummary(BaseModel):
    """Lightweight version for list view."""
    id: str
    project_id: Optional[str] = None
    pr_url: str
    repo_name: str
    pr_number: int
    status: str
    created_at: str
    total_patches: int
    total_files: int
    author_login: Optional[str] = None


# ============================================================================
# Chat sessions — ChatGPT-style persistent conversations
# ============================================================================
class ChatSessionSummary(BaseModel):
    """One row in the sidebar list."""
    id: str
    title: str
    user_login: str
    message_count: int = 0
    last_message_preview: str = ""
    created_at: str
    updated_at: str


class ChatMessageRecord(BaseModel):
    """A single message inside a session."""
    id: str
    session_id: str
    role: str               # "user" | "assistant"
    content: str            # text content
    render_data: Optional[dict] = None  # structured tool result for rich UI
    tool_name: Optional[str] = None     # if this message is a tool result
    created_at: str


class ChatSession(BaseModel):
    """Full session with all messages."""
    id: str
    title: str
    user_login: str
    created_at: str
    updated_at: str
    messages: List[ChatMessageRecord] = Field(default_factory=list)


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None  # auto-generated if absent


class RenameSessionRequest(BaseModel):
    title: str
