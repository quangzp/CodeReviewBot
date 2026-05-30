"""
Schemas for the institutional memory layer.

Each fact in Neo4j has a bi-temporal model:
- valid_from: when the fact became true in reality
- valid_to: when it stopped being true (None = still valid)
- recorded_at: when we learned about it
- invalidated_at: when we marked it false (None = still believed)
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Developer(BaseModel):
    """A code author the bot has seen."""
    login: str                          # GitHub login, primary key
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    first_seen_at: datetime
    last_seen_at: datetime
    pr_count: int = 0


class BugPattern(BaseModel):
    """A recurring bug pattern observed across reviews."""
    id: str                             # canonical-kebab-case e.g. "missing-null-check"
    name: str                           # human-readable e.g. "Missing null check"
    description: str
    severity: str = "medium"            # low | medium | high
    example_diff: Optional[str] = None  # one example to make it concrete
    occurrence_count: int = 0
    first_seen_at: datetime
    last_seen_at: datetime


class ModuleStats(BaseModel):
    """Bug statistics for a module/file."""
    path: str                           # e.g. "src/auth/login.py"
    project_id: str                     # owner_repo
    recent_bug_count: int = 0           # bugs in last 6 months
    total_bug_count: int = 0
    last_bug_at: Optional[datetime] = None


class DevPatternEdge(BaseModel):
    """
    A learned fact: 'developer X tends to make bug pattern Y'.
    This is the core unit of dev profile knowledge.
    """
    developer_login: str
    pattern_id: str
    confidence: float                   # 0.0 - 1.0
    evidence_count: int                 # how many PRs we've seen this in
    first_observed_at: datetime
    last_observed_at: datetime
    invalidated_at: Optional[datetime] = None  # set when pattern stopped


class DeveloperProfile(BaseModel):
    """Full profile for a developer — what the bot has learned about them."""
    developer: Developer
    patterns: list["PatternWithEvidence"] = Field(default_factory=list)
    recent_pr_count_30d: int = 0
    recent_pr_count_90d: int = 0
    favorite_modules: list[str] = Field(default_factory=list)  # paths they touch most


class PatternWithEvidence(BaseModel):
    """A bug pattern attached to a developer, with confidence + evidence."""
    pattern: BugPattern
    confidence: float
    evidence_count: int
    last_observed_at: datetime
    is_active: bool                     # invalidated_at IS NULL
    example_pr_urls: list[str] = Field(default_factory=list)


class ReviewFact(BaseModel):
    """
    A fact extracted from a completed review, to be persisted.
    Output of the LLM extractor.
    """
    developer_login: str
    repo_name: str                      # owner/repo
    pr_number: int
    pr_url: str
    files_touched: list[str]
    bug_patterns: list[str]             # pattern IDs detected
    summary: str                        # one-line summary of what was reviewed
    reviewed_at: datetime
