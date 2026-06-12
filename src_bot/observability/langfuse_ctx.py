from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LangfuseTraceContext:
    trace_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


_current_trace: ContextVar[Optional[LangfuseTraceContext]] = ContextVar(
    "langfuse_trace", default=None
)


def set_trace_context(ctx: LangfuseTraceContext) -> None:
    _current_trace.set(ctx)


def get_trace_context() -> Optional[LangfuseTraceContext]:
    return _current_trace.get()


def clear_trace_context() -> None:
    _current_trace.set(None)


@contextmanager
def langfuse_trace(
    trace_id: str,
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
):
    """Set Langfuse trace context for the duration of a pipeline run.

    Usage:
        with langfuse_trace(trace_id=review.id, tags=["pr_review"]):
            # all get_llm() calls inside here attach to this trace
            ...
    """
    ctx = LangfuseTraceContext(
        trace_id=trace_id,
        session_id=session_id,
        user_id=user_id,
        tags=tags or [],
        metadata=metadata or {},
    )
    token = _current_trace.set(ctx)
    try:
        yield ctx
    finally:
        _current_trace.reset(token)
