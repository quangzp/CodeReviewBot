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
def langfuse_span(name: str, *, metadata: Optional[dict] = None):
    """Create a Langfuse span around a pipeline phase.

    No-op when LANGFUSE_ENABLED=false or langfuse is not installed.
    Use inside an active langfuse_trace() context.
    """
    try:
        from src_bot.config.config import configs
        if not configs.LANGFUSE_ENABLED:
            yield
            return

        from langfuse import Langfuse

        ctx = get_trace_context()
        if not ctx:
            yield
            return

        langfuse = Langfuse(
            public_key=configs.LANGFUSE_PUBLIC_KEY,
            secret_key=configs.LANGFUSE_SECRET_KEY,
            host=configs.LANGFUSE_HOST,
        )
        trace = langfuse.trace(id=ctx.trace_id)
        span = trace.span(name=name, metadata=metadata or {})
        try:
            yield span
        finally:
            span.end()
    except Exception:
        yield


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
