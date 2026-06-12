from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional


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

# Module-level singleton — avoids opening a new HTTP connection pool per phase call.
_langfuse_client: Optional[Any] = None


def _get_langfuse_client() -> Optional[Any]:
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    try:
        from src_bot.config.config import configs
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=configs.LANGFUSE_PUBLIC_KEY,
            secret_key=configs.LANGFUSE_SECRET_KEY,
            host=configs.LANGFUSE_HOST,
        )
        return _langfuse_client
    except Exception:
        return None


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

    Exactly ONE yield — exceptions from setup are swallowed (observability
    must never break the pipeline), exceptions from the body propagate normally.
    """
    span = None
    try:
        from src_bot.config.config import configs
        if configs.LANGFUSE_ENABLED:
            ctx = get_trace_context()
            if ctx:
                client = _get_langfuse_client()
                if client:
                    trace = client.trace(id=ctx.trace_id)
                    span = trace.span(name=name, metadata=metadata or {})
    except Exception:
        pass  # setup failure → no-op span, pipeline continues unaffected

    try:
        yield span
    finally:
        if span is not None:
            try:
                span.end()
            except Exception:
                pass


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
