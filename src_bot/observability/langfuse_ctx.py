from __future__ import annotations

import os
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

# Module-level singleton — one HTTP connection pool for the process lifetime
_langfuse_client: Optional[Any] = None


def _ensure_env(configs) -> None:
    """Set Langfuse env vars from config so v3 SDK can read them."""
    if configs.LANGFUSE_PUBLIC_KEY:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", configs.LANGFUSE_PUBLIC_KEY)
    if configs.LANGFUSE_SECRET_KEY:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", configs.LANGFUSE_SECRET_KEY)
    if configs.LANGFUSE_HOST:
        os.environ.setdefault("LANGFUSE_HOST", configs.LANGFUSE_HOST)


def _get_langfuse_client() -> Optional[Any]:
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    try:
        from src_bot.config.config import configs
        if not configs.LANGFUSE_ENABLED:
            return None
        _ensure_env(configs)
        from langfuse import Langfuse
        _langfuse_client = Langfuse()
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
def langfuse_trace(
    trace_id: str,
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
):
    """Open a root Langfuse span (= trace) for the duration of a pipeline run.

    Langfuse v3 creates a trace automatically when the first span is started.
    update_current_trace() sets user/session/tag metadata on that trace.

    Usage:
        with langfuse_trace(trace_id=review.id, user_id=author, tags=["pr_review"]):
            with langfuse_span("phase1"):
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

    # Try to open a Langfuse root span; fall back to no-op if unavailable
    span_cm = None
    entered = False
    try:
        from src_bot.config.config import configs
        if configs.LANGFUSE_ENABLED:
            client = _get_langfuse_client()
            if client:
                span_cm = client.start_as_current_span(
                    name=f"pipeline:{trace_id[:8]}", as_type="span"
                )
                span_cm.__enter__()
                entered = True
                try:
                    client.update_current_trace(
                        user_id=user_id,
                        session_id=session_id,
                        tags=tags,
                        metadata=metadata,
                    )
                except Exception:
                    pass
    except Exception:
        pass

    try:
        yield ctx
    finally:
        if entered and span_cm is not None:
            try:
                span_cm.__exit__(None, None, None)
            except Exception:
                pass
        _current_trace.reset(token)


@contextmanager
def langfuse_span(name: str, *, metadata: Optional[dict] = None):
    """Create a child Langfuse span around a pipeline phase (v3 API).

    No-op when LANGFUSE_ENABLED=false or langfuse is not installed.
    Must be called inside an active langfuse_trace() context.
    Exceptions from the body propagate normally; setup failures are swallowed.
    """
    span_cm = None
    entered = False
    try:
        from src_bot.config.config import configs
        if configs.LANGFUSE_ENABLED:
            client = _get_langfuse_client()
            if client:
                span_cm = client.start_as_current_span(name=name, as_type="span")
                span_cm.__enter__()
                entered = True
                if metadata:
                    try:
                        client.update_current_span(metadata=metadata)
                    except Exception:
                        pass
    except Exception:
        pass

    try:
        yield span_cm
    finally:
        if entered and span_cm is not None:
            try:
                span_cm.__exit__(None, None, None)
            except Exception:
                pass
