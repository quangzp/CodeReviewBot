from __future__ import annotations

from typing import Optional

from src_bot.observability.langfuse_ctx import get_trace_context


def get_langfuse_callback() -> Optional[object]:
    """Return a Langfuse CallbackHandler if observability is enabled, else None.

    Reads the current trace context from the contextvar so each pipeline run
    attaches its LLM calls to the correct trace/session/user.

    Returns None (no-op) when LANGFUSE_ENABLED=false (the default).
    """
    try:
        from src_bot.config.config import configs
        if not configs.LANGFUSE_ENABLED:
            return None

        from langfuse.callback import CallbackHandler

        ctx = get_trace_context()
        kwargs: dict = {
            "public_key": configs.LANGFUSE_PUBLIC_KEY,
            "secret_key": configs.LANGFUSE_SECRET_KEY,
            "host": configs.LANGFUSE_HOST,
        }
        if ctx:
            if ctx.trace_id:
                kwargs["trace_id"] = ctx.trace_id
            if ctx.session_id:
                kwargs["session_id"] = ctx.session_id
            if ctx.user_id:
                kwargs["user_id"] = ctx.user_id
            if ctx.tags:
                kwargs["tags"] = ctx.tags
            if ctx.metadata:
                kwargs["metadata"] = ctx.metadata

        return CallbackHandler(**kwargs)

    except ImportError:
        # langfuse not installed — silently skip
        return None
    except Exception:
        # Never let observability break the pipeline
        return None
