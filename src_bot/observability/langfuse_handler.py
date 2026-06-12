from __future__ import annotations

import os
from typing import Optional


def get_langfuse_callback() -> Optional[object]:
    """Return a Langfuse CallbackHandler (v3) if observability is enabled, else None.

    Langfuse v3 reads credentials from env vars, not constructor arguments.
    We ensure env vars are set from config before constructing the handler.
    """
    try:
        from src_bot.config.config import configs
        if not configs.LANGFUSE_ENABLED:
            return None

        # v3: credentials must be in env vars — set from config if not already present
        if configs.LANGFUSE_PUBLIC_KEY:
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", configs.LANGFUSE_PUBLIC_KEY)
        if configs.LANGFUSE_SECRET_KEY:
            os.environ.setdefault("LANGFUSE_SECRET_KEY", configs.LANGFUSE_SECRET_KEY)
        if configs.LANGFUSE_HOST:
            os.environ.setdefault("LANGFUSE_HOST", configs.LANGFUSE_HOST)

        from langfuse.langchain import CallbackHandler
        return CallbackHandler()

    except ImportError:
        return None
    except Exception:
        return None
