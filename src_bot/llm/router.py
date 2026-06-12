from __future__ import annotations

"""
Swappable LLM Router — supports multiple backends for development and paper experiments.

Usage:
    from src_bot.llm.router import get_llm

    # Returns the configured LLM based on LLM_PROVIDER env var
    llm = get_llm()

Supported providers (set via LLM_PROVIDER env var):
    - "ollama"    : Local model via Ollama (free, default for development)
    - "groq"      : Groq API (free tier, fast inference, llama-3.3-70b)
    - "openai"    : OpenAI API (GPT-4o, for paper experiments)
    - "together"  : Together.ai API (gpt-oss-20b, open-source models)
    - "vllm"      : Self-hosted vLLM server (free, OpenAI-compatible API)
    - "vllm"      : Self-hosted vLLM server (OpenAI-compatible API)

Examples:
    LLM_PROVIDER=ollama   LLM_MODEL=deepseek-coder:6.7b-instruct
    LLM_PROVIDER=groq     LLM_MODEL=llama-3.3-70b-versatile
    LLM_PROVIDER=openai   LLM_MODEL=gpt-4o
    LLM_PROVIDER=together LLM_MODEL=openai/gpt-oss-20b
    LLM_PROVIDER=vllm     LLM_MODEL=R2E-Gym/R2EGym-32B  VLLM_API_BASE=http://localhost:8000/v1
"""

import os
import time
import logging
from functools import wraps
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class _RateLimitedLLM:
    """
    Lightweight proxy wrapper that adds exponential backoff on rate-limit
    errors to any LangChain LLM. Delegates all attribute access to the
    underlying model except .invoke (and .ainvoke), which it intercepts.

    We can't monkey-patch .invoke directly because LangChain's ChatGroq
    is a pydantic v2 model that rejects unknown attribute assignment.
    """

    def __init__(self, llm: BaseChatModel, max_retries: int = 5):
        self._llm = llm
        self._max_retries = max_retries

    def __getattr__(self, name):
        # Forward everything we don't override
        return getattr(self._llm, name)

    @staticmethod
    def _is_permanent_error(err_lower: str) -> bool:
        return any(k in err_lower for k in (
            "401", "403", "invalid api key", "invalid_api_key",
            "authentication", "permission denied",
            "model_decommissioned", "model decommissioned",
            "is no longer supported", "has been decommissioned",
        ))

    def invoke(self, input, *args, **kwargs):
        delay = 5
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._llm.invoke(input, *args, **kwargs)
            except Exception as e:
                err = str(e).lower()
                if self._is_permanent_error(err):
                    raise  # fail fast — retrying will not help
                is_rate_limit = (
                    "429" in err
                    or "rate_limit" in err
                    or "rate limit" in err
                    or "too many" in err
                )
                if is_rate_limit and attempt < self._max_retries:
                    logger.warning(
                        f"Rate limit hit (attempt {attempt}/{self._max_retries}), "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    raise
        raise RuntimeError("Max retries exceeded")

    async def ainvoke(self, input, *args, **kwargs):
        import asyncio
        delay = 5
        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._llm.ainvoke(input, *args, **kwargs)
            except Exception as e:
                err = str(e).lower()
                if self._is_permanent_error(err):
                    raise  # fail fast — retrying will not help
                is_rate_limit = (
                    "429" in err
                    or "rate_limit" in err
                    or "rate limit" in err
                    or "too many" in err
                )
                if is_rate_limit and attempt < self._max_retries:
                    logger.warning(
                        f"Rate limit hit (attempt {attempt}/{self._max_retries}), "
                        f"retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    raise
        raise RuntimeError("Max retries exceeded")


def _with_rate_limit_retry(llm: BaseChatModel, max_retries: int = 5):
    """Wrap an LLM with rate-limit retry. Returns a proxy with the same API."""
    return _RateLimitedLLM(llm, max_retries=max_retries)


class _FallbackLLM:
    """
    Proxy that tries the primary LLM and falls back to a lazily-built
    secondary on connection-class errors (server down, refused, timeout).
    """

    def __init__(self, primary, fallback_factory):
        self._primary = primary
        self._fallback_factory = fallback_factory
        self._fallback = None

    def __getattr__(self, name):
        return getattr(self._primary, name)

    def _get_fallback(self):
        if self._fallback is None:
            self._fallback = self._fallback_factory()
            logger.info("Fallback LLM created")
        return self._fallback

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        err = str(exc).lower()
        return any(k in err for k in (
            "connection", "refused", "timeout", "unreachable",
            "name or service not known", "nodename nor servname",
        ))

    def invoke(self, input, *args, **kwargs):
        try:
            return self._primary.invoke(input, *args, **kwargs)
        except Exception as e:
            if self._is_connection_error(e):
                logger.warning(f"Primary LLM down ({e}), falling back")
                return self._get_fallback().invoke(input, *args, **kwargs)
            raise

    async def ainvoke(self, input, *args, **kwargs):
        try:
            return await self._primary.ainvoke(input, *args, **kwargs)
        except Exception as e:
            if self._is_connection_error(e):
                logger.warning(f"Primary LLM down ({e}), falling back")
                return await self._get_fallback().ainvoke(input, *args, **kwargs)
            raise


# Groq free-tier models (as of 2025):
#   llama-3.3-70b-versatile   — best quality, 70B params
#   llama-3.1-8b-instant      — faster, 8B params
#   gemma2-9b-it              — Google Gemma 9B
#   deepseek-r1-distill-llama-70b — DeepSeek R1 distilled
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
VLLM_DEFAULT_MODEL = "R2E-Gym/R2EGym-32B"


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0,
    role: str | None = None,
) -> BaseChatModel:
    """
    Create and return an LLM instance based on the configured provider.

    Args:
        provider: Override the provider (LLM_PROVIDER env var).
        model: Override the model (LLM_MODEL env var).
        temperature: Sampling temperature (0 = deterministic).
        role: Optional routing hint — "fast_gate" | "reason" | "generation" | "chat".
              When set, uses role-specific env vars before falling back to defaults.
              - fast_gate:  classifier + binary decisions  (FAST_LLM_PROVIDER / FAST_LLM_MODEL)
              - reason:     Phase 2 fault analysis, Planner, Reflexion (REASON_LLM_*)
                            falls back to GEN_LLM_* when REASON_LLM_* not set
              - generation: Phase 3 patch writing only    (GEN_LLM_PROVIDER / GEN_LLM_MODEL)
              - chat:       agent loop + tool routing      (CHAT_LLM_* → FAST_LLM_* fallback)

    Returns:
        A LangChain-compatible chat model.
    """
    from src_bot.config.config import configs

    # Role-based provider/model resolution — overrides env defaults when set
    if role and not provider and not model:
        if role == "fast_gate":
            provider = configs.FAST_LLM_PROVIDER or None
            model = configs.FAST_LLM_MODEL or None
        elif role == "chat":
            # chat prefers CHAT_LLM_* but falls back to FAST_LLM_* for backward compat
            provider = configs.CHAT_LLM_PROVIDER or configs.FAST_LLM_PROVIDER or None
            model = configs.CHAT_LLM_MODEL or configs.FAST_LLM_MODEL or None
        elif role == "reason":
            # Phase 2 fault analysis, Planner contracts, Reflexion — reasoning-specialized
            # Falls back to generation config when REASON_LLM_* not set
            provider = configs.REASON_LLM_PROVIDER or configs.GEN_LLM_PROVIDER or None
            model = configs.REASON_LLM_MODEL or configs.GEN_LLM_MODEL or None
        elif role == "generation":
            provider = configs.GEN_LLM_PROVIDER or None
            model = configs.GEN_LLM_MODEL or None

    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    model = model or os.getenv("LLM_MODEL", "")

    # vLLM base URL selection based on role
    vllm_base = ""
    if role == "fast_gate" and configs.VLLM_FAST_BASE:
        vllm_base = configs.VLLM_FAST_BASE
    elif role == "chat" and (configs.VLLM_CHAT_BASE or configs.VLLM_GEN_BASE):
        # chat shares the gen endpoint when no dedicated VLLM_CHAT_BASE is set
        vllm_base = configs.VLLM_CHAT_BASE or configs.VLLM_GEN_BASE
    elif role == "reason" and (configs.VLLM_REASON_BASE or configs.VLLM_GEN_BASE):
        # reason shares gen endpoint unless a dedicated VLLM_REASON_BASE is set
        vllm_base = configs.VLLM_REASON_BASE or configs.VLLM_GEN_BASE
    elif role == "generation" and configs.VLLM_GEN_BASE:
        vllm_base = configs.VLLM_GEN_BASE

    if provider == "ollama":
        return _create_ollama(model or "deepseek-coder:6.7b-instruct", temperature)
    elif provider == "groq":
        # Groq free tier has rate limits — wrap with auto-retry backoff
        llm = _create_groq(model or GROQ_DEFAULT_MODEL, temperature)
        return _with_rate_limit_retry(llm)
    elif provider == "openai":
        llm = _create_openai(model or "gpt-4o", temperature)
        return _with_rate_limit_retry(llm)
    elif provider == "together":
        llm = _create_together(model or "openai/gpt-oss-20b", temperature)
        return _with_rate_limit_retry(llm)
    elif provider == "vllm":
        primary = _create_vllm(model or VLLM_DEFAULT_MODEL, temperature, vllm_base)
        # When using vLLM with a role, auto-fallback to Groq on connection errors
        if role:
            fallback_model = model or GROQ_DEFAULT_MODEL
            def _groq_fallback():
                return _with_rate_limit_retry(_create_groq(fallback_model, temperature))
            return _FallbackLLM(primary, _groq_fallback)
        return primary
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Supported: ollama, groq, openai, together, vllm"
        )


def _attach_langfuse(llm: BaseChatModel) -> BaseChatModel:
    """Attach Langfuse callback to a base LangChain model if observability is enabled.

    Must be called BEFORE wrapping in _RateLimitedLLM / _FallbackLLM so the
    callback is on the underlying model that actually makes LLM calls.
    """
    try:
        from src_bot.observability.langfuse_handler import get_langfuse_callback
        handler = get_langfuse_callback()
        if handler:
            existing = list(getattr(llm, "callbacks", None) or [])
            llm.callbacks = existing + [handler]
    except Exception:
        pass  # observability must never break the pipeline
    return llm


def _create_ollama(model: str, temperature: float) -> BaseChatModel:
    """Local Ollama model — free, good for development."""
    from langchain_ollama import ChatOllama

    return _attach_langfuse(ChatOllama(model=model, temperature=temperature))


def _create_groq(model: str, temperature: float) -> BaseChatModel:
    """
    Groq API — free tier, very fast inference.

    Free tier limits (as of 2025):
      - llama-3.3-70b: 30 req/min, 6000 tokens/min
      - llama-3.1-8b:  30 req/min, 20000 tokens/min

    Get key at: https://console.groq.com/keys
    """
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is required when LLM_PROVIDER=groq. "
            "Get a free key at https://console.groq.com/keys"
        )
    return _attach_langfuse(ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key,
    ))


def _create_openai(model: str, temperature: float) -> BaseChatModel:
    """OpenAI API — GPT-4o for paper experiments."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
            "Set it in your .env file."
        )
    return _attach_langfuse(ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    ))


def _create_together(model: str, temperature: float) -> BaseChatModel:
    """Together.ai API — for open-source models like gpt-oss-20b."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError(
            "TOGETHER_API_KEY is required when LLM_PROVIDER=together. "
            "Set it in your .env file."
        )
    return _attach_langfuse(ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
    ))


def _create_vllm(model: str, temperature: float, base_url: str = "") -> BaseChatModel:
    """
    Self-hosted vLLM server — free, no rate limits.

    vLLM exposes an OpenAI-compatible API, so we reuse ChatOpenAI.
    Set VLLM_API_BASE to your server URL (default: http://localhost:8000/v1).
    For dual-endpoint setup use VLLM_FAST_BASE / VLLM_GEN_BASE — the router
    selects the correct URL based on the role= parameter.

    Recommended models:
      fast_gate / chat : llama-3.1-8b-instant on RTX 3090 (VLLM_FAST_BASE)
      generation       : llama-3.3-70b on A100 80GB (VLLM_GEN_BASE)

    Setup on Vast.ai / Lambda / RunPod:
        pip install vllm
        python -m vllm.entrypoints.openai.api_server \\
            --model <model> --dtype float16 --max-model-len 16384 --port 8000
    """
    from langchain_openai import ChatOpenAI

    resolved_base = base_url or os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
    return _attach_langfuse(ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key="dummy",          # vLLM doesn't require auth
        base_url=resolved_base,
        max_tokens=4096,
    ))
