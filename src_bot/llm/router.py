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

    def invoke(self, input, *args, **kwargs):
        delay = 5
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._llm.invoke(input, *args, **kwargs)
            except Exception as e:
                err = str(e).lower()
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
        # Async variant — same backoff loop using asyncio.sleep
        import asyncio
        delay = 5
        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._llm.ainvoke(input, *args, **kwargs)
            except Exception as e:
                err = str(e).lower()
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
) -> BaseChatModel:
    """
    Create and return an LLM instance based on the configured provider.

    Args:
        provider: Override LLM_PROVIDER env var.
        model: Override LLM_MODEL env var.
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        A LangChain-compatible chat model.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    model = model or os.getenv("LLM_MODEL", "")

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
        # vLLM is self-hosted — no rate limits, no retry wrapper needed
        return _create_vllm(model or VLLM_DEFAULT_MODEL, temperature)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Supported: ollama, groq, openai, together, vllm"
        )


def _create_ollama(model: str, temperature: float) -> BaseChatModel:
    """Local Ollama model — free, good for development."""
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=temperature)


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
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )


def _create_openai(model: str, temperature: float) -> BaseChatModel:
    """OpenAI API — GPT-4o for paper experiments."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
            "Set it in your .env file."
        )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )


def _create_together(model: str, temperature: float) -> BaseChatModel:
    """Together.ai API — for open-source models like gpt-oss-20b."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError(
            "TOGETHER_API_KEY is required when LLM_PROVIDER=together. "
            "Set it in your .env file."
        )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
    )


def _create_vllm(model: str, temperature: float) -> BaseChatModel:
    """
    Self-hosted vLLM server — free, no rate limits.

    vLLM exposes an OpenAI-compatible API, so we reuse ChatOpenAI.
    Set VLLM_API_BASE to your server URL (default: http://localhost:8000/v1).

    Recommended model: R2E-Gym/R2EGym-32B (fine-tuned for SWE-bench, ~34% pass@1)

    Setup on Vast.ai / Lambda / RunPod:
        pip install vllm
        python -m vllm.entrypoints.openai.api_server \\
            --model R2E-Gym/R2EGym-32B \\
            --dtype float16 \\
            --max-model-len 16384 \\
            --port 8000
    """
    from langchain_openai import ChatOpenAI

    base_url = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key="dummy",          # vLLM doesn't require auth
        base_url=base_url,
        max_tokens=4096,
    )
