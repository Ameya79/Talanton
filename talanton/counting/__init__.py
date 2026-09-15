"""Universal token counting dispatcher for Talanton.

Routes counting requests across OpenAI, Anthropic, HuggingFace,
and unknown models with guaranteed graceful degradation to heuristic fallback.
"""

from __future__ import annotations

import logging
from typing import Any

from talanton.counting.fallback import count_fallback
from talanton.counting.openai import count_openai
from talanton.providers import detect_provider

logger = logging.getLogger("talanton.counting")


def count_tokens(text: str | list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Counts tokens for any supported model, returning a normalized contract.

    Routes to provider-specific adapters (OpenAI tiktoken, Anthropic API,
    or HuggingFace local tokenizers). Accounts for chat-format message
    overhead per provider.

    Degrades gracefully to a heuristic estimator (chars / 4) with exact=False
    if the model is unknown or if a provider API/dependency fails.

    Data contract:
        {
            "tokens": int,
            "exact": bool,
            "provider": str,  # 'openai' | 'anthropic' | 'huggingface' | 'unknown'
            "model": str
        }

    Args:
        text: Raw string prompt or chat-formatted list of message dicts.
        model: Model name string (e.g. 'gpt-4o', 'claude-sonnet-4.5').

    Returns:
        Standardized token count dictionary.
    """
    provider = detect_provider(model)

    if provider == "openai":
        try:
            return count_openai(text, model)
        except Exception as e:
            logger.warning("OpenAI counting failed for model %r: %s. Falling back to heuristic.", model, e)
            return count_fallback(text, model, provider="openai")

    elif provider == "anthropic":
        try:
            from talanton.counting.anthropic import count_anthropic
            return count_anthropic(text, model)
        except Exception as e:
            logger.warning("Anthropic counting failed for model %r: %s. Falling back to heuristic.", model, e)
            return count_fallback(text, model, provider="anthropic")

    elif provider == "huggingface":
        try:
            from talanton.counting.huggingface import count_huggingface
            return count_huggingface(text, model)
        except Exception as e:
            logger.warning("HuggingFace counting failed for model %r: %s. Falling back to heuristic.", model, e)
            return count_fallback(text, model, provider="huggingface")

    else:
        return count_fallback(text, model, provider="unknown")


__all__ = ["count_tokens", "count_fallback", "count_openai"]
