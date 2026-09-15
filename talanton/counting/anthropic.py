"""Anthropic token counting adapter.

Calls Anthropic's official count_tokens endpoint when ANTHROPIC_API_KEY is available,
mitigating rate limits with a local disk cache (~/.talanton/cache.json).
Gracefully falls back to heuristic estimation (exact=False) if the key is missing or the call fails.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from talanton.counting.fallback import count_fallback

logger = logging.getLogger("talanton.counting.anthropic")

_CACHE_DIR = Path.home() / ".talanton"
_CACHE_FILE = _CACHE_DIR / "cache.json"
_LOCAL_CACHE: dict[str, int] | None = None


def _get_cache() -> dict[str, int]:
    """Loads cache dictionary from disk."""
    global _LOCAL_CACHE
    if _LOCAL_CACHE is None:
        _LOCAL_CACHE = {}
        if _CACHE_FILE.exists():
            try:
                with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                    _LOCAL_CACHE = json.load(f)
            except Exception as e:
                logger.debug("Failed to read token cache file: %s", e)
                _LOCAL_CACHE = {}
    return _LOCAL_CACHE


def _set_cache(cache_key: str, tokens: int) -> None:
    """Saves a token count to local disk cache."""
    cache = _get_cache()
    cache[cache_key] = tokens
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        logger.debug("Failed to write to token cache file: %s", e)


def _compute_cache_key(text: str | list[dict[str, Any]], model: str) -> str:
    """Computes a deterministic hash for a given text and model."""
    serialized = json.dumps(text, sort_keys=True) if not isinstance(text, str) else text
    raw = f"{model}:{serialized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def count_anthropic(
    text: str | list[dict[str, Any]],
    model: str
) -> dict[str, Any]:
    """Counts tokens using Anthropic's count_tokens endpoint or local cache.

    Requires ANTHROPIC_API_KEY environment variable. If missing or if the API call
    fails, falls back cleanly to heuristic estimation with exact=False.

    Args:
        text: Raw text string or chat message list.
        model: Anthropic model identifier (e.g. 'claude-sonnet-4.5').

    Returns:
        {'tokens': int, 'exact': bool, 'provider': 'anthropic', 'model': str}
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    cache_key = _compute_cache_key(text, model)
    cache = _get_cache()

    # Check local disk cache first (rate-limit mitigation)
    if cache_key in cache:
        return {
            "tokens": cache[cache_key],
            "exact": True,
            "provider": "anthropic",
            "model": model,
        }

    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Anthropic does not publish offline tokenizer rules; exact counting requires "
            "calling Anthropic's count_tokens endpoint. Falling back to heuristic estimate (exact=False)."
        )
        return count_fallback(text, model, provider="anthropic")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Build messages payload
        system_content = None
        messages_payload: list[dict[str, Any]] = []

        if isinstance(text, str):
            messages_payload = [{"role": "user", "content": text}]
        elif isinstance(text, list):
            for msg in text:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_content = content
                else:
                    messages_payload.append({"role": role, "content": content})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages_payload if messages_payload else [{"role": "user", "content": ""}]
        }
        if system_content:
            kwargs["system"] = system_content

        response = client.messages.count_tokens(**kwargs)
        tokens = int(response.input_tokens)

        # Cache valid response
        _set_cache(cache_key, tokens)

        return {
            "tokens": tokens,
            "exact": True,
            "provider": "anthropic",
            "model": model,
        }

    except Exception as e:
        logger.warning(
            "Anthropic API token counting request failed (%s). "
            "Degrading gracefully to heuristic estimate (exact=False).",
            e
        )
        return count_fallback(text, model, provider="anthropic")
