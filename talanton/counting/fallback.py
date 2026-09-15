"""Fallback heuristic token estimator.

Uses the standard English approximation (~4 characters per token).
Always returns exact=False.
"""

from __future__ import annotations

from typing import Any


def count_fallback(
    text: str | list[dict[str, Any]],
    model: str,
    provider: str = "unknown"
) -> dict[str, Any]:
    """Estimates token count using character length heuristic (chars / 4).

    Always returns exact=False.

    Args:
        text: Raw text string or chat-formatted list of message dicts.
        model: Model name.
        provider: Provider identifier ('unknown', 'openai', etc.).

    Returns:
        Token count dict: {'tokens': int, 'exact': False, 'provider': str, 'model': str}
    """
    if isinstance(text, str):
        char_count = len(text)
    elif isinstance(text, list):
        # Extract text from message dicts plus role overhead approximation
        total_chars = 0
        for msg in text:
            if isinstance(msg, dict):
                content = str(msg.get("content", ""))
                role = str(msg.get("role", ""))
                name = str(msg.get("name", ""))
                total_chars += len(content) + len(role) + len(name) + 8
            else:
                total_chars += len(str(msg))
        char_count = total_chars
    else:
        char_count = len(str(text))

    estimated_tokens = max(1, round(char_count / 4.0)) if char_count > 0 else 0

    return {
        "tokens": estimated_tokens,
        "exact": False,
        "provider": provider,
        "model": model,
    }
