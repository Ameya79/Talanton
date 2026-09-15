"""OpenAI token counting adapter using tiktoken.

Accurately computes token counts for both raw text strings and
chat-formatted message lists, matching OpenAI's official cookbook formula.
"""

from __future__ import annotations

from typing import Any
import tiktoken

from talanton.providers import get_model_info


def _get_encoding(model: str) -> tiktoken.Encoding:
    """Gets tiktoken Encoding object for a model with graceful fallbacks."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        info = get_model_info(model)
        if info and info.get("tokenizer"):
            tokenizer_id = info["tokenizer"]
            try:
                return tiktoken.get_encoding(tokenizer_id)
            except Exception:
                pass

        # Fallback based on model family
        model_lower = model.lower()
        if "4o" in model_lower or "o1" in model_lower or "o3" in model_lower:
            return tiktoken.get_encoding("o200k_base")
        return tiktoken.get_encoding("cl100k_base")


def count_openai(
    text: str | list[dict[str, Any]],
    model: str
) -> dict[str, Any]:
    """Counts tokens for OpenAI models with exact chat-template accounting.

    Implements the OpenAI cookbook recipe for messages:
    - 3 tokens per message (<|im_start|>{role}\\n{content}<|im_end|>)
    - 1 additional token if a message has a 'name' field
    - 3 tokens to prime the assistant's reply (<|im_start|>assistant<|message|>)

    Args:
        text: Raw text string or chat messages list.
        model: OpenAI model identifier (e.g. 'gpt-4o', 'gpt-3.5-turbo').

    Returns:
        {'tokens': int, 'exact': True, 'provider': 'openai', 'model': str}
    """
    encoding = _get_encoding(model)

    if isinstance(text, str):
        tokens = len(encoding.encode(text))
        return {
            "tokens": tokens,
            "exact": True,
            "provider": "openai",
            "model": model,
        }

    if isinstance(text, list):
        if "gpt-3.5-turbo-0301" in model.lower():
            tokens_per_message = 4
            tokens_per_name = -1
        else:
            tokens_per_message = 3
            tokens_per_name = 1

        num_tokens = 0
        for message in text:
            num_tokens += tokens_per_message
            for key, value in message.items():
                content_str = str(value) if value is not None else ""
                num_tokens += len(encoding.encode(content_str))
                if key == "name":
                    num_tokens += tokens_per_name

        # Every reply is primed with <|start|>assistant<|message|>
        num_tokens += 3

        return {
            "tokens": num_tokens,
            "exact": True,
            "provider": "openai",
            "model": model,
        }

    # Fallback for unexpected types
    tokens = len(encoding.encode(str(text)))
    return {
        "tokens": tokens,
        "exact": True,
        "provider": "openai",
        "model": model,
    }
