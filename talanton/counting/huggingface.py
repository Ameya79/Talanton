"""HuggingFace local tokenizer adapter for Talanton.

Loads and caches open-weight tokenizers (e.g. Llama 3, Mistral) locally.
Explicitly warns users before downloading any uncached tokenizer files from the Hub.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from talanton.counting.fallback import count_fallback
from talanton.providers import get_model_info

logger = logging.getLogger("talanton.counting.huggingface")

_TOKENIZER_CACHE: dict[str, Any] = {}


def _is_cached_locally(model_id: str) -> bool:
    """Checks if model tokenizer files already exist in local HF cache."""
    try:
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        hf_cache = Path(HUGGINGFACE_HUB_CACHE)
    except Exception:
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"

    model_dir_name = f"models--{model_id.replace('/', '--')}"
    target_dir = hf_cache / model_dir_name
    return target_dir.exists() and any(target_dir.iterdir()) if target_dir.exists() else False


def _load_hf_tokenizer(model_id: str) -> Any:
    """Loads a HuggingFace tokenizer with caching and download warning."""
    if model_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_id]

    if not _is_cached_locally(model_id):
        print(
            f"[Talanton Notice] Tokenizer for '{model_id}' is not cached locally. "
            "Downloading tokenizer configuration from Hugging Face Hub (this may take a few seconds)..."
        )

    # Attempt loading with tokenizers library first (fast Rust implementation)
    tokenizer = None
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_pretrained(model_id)
    except Exception as e1:
        logger.debug("tokenizers.Tokenizer.from_pretrained failed: %s", e1)
        # Attempt loading with transformers AutoTokenizer if installed
        try:
            from transformers import AutoTokenizer  # type: ignore
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e2:
            logger.debug("transformers.AutoTokenizer.from_pretrained failed: %s", e2)
            raise RuntimeError(f"Could not load tokenizer for '{model_id}': {e1}; {e2}")

    _TOKENIZER_CACHE[model_id] = tokenizer
    return tokenizer


def count_huggingface(
    text: str | list[dict[str, Any]],
    model: str
) -> dict[str, Any]:
    """Counts tokens using a local HuggingFace tokenizer.

    Warns before downloading uncached tokenizer files and caches loaded instances in-process.
    Falls back gracefully to heuristic estimation if dependencies fail or files cannot be fetched.

    Args:
        text: Prompt text or chat messages list.
        model: HuggingFace model repo ID (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct').

    Returns:
        {'tokens': int, 'exact': bool, 'provider': 'huggingface', 'model': str}
    """
    # Resolve canonical tokenizer ID from registry if available
    info = get_model_info(model)
    model_id = (info.get("tokenizer") if info else None) or model

    try:
        tok = _load_hf_tokenizer(model_id)

        if isinstance(text, str):
            if hasattr(tok, "encode"):
                encoded = tok.encode(text)
                # Handle both tokenizers.Encoding and list[int] from transformers
                tokens = len(encoded.ids) if hasattr(encoded, "ids") else len(encoded)
            else:
                tokens = round(len(text) / 4)
        elif isinstance(text, list):
            # Check if tokenizer has apply_chat_template (transformers)
            if hasattr(tok, "apply_chat_template"):
                try:
                    ids = tok.apply_chat_template(text, tokenize=True)
                    tokens = len(ids)
                except Exception:
                    # Fallback to manual template overhead
                    tokens = sum(
                        len(tok.encode(msg.get("content", "")).ids
                            if hasattr(tok.encode(msg.get("content", "")), "ids")
                            else tok.encode(msg.get("content", ""))) + 4
                        for msg in text if isinstance(msg, dict)
                    )
            else:
                # Approximate role markers: <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
                total = 0
                for msg in text:
                    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    enc = tok.encode(content)
                    count = len(enc.ids) if hasattr(enc, "ids") else len(enc)
                    total += count + 4  # 4 tokens for Llama 3 header markers
                tokens = total
        else:
            enc = tok.encode(str(text))
            tokens = len(enc.ids) if hasattr(enc, "ids") else len(enc)

        return {
            "tokens": int(tokens),
            "exact": True,
            "provider": "huggingface",
            "model": model,
        }

    except Exception as e:
        logger.warning(
            "HuggingFace token counting failed for model %r (%s). "
            "Degrading gracefully to heuristic fallback (exact=False).",
            model,
            e
        )
        return count_fallback(text, model, provider="huggingface")
