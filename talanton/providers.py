"""Model to provider detection for Talanton.

Provides lookup functions mapping model names to provider identifiers
('openai', 'anthropic', 'huggingface', 'unknown') backed by models.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_MODELS_PATH = Path(__file__).resolve().parent / "models.json"
_MODELS_CACHE: dict[str, dict[str, Any]] | None = None
_LOWER_MODELS_CACHE: dict[str, str] | None = None


def _load_models() -> dict[str, dict[str, Any]]:
    """Loads and caches the models lookup table from models.json."""
    global _MODELS_CACHE, _LOWER_MODELS_CACHE
    if _MODELS_CACHE is None:
        try:
            with open(_MODELS_PATH, "r", encoding="utf-8") as f:
                _MODELS_CACHE = json.load(f)
        except Exception:
            _MODELS_CACHE = {}

        _LOWER_MODELS_CACHE = {k.lower(): k for k in _MODELS_CACHE}
    return _MODELS_CACHE


def detect_provider(model: str) -> str:
    """Detects provider for a given model name.

    Returns 'openai' | 'anthropic' | 'huggingface' | 'unknown' based on a
    maintained lookup table (models.json), not string-matching heuristics.

    Unknown model names return 'unknown' and never raise an exception.

    Args:
        model: Name of the model (e.g. 'gpt-4o', 'claude-sonnet-4.5').

    Returns:
        Provider name or 'unknown'.
    """
    if not isinstance(model, str):
        return "unknown"

    cleaned_model = model.strip()
    if not cleaned_model:
        return "unknown"

    models_table = _load_models()

    # 1. Exact match
    if cleaned_model in models_table:
        return models_table[cleaned_model].get("provider", "unknown")

    # 2. Case-insensitive match fallback
    assert _LOWER_MODELS_CACHE is not None
    lower_key = cleaned_model.lower()
    if lower_key in _LOWER_MODELS_CACHE:
        canonical_key = _LOWER_MODELS_CACHE[lower_key]
        return models_table[canonical_key].get("provider", "unknown")

    return "unknown"


def get_model_info(model: str) -> dict[str, Any] | None:
    """Retrieves metadata for a known model from models.json.

    Args:
        model: Name of the model.

    Returns:
        Metadata dict (e.g. {'provider': 'openai', 'tokenizer': 'o200k_base'})
        or None if model is unknown.
    """
    if not isinstance(model, str):
        return None

    cleaned_model = model.strip()
    if not cleaned_model:
        return None

    models_table = _load_models()
    if cleaned_model in models_table:
        return dict(models_table[cleaned_model])

    assert _LOWER_MODELS_CACHE is not None
    lower_key = cleaned_model.lower()
    if lower_key in _LOWER_MODELS_CACHE:
        canonical_key = _LOWER_MODELS_CACHE[lower_key]
        return dict(models_table[canonical_key])

    return None


def is_supported_model(model: str) -> bool:
    """Checks whether a model is registered in models.json."""
    return detect_provider(model) != "unknown"


def list_supported_models(provider: str | None = None) -> list[str]:
    """Returns a list of all supported model names, optionally filtered by provider.

    Args:
        provider: Optional provider filter ('openai', 'anthropic', 'huggingface').

    Returns:
        Sorted list of model names.
    """
    models_table = _load_models()
    if provider is None:
        return sorted(models_table.keys())

    target_provider = provider.strip().lower()
    return sorted(
        name for name, info in models_table.items()
        if info.get("provider", "").lower() == target_provider
    )
