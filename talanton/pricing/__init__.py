"""Pricing table loader and query interface for Talanton.

Provides model pricing lookup in USD per 1,000,000 tokens.
Raises PricingNotFoundError if a model has no pricing entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PRICING_PATH = Path(__file__).resolve().parent / "pricing_table.json"
_PRICING_CACHE: dict[str, dict[str, float]] | None = None
_LOWER_PRICING_CACHE: dict[str, str] | None = None


class PricingNotFoundError(KeyError):
    """Raised when pricing data is not found for a requested model."""
    def __init__(self, model: str):
        super().__init__(
            f"No pricing data found for model '{model}'. "
            "A wrong cost number is worse than a loud error. "
            "Please check models in the pricing table or provide explicit pricing."
        )
        self.model = model


def _load_pricing() -> dict[str, dict[str, float]]:
    """Loads and caches the pricing table from pricing_table.json."""
    global _PRICING_CACHE, _LOWER_PRICING_CACHE
    if _PRICING_CACHE is None:
        try:
            with open(_PRICING_PATH, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception:
            raw_data = {}

        # Filter out metadata keys like '_metadata'
        _PRICING_CACHE = {
            k: v for k, v in raw_data.items()
            if not k.startswith("_") and isinstance(v, dict)
        }
        _LOWER_PRICING_CACHE = {k.lower(): k for k in _PRICING_CACHE}
    return _PRICING_CACHE


def get_price(model: str) -> dict[str, float]:
    """Retrieves pricing for a model in USD per 1,000,000 tokens.

    Args:
        model: Model name identifier.

    Returns:
        Dict with 'input' and 'output' price in $/1M tokens:
        {'input': float, 'output': float}

    Raises:
        PricingNotFoundError: If model is not found in the pricing table.
    """
    if not isinstance(model, str):
        raise PricingNotFoundError(str(model))

    cleaned_model = model.strip()
    if not cleaned_model:
        raise PricingNotFoundError("")

    pricing = _load_pricing()

    # Exact match
    if cleaned_model in pricing:
        entry = pricing[cleaned_model]
        return {"input": float(entry["input"]), "output": float(entry["output"])}

    # Case-insensitive match
    assert _LOWER_PRICING_CACHE is not None
    lower_key = cleaned_model.lower()
    if lower_key in _LOWER_PRICING_CACHE:
        canonical_key = _LOWER_PRICING_CACHE[lower_key]
        entry = pricing[canonical_key]
        return {"input": float(entry["input"]), "output": float(entry["output"])}

    raise PricingNotFoundError(cleaned_model)


def list_priced_models() -> list[str]:
    """Returns sorted list of all models with available pricing."""
    pricing = _load_pricing()
    return sorted(pricing.keys())


__all__ = ["get_price", "PricingNotFoundError", "list_priced_models"]
