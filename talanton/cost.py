"""Cost calculation and model comparison for Talanton.

Translates input/output token counts into dollar spend based on curated pricing,
propagating exactness flags and sorting candidate models by cost.
"""

from __future__ import annotations

from typing import Any
from talanton.counting import count_tokens
from talanton.pricing import get_price


def calculate_cost(
    text: str | list[dict[str, Any]] | int,
    model: str,
    expected_output_tokens: int = 0
) -> dict[str, Any]:
    """Calculates dollar cost for a prompt and optional expected output tokens.

    Data contract:
        {
            "input_tokens": int,
            "output_tokens": int,
            "input_cost": float,
            "output_cost": float,
            "total_cost": float,
            "exact": bool,
            "model": str
        }

    Args:
        text: Raw prompt string, chat messages list, or integer token count directly.
        model: Model name string (must exist in pricing table).
        expected_output_tokens: Expected completion token count (default 0).

    Returns:
        Standardized cost breakdown dictionary.

    Raises:
        PricingNotFoundError: If model is not found in pricing table.
        ValueError: If expected_output_tokens is negative.
    """
    if expected_output_tokens < 0:
        raise ValueError(f"expected_output_tokens must be non-negative, got {expected_output_tokens}")

    if isinstance(text, int):
        if text < 0:
            raise ValueError(f"Input tokens must be non-negative, got {text}")
        input_tokens = text
        exact = True
    else:
        count_res = count_tokens(text, model)
        input_tokens = count_res["tokens"]
        exact = count_res["exact"]

    pricing = get_price(model)
    input_rate = pricing["input"]    # USD per 1M tokens
    output_rate = pricing["output"]  # USD per 1M tokens

    input_cost = (input_tokens / 1_000_000.0) * input_rate
    output_cost = (expected_output_tokens / 1_000_000.0) * output_rate
    total_cost = round(input_cost + output_cost, 8)

    return {
        "input_tokens": input_tokens,
        "output_tokens": expected_output_tokens,
        "input_cost": round(input_cost, 8),
        "output_cost": round(output_cost, 8),
        "total_cost": total_cost,
        "exact": exact,
        "model": model,
    }


def compare_models(
    text: str | list[dict[str, Any]] | int,
    models: list[str],
    expected_output_tokens: int = 0
) -> list[dict[str, Any]]:
    """Compares costs across candidate models for the same prompt and expected output.

    Args:
        text: Prompt text, chat messages list, or direct token count.
        models: List of model names to compare.
        expected_output_tokens: Expected output length in tokens (default 0).

    Returns:
        List of cost dicts sorted ascending by total_cost (cheapest first).
    """
    if not models:
        return []

    results = []
    for model in models:
        cost_entry = calculate_cost(text, model, expected_output_tokens=expected_output_tokens)
        results.append(cost_entry)

    # Sort ascending by total_cost, then input_cost
    results.sort(key=lambda item: (item["total_cost"], item["input_cost"]))
    return results
