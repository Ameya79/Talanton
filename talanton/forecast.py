"""Volume spend forecasting and scale switch analysis for Talanton.

Projects future API expenditure over time under user-supplied volume and growth assumptions.
Disclaimer: This uses mathematical compounding to project user-supplied assumptions;
it does not predict real-world user adoption or business growth.
"""

from __future__ import annotations

from typing import Any
from talanton.pricing import get_price

DAYS_PER_MONTH = 30


def forecast(
    model: str,
    calls_per_day: float | int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    months: int = 6,
    growth_rate: float = 0.0
) -> list[dict[str, Any]]:
    """Projects month-by-month API spend under a user-supplied growth assumption.

    NOTE: This is a plain compounding formula (calls_m = calls_0 * (1 + growth)^(m-1)).
    It does NOT predict real-world growth; it strictly projects the mathematical
    consequences of a user-supplied assumption.

    Data contract:
        [
            {"month": int, "cost": float, "calls_per_day": float},
            ...
        ]

    Args:
        model: Model name string.
        calls_per_day: Initial daily request volume.
        avg_input_tokens: Average prompt tokens per call.
        avg_output_tokens: Average completion tokens per call.
        months: Forecast horizon in months (must be a positive integer, default 6).
        growth_rate: Monthly compounding growth rate between 0.0 and 1.0 (e.g. 0.15 for 15%).

    Returns:
        List of monthly cost projections.

    Raises:
        ValueError: If inputs fall outside valid ranges.
        PricingNotFoundError: If model is not found in the pricing table.
    """
    if not isinstance(months, int) or months <= 0:
        raise ValueError(f"months must be a positive integer, got {months}")
    if calls_per_day < 0:
        raise ValueError(f"calls_per_day must be non-negative, got {calls_per_day}")
    if avg_input_tokens < 0 or avg_output_tokens < 0:
        raise ValueError("Token counts must be non-negative")
    if not (0.0 <= growth_rate <= 1.0):
        raise ValueError(f"growth_rate must be between 0.0 and 1.0, got {growth_rate}")

    pricing = get_price(model)
    input_rate = pricing["input"]    # $/1M
    output_rate = pricing["output"]  # $/1M

    cost_per_call = (avg_input_tokens / 1_000_000.0) * input_rate + \
                    (avg_output_tokens / 1_000_000.0) * output_rate

    projection: list[dict[str, Any]] = []

    for m in range(1, months + 1):
        # Month 1 uses initial calls_per_day; subsequent months compound
        current_calls_per_day = float(calls_per_day) * ((1.0 + growth_rate) ** (m - 1))
        monthly_calls = current_calls_per_day * DAYS_PER_MONTH
        monthly_cost = round(monthly_calls * cost_per_call, 2)

        projection.append({
            "month": m,
            "cost": monthly_cost,
            "calls_per_day": round(current_calls_per_day, 1),
        })

    return projection


def compare_at_scale(
    models: list[str],
    calls_per_day: float | int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    months: int = 6,
    growth_rate: float = 0.0
) -> dict[str, Any]:
    """Compares total projected spend across candidate models over a time horizon.

    Calculates the cumulative spend per model, identifies the cheapest model,
    and quantifies the potential dollar savings achieved by choosing or switching
    to that cheapest option compared to the highest-cost alternative.

    Data contract:
        {
            "totals": {model: total_cost, ...},
            "cheapest": str,
            "savings_by_switching": float
        }

    Args:
        models: List of model identifiers.
        calls_per_day: Initial daily request volume.
        avg_input_tokens: Average input tokens per call.
        avg_output_tokens: Average output tokens per call.
        months: Horizon in months (default 6).
        growth_rate: Monthly compounding rate (default 0.0).

    Returns:
        Summary comparison dictionary.
    """
    if not models:
        return {"totals": {}, "cheapest": "", "savings_by_switching": 0.0}

    totals: dict[str, float] = {}

    for model in models:
        monthly_data = forecast(
            model=model,
            calls_per_day=calls_per_day,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            months=months,
            growth_rate=growth_rate,
        )
        total_spend = sum(entry["cost"] for entry in monthly_data)
        totals[model] = round(total_spend, 2)

    cheapest_model = min(totals, key=totals.get)  # type: ignore
    most_expensive = max(totals.values()) if totals else 0.0
    cheapest_cost = totals[cheapest_model]
    savings = round(most_expensive - cheapest_cost, 2)

    return {
        "totals": totals,
        "cheapest": cheapest_model,
        "savings_by_switching": savings,
    }
