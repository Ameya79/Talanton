"""Tests for talanton.cost calculation and model comparison."""

import pytest
from talanton.cost import calculate_cost, compare_models
from talanton.pricing import PricingNotFoundError, get_price


def test_calculate_cost_gpt_4o_hand_math():
    """Verify cost math against hand-calculated values for gpt-4o.

    Pricing: $2.50 / 1M input, $10.00 / 1M output.
    Input tokens: 1,000  -> 1,000 * 2.50 / 1,000,000 = $0.0025
    Output tokens: 500   -> 500 * 10.00 / 1,000,000  = $0.0050
    Total: $0.0075
    """
    result = calculate_cost(text=1000, model="gpt-4o", expected_output_tokens=500)

    assert result["input_tokens"] == 1000
    assert result["output_tokens"] == 500
    assert pytest.approx(result["input_cost"], rel=1e-6) == 0.0025
    assert pytest.approx(result["output_cost"], rel=1e-6) == 0.0050
    assert pytest.approx(result["total_cost"], rel=1e-6) == 0.0075
    assert result["exact"] is True
    assert result["model"] == "gpt-4o"


def test_calculate_cost_gpt_4o_mini_hand_math():
    """Verify cost math for gpt-4o-mini.

    Pricing: $0.15 / 1M input, $0.60 / 1M output.
    Input tokens: 10,000 -> 10,000 * 0.15 / 1,000,000 = $0.0015
    Output tokens: 2,000 -> 2,000 * 0.60 / 1,000,000  = $0.0012
    Total: $0.0027
    """
    result = calculate_cost(text=10000, model="gpt-4o-mini", expected_output_tokens=2000)

    assert pytest.approx(result["input_cost"], rel=1e-6) == 0.0015
    assert pytest.approx(result["output_cost"], rel=1e-6) == 0.0012
    assert pytest.approx(result["total_cost"], rel=1e-6) == 0.0027


def test_calculate_cost_claude_3_5_haiku_hand_math():
    """Verify cost math for claude-3-5-haiku-latest.

    Pricing: $0.80 / 1M input, $4.00 / 1M output.
    Input tokens: 5,000 -> 5,000 * 0.80 / 1,000,000 = $0.0040
    Output tokens: 1,000 -> 1,000 * 4.00 / 1,000,000 = $0.0040
    Total: $0.0080
    """
    result = calculate_cost(text=5000, model="claude-3-5-haiku-latest", expected_output_tokens=1000)

    assert pytest.approx(result["input_cost"], rel=1e-6) == 0.0040
    assert pytest.approx(result["output_cost"], rel=1e-6) == 0.0040
    assert pytest.approx(result["total_cost"], rel=1e-6) == 0.0080


def test_compare_models_sorts_cheapest_first():
    """Verify compare_models ranks results strictly by total_cost ascending."""
    prompt = "Analyze the financial statement and summarize EBITDA growth."
    models = ["gpt-4o", "gpt-4o-mini", "claude-3-5-haiku-latest"]

    results = compare_models(prompt, models, expected_output_tokens=100)

    assert len(results) == 3
    # Check that costs are strictly non-decreasing
    for i in range(len(results) - 1):
        assert results[i]["total_cost"] <= results[i + 1]["total_cost"]

    # gpt-4o-mini should be cheaper than gpt-4o
    model_order = [r["model"] for r in results]
    assert model_order.index("gpt-4o-mini") < model_order.index("gpt-4o")


def test_pricing_not_found_raises_typed_error():
    """Verify missing pricing raises PricingNotFoundError and does NOT return $0."""
    with pytest.raises(PricingNotFoundError):
        calculate_cost("test", "nonexistent-model-without-pricing")


def test_invalid_negative_tokens():
    """Verify negative tokens raises ValueError."""
    with pytest.raises(ValueError):
        calculate_cost(100, "gpt-4o", expected_output_tokens=-5)

    with pytest.raises(ValueError):
        calculate_cost(-10, "gpt-4o")
