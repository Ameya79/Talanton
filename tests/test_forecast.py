"""Tests for talanton.forecast volume projections and scale switch comparison."""

import pytest
from talanton.forecast import forecast, compare_at_scale


def test_forecast_flat_growth_hand_math():
    """Verify forecast at 0% growth matches hand-calculated numbers.

    gpt-4o ($2.50/1M in, $10.00/1M out).
    800 in, 200 out -> cost per call = $0.004.
    1,000 calls/day -> 30,000 calls/month -> $120.00/month.
    """
    results = forecast(
        model="gpt-4o",
        calls_per_day=1000,
        avg_input_tokens=800,
        avg_output_tokens=200,
        months=6,
        growth_rate=0.0
    )

    assert len(results) == 6
    for idx, row in enumerate(results, start=1):
        assert row["month"] == idx
        assert row["calls_per_day"] == 1000.0
        assert pytest.approx(row["cost"], rel=1e-2) == 120.00


def test_forecast_compounding_growth_hand_math():
    """Verify forecast with 10% monthly compounding growth.

    Month 1: 1,000 calls/day -> $120.00
    Month 2: 1,100 calls/day -> $132.00
    Month 3: 1,210 calls/day -> $145.20
    """
    results = forecast(
        model="gpt-4o",
        calls_per_day=1000,
        avg_input_tokens=800,
        avg_output_tokens=200,
        months=3,
        growth_rate=0.10
    )

    assert len(results) == 3
    assert pytest.approx(results[0]["cost"], rel=1e-2) == 120.00
    assert pytest.approx(results[1]["cost"], rel=1e-2) == 132.00
    assert pytest.approx(results[2]["cost"], rel=1e-2) == 145.20


def test_compare_at_scale_savings():
    """Verify compare_at_scale identifies cheapest and calculates accurate savings."""
    res = compare_at_scale(
        models=["gpt-4o", "gpt-4o-mini"],
        calls_per_day=1000,
        avg_input_tokens=800,
        avg_output_tokens=200,
        months=6,
        growth_rate=0.0
    )

    totals = res["totals"]
    assert "gpt-4o" in totals
    assert "gpt-4o-mini" in totals
    assert pytest.approx(totals["gpt-4o"], rel=1e-2) == 720.00
    assert pytest.approx(totals["gpt-4o-mini"], rel=1e-2) == 43.20
    assert res["cheapest"] == "gpt-4o-mini"
    assert pytest.approx(res["savings_by_switching"], rel=1e-2) == 676.80


def test_forecast_boundary_validation():
    """Verify input validation rules."""
    with pytest.raises(ValueError):
        forecast("gpt-4o", calls_per_day=-10, avg_input_tokens=100, avg_output_tokens=100)

    with pytest.raises(ValueError):
        forecast("gpt-4o", calls_per_day=100, avg_input_tokens=100, avg_output_tokens=100, months=0)

    with pytest.raises(ValueError):
        forecast("gpt-4o", calls_per_day=100, avg_input_tokens=100, avg_output_tokens=100, growth_rate=1.5)
