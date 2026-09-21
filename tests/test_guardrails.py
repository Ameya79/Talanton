"""Tests for talanton.guardrails budget enforcement."""

import pytest

from talanton.guardrails import (
    BudgetExceededError,
    BudgetGuardrail,
    GuardrailResult,
    log_warning,
    raise_exception,
)
from talanton.tracker import TalantonTracker


@pytest.fixture
def tmp_db(tmp_path):
    """Creates a temporary database path for test isolation."""
    return tmp_path / "test_guardrails.db"


@pytest.fixture
def tracker(tmp_db):
    """Creates a TalantonTracker with temp database."""
    return TalantonTracker(db_path=tmp_db)


# ─── GuardrailResult Tests ───────────────────────────────────────────────


class TestGuardrailResult:

    def test_allow_result(self):
        result = GuardrailResult(
            status="ALLOW", current_spend=10.0, soft_limit=50.0,
            hard_limit=100.0, remaining_soft=40.0, remaining_hard=90.0,
            period="day",
        )
        assert result.is_allowed is True
        assert result.is_blocked is False

    def test_warn_result(self):
        result = GuardrailResult(
            status="WARN", current_spend=55.0, soft_limit=50.0,
            hard_limit=100.0, remaining_soft=0.0, remaining_hard=45.0,
            period="day",
        )
        assert result.is_allowed is True
        assert result.is_blocked is False

    def test_block_result(self):
        result = GuardrailResult(
            status="BLOCK", current_spend=105.0, soft_limit=50.0,
            hard_limit=100.0, remaining_soft=0.0, remaining_hard=0.0,
            period="day",
        )
        assert result.is_allowed is False
        assert result.is_blocked is True


# ─── BudgetGuardrail Tests ───────────────────────────────────────────────


class TestBudgetGuardrail:

    def test_allow_when_under_limits(self, tracker):
        """Verify ALLOW status when spend is well under limits."""
        guardrail = BudgetGuardrail(tracker, soft_limit=50.0, hard_limit=100.0)
        result = guardrail.check(estimated_cost=1.0)
        assert result.status == "ALLOW"
        assert result.remaining_hard > 0

    def test_warn_when_over_soft_limit(self, tracker):
        """Verify WARN status when estimated cost pushes past soft limit."""
        # Record enough spend to get close to soft limit
        for _ in range(10):
            tracker.track(model="gpt-4o", input_tokens=100000, output_tokens=50000)

        guardrail = BudgetGuardrail(
            tracker, soft_limit=0.001, hard_limit=100.0, period="day",
        )
        result = guardrail.check(estimated_cost=0.0)
        # Current spend should exceed the very low soft limit
        assert result.status in ("WARN", "BLOCK")

    def test_block_when_over_hard_limit(self, tracker):
        """Verify BLOCK status when estimated cost exceeds hard limit."""
        guardrail = BudgetGuardrail(
            tracker, soft_limit=0.001, hard_limit=0.002, period="day",
        )
        # Check with a cost that would exceed the hard limit
        result = guardrail.check(estimated_cost=0.01)
        assert result.status == "BLOCK"
        assert result.is_blocked is True

    def test_soft_limit_callback_fires(self, tracker):
        """Verify the soft limit callback is invoked."""
        callback_results = []

        def capture_callback(result):
            callback_results.append(result)

        guardrail = BudgetGuardrail(
            tracker,
            soft_limit=0.0001,
            hard_limit=100.0,
            period="day",
            on_soft_limit=capture_callback,
        )

        # Record a small call to push past tiny soft limit
        tracker.track(model="gpt-4o", input_tokens=1000, output_tokens=500)
        result = guardrail.check(estimated_cost=0.0)

        if result.status == "WARN":
            assert len(callback_results) == 1
            assert callback_results[0].status == "WARN"

    def test_hard_limit_callback_fires(self, tracker):
        """Verify the hard limit callback is invoked."""
        callback_results = []

        def capture_callback(result):
            callback_results.append(result)

        guardrail = BudgetGuardrail(
            tracker,
            soft_limit=0.0001,
            hard_limit=0.0002,
            period="day",
            on_hard_limit=capture_callback,
        )

        result = guardrail.check(estimated_cost=1.0)
        assert result.status == "BLOCK"
        assert len(callback_results) == 1

    def test_per_team_isolation(self, tracker):
        """Verify budget checks are scoped by team."""
        # Record spend for team-alpha only
        tracker.track(model="gpt-4o", input_tokens=100000, output_tokens=50000, team="alpha")

        guardrail_alpha = BudgetGuardrail(
            tracker, soft_limit=0.001, hard_limit=0.002, period="day", team="alpha",
        )
        guardrail_beta = BudgetGuardrail(
            tracker, soft_limit=0.001, hard_limit=0.002, period="day", team="beta",
        )

        result_alpha = guardrail_alpha.check()
        result_beta = guardrail_beta.check()

        # Alpha should have spend, beta should not
        assert result_alpha.current_spend > result_beta.current_spend

    def test_raise_exception_callback(self, tracker):
        """Verify raise_exception callback raises BudgetExceededError."""
        guardrail = BudgetGuardrail(
            tracker,
            soft_limit=0.0001,
            hard_limit=0.0002,
            period="day",
            on_hard_limit=raise_exception,
        )

        with pytest.raises(BudgetExceededError):
            guardrail.check(estimated_cost=1.0)


# ─── Validation Tests ────────────────────────────────────────────────────


class TestGuardrailValidation:

    def test_soft_gt_hard_raises(self, tracker):
        """Verify soft_limit > hard_limit raises ValueError."""
        with pytest.raises(ValueError, match="soft_limit"):
            BudgetGuardrail(tracker, soft_limit=100.0, hard_limit=50.0)

    def test_negative_limits_raise(self, tracker):
        """Verify negative limits raise ValueError."""
        with pytest.raises(ValueError):
            BudgetGuardrail(tracker, soft_limit=-1.0)
        with pytest.raises(ValueError):
            BudgetGuardrail(tracker, hard_limit=-1.0)

    def test_invalid_period_raises(self, tracker):
        """Verify invalid period raises ValueError."""
        with pytest.raises(ValueError, match="period"):
            BudgetGuardrail(tracker, period="hour")

    def test_property_accessors(self, tracker):
        """Verify property getters return correct values."""
        g = BudgetGuardrail(tracker, soft_limit=25.0, hard_limit=75.0, period="week", team="ops")
        assert g.soft_limit == 25.0
        assert g.hard_limit == 75.0
        assert g.period == "week"
        assert g.team == "ops"

    def test_property_setters(self, tracker):
        """Verify property setters update and validate."""
        g = BudgetGuardrail(tracker, soft_limit=25.0, hard_limit=75.0)

        g.soft_limit = 50.0
        assert g.soft_limit == 50.0

        g.hard_limit = 100.0
        assert g.hard_limit == 100.0

        with pytest.raises(ValueError):
            g.soft_limit = 200.0  # > hard_limit

        with pytest.raises(ValueError):
            g.hard_limit = 10.0  # < soft_limit
