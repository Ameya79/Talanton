"""Talanton Budget Guardrails — Hard/soft spend limits, alerts, and per-team quotas.

Enforces budget policies on LLM spend by querying the telemetry store for
current period spend and returning ALLOW / WARN / BLOCK decisions. Supports
pluggable alert callbacks for logging, exceptions, and webhook notifications.

Usage:
    from talanton import TalantonTracker, BudgetGuardrail

    tracker = TalantonTracker()
    guardrail = BudgetGuardrail(
        tracker=tracker,
        soft_limit=50.0,
        hard_limit=100.0,
        period="day",
        on_soft_limit=log_warning,
    )

    result = guardrail.check(estimated_cost=0.05)
    if result.status == "BLOCK":
        raise Exception("Budget exceeded!")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("talanton.guardrails")


@dataclass
class GuardrailResult:
    """Result of a budget guardrail check.

    Attributes:
        status: 'ALLOW', 'WARN', or 'BLOCK'.
        current_spend: Current period spend in USD.
        soft_limit: Configured soft limit in USD.
        hard_limit: Configured hard limit in USD.
        remaining_soft: Remaining budget before soft limit.
        remaining_hard: Remaining budget before hard limit.
        period: The budget period ('day', 'week', 'month').
        team: The team this check applies to (None for global).
    """

    status: str  # 'ALLOW' | 'WARN' | 'BLOCK'
    current_spend: float
    soft_limit: float
    hard_limit: float
    remaining_soft: float
    remaining_hard: float
    period: str
    team: str | None = None

    @property
    def is_allowed(self) -> bool:
        """Returns True if the call should proceed (ALLOW or WARN)."""
        return self.status in ("ALLOW", "WARN")

    @property
    def is_blocked(self) -> bool:
        """Returns True if the call should be blocked."""
        return self.status == "BLOCK"


# ─── Pre-built alert callbacks ──────────────────────────────────────────────


def log_warning(result: GuardrailResult) -> None:
    """Logs a warning when soft or hard limit is reached."""
    if result.status == "WARN":
        logger.warning(
            "⚠️ SOFT BUDGET LIMIT reached: $%.4f / $%.2f (%s, team=%s). "
            "Remaining before hard limit: $%.4f",
            result.current_spend,
            result.soft_limit,
            result.period,
            result.team or "global",
            result.remaining_hard,
        )
    elif result.status == "BLOCK":
        logger.error(
            "🛑 HARD BUDGET LIMIT exceeded: $%.4f / $%.2f (%s, team=%s). "
            "Call BLOCKED.",
            result.current_spend,
            result.hard_limit,
            result.period,
            result.team or "global",
        )


def raise_exception(result: GuardrailResult) -> None:
    """Raises a BudgetExceededError when the hard limit is reached."""
    if result.status == "BLOCK":
        raise BudgetExceededError(result)


def webhook_alert(url: str) -> Callable[[GuardrailResult], None]:
    """Returns a callback that POSTs guardrail alerts to a webhook URL.

    Args:
        url: The webhook endpoint URL.

    Returns:
        A callback function compatible with on_soft_limit / on_hard_limit.
    """

    def _send(result: GuardrailResult) -> None:
        try:
            import httpx

            payload = {
                "event": "talanton_budget_alert",
                "status": result.status,
                "current_spend": result.current_spend,
                "soft_limit": result.soft_limit,
                "hard_limit": result.hard_limit,
                "remaining_hard": result.remaining_hard,
                "period": result.period,
                "team": result.team,
            }
            httpx.post(url, json=payload, timeout=5.0)
        except Exception as e:
            logger.error("Failed to send webhook alert to %s: %s", url, e)

    return _send


# ─── Exceptions ──────────────────────────────────────────────────────────────


class BudgetExceededError(Exception):
    """Raised when an LLM call is blocked by a hard budget limit."""

    def __init__(self, result: GuardrailResult) -> None:
        self.result = result
        super().__init__(
            f"Budget BLOCKED: ${result.current_spend:.4f} spent "
            f"(hard limit: ${result.hard_limit:.2f}, "
            f"period: {result.period}, team: {result.team or 'global'})"
        )


# ─── Main Guardrail Class ────────────────────────────────────────────────────


class BudgetGuardrail:
    """Enforces spend limits with configurable alert callbacks.

    Queries the tracker's telemetry store for current period spend and
    returns a GuardrailResult indicating whether the next call should
    proceed (ALLOW), proceed with warning (WARN), or be blocked (BLOCK).

    Args:
        tracker: A TalantonTracker instance for querying spend data.
        soft_limit: Dollar threshold that triggers a warning (default 50.0).
        hard_limit: Dollar threshold that blocks calls (default 100.0).
        period: Budget period — 'day', 'week', or 'month' (default 'day').
        team: Optional team to scope the budget to.
        on_soft_limit: Callback invoked when soft limit is reached.
        on_hard_limit: Callback invoked when hard limit is reached.
    """

    def __init__(
        self,
        tracker: Any,  # TalantonTracker — using Any to avoid circular import
        *,
        soft_limit: float = 50.0,
        hard_limit: float = 100.0,
        period: str = "day",
        team: str | None = None,
        on_soft_limit: Callable[[GuardrailResult], None] | None = None,
        on_hard_limit: Callable[[GuardrailResult], None] | None = None,
    ) -> None:
        if soft_limit < 0:
            raise ValueError(f"soft_limit must be non-negative, got {soft_limit}")
        if hard_limit < 0:
            raise ValueError(f"hard_limit must be non-negative, got {hard_limit}")
        if soft_limit > hard_limit:
            raise ValueError(
                f"soft_limit (${soft_limit}) must be <= hard_limit (${hard_limit})"
            )
        if period not in ("day", "week", "month"):
            raise ValueError(f"period must be 'day', 'week', or 'month', got '{period}'")

        self._tracker = tracker
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit
        self._period = period
        self._team = team
        self._on_soft_limit = on_soft_limit or log_warning
        self._on_hard_limit = on_hard_limit or log_warning

    def check(self, estimated_cost: float = 0.0) -> GuardrailResult:
        """Checks whether the next call should proceed based on current spend.

        Args:
            estimated_cost: The estimated cost of the next call.
                Used to determine if the call would push spend over the limit.

        Returns:
            GuardrailResult with status 'ALLOW', 'WARN', or 'BLOCK'.
        """
        current_spend = self._tracker.store.get_total_spend(
            period=self._period,
            team=self._team,
        )

        projected_spend = current_spend + estimated_cost
        remaining_soft = max(0.0, self._soft_limit - current_spend)
        remaining_hard = max(0.0, self._hard_limit - current_spend)

        if projected_spend >= self._hard_limit:
            status = "BLOCK"
        elif projected_spend >= self._soft_limit:
            status = "WARN"
        else:
            status = "ALLOW"

        result = GuardrailResult(
            status=status,
            current_spend=round(current_spend, 8),
            soft_limit=self._soft_limit,
            hard_limit=self._hard_limit,
            remaining_soft=round(remaining_soft, 8),
            remaining_hard=round(remaining_hard, 8),
            period=self._period,
            team=self._team,
        )

        # Fire callbacks
        if status == "WARN" and self._on_soft_limit:
            try:
                self._on_soft_limit(result)
            except Exception as e:
                logger.error("Soft limit callback failed: %s", e)
        elif status == "BLOCK" and self._on_hard_limit:
            try:
                self._on_hard_limit(result)
            except BudgetExceededError:
                raise
            except Exception as e:
                logger.error("Hard limit callback failed: %s", e)

        return result

    @property
    def soft_limit(self) -> float:
        """Returns the configured soft limit."""
        return self._soft_limit

    @soft_limit.setter
    def soft_limit(self, value: float) -> None:
        """Updates the soft limit."""
        if value < 0:
            raise ValueError(f"soft_limit must be non-negative, got {value}")
        if value > self._hard_limit:
            raise ValueError(f"soft_limit (${value}) must be <= hard_limit (${self._hard_limit})")
        self._soft_limit = value

    @property
    def hard_limit(self) -> float:
        """Returns the configured hard limit."""
        return self._hard_limit

    @hard_limit.setter
    def hard_limit(self, value: float) -> None:
        """Updates the hard limit."""
        if value < 0:
            raise ValueError(f"hard_limit must be non-negative, got {value}")
        if value < self._soft_limit:
            raise ValueError(f"hard_limit (${value}) must be >= soft_limit (${self._soft_limit})")
        self._hard_limit = value

    @property
    def period(self) -> str:
        """Returns the budget period."""
        return self._period

    @property
    def team(self) -> str | None:
        """Returns the team scope."""
        return self._team
