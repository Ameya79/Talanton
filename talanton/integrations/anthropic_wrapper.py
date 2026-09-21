"""Talanton Anthropic Integration — Auto-tracking wrapper for anthropic.Anthropic.

Wraps the Anthropic client to automatically track every messages.create() call
with token counts, costs, and optional budget guardrail checks.

Usage:
    import anthropic
    from talanton import TalantonTracker
    from talanton.integrations import TalantonAnthropic

    tracker = TalantonTracker()
    client = TalantonAnthropic(anthropic.Anthropic(), tracker=tracker)

    # Use exactly like normal Anthropic client — tracking is automatic
    response = client.messages.create(
        model="claude-sonnet-4.5",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
    )
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("talanton.integrations.anthropic")


class _TrackedMessages:
    """Wraps Anthropic's client.messages to intercept create() calls."""

    def __init__(
        self,
        original_messages: Any,
        tracker: Any,
        guardrail: Any | None = None,
        team: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._original = original_messages
        self._tracker = tracker
        self._guardrail = guardrail
        self._team = team
        self._session_id = session_id

    def create(self, **kwargs: Any) -> Any:
        """Intercepts messages.create() with auto-tracking and guardrail checks.

        Args:
            **kwargs: All arguments passed to the original create() method.

        Returns:
            The original Anthropic Message response.

        Raises:
            BudgetExceededError: If a guardrail blocks the call.
        """
        model = kwargs.get("model", "unknown")

        # Pre-call guardrail check
        if self._guardrail is not None:
            from talanton.cost import calculate_cost
            from talanton.counting import count_tokens

            messages = kwargs.get("messages", [])
            try:
                count_result = count_tokens(messages, model)
                estimated_output = kwargs.get("max_tokens", 1024)
                cost_estimate = calculate_cost(
                    count_result["tokens"], model, expected_output_tokens=estimated_output
                )
                check = self._guardrail.check(estimated_cost=cost_estimate["total_cost"])
                if check.is_blocked:
                    from talanton.guardrails import BudgetExceededError
                    raise BudgetExceededError(check)
            except ImportError:
                pass
            except Exception as e:
                if "BudgetExceededError" in type(e).__name__:
                    raise
                logger.warning("Guardrail pre-check failed: %s", e)

        # Execute the actual API call
        response = self._original.create(**kwargs)

        # Post-call tracking
        if self._tracker is not None:
            try:
                self._tracker.track_from_response(
                    response,
                    model=model,
                    provider="anthropic",
                    team=self._team,
                    session_id=self._session_id,
                )
            except Exception as e:
                logger.warning("Failed to track Anthropic call: %s", e)

        return response

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the original messages object."""
        return getattr(self._original, name)


class TalantonAnthropic:
    """Drop-in wrapper for anthropic.Anthropic with automatic cost tracking.

    Wraps the client so that every call to `client.messages.create()`
    is automatically recorded to the Talanton telemetry store.

    Args:
        client: An anthropic.Anthropic() instance.
        tracker: A TalantonTracker instance. Created automatically if not provided.
        guardrail: Optional BudgetGuardrail for pre-call spend checks.
        team: Default team identifier for tracked calls.
        session_id: Default session identifier.
    """

    def __init__(
        self,
        client: Any,
        tracker: Any | None = None,
        guardrail: Any | None = None,
        team: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._client = client
        if tracker is None:
            from talanton.tracker import TalantonTracker
            tracker = TalantonTracker(team=team, session_id=session_id)
        self._tracker = tracker
        self._guardrail = guardrail
        self._messages = _TrackedMessages(
            client.messages,
            tracker=tracker,
            guardrail=guardrail,
            team=team,
            session_id=session_id,
        )

    @property
    def messages(self) -> _TrackedMessages:
        """Returns the tracked messages interface."""
        return self._messages

    @property
    def tracker(self) -> Any:
        """Returns the underlying TalantonTracker."""
        return self._tracker

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the original Anthropic client."""
        return getattr(self._client, name)
