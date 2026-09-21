"""Talanton OpenAI Integration — Auto-tracking wrapper for openai.OpenAI.

Wraps the OpenAI client to automatically track every chat completion call
with token counts, costs, and optional budget guardrail checks.

Usage:
    import openai
    from talanton import TalantonTracker
    from talanton.integrations import TalantonOpenAI

    tracker = TalantonTracker()
    client = TalantonOpenAI(openai.OpenAI(), tracker=tracker)

    # Use exactly like normal OpenAI client — tracking is automatic
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("talanton.integrations.openai")


class _TrackedCompletions:
    """Wraps OpenAI's chat.completions to intercept create() calls."""

    def __init__(
        self,
        original_completions: Any,
        tracker: Any,
        guardrail: Any | None = None,
        team: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._original = original_completions
        self._tracker = tracker
        self._guardrail = guardrail
        self._team = team
        self._session_id = session_id

    def create(self, **kwargs: Any) -> Any:
        """Intercepts chat.completions.create() with auto-tracking and guardrail checks.

        Args:
            **kwargs: All arguments passed to the original create() method.

        Returns:
            The original OpenAI ChatCompletion response.

        Raises:
            BudgetExceededError: If a guardrail is configured and the hard limit is exceeded.
        """
        model = kwargs.get("model", "unknown")

        # Pre-call guardrail check
        if self._guardrail is not None:
            from talanton.cost import calculate_cost
            from talanton.counting import count_tokens

            messages = kwargs.get("messages", [])
            try:
                count_result = count_tokens(messages, model)
                estimated_output = kwargs.get("max_tokens", 500)
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
                    provider="openai",
                    team=self._team,
                    session_id=self._session_id,
                )
            except Exception as e:
                logger.warning("Failed to track OpenAI call: %s", e)

        return response

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the original completions object."""
        return getattr(self._original, name)


class _TrackedChat:
    """Wraps OpenAI's client.chat to inject tracked completions."""

    def __init__(
        self,
        original_chat: Any,
        tracker: Any,
        guardrail: Any | None = None,
        team: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._original = original_chat
        self._completions = _TrackedCompletions(
            original_chat.completions,
            tracker=tracker,
            guardrail=guardrail,
            team=team,
            session_id=session_id,
        )

    @property
    def completions(self) -> _TrackedCompletions:
        return self._completions

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the original chat object."""
        return getattr(self._original, name)


class TalantonOpenAI:
    """Drop-in wrapper for openai.OpenAI with automatic cost tracking.

    Wraps the client so that every call to `client.chat.completions.create()`
    is automatically recorded to the Talanton telemetry store.

    Args:
        client: An openai.OpenAI() instance.
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
        self._chat = _TrackedChat(
            client.chat,
            tracker=tracker,
            guardrail=guardrail,
            team=team,
            session_id=session_id,
        )

    @property
    def chat(self) -> _TrackedChat:
        """Returns the tracked chat interface."""
        return self._chat

    @property
    def tracker(self) -> Any:
        """Returns the underlying TalantonTracker."""
        return self._tracker

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the original OpenAI client."""
        return getattr(self._client, name)
