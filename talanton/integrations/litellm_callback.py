"""Talanton LiteLLM Integration — Custom logger for automatic cost tracking.

Implements a LiteLLM-compatible custom logger that auto-records every LLM call
to the Talanton telemetry store with token counts and costs.

Usage:
    import litellm
    from talanton import TalantonTracker
    from talanton.integrations import TalantonLiteLLMCallback

    tracker = TalantonTracker()
    litellm.callbacks = [TalantonLiteLLMCallback(tracker=tracker)]

    # All litellm.completion() calls are now auto-tracked
    response = litellm.completion(model="gpt-4o", messages=[...])
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("talanton.integrations.litellm")


class TalantonLiteLLMCallback:
    """LiteLLM-compatible custom logger for automatic cost tracking.

    Implements log_success_event() and log_failure_event() to record
    every LiteLLM call to the Talanton telemetry store.

    Args:
        tracker: A TalantonTracker instance. Created automatically if not provided.
        guardrail: Optional BudgetGuardrail for spend checks.
        team: Default team identifier for tracked calls.
        session_id: Default session identifier.
    """

    def __init__(
        self,
        tracker: Any | None = None,
        guardrail: Any | None = None,
        team: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if tracker is None:
            from talanton.tracker import TalantonTracker
            tracker = TalantonTracker(team=team, session_id=session_id)
        self._tracker = tracker
        self._guardrail = guardrail
        self._team = team
        self._session_id = session_id

    def log_success_event(self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
        """Called by LiteLLM after a successful LLM call.

        Extracts token usage from the response and records it to the tracker.

        Args:
            kwargs: The original kwargs passed to litellm.completion().
            response_obj: The LiteLLM ModelResponse object.
            start_time: Call start timestamp.
            end_time: Call end timestamp.
        """
        model = kwargs.get("model", "unknown")
        input_tokens = 0
        output_tokens = 0

        # Extract from LiteLLM response object
        if hasattr(response_obj, "usage") and response_obj.usage is not None:
            usage = response_obj.usage
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        elif isinstance(response_obj, dict):
            usage = response_obj.get("usage", {})
            if isinstance(usage, dict):
                input_tokens = int(usage.get("prompt_tokens", 0))
                output_tokens = int(usage.get("completion_tokens", 0))

        try:
            self._tracker.track(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                team=self._team,
                session_id=self._session_id,
                metadata={
                    "source": "litellm",
                    "start_time": str(start_time) if start_time else None,
                    "end_time": str(end_time) if end_time else None,
                },
            )
        except Exception as e:
            logger.warning("Failed to track LiteLLM call: %s", e)

    def log_failure_event(self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
        """Called by LiteLLM after a failed LLM call.

        Logs the failure for debugging but does not record costs.

        Args:
            kwargs: The original kwargs passed to litellm.completion().
            response_obj: The error response.
            start_time: Call start timestamp.
            end_time: Call end timestamp.
        """
        model = kwargs.get("model", "unknown")
        logger.debug(
            "LiteLLM call failed for model '%s'. Not recording cost. Response: %s",
            model,
            response_obj,
        )

    def log_pre_api_call(self, model: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> None:
        """Called by LiteLLM before making an API call. Used for guardrail checks.

        Args:
            model: The model being called.
            messages: The messages being sent.
            kwargs: Additional kwargs.
        """
        if self._guardrail is not None:
            from talanton.cost import calculate_cost
            from talanton.counting import count_tokens

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

    @property
    def tracker(self) -> Any:
        """Returns the underlying TalantonTracker."""
        return self._tracker
