"""Talanton LangChain Integration — Callback handler for automatic cost tracking.

Implements a LangChain BaseCallbackHandler that auto-records every LLM call
to the Talanton telemetry store with token counts and costs.

Usage:
    from langchain_openai import ChatOpenAI
    from talanton import TalantonTracker
    from talanton.integrations import TalantonCallbackHandler

    tracker = TalantonTracker()
    handler = TalantonCallbackHandler(tracker=tracker)

    llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
    response = llm.invoke("Hello!")
    # Call is automatically tracked with tokens and costs
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

logger = logging.getLogger("talanton.integrations.langchain")


class TalantonCallbackHandler:
    """LangChain-compatible callback handler for automatic cost tracking.

    Implements the LangChain callback interface (on_llm_start, on_llm_end)
    to auto-record every LLM invocation to the Talanton telemetry store.

    This handler is compatible with both LangChain and LangChain Expression
    Language (LCEL) chains.

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
        # Track pending runs to associate start/end events
        self._pending_runs: dict[str, dict[str, Any]] = {}

    # ── LangChain Callback Interface ──────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call starts. Stores the model info for on_llm_end.

        Args:
            serialized: Serialized LLM config dict.
            prompts: List of prompt strings.
            run_id: Unique run identifier.
            **kwargs: Additional LangChain kwargs (parent_run_id, tags, etc.).
        """
        run_key = str(run_id) if run_id else "unknown"
        model = kwargs.get("invocation_params", {}).get("model_name", "")
        if not model:
            model = kwargs.get("invocation_params", {}).get("model", "")
        if not model:
            # Try to extract from serialized config
            model = serialized.get("kwargs", {}).get("model_name", "")
            if not model:
                model = serialized.get("kwargs", {}).get("model", "unknown")

        self._pending_runs[run_key] = {
            "model": model,
            "prompts": prompts,
        }

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call completes. Records tokens and cost.

        Args:
            response: The LLMResult from LangChain.
            run_id: Unique run identifier.
            **kwargs: Additional LangChain kwargs.
        """
        run_key = str(run_id) if run_id else "unknown"
        run_info = self._pending_runs.pop(run_key, {})
        model = run_info.get("model", "unknown")

        # Extract token usage from LangChain's LLMResult
        input_tokens = 0
        output_tokens = 0

        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
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
                metadata={"source": "langchain", "run_id": run_key},
            )
        except Exception as e:
            logger.warning("Failed to track LangChain LLM call: %s", e)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call errors. Cleans up pending run state.

        Args:
            error: The exception that occurred.
            run_id: Unique run identifier.
            **kwargs: Additional kwargs.
        """
        run_key = str(run_id) if run_id else "unknown"
        self._pending_runs.pop(run_key, None)
        logger.debug("LangChain LLM call errored (run=%s): %s", run_key, error)

    # ── No-op stubs for other LangChain events ───────────────────────────
    # LangChain requires these to exist on callback handlers.

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        pass

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        pass

    @property
    def tracker(self) -> Any:
        """Returns the underlying TalantonTracker."""
        return self._tracker
