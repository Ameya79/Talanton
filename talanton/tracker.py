"""Talanton Tracker — Live cost observability engine for LLM calls.

The central entry point for auto-tracking every LLM call in production.
Records tokens, costs, model, team, and timestamp to a local SQLite store.
Supports auto-extraction from OpenAI and Anthropic response objects.

Usage:
    from talanton import TalantonTracker

    tracker = TalantonTracker()
    tracker.track(model="gpt-4o", input_tokens=500, output_tokens=200)
    tracker.track_from_response(response, model="gpt-4o", provider="openai")

    summary = tracker.get_summary(period="day")
    calls = tracker.get_calls(limit=50, model="gpt-4o")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from talanton.pricing import PricingNotFoundError, get_price
from talanton.providers import detect_provider
from talanton.store import TelemetryStore

logger = logging.getLogger("talanton.tracker")


class TalantonTracker:
    """Live cost observability engine for LLM API calls.

    Tracks every call with token counts, dollar costs, model info, and
    optional team/session identifiers. All data persists to a local SQLite
    database for querying, aggregation, and export.

    Args:
        db_path: Optional custom path for the SQLite database.
            Defaults to ~/.talanton/telemetry.db.
        team: Default team identifier applied to all tracked calls
            unless overridden per-call.
        session_id: Default session identifier for grouping related calls.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        team: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._store = TelemetryStore(db_path=db_path)
        self._default_team = team
        self._default_session_id = session_id

    def track(
        self,
        *,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        team: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Records a single LLM call with automatic cost calculation.

        Looks up pricing from Talanton's pricing table and calculates
        input/output/total costs automatically. If pricing is not found,
        records with zero costs and logs a warning.

        Args:
            model: Model identifier (e.g. 'gpt-4o', 'claude-sonnet-4.5').
            input_tokens: Number of input/prompt tokens consumed.
            output_tokens: Number of output/completion tokens consumed.
            team: Team identifier (overrides tracker default).
            session_id: Session identifier (overrides tracker default).
            metadata: Optional dict of extra metadata to store.

        Returns:
            Dict with the recorded call details including computed costs:
            {
                'call_id': str,
                'model': str,
                'provider': str,
                'input_tokens': int,
                'output_tokens': int,
                'input_cost': float,
                'output_cost': float,
                'total_cost': float,
                'team': str | None,
                'session_id': str | None,
            }
        """
        provider = detect_provider(model)
        input_cost = 0.0
        output_cost = 0.0
        total_cost = 0.0

        try:
            pricing = get_price(model)
            input_cost = round((input_tokens / 1_000_000.0) * pricing["input"], 8)
            output_cost = round((output_tokens / 1_000_000.0) * pricing["output"], 8)
            total_cost = round(input_cost + output_cost, 8)
        except PricingNotFoundError:
            logger.warning(
                "No pricing data for model '%s'. Recording call with $0.00 cost. "
                "Add pricing to pricing_table.json for accurate cost tracking.",
                model,
            )

        effective_team = team or self._default_team
        effective_session = session_id or self._default_session_id

        call_id = self._store.record_call(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            team=effective_team,
            session_id=effective_session,
            metadata=metadata,
        )

        return {
            "call_id": call_id,
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "team": effective_team,
            "session_id": effective_session,
        }

    def track_batch(
        self,
        calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Tracks multiple LLM calls in a single high-throughput batch.

        Auto-calculates costs for each call and commits them in an atomic SQLite transaction.

        Args:
            calls: List of dicts with model, input_tokens, output_tokens, etc.

        Returns:
            List of recorded call dicts with computed costs and call_ids.
        """
        if not calls:
            return []

        prepared = []
        for c in calls:
            m = c.get("model", "unknown")
            provider = detect_provider(m)
            inp = int(c.get("input_tokens", 0))
            out = int(c.get("output_tokens", 0))
            inc = 0.0
            outc = 0.0
            totc = 0.0
            try:
                pricing = get_price(m)
                inc = round((inp / 1_000_000.0) * pricing["input"], 8)
                outc = round((out / 1_000_000.0) * pricing["output"], 8)
                totc = round(inc + outc, 8)
            except PricingNotFoundError:
                pass

            team = c.get("team") or self._default_team
            session_id = c.get("session_id") or self._default_session_id
            prepared.append({
                "model": m,
                "provider": provider,
                "input_tokens": inp,
                "output_tokens": out,
                "input_cost": inc,
                "output_cost": outc,
                "total_cost": totc,
                "team": team,
                "session_id": session_id,
                "metadata": c.get("metadata"),
                "timestamp": c.get("timestamp"),
            })

        call_ids = self._store.record_calls_batch(prepared)
        results = []
        for i, item in enumerate(prepared):
            res = dict(item)
            res["call_id"] = call_ids[i]
            results.append(res)
        return results

    def track_from_response(
        self,
        response: Any,
        *,
        model: str | None = None,
        provider: str | None = None,
        team: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Auto-extracts token usage from an OpenAI or Anthropic response and records it.

        Supports:
            - OpenAI ChatCompletion responses (response.usage.prompt_tokens, .completion_tokens)
            - Anthropic Message responses (response.usage.input_tokens, .output_tokens)
            - Dict-based responses with 'usage' key

        Args:
            response: The raw API response object from OpenAI or Anthropic.
            model: Model name override. Auto-detected from response if not provided.
            provider: Provider override. Auto-detected from model if not provided.
            team: Team identifier override.
            session_id: Session identifier override.
            metadata: Additional metadata dict.

        Returns:
            Recorded call details dict (same as track()).

        Raises:
            ValueError: If token usage cannot be extracted from the response.
        """
        input_tokens = 0
        output_tokens = 0
        detected_model = model

        # Try OpenAI-style response (object with .usage attribute)
        if hasattr(response, "usage") and response.usage is not None:
            usage = response.usage

            # OpenAI format: prompt_tokens, completion_tokens
            if hasattr(usage, "prompt_tokens"):
                input_tokens = int(usage.prompt_tokens or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            # Anthropic format: input_tokens, output_tokens
            elif hasattr(usage, "input_tokens"):
                input_tokens = int(usage.input_tokens or 0)
                output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

            # Try to auto-detect model from response
            if detected_model is None and hasattr(response, "model"):
                detected_model = str(response.model)

        # Try dict-based response
        elif isinstance(response, dict):
            usage = response.get("usage", {})
            if isinstance(usage, dict):
                # OpenAI dict format
                input_tokens = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0))
                output_tokens = int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0))

            if detected_model is None:
                detected_model = response.get("model", "unknown")

        if detected_model is None:
            raise ValueError(
                "Could not detect model from response. Please provide the 'model' argument explicitly."
            )

        if input_tokens == 0 and output_tokens == 0:
            logger.warning(
                "Could not extract token usage from response for model '%s'. "
                "Recording with 0 tokens. Pass input_tokens/output_tokens to track() instead.",
                detected_model,
            )

        return self.track(
            model=detected_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            team=team,
            session_id=session_id,
            metadata=metadata,
        )

    def get_summary(
        self,
        period: str = "day",
        team: str | None = None,
    ) -> dict[str, Any]:
        """Returns an aggregated spend summary for a given period.

        Args:
            period: 'day', 'week', or 'month'.
            team: Optional team filter.

        Returns:
            {
                'period': str,
                'total_spend': float,
                'by_model': [{'model': str, 'total_calls': int, 'total_cost': float, ...}],
                'by_team': [{'team': str, 'total_calls': int, 'total_cost': float, ...}],
            }
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)

        if period == "day":
            since = now - timedelta(days=1)
        elif period == "week":
            since = now - timedelta(weeks=1)
        elif period == "month":
            since = now - timedelta(days=30)
        else:
            raise ValueError(f"Invalid period '{period}'. Must be 'day', 'week', or 'month'.")

        total_spend = self._store.get_total_spend(period=period, team=team)
        by_model = self._store.aggregate_by_model(since=since)
        by_team = self._store.aggregate_by_team(since=since)

        total_calls = sum(m.get("total_calls", 0) for m in by_model)
        total_tokens = sum(
            m.get("total_input_tokens", 0) + m.get("total_output_tokens", 0)
            for m in by_model
        )

        return {
            "period": period,
            "total_spend": round(total_spend, 8),
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "by_model": by_model,
            "by_team": by_team,
        }

    def get_total_spend(
        self,
        period: str = "day",
        team: str | None = None,
    ) -> float:
        """Returns total dollar spend for a given period.

        Args:
            period: 'day', 'week', or 'month'.
            team: Optional team filter.

        Returns:
            Total spend in USD as float.
        """
        return self._store.get_total_spend(period=period, team=team)

    get_spend = get_total_spend

    def get_calls(
        self,
        limit: int = 100,
        model: str | None = None,
        team: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Queries recent tracked calls with optional filters.

        Args:
            limit: Maximum results (default 100).
            model: Filter by model name.
            team: Filter by team.
            session_id: Filter by session.

        Returns:
            List of call record dicts ordered by timestamp descending.
        """
        return self._store.query_calls(
            limit=limit,
            model=model,
            team=team,
            session_id=session_id,
        )

    def export(self, fmt: str = "json", **kwargs: Any) -> str:
        """Exports tracked call data.

        Args:
            fmt: 'json' or 'csv'.
            **kwargs: Passed to store export methods (since, until).

        Returns:
            Formatted string of exported data.
        """
        if fmt == "csv":
            return self._store.export_csv(**kwargs)
        elif fmt == "json":
            return self._store.export_json(**kwargs)
        else:
            raise ValueError(f"Unsupported export format '{fmt}'. Use 'json' or 'csv'.")

    @property
    def store(self) -> TelemetryStore:
        """Direct access to the underlying TelemetryStore for advanced queries."""
        return self._store
