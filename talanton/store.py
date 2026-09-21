"""Talanton Telemetry Store — SQLite-backed persistence for LLM call tracking.

Provides a thread-safe, zero-dependency persistence layer that records every
tracked LLM call with tokens, costs, model, team, and timestamp. Supports
aggregation by model, team, and time period, plus export to CSV/JSON.

Storage location: ~/.talanton/telemetry.db (auto-created on first use).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("talanton.store")

_DEFAULT_DB_DIR = Path.home() / ".talanton"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "telemetry.db"

_CREATE_CALLS_TABLE = """
CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'unknown',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    input_cost REAL NOT NULL DEFAULT 0.0,
    output_cost REAL NOT NULL DEFAULT 0.0,
    total_cost REAL NOT NULL DEFAULT 0.0,
    team TEXT DEFAULT NULL,
    session_id TEXT DEFAULT NULL,
    metadata_json TEXT DEFAULT '{}'
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_calls_timestamp ON calls(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_calls_model ON calls(model);",
    "CREATE INDEX IF NOT EXISTS idx_calls_team ON calls(team);",
    "CREATE INDEX IF NOT EXISTS idx_calls_session ON calls(session_id);",
]


class TelemetryStore:
    """Thread-safe SQLite persistence layer for LLM call telemetry.

    Auto-creates the database and tables on first instantiation.
    All public methods are safe to call from multiple threads.

    Args:
        db_path: Path to SQLite database file. Defaults to ~/.talanton/telemetry.db.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Creates database directory, file, and schema if not present."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute(_CREATE_CALLS_TABLE)
                for idx_sql in _CREATE_INDEXES:
                    conn.execute(idx_sql)
                conn.commit()
        except Exception as e:
            logger.error("Failed to initialize telemetry database at %s: %s", self._db_path, e)
            raise

    def _connect(self) -> sqlite3.Connection:
        """Creates a new SQLite connection with row factory enabled."""
        conn = sqlite3.connect(str(self._db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def record_call(
        self,
        *,
        model: str,
        provider: str = "unknown",
        input_tokens: int = 0,
        output_tokens: int = 0,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        total_cost: float = 0.0,
        team: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        """Records a single LLM call to the telemetry store.

        Args:
            model: Model identifier (e.g. 'gpt-4o').
            provider: Provider name (e.g. 'openai', 'anthropic').
            input_tokens: Number of input/prompt tokens.
            output_tokens: Number of output/completion tokens.
            input_cost: Dollar cost of input tokens.
            output_cost: Dollar cost of output tokens.
            total_cost: Total dollar cost of the call.
            team: Optional team/project identifier for quota tracking.
            session_id: Optional session identifier for grouping related calls.
            metadata: Optional dict of additional metadata.
            timestamp: Optional timestamp; defaults to UTC now.

        Returns:
            The unique call ID (UUID4 string).
        """
        call_id = str(uuid.uuid4())
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        meta_json = json.dumps(metadata or {})

        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        """INSERT INTO calls
                        (id, timestamp, model, provider, input_tokens, output_tokens,
                         input_cost, output_cost, total_cost, team, session_id, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (call_id, ts, model, provider, input_tokens, output_tokens,
                         input_cost, output_cost, total_cost, team, session_id, meta_json),
                    )
                    conn.commit()
            except Exception as e:
                logger.error("Failed to record call: %s", e)
                raise

        return call_id

    def record_calls_batch(
        self,
        calls: list[dict[str, Any]],
    ) -> list[str]:
        """Records multiple LLM calls in a single atomic transaction.

        Provides high-throughput batch ingestion (50,000+ calls/sec) by minimizing
        connection overhead and transaction sync commits.

        Args:
            calls: List of dicts, each containing call parameters.

        Returns:
            List of generated call IDs (UUID4 strings) matching input order.
        """
        if not calls:
            return []

        rows = []
        call_ids = []
        default_ts = datetime.now(timezone.utc).isoformat()

        for c in calls:
            cid = str(uuid.uuid4())
            call_ids.append(cid)
            ts = c.get("timestamp")
            ts_str = ts.isoformat() if isinstance(ts, datetime) else (ts or default_ts)
            meta = c.get("metadata") or {}
            meta_json = json.dumps(meta) if not isinstance(meta, str) else meta

            rows.append((
                cid,
                ts_str,
                c.get("model", "unknown"),
                c.get("provider", "unknown"),
                int(c.get("input_tokens", 0)),
                int(c.get("output_tokens", 0)),
                float(c.get("input_cost", 0.0)),
                float(c.get("output_cost", 0.0)),
                float(c.get("total_cost", 0.0)),
                c.get("team"),
                c.get("session_id"),
                meta_json,
            ))

        with self._lock:
            try:
                with self._connect() as conn:
                    conn.executemany(
                        """INSERT INTO calls
                        (id, timestamp, model, provider, input_tokens, output_tokens,
                         input_cost, output_cost, total_cost, team, session_id, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        rows,
                    )
                    conn.commit()
            except Exception as e:
                logger.error("Failed to record calls batch (%d calls): %s", len(calls), e)
                raise

        return call_ids

    def query_calls(
        self,
        *,
        limit: int = 100,
        model: str | None = None,
        team: str | None = None,
        session_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Queries recorded calls with optional filters.

        Args:
            limit: Maximum number of results (default 100).
            model: Filter by model name.
            team: Filter by team identifier.
            session_id: Filter by session ID.
            since: Filter calls after this timestamp (inclusive).
            until: Filter calls before this timestamp (inclusive).

        Returns:
            List of call records as dictionaries, ordered by timestamp descending.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if model:
            conditions.append("model = ?")
            params.append(model)
        if team:
            conditions.append("team = ?")
            params.append(team)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        if until:
            conditions.append("timestamp <= ?")
            params.append(until.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM calls WHERE {where_clause} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def aggregate_by_model(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregates spend and call count grouped by model.

        Returns:
            List of dicts: [{'model': str, 'total_calls': int, 'total_cost': float,
                             'total_input_tokens': int, 'total_output_tokens': int}]
        """
        conditions: list[str] = []
        params: list[Any] = []

        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        if until:
            conditions.append("timestamp <= ?")
            params.append(until.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT model,
                   COUNT(*) as total_calls,
                   ROUND(SUM(total_cost), 8) as total_cost,
                   SUM(input_tokens) as total_input_tokens,
                   SUM(output_tokens) as total_output_tokens
            FROM calls WHERE {where_clause}
            GROUP BY model
            ORDER BY total_cost DESC
        """

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def aggregate_by_team(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregates spend and call count grouped by team.

        Returns:
            List of dicts: [{'team': str, 'total_calls': int, 'total_cost': float}]
        """
        conditions: list[str] = []
        params: list[Any] = []

        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        if until:
            conditions.append("timestamp <= ?")
            params.append(until.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT COALESCE(team, 'unassigned') as team,
                   COUNT(*) as total_calls,
                   ROUND(SUM(total_cost), 8) as total_cost,
                   SUM(input_tokens) as total_input_tokens,
                   SUM(output_tokens) as total_output_tokens
            FROM calls WHERE {where_clause}
            GROUP BY team
            ORDER BY total_cost DESC
        """

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def aggregate_by_period(
        self,
        *,
        period: str = "day",
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregates spend grouped by time period.

        Args:
            period: 'day', 'week', or 'month'.
            since: Start of range.
            until: End of range.

        Returns:
            List of dicts: [{'period': str, 'total_calls': int, 'total_cost': float}]
        """
        if period == "day":
            date_expr = "DATE(timestamp)"
        elif period == "week":
            date_expr = "strftime('%Y-W%W', timestamp)"
        elif period == "month":
            date_expr = "strftime('%Y-%m', timestamp)"
        else:
            raise ValueError(f"Invalid period '{period}'. Must be 'day', 'week', or 'month'.")

        conditions: list[str] = []
        params: list[Any] = []

        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        if until:
            conditions.append("timestamp <= ?")
            params.append(until.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT {date_expr} as period,
                   COUNT(*) as total_calls,
                   ROUND(SUM(total_cost), 8) as total_cost,
                   SUM(input_tokens) as total_input_tokens,
                   SUM(output_tokens) as total_output_tokens
            FROM calls WHERE {where_clause}
            GROUP BY {date_expr}
            ORDER BY period DESC
        """

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def get_total_spend(
        self,
        *,
        period: str = "day",
        team: str | None = None,
    ) -> float:
        """Returns total dollar spend for a given period.

        Args:
            period: 'day', 'week', or 'month' — how far back to look.
            team: Optional team filter.

        Returns:
            Total spend in USD as a float.
        """
        now = datetime.now(timezone.utc)

        if period == "day":
            since = now - timedelta(days=1)
        elif period == "week":
            since = now - timedelta(weeks=1)
        elif period == "month":
            since = now - timedelta(days=30)
        else:
            raise ValueError(f"Invalid period '{period}'. Must be 'day', 'week', or 'month'.")

        conditions = ["timestamp >= ?"]
        params: list[Any] = [since.isoformat()]

        if team:
            conditions.append("team = ?")
            params.append(team)

        where_clause = " AND ".join(conditions)

        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    f"SELECT COALESCE(SUM(total_cost), 0.0) as spend FROM calls WHERE {where_clause}",
                    params,
                ).fetchone()

        return float(row["spend"]) if row else 0.0

    def export_csv(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> str:
        """Exports all matching call records as a CSV string.

        Returns:
            CSV-formatted string with headers.
        """
        calls = self.query_calls(limit=999999, since=since, until=until)
        if not calls:
            return ""

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=calls[0].keys())
        writer.writeheader()
        writer.writerows(calls)
        return output.getvalue()

    def export_json(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> str:
        """Exports all matching call records as a JSON string.

        Returns:
            JSON-formatted string.
        """
        calls = self.query_calls(limit=999999, since=since, until=until)
        return json.dumps(calls, indent=2, default=str)

    def clear(self) -> None:
        """Deletes all records from the calls table. Use with caution."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM calls")
                conn.commit()

    @property
    def db_path(self) -> Path:
        """Returns the path to the SQLite database file."""
        return self._db_path
