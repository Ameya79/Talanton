"""Tests for talanton.tracker and talanton.store live cost tracking."""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from talanton.store import TelemetryStore
from talanton.tracker import TalantonTracker


@pytest.fixture
def tmp_db(tmp_path):
    """Creates a temporary database path for test isolation."""
    return tmp_path / "test_telemetry.db"


@pytest.fixture
def store(tmp_db):
    """Creates a fresh TelemetryStore with a temp database."""
    return TelemetryStore(db_path=tmp_db)


@pytest.fixture
def tracker(tmp_db):
    """Creates a fresh TalantonTracker with a temp database."""
    return TalantonTracker(db_path=tmp_db)


# ─── TelemetryStore Tests ─────────────────────────────────────────────────


class TestTelemetryStore:

    def test_record_and_query(self, store):
        """Verify recording a call and querying it back."""
        call_id = store.record_call(
            model="gpt-4o",
            provider="openai",
            input_tokens=500,
            output_tokens=200,
            input_cost=0.00125,
            output_cost=0.002,
            total_cost=0.00325,
            team="backend",
        )

        assert call_id  # UUID string
        calls = store.query_calls(limit=10)
        assert len(calls) == 1
        assert calls[0]["model"] == "gpt-4o"
        assert calls[0]["input_tokens"] == 500
        assert calls[0]["output_tokens"] == 200
        assert calls[0]["team"] == "backend"

    def test_query_filter_by_model(self, store):
        """Verify filtering calls by model."""
        store.record_call(model="gpt-4o", provider="openai", total_cost=0.01)
        store.record_call(model="claude-sonnet-4.5", provider="anthropic", total_cost=0.02)
        store.record_call(model="gpt-4o", provider="openai", total_cost=0.015)

        gpt_calls = store.query_calls(model="gpt-4o")
        assert len(gpt_calls) == 2
        assert all(c["model"] == "gpt-4o" for c in gpt_calls)

    def test_query_filter_by_team(self, store):
        """Verify filtering calls by team."""
        store.record_call(model="gpt-4o", provider="openai", team="frontend", total_cost=0.01)
        store.record_call(model="gpt-4o", provider="openai", team="backend", total_cost=0.02)

        frontend = store.query_calls(team="frontend")
        assert len(frontend) == 1
        assert frontend[0]["team"] == "frontend"

    def test_aggregate_by_model(self, store):
        """Verify model-level aggregation."""
        store.record_call(model="gpt-4o", provider="openai", input_tokens=100, output_tokens=50, total_cost=0.01)
        store.record_call(model="gpt-4o", provider="openai", input_tokens=200, output_tokens=100, total_cost=0.02)
        store.record_call(model="claude-sonnet-4.5", provider="anthropic", input_tokens=300, output_tokens=150, total_cost=0.03)

        agg = store.aggregate_by_model()
        assert len(agg) == 2

        # Sorted by total_cost DESC
        assert agg[0]["model"] in ("gpt-4o", "claude-sonnet-4.5")

    def test_aggregate_by_team(self, store):
        """Verify team-level aggregation."""
        store.record_call(model="gpt-4o", provider="openai", team="alpha", total_cost=0.05)
        store.record_call(model="gpt-4o", provider="openai", team="alpha", total_cost=0.03)
        store.record_call(model="gpt-4o", provider="openai", team="beta", total_cost=0.01)

        agg = store.aggregate_by_team()
        assert len(agg) == 2

    def test_get_total_spend(self, store):
        """Verify total spend calculation for a period."""
        store.record_call(model="gpt-4o", provider="openai", total_cost=0.05)
        store.record_call(model="gpt-4o", provider="openai", total_cost=0.03)

        spend = store.get_total_spend(period="day")
        assert pytest.approx(spend, rel=1e-4) == 0.08

    def test_get_total_spend_by_team(self, store):
        """Verify team-scoped spend calculation."""
        store.record_call(model="gpt-4o", provider="openai", team="alpha", total_cost=0.10)
        store.record_call(model="gpt-4o", provider="openai", team="beta", total_cost=0.20)

        alpha_spend = store.get_total_spend(period="day", team="alpha")
        assert pytest.approx(alpha_spend, rel=1e-4) == 0.10

    def test_clear(self, store):
        """Verify clear deletes all records."""
        store.record_call(model="gpt-4o", provider="openai", total_cost=0.01)
        store.record_call(model="gpt-4o", provider="openai", total_cost=0.02)
        assert len(store.query_calls()) == 2

        store.clear()
        assert len(store.query_calls()) == 0

    def test_export_json(self, store):
        """Verify JSON export contains recorded calls."""
        store.record_call(model="gpt-4o", provider="openai", total_cost=0.01)
        json_str = store.export_json()
        assert "gpt-4o" in json_str
        assert "0.01" in json_str

    def test_export_csv(self, store):
        """Verify CSV export contains headers and data."""
        store.record_call(model="gpt-4o", provider="openai", total_cost=0.01)
        csv_str = store.export_csv()
        assert "model" in csv_str  # Header
        assert "gpt-4o" in csv_str


# ─── TalantonTracker Tests ────────────────────────────────────────────────


class TestTalantonTracker:

    def test_track_auto_calculates_cost(self, tracker):
        """Verify track() auto-calculates cost from pricing table."""
        result = tracker.track(model="gpt-4o", input_tokens=1000, output_tokens=500)

        assert result["model"] == "gpt-4o"
        assert result["provider"] == "openai"
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500
        # gpt-4o: $2.50/1M input, $10.00/1M output
        assert pytest.approx(result["input_cost"], rel=1e-4) == 0.0025
        assert pytest.approx(result["output_cost"], rel=1e-4) == 0.005
        assert result["call_id"]  # Non-empty UUID

    def test_track_with_team(self, tracker):
        """Verify team assignment in tracked calls."""
        result = tracker.track(model="gpt-4o", input_tokens=100, team="ml-team")
        assert result["team"] == "ml-team"

        calls = tracker.get_calls(team="ml-team")
        assert len(calls) == 1
        assert calls[0]["team"] == "ml-team"

    def test_track_unknown_model_records_zero_cost(self, tracker):
        """Verify unknown models record with zero cost and don't crash."""
        result = tracker.track(model="nonexistent-xyz", input_tokens=100, output_tokens=50)
        assert result["total_cost"] == 0.0
        assert result["model"] == "nonexistent-xyz"

    def test_track_from_response_openai_style(self, tracker):
        """Verify auto-extraction from OpenAI-style response objects."""

        class MockUsage:
            prompt_tokens = 150
            completion_tokens = 75

        class MockResponse:
            usage = MockUsage()
            model = "gpt-4o"

        result = tracker.track_from_response(MockResponse())
        assert result["input_tokens"] == 150
        assert result["output_tokens"] == 75
        assert result["model"] == "gpt-4o"

    def test_track_from_response_anthropic_style(self, tracker):
        """Verify auto-extraction from Anthropic-style response objects."""

        class MockUsage:
            input_tokens = 200
            output_tokens = 100

        class MockResponse:
            usage = MockUsage()
            model = "claude-sonnet-4.5"

        result = tracker.track_from_response(MockResponse(), model="claude-sonnet-4.5")
        assert result["input_tokens"] == 200
        assert result["output_tokens"] == 100

    def test_track_from_response_dict_style(self, tracker):
        """Verify auto-extraction from dict-based responses."""
        response = {
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 300,
                "completion_tokens": 150,
            },
        }
        result = tracker.track_from_response(response)
        assert result["input_tokens"] == 300
        assert result["output_tokens"] == 150

    def test_get_summary(self, tracker):
        """Verify summary aggregation."""
        tracker.track(model="gpt-4o", input_tokens=1000, output_tokens=500)
        tracker.track(model="gpt-4o-mini", input_tokens=2000, output_tokens=1000)

        summary = tracker.get_summary(period="day")
        assert summary["total_spend"] > 0
        assert len(summary["by_model"]) == 2

    def test_default_team(self, tmp_db):
        """Verify default team is applied to all calls."""
        t = TalantonTracker(db_path=tmp_db, team="default-team")
        result = t.track(model="gpt-4o", input_tokens=100)
        assert result["team"] == "default-team"

    def test_export_json(self, tracker):
        """Verify JSON export through tracker."""
        tracker.track(model="gpt-4o", input_tokens=100)
        data = tracker.export(fmt="json")
        assert "gpt-4o" in data

    def test_export_csv(self, tracker):
        """Verify CSV export through tracker."""
        tracker.track(model="gpt-4o", input_tokens=100)
        data = tracker.export(fmt="csv")
        assert "model" in data


# ─── Thread Safety Test ───────────────────────────────────────────────────


def test_concurrent_writes(tmp_db):
    """Verify thread-safe concurrent writes to the store."""
    import threading

    tracker = TalantonTracker(db_path=tmp_db)
    errors = []

    def write_batch(thread_id):
        try:
            for i in range(10):
                tracker.track(
                    model="gpt-4o",
                    input_tokens=100 * (thread_id + 1),
                    output_tokens=50,
                    team=f"thread-{thread_id}",
                )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write_batch, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent write errors: {errors}"

    calls = tracker.get_calls(limit=999)
    assert len(calls) == 50  # 5 threads × 10 calls
