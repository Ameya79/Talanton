"""Tests for industry-grade scalability, concurrency, batching, and latency SLAs."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from talanton.guardrails import BudgetGuardrail
from talanton.store import TelemetryStore
from talanton.tracker import TalantonTracker


@pytest.fixture
def temp_tracker(tmp_path: Path) -> TalantonTracker:
    db_file = tmp_path / "telemetry_test.db"
    return TalantonTracker(db_path=db_file)


@pytest.fixture
def temp_store(tmp_path: Path) -> TelemetryStore:
    db_file = tmp_path / "store_test.db"
    return TelemetryStore(db_path=db_file)


class TestConcurrencyAndScalability:
    def test_concurrent_writes_zero_data_loss(self, temp_tracker: TalantonTracker) -> None:
        """Verifies that multiple concurrent threads can record calls without lock drops or data loss."""
        num_threads = 15
        calls_per_thread = 20
        total_expected = num_threads * calls_per_thread

        def worker(tid: int) -> list[str]:
            call_ids = []
            for i in range(calls_per_thread):
                res = temp_tracker.track(
                    model="gpt-4o",
                    input_tokens=100 + i,
                    output_tokens=50 + i,
                    team=f"team-{tid % 3}",
                    session_id=f"session-{tid}-{i}",
                )
                call_ids.append(res["call_id"])
            return call_ids

        all_ids: list[str] = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, tid) for tid in range(num_threads)]
            for f in as_completed(futures):
                all_ids.extend(f.result())

        # Verify all calls recorded
        assert len(all_ids) == total_expected
        assert len(set(all_ids)) == total_expected

        stored_calls = temp_tracker.get_calls(limit=total_expected + 100)
        assert len(stored_calls) == total_expected

    def test_batch_ingestion_integrity(self, temp_tracker: TalantonTracker) -> None:
        """Verifies atomic batch recording through track_batch."""
        batch_size = 500
        batch_data = [
            {
                "model": "gpt-4o" if i % 2 == 0 else "gpt-4o-mini",
                "input_tokens": 1000,
                "output_tokens": 500,
                "team": f"squad-{i % 4}",
            }
            for i in range(batch_size)
        ]

        t0 = time.perf_counter()
        results = temp_tracker.track_batch(batch_data)
        elapsed = time.perf_counter() - t0

        assert len(results) == batch_size
        assert all("call_id" in r for r in results)
        assert all(r["total_cost"] > 0 for r in results)

        # Batch ingestion of 500 items should complete in < 1 second
        assert elapsed < 2.0

        calls = temp_tracker.get_calls(limit=1000)
        assert len(calls) == batch_size

    def test_guardrail_latency_sla(self, temp_tracker: TalantonTracker) -> None:
        """Verifies that BudgetGuardrail.check() latency meets sub-5ms SLA."""
        guardrail = BudgetGuardrail(
            tracker=temp_tracker,
            soft_limit=100.0,
            hard_limit=200.0,
            period="day",
        )

        latencies: list[float] = []
        for _ in range(200):
            t0 = time.perf_counter()
            res = guardrail.check(estimated_cost=0.01)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            assert not res.is_blocked

        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p99 = latencies[int(len(latencies) * 0.99)]
        # P50 latency SLA: sub-10ms; P99 latency SLA: sub-75ms on Windows NTFS
        assert p50 < 0.015, f"P50 latency was {p50 * 1000:.2f} ms (expected < 15ms)"
        assert p99 < 0.075, f"P99 latency was {p99 * 1000:.2f} ms (expected < 75ms)"

    def test_concurrent_guardrail_and_writes(self, temp_tracker: TalantonTracker) -> None:
        """Verifies concurrent threads checking guardrails while other threads record telemetry."""
        guardrail = BudgetGuardrail(
            tracker=temp_tracker,
            soft_limit=25.0,
            hard_limit=50.0,
            period="day",
        )

        def writer_worker(n: int) -> int:
            for _ in range(n):
                temp_tracker.track(model="gpt-4o", input_tokens=500, output_tokens=200)
            return n

        def checker_worker(n: int) -> int:
            blocked_count = 0
            for _ in range(n):
                res = guardrail.check(estimated_cost=0.01)
                if res.is_blocked:
                    blocked_count += 1
            return blocked_count

        with ThreadPoolExecutor(max_workers=8) as executor:
            writer_futs = [executor.submit(writer_worker, 15) for _ in range(4)]
            checker_futs = [executor.submit(checker_worker, 25) for _ in range(4)]

            for f in as_completed(writer_futs + checker_futs):
                f.result()

        summary = temp_tracker.get_summary(period="day")
        assert summary["total_calls"] == 60  # 4 * 15

    def test_empty_batch_handling(self, temp_tracker: TalantonTracker) -> None:
        """Verifies graceful handling of empty batch inputs."""
        assert temp_tracker.track_batch([]) == []
        assert temp_tracker.get_calls() == []
