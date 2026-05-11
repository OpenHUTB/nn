from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dispatcher import compare_dispatchers, run_dispatch
from main import build_metrics
from scenario import make_demo_scenario


def test_destination_aware_reduces_waiting_time() -> None:
    scenario = make_demo_scenario(seed=31)
    baseline, optimized = compare_dispatchers(scenario)

    assert optimized.mean_wait_s < baseline.mean_wait_s
    assert optimized.p90_wait_s < baseline.p90_wait_s
    assert optimized.long_wait_count <= baseline.long_wait_count


def test_all_requests_are_served_once() -> None:
    scenario = make_demo_scenario(seed=31)
    result = run_dispatch(scenario, "destination_aware")

    assert len(result.records) == len(scenario.requests)
    assert sorted(record.request_id for record in result.records) == list(range(1, len(scenario.requests) + 1))
    assert all(record.dropoff_s > record.pickup_s for record in result.records)


def test_main_exports_metrics_and_visualizations(tmp_path: Path) -> None:
    metrics = build_metrics(tmp_path, seed=31)

    assert metrics["improvement"]["mean_wait_percent"] > 5
    assert (tmp_path / "wait_time_distribution.png").exists()
    assert (tmp_path / "elevator_timeline.png").exists()
    assert (tmp_path / "floor_demand_heatmap.png").exists()
    assert (tmp_path / "metric_comparison.png").exists()
    assert (tmp_path / "metrics.json").exists()
