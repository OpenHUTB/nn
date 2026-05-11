from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from controller import compare_strategies, simulate_corridor
from main import build_metrics
from scenario import make_demo_scenario


def test_adaptive_holding_reduces_bunching() -> None:
    scenario = make_demo_scenario()
    baseline, optimized = compare_strategies(scenario)

    assert optimized.headway_std_min < baseline.headway_std_min
    assert optimized.bunching_count <= baseline.bunching_count
    assert optimized.service_reliability_score > baseline.service_reliability_score


def test_bus_runs_have_monotonic_departures() -> None:
    scenario = make_demo_scenario()
    result = simulate_corridor(scenario, "adaptive_holding")

    assert len(result.runs) == scenario.bus_count
    for run in result.runs:
        assert all(later > earlier for earlier, later in zip(run.departure_min, run.departure_min[1:]))


def test_main_exports_metrics_and_visualizations(tmp_path: Path) -> None:
    metrics = build_metrics(tmp_path)

    assert metrics["improvement"]["headway_std_percent"] > 5
    assert (tmp_path / "time_space_diagram.png").exists()
    assert (tmp_path / "headway_heatmap.png").exists()
    assert (tmp_path / "passenger_wait_by_stop.png").exists()
    assert (tmp_path / "metric_comparison.png").exists()
    assert (tmp_path / "metrics.json").exists()
