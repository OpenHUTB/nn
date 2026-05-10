from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from controller import compare_controllers, simulate_platoon
from main import build_metrics
from scenario import make_demo_scenario


def test_cacc_reduces_gap_error() -> None:
    scenario = make_demo_scenario()
    baseline, optimized = compare_controllers(scenario)

    assert optimized.mean_abs_gap_error < baseline.mean_abs_gap_error
    assert optimized.max_abs_gap_error < baseline.max_abs_gap_error


def test_simulation_keeps_safe_positive_gaps() -> None:
    scenario = make_demo_scenario()
    result = simulate_platoon(scenario, "cacc_feedforward")

    assert result.min_gap > scenario.standstill_gap * 0.65
    assert result.positions.shape[1] == scenario.vehicle_count
    assert result.gaps.shape[1] == scenario.vehicle_count - 1


def test_main_exports_assets(tmp_path: Path) -> None:
    metrics = build_metrics(tmp_path)

    assert metrics["improvement"]["mean_gap_error_percent"] > 10
    assert (tmp_path / "speed_profiles.png").exists()
    assert (tmp_path / "cacc_gap_error_heatmap.png").exists()
    assert (tmp_path / "metric_comparison.png").exists()
    assert (tmp_path / "space_time_trajectories.png").exists()
    assert (tmp_path / "metrics.json").exists()
