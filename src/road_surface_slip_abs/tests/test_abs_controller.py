from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from controller import compare_controllers, tire_grip_factor
from main import build_metrics
from scenario import make_demo_scenario


def test_tire_grip_peaks_near_optimal_slip() -> None:
    assert tire_grip_factor(0.18) > tire_grip_factor(0.02)
    assert tire_grip_factor(0.18) > tire_grip_factor(0.95)


def test_adaptive_abs_reduces_slip_and_distance() -> None:
    scenario = make_demo_scenario()
    baseline, optimized = compare_controllers(scenario)

    assert optimized.stopping_distance_m < baseline.stopping_distance_m
    assert optimized.max_slip_ratio < baseline.max_slip_ratio * 0.45
    assert optimized.mean_abs_slip_error < baseline.mean_abs_slip_error * 0.35


def test_main_exports_metrics_and_visualizations(tmp_path: Path) -> None:
    metrics = build_metrics(tmp_path)

    assert metrics["improvement"]["stopping_distance_percent"] > 5
    assert (tmp_path / "speed_distance_comparison.png").exists()
    assert (tmp_path / "friction_estimation.png").exists()
    assert (tmp_path / "slip_control_response.png").exists()
    assert (tmp_path / "metric_comparison.png").exists()
    assert (tmp_path / "metrics.json").exists()
