from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calibration import estimate_lidar_delay, run_calibration
from main import build_metrics
from scenario import make_demo_scenario


def test_estimated_delay_is_close_to_truth() -> None:
    scenario = make_demo_scenario(seed=7)
    estimated_delay, offsets, errors = estimate_lidar_delay(scenario.camera, scenario.lidar)

    assert len(offsets) == len(errors)
    assert abs(estimated_delay - scenario.true_lidar_delay) <= 0.04


def test_synchronization_reduces_fusion_error() -> None:
    scenario = make_demo_scenario(seed=7)
    result = run_calibration(scenario)

    assert result.after_rmse < result.before_rmse * 0.65
    assert result.fused_positions.shape == result.unsynced_positions.shape
    assert result.fused_positions.shape[1] == 2


def test_demo_exports_metrics_and_visualizations(tmp_path: Path) -> None:
    metrics = build_metrics(seed=7, output_dir=tmp_path)

    assert metrics["rmse_improvement_percent"] > 35
    assert (tmp_path / "trajectory_alignment.png").exists()
    assert (tmp_path / "offset_search_curve.png").exists()
    assert (tmp_path / "rmse_comparison.png").exists()
    assert (tmp_path / "metrics.json").exists()
