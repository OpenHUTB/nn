from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from main import build_metrics
from scenario import make_demo_scenario
from scheduler import compare_schedulers


def test_load_aware_scheduler_reduces_grid_peak() -> None:
    scenario = make_demo_scenario(seed=23)
    baseline, optimized = compare_schedulers(scenario)

    assert optimized.completed_count == baseline.completed_count
    assert optimized.peak_grid_load_kw < baseline.peak_grid_load_kw
    assert optimized.overload_minutes <= baseline.overload_minutes


def test_scheduler_keeps_all_chargers_within_capacity() -> None:
    scenario = make_demo_scenario(seed=23)
    _, optimized = compare_schedulers(scenario)

    assert optimized.charger_occupancy.max() <= scenario.charger_count
    assert optimized.station_load_kw.max() <= scenario.charger_count * scenario.charger_power_kw


def test_main_exports_metrics_and_visualizations(tmp_path: Path) -> None:
    metrics = build_metrics(tmp_path, seed=23)

    assert metrics["improvement"]["peak_grid_load_percent"] > 5
    assert (tmp_path / "load_profile_comparison.png").exists()
    assert (tmp_path / "charger_occupancy.png").exists()
    assert (tmp_path / "waiting_time_distribution.png").exists()
    assert (tmp_path / "metric_comparison.png").exists()
    assert (tmp_path / "metrics.json").exists()
