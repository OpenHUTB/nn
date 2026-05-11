from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from allocator import compare_allocators
from main import build_metrics
from scenario import make_demo_scenario


def test_balanced_allocator_reduces_zone_imbalance() -> None:
    scenario = make_demo_scenario(seed=41)
    baseline, optimized = compare_allocators(scenario)

    assert optimized.served_count >= baseline.served_count
    assert optimized.zone_imbalance_index < baseline.zone_imbalance_index
    assert optimized.peak_zone_occupancy_ratio <= baseline.peak_zone_occupancy_ratio


def test_special_requests_are_served_when_possible() -> None:
    scenario = make_demo_scenario(seed=41)
    _, optimized = compare_allocators(scenario)

    charger_requests = sum(request.needs_charger for request in scenario.requests)
    accessible_requests = sum(request.needs_accessible for request in scenario.requests)
    assert optimized.charger_mismatch_count <= charger_requests
    assert optimized.accessible_mismatch_count <= accessible_requests


def test_main_exports_metrics_and_visualizations(tmp_path: Path) -> None:
    metrics = build_metrics(tmp_path, seed=41)

    assert metrics["improvement"]["zone_imbalance_percent"] > 5
    assert (tmp_path / "parking_layout_allocation.png").exists()
    assert (tmp_path / "zone_occupancy_curve.png").exists()
    assert (tmp_path / "walking_distance_distribution.png").exists()
    assert (tmp_path / "metric_comparison.png").exists()
    assert (tmp_path / "metrics.json").exists()
