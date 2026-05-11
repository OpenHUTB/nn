"""Run the bus headway control and bunching suppression demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from controller import compare_strategies
from scenario import make_demo_scenario
from visualization import create_visualizations


def _improvement(before: float, after: float) -> float:
    return (before - after) / max(before, 1e-9) * 100.0


def build_metrics(output_dir: Path) -> dict[str, object]:
    scenario = make_demo_scenario()
    baseline, optimized = compare_strategies(scenario)
    outputs = create_visualizations(scenario, baseline, optimized, output_dir)

    metrics = {
        "project": "bus_headway_control",
        "bus_count": scenario.bus_count,
        "stop_count": len(scenario.stop_positions_km),
        "target_headway_min": scenario.target_headway_min,
        "injected_delay_min": scenario.delay_min,
        "baseline_no_control": {
            "mean_headway_min": round(baseline.mean_headway_min, 4),
            "headway_std_min": round(baseline.headway_std_min, 4),
            "headway_cv": round(baseline.headway_cv, 4),
            "bunching_count": baseline.bunching_count,
            "average_passenger_wait_min": round(baseline.average_passenger_wait_min, 4),
            "total_holding_min": round(baseline.total_holding_min, 4),
            "max_load": baseline.max_load,
            "service_reliability_score": round(baseline.service_reliability_score, 4),
        },
        "optimized_adaptive_holding": {
            "mean_headway_min": round(optimized.mean_headway_min, 4),
            "headway_std_min": round(optimized.headway_std_min, 4),
            "headway_cv": round(optimized.headway_cv, 4),
            "bunching_count": optimized.bunching_count,
            "average_passenger_wait_min": round(optimized.average_passenger_wait_min, 4),
            "total_holding_min": round(optimized.total_holding_min, 4),
            "max_load": optimized.max_load,
            "service_reliability_score": round(optimized.service_reliability_score, 4),
        },
        "improvement": {
            "headway_std_percent": round(_improvement(baseline.headway_std_min, optimized.headway_std_min), 2),
            "bunching_count_percent": round(_improvement(float(baseline.bunching_count), float(optimized.bunching_count)), 2),
            "average_wait_percent": round(
                _improvement(baseline.average_passenger_wait_min, optimized.average_passenger_wait_min), 2
            ),
            "reliability_score_delta": round(optimized.service_reliability_score - baseline.service_reliability_score, 2),
        },
        "generated_files": [path.name for path in outputs],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bus headway control and bunching suppression demo")
    parser.add_argument("--output", type=Path, default=Path("assets"), help="directory for generated assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics(args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
