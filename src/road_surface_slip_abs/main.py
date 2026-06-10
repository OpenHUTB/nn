"""Run the road-surface friction estimation and adaptive ABS demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from controller import compare_controllers
from scenario import make_demo_scenario
from visualization import create_visualizations


def _improvement(before: float, after: float) -> float:
    return (before - after) / max(before, 1e-9) * 100.0


def build_metrics(output_dir: Path) -> dict[str, object]:
    scenario = make_demo_scenario()
    baseline, optimized = compare_controllers(scenario)
    outputs = create_visualizations(scenario, baseline, optimized, output_dir)

    metrics = {
        "project": "road_surface_slip_abs",
        "initial_speed_mps": scenario.initial_speed_mps,
        "optimal_slip": scenario.optimal_slip,
        "road_segments": [
            {
                "name": segment.name,
                "start_m": segment.start_m,
                "end_m": segment.end_m,
                "friction_mu": segment.friction_mu,
            }
            for segment in scenario.road_segments
        ],
        "baseline_locked_brake": {
            "stopping_distance_m": round(baseline.stopping_distance_m, 4),
            "stopping_time_s": round(baseline.stopping_time_s, 4),
            "max_slip_ratio": round(baseline.max_slip_ratio, 4),
            "mean_abs_slip_error": round(baseline.mean_abs_slip_error, 4),
            "comfort_cost": round(baseline.comfort_cost, 4),
        },
        "optimized_adaptive_abs": {
            "stopping_distance_m": round(optimized.stopping_distance_m, 4),
            "stopping_time_s": round(optimized.stopping_time_s, 4),
            "max_slip_ratio": round(optimized.max_slip_ratio, 4),
            "mean_abs_slip_error": round(optimized.mean_abs_slip_error, 4),
            "comfort_cost": round(optimized.comfort_cost, 4),
            "low_mu_detection_time_s": optimized.low_mu_detection_time_s,
        },
        "improvement": {
            "stopping_distance_percent": round(_improvement(baseline.stopping_distance_m, optimized.stopping_distance_m), 2),
            "stopping_time_percent": round(_improvement(baseline.stopping_time_s, optimized.stopping_time_s), 2),
            "max_slip_percent": round(_improvement(baseline.max_slip_ratio, optimized.max_slip_ratio), 2),
            "slip_error_percent": round(_improvement(baseline.mean_abs_slip_error, optimized.mean_abs_slip_error), 2),
        },
        "generated_files": [path.name for path in outputs],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Road friction estimation and adaptive ABS braking demo")
    parser.add_argument("--output", type=Path, default=Path("assets"), help="directory for generated assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics(args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
