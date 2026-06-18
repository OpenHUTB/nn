"""Run the platoon cooperative adaptive cruise control demo."""

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
        "project": "platoon_cacc_gap_control",
        "vehicle_count": scenario.vehicle_count,
        "communication_delay_s": scenario.communication_delay_s,
        "dropout_windows": scenario.dropout_windows,
        "baseline": {
            "min_gap_m": round(baseline.min_gap, 4),
            "mean_abs_gap_error_m": round(baseline.mean_abs_gap_error, 4),
            "max_abs_gap_error_m": round(baseline.max_abs_gap_error, 4),
            "speed_std_amplification": round(baseline.speed_std_amplification, 4),
            "comfort_cost": round(baseline.comfort_cost, 4),
            "emergency_brake_count": baseline.emergency_brake_count,
        },
        "optimized": {
            "min_gap_m": round(optimized.min_gap, 4),
            "mean_abs_gap_error_m": round(optimized.mean_abs_gap_error, 4),
            "max_abs_gap_error_m": round(optimized.max_abs_gap_error, 4),
            "speed_std_amplification": round(optimized.speed_std_amplification, 4),
            "comfort_cost": round(optimized.comfort_cost, 4),
            "emergency_brake_count": optimized.emergency_brake_count,
        },
        "improvement": {
            "mean_gap_error_percent": round(_improvement(baseline.mean_abs_gap_error, optimized.mean_abs_gap_error), 2),
            "max_gap_error_percent": round(_improvement(baseline.max_abs_gap_error, optimized.max_abs_gap_error), 2),
            "comfort_cost_percent": round(_improvement(baseline.comfort_cost, optimized.comfort_cost), 2),
        },
        "generated_files": [path.name for path in outputs],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Platoon CACC gap-control simulation")
    parser.add_argument("--output", type=Path, default=Path("assets"), help="directory for generated assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics(args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
