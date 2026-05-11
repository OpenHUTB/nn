"""Run the EV charging station scheduling demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenario import make_demo_scenario
from scheduler import compare_schedulers
from visualization import create_visualizations


def _improvement(before: float, after: float) -> float:
    return (before - after) / max(before, 1e-9) * 100.0


def build_metrics(output_dir: Path, seed: int = 23) -> dict[str, object]:
    scenario = make_demo_scenario(seed=seed)
    baseline, optimized = compare_schedulers(scenario)
    outputs = create_visualizations(scenario, baseline, optimized, output_dir)

    metrics = {
        "project": "ev_charging_station_scheduler",
        "seed": seed,
        "vehicle_count": len(scenario.requests),
        "charger_count": scenario.charger_count,
        "charger_power_kw": scenario.charger_power_kw,
        "grid_limit_kw": scenario.grid_limit_kw,
        "baseline_fcfs": {
            "completed_count": baseline.completed_count,
            "average_wait_min": round(baseline.average_wait_min, 4),
            "deadline_miss_count": baseline.deadline_miss_count,
            "peak_grid_load_kw": round(baseline.peak_grid_load_kw, 4),
            "overload_minutes": baseline.overload_minutes,
            "solar_utilization_percent": round(baseline.solar_utilization_percent, 4),
            "load_variance": round(baseline.load_variance, 4),
        },
        "optimized_load_aware": {
            "completed_count": optimized.completed_count,
            "average_wait_min": round(optimized.average_wait_min, 4),
            "deadline_miss_count": optimized.deadline_miss_count,
            "peak_grid_load_kw": round(optimized.peak_grid_load_kw, 4),
            "overload_minutes": optimized.overload_minutes,
            "solar_utilization_percent": round(optimized.solar_utilization_percent, 4),
            "load_variance": round(optimized.load_variance, 4),
        },
        "improvement": {
            "peak_grid_load_percent": round(_improvement(baseline.peak_grid_load_kw, optimized.peak_grid_load_kw), 2),
            "overload_minutes_percent": round(_improvement(float(baseline.overload_minutes), float(optimized.overload_minutes)), 2),
            "load_variance_percent": round(_improvement(baseline.load_variance, optimized.load_variance), 2),
            "solar_utilization_delta_percent": round(optimized.solar_utilization_percent - baseline.solar_utilization_percent, 2),
        },
        "generated_files": [path.name for path in outputs],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EV charging station load-aware scheduling demo")
    parser.add_argument("--seed", type=int, default=23, help="random seed for deterministic vehicle requests")
    parser.add_argument("--output", type=Path, default=Path("assets"), help="directory for generated assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics(output_dir=args.output, seed=args.seed)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
