"""Run the elevator group dispatch optimization demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dispatcher import compare_dispatchers
from scenario import make_demo_scenario
from visualization import create_visualizations


def _improvement(before: float, after: float) -> float:
    return (before - after) / max(before, 1e-9) * 100.0


def build_metrics(output_dir: Path, seed: int = 31) -> dict[str, object]:
    scenario = make_demo_scenario(seed=seed)
    baseline, optimized = compare_dispatchers(scenario)
    outputs = create_visualizations(scenario, baseline, optimized, output_dir)

    metrics = {
        "project": "elevator_group_dispatch",
        "seed": seed,
        "request_count": len(scenario.requests),
        "passenger_throughput": optimized.passenger_throughput,
        "elevator_count": scenario.elevator_count,
        "floor_count": scenario.floor_count,
        "baseline_nearest_car": {
            "mean_wait_s": round(baseline.mean_wait_s, 4),
            "p90_wait_s": round(baseline.p90_wait_s, 4),
            "max_wait_s": round(baseline.max_wait_s, 4),
            "mean_ride_s": round(baseline.mean_ride_s, 4),
            "total_distance_floors": round(baseline.total_distance_floors, 4),
            "empty_distance_floors": round(baseline.empty_distance_floors, 4),
            "energy_index": round(baseline.energy_index, 4),
            "long_wait_count": baseline.long_wait_count,
        },
        "optimized_destination_aware": {
            "mean_wait_s": round(optimized.mean_wait_s, 4),
            "p90_wait_s": round(optimized.p90_wait_s, 4),
            "max_wait_s": round(optimized.max_wait_s, 4),
            "mean_ride_s": round(optimized.mean_ride_s, 4),
            "total_distance_floors": round(optimized.total_distance_floors, 4),
            "empty_distance_floors": round(optimized.empty_distance_floors, 4),
            "energy_index": round(optimized.energy_index, 4),
            "long_wait_count": optimized.long_wait_count,
        },
        "improvement": {
            "mean_wait_percent": round(_improvement(baseline.mean_wait_s, optimized.mean_wait_s), 2),
            "p90_wait_percent": round(_improvement(baseline.p90_wait_s, optimized.p90_wait_s), 2),
            "empty_distance_percent": round(_improvement(baseline.empty_distance_floors, optimized.empty_distance_floors), 2),
            "energy_index_percent": round(_improvement(baseline.energy_index, optimized.energy_index), 2),
            "long_wait_count_percent": round(_improvement(float(baseline.long_wait_count), float(optimized.long_wait_count)), 2),
        },
        "generated_files": [path.name for path in outputs],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Elevator group dispatch optimization demo")
    parser.add_argument("--seed", type=int, default=31, help="random seed for deterministic passenger requests")
    parser.add_argument("--output", type=Path, default=Path("assets"), help="directory for generated assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics(output_dir=args.output, seed=args.seed)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
