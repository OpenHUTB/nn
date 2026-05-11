"""Run the smart parking allocation demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from allocator import compare_allocators
from scenario import make_demo_scenario
from visualization import create_visualizations


def _improvement(before: float, after: float) -> float:
    return (before - after) / max(before, 1e-9) * 100.0


def build_metrics(output_dir: Path, seed: int = 41) -> dict[str, object]:
    scenario = make_demo_scenario(seed=seed)
    baseline, optimized = compare_allocators(scenario)
    outputs = create_visualizations(scenario, baseline, optimized, output_dir)

    metrics = {
        "project": "smart_parking_allocation",
        "seed": seed,
        "spot_count": len(scenario.spots),
        "vehicle_count": len(scenario.requests),
        "baseline_nearest_spot": {
            "served_count": baseline.served_count,
            "rejected_count": baseline.rejected_count,
            "average_walking_distance_m": round(baseline.average_walking_distance_m, 4),
            "p90_walking_distance_m": round(baseline.p90_walking_distance_m, 4),
            "average_wait_min": round(baseline.average_wait_min, 4),
            "peak_zone_occupancy_ratio": round(baseline.peak_zone_occupancy_ratio, 4),
            "zone_imbalance_index": round(baseline.zone_imbalance_index, 4),
            "traffic_distance_m": round(baseline.traffic_distance_m, 4),
        },
        "optimized_balanced_allocation": {
            "served_count": optimized.served_count,
            "rejected_count": optimized.rejected_count,
            "average_walking_distance_m": round(optimized.average_walking_distance_m, 4),
            "p90_walking_distance_m": round(optimized.p90_walking_distance_m, 4),
            "average_wait_min": round(optimized.average_wait_min, 4),
            "peak_zone_occupancy_ratio": round(optimized.peak_zone_occupancy_ratio, 4),
            "zone_imbalance_index": round(optimized.zone_imbalance_index, 4),
            "traffic_distance_m": round(optimized.traffic_distance_m, 4),
        },
        "improvement": {
            "average_walking_distance_percent": round(
                _improvement(baseline.average_walking_distance_m, optimized.average_walking_distance_m), 2
            ),
            "p90_walking_distance_percent": round(
                _improvement(baseline.p90_walking_distance_m, optimized.p90_walking_distance_m), 2
            ),
            "zone_imbalance_percent": round(_improvement(baseline.zone_imbalance_index, optimized.zone_imbalance_index), 2),
            "traffic_distance_percent": round(_improvement(baseline.traffic_distance_m, optimized.traffic_distance_m), 2),
        },
        "generated_files": [path.name for path in outputs],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart parking-space allocation demo")
    parser.add_argument("--seed", type=int, default=41, help="random seed for deterministic parking demand")
    parser.add_argument("--output", type=Path, default=Path("assets"), help="directory for generated assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics(args.output, seed=args.seed)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
