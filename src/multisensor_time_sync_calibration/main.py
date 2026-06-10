"""Run the multi-sensor timestamp calibration demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration import run_calibration
from scenario import make_demo_scenario
from visualization import create_visualizations


def build_metrics(seed: int, output_dir: Path) -> dict[str, object]:
    scenario = make_demo_scenario(seed=seed)
    result = run_calibration(scenario)
    outputs = create_visualizations(scenario, result, output_dir)

    improvement = (result.before_rmse - result.after_rmse) / result.before_rmse * 100.0
    metrics = {
        "project": "multisensor_time_sync_calibration",
        "seed": seed,
        "true_lidar_delay_s": round(scenario.true_lidar_delay, 4),
        "estimated_lidar_delay_s": round(result.estimated_lidar_delay, 4),
        "delay_absolute_error_s": round(abs(result.estimated_lidar_delay - scenario.true_lidar_delay), 4),
        "before_sync_rmse_m": round(result.before_rmse, 4),
        "after_sync_rmse_m": round(result.after_rmse, 4),
        "rmse_improvement_percent": round(improvement, 2),
        "generated_files": [str(path.name) for path in outputs],
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-sensor time synchronization calibration demo")
    parser.add_argument("--seed", type=int, default=7, help="random seed for deterministic sensor noise")
    parser.add_argument("--output", type=Path, default=Path("assets"), help="directory for generated images and metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics(args.seed, args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
