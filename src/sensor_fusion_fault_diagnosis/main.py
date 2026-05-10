"""Run the sensor fusion fault diagnosis demo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fusion import diagnosis_scores, run_fusion
from scenario import SENSORS, fault_labels, make_demo_scenario
from visualization import save_residual_plot, save_rmse_chart, save_trajectory_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sensor fusion fault diagnosis demo")
    parser.add_argument("--output", default="assets", help="output directory")
    parser.add_argument("--seed", type=int, default=42, help="random seed for synthetic measurements")
    return parser.parse_args()


def run(output_dir: str | Path = "assets", seed: int = 42) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = make_demo_scenario(seed=seed)
    result = run_fusion(scenario)
    labels = fault_labels(scenario)
    scores = diagnosis_scores(result, labels)

    trajectory_plot = save_trajectory_plot(scenario, result, output_dir / "fusion_trajectory.png")
    residual_plot = save_residual_plot(scenario, result, output_dir / "fault_residuals.png")
    rmse_chart = save_rmse_chart(result, output_dir / "rmse_comparison.png")

    metrics = {
        "steps": scenario.steps,
        "sensors": list(SENSORS),
        "fault_windows": [
            {
                "sensor": fault.sensor,
                "start": fault.start,
                "end": fault.end,
                "type": fault.fault_type,
            }
            for fault in scenario.faults
        ],
        "rmse_naive": round(result.rmse_naive, 4),
        "rmse_robust": round(result.rmse_robust, 4),
        "rmse_improvement_percent": round((result.rmse_naive - result.rmse_robust) / result.rmse_naive * 100.0, 2),
        "diagnosis": scores,
        "outputs": {
            "trajectory": str(trajectory_plot),
            "residuals": str(residual_plot),
            "rmse_chart": str(rmse_chart),
        },
    }
    write_outputs(output_dir, metrics, result)
    print_summary(metrics)
    return metrics


def write_outputs(output_dir: Path, metrics: dict, result) -> None:
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

    with open(output_dir / "fault_flags.csv", "w", encoding="utf-8", newline="") as file_obj:
        fieldnames = ["step", *[f"{sensor}_fault" for sensor in SENSORS]]
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(len(result.estimates)):
            row = {"step": step}
            for sensor in SENSORS:
                row[f"{sensor}_fault"] = int(result.fault_flags[sensor][step])
            writer.writerow(row)


def print_summary(metrics: dict) -> None:
    print("Sensor fusion fault diagnosis demo finished")
    print(f"Steps: {metrics['steps']}  Sensors: {', '.join(metrics['sensors'])}")
    print(f"RMSE naive: {metrics['rmse_naive']}  RMSE robust: {metrics['rmse_robust']}")
    print(f"RMSE improvement: {metrics['rmse_improvement_percent']}%")
    for sensor, score in metrics["diagnosis"].items():
        print(f"{sensor}: precision={score['precision']} recall={score['recall']}")


def main() -> None:
    args = parse_args()
    run(args.output, args.seed)


if __name__ == "__main__":
    main()
