"""Run the energy-aware drone delivery demo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from planner import battery_trace, naive_distance_plan, plan_mission
from scenario import make_demo_scenario
from visualization import save_battery_profile, save_energy_breakdown, save_route_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Energy-aware drone delivery demo")
    parser.add_argument("--output", default="assets", help="output directory")
    return parser.parse_args()


def run(output_dir: str | Path = "assets") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = make_demo_scenario()
    plan = plan_mission(scenario)
    baseline = naive_distance_plan(scenario)

    route_map = save_route_map(scenario, plan, output_dir / "energy_aware_route.png")
    battery_profile = save_battery_profile(scenario, plan, baseline, output_dir / "battery_profile.png")
    energy_breakdown = save_energy_breakdown(plan, output_dir / "energy_breakdown.png")

    smart_trace = battery_trace(scenario, plan)
    naive_trace = battery_trace(scenario, baseline)
    metrics = {
        "tasks": len(scenario.tasks),
        "charging_stations": len(scenario.stations),
        "feasible": plan.feasible,
        "task_order": plan.task_order,
        "charges": plan.charges,
        "total_steps": plan.total_steps,
        "total_energy": round(plan.total_energy, 3),
        "minimum_battery": round(min(smart_trace), 3),
        "naive_total_energy": round(baseline.total_energy, 3),
        "naive_minimum_battery": round(min(naive_trace), 3),
        "outputs": {
            "route_map": str(route_map),
            "battery_profile": str(battery_profile),
            "energy_breakdown": str(energy_breakdown),
        },
    }
    write_outputs(output_dir, metrics, smart_trace)
    print_summary(metrics)
    return metrics


def write_outputs(output_dir: Path, metrics: dict, battery_values: list[float]) -> None:
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

    with open(output_dir / "battery_trace.csv", "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["segment", "battery"])
        writer.writeheader()
        for index, value in enumerate(battery_values):
            writer.writerow({"segment": index, "battery": round(value, 3)})


def print_summary(metrics: dict) -> None:
    print("Energy-aware drone delivery demo finished")
    print(f"Tasks: {metrics['tasks']}  Charging stations: {metrics['charging_stations']}")
    print(f"Feasible: {metrics['feasible']}  Charges: {metrics['charges']}")
    print(f"Task order: {', '.join(metrics['task_order'])}")
    print(f"Total steps: {metrics['total_steps']}  Total energy: {metrics['total_energy']}")
    print(f"Minimum battery: {metrics['minimum_battery']}")
    print(f"Naive minimum battery: {metrics['naive_minimum_battery']}")


def main() -> None:
    args = parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
