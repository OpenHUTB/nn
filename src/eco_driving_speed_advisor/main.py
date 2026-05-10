"""Run the eco-driving speed advisor demo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from advisor import plan_eco_speed, signal_compliance, simulate_cruise_baseline
from scenario import make_demo_scenario
from visualization import save_metrics_chart, save_speed_profile, save_time_space_diagram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eco-driving speed advisor demo")
    parser.add_argument("--output", default="assets", help="output directory")
    return parser.parse_args()


def run(output_dir: str | Path = "assets") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = make_demo_scenario()
    eco = plan_eco_speed(scenario)
    baseline = simulate_cruise_baseline(scenario)

    speed_plot = save_speed_profile(eco, baseline, output_dir / "speed_profile.png")
    time_space_plot = save_time_space_diagram(scenario, eco, baseline, output_dir / "time_space_diagram.png")
    metrics_plot = save_metrics_chart(scenario, eco, baseline, output_dir / "eco_metrics.png")

    eco_green = signal_compliance(scenario, eco)
    baseline_green = signal_compliance(scenario, baseline)
    metrics = {
        "signals": len(scenario.lights),
        "corridor_length_m": scenario.length_m,
        "eco_travel_time_s": round(float(eco.time_s[-1]), 3),
        "baseline_travel_time_s": round(float(baseline.time_s[-1]), 3),
        "eco_energy": round(eco.energy, 3),
        "baseline_energy": round(baseline.energy, 3),
        "energy_saving_percent": round((baseline.energy - eco.energy) / baseline.energy * 100.0, 2),
        "eco_stops": eco.stops,
        "baseline_stops": baseline.stops,
        "eco_green_passes": int(sum(eco_green.values())),
        "baseline_green_passes": int(sum(baseline_green.values())),
        "eco_pass_times": {key: round(value, 3) for key, value in eco.pass_times.items()},
        "baseline_pass_times": {key: round(value, 3) for key, value in baseline.pass_times.items()},
        "outputs": {
            "speed_profile": str(speed_plot),
            "time_space_diagram": str(time_space_plot),
            "metrics_chart": str(metrics_plot),
        },
    }
    write_outputs(output_dir, metrics, eco, baseline)
    print_summary(metrics)
    return metrics


def write_outputs(output_dir: Path, metrics: dict, eco, baseline) -> None:
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

    with open(output_dir / "speed_profile.csv", "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["profile", "time_s", "position_m", "speed_mps"])
        writer.writeheader()
        for profile in (eco, baseline):
            for time_s, position_m, speed_mps in zip(profile.time_s, profile.position_m, profile.speed_mps):
                writer.writerow(
                    {
                        "profile": profile.name,
                        "time_s": round(float(time_s), 3),
                        "position_m": round(float(position_m), 3),
                        "speed_mps": round(float(speed_mps), 3),
                    }
                )


def print_summary(metrics: dict) -> None:
    print("Eco-driving speed advisor demo finished")
    print(f"Signals: {metrics['signals']}  Corridor: {metrics['corridor_length_m']} m")
    print(f"Eco travel time: {metrics['eco_travel_time_s']} s")
    print(f"Baseline travel time: {metrics['baseline_travel_time_s']} s")
    print(f"Energy saving: {metrics['energy_saving_percent']}%")
    print(f"Stops eco/baseline: {metrics['eco_stops']} / {metrics['baseline_stops']}")
    print(f"Green passes eco/baseline: {metrics['eco_green_passes']} / {metrics['baseline_green_passes']}")


def main() -> None:
    args = parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
