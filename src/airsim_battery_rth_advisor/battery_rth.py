"""AirSim drone battery return-to-home advisor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path(__file__).with_name("sample_data") / "airsim_battery_log.csv"


def load_rows(path: Path = DATA) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def advise(rows: list[dict[str, float]]) -> list[dict[str, float | str]]:
    scored = []
    for row in rows:
        speed = max(row["ground_speed_mps"], 0.6)
        return_time = row["distance_to_home_m"] / speed
        burn_rate = 0.036 + 0.004 * row["wind_mps"] + 0.006 * row["payload_kg"] + 0.0006 * row["altitude_m"]
        required_battery = return_time * burn_rate + 8.0
        margin = row["battery_percent"] - required_battery
        action = "return_now" if margin < 8 else "prepare_return" if margin < 18 else "continue_mission"
        scored.append({**row, "estimated_return_time_s": round(return_time, 2), "required_battery_percent": round(required_battery, 2), "battery_margin": round(margin, 2), "rth_action": action})
    return scored


def plot(scored: list[dict[str, float | str]], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    t = np.array([float(r["time_s"]) for r in scored])
    battery = np.array([float(r["battery_percent"]) for r in scored])
    required = np.array([float(r["required_battery_percent"]) for r in scored])
    margin = np.array([float(r["battery_margin"]) for r in scored])
    paths = []
    path = output / "airsim_battery_requirement.png"
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, battery, marker="o", label="battery", color="#27ae60")
    plt.plot(t, required, marker="s", label="required to home", color="#eb5757")
    plt.xlabel("time (s)")
    plt.ylabel("battery (%)")
    plt.title("AirSim return-to-home battery requirement")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    path = output / "airsim_battery_margin.png"
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(t, margin, width=18, color=["#eb5757" if m < 8 else "#f2994a" if m < 18 else "#2f80ed" for m in margin])
    plt.axhline(8, color="#eb5757", linestyle="--")
    plt.xlabel("time (s)")
    plt.ylabel("battery margin (%)")
    plt.title("RTH safety margin")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    return paths


def run(output: Path, data: Path = DATA) -> dict[str, object]:
    scored = advise(load_rows(data))
    files = plot(scored, output)
    csv_path = output / "airsim_rth_advice.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)
    files.append(csv_path)
    margins = np.array([float(r["battery_margin"]) for r in scored])
    report = {"source": "AirSim battery telemetry log", "records": len(scored), "min_battery_margin": round(float(margins.min()), 2), "return_now_frames": sum(r["rth_action"] == "return_now" for r in scored), "generated_files": [p.name for p in files]}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/pr_assets/airsim_battery_rth_advisor"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
