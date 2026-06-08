"""AirSim drone LiDAR safety monitor from simulator flight logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path(__file__).with_name("sample_data") / "airsim_lidar_log.csv"


def load_log(path: Path = DATA) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def score_rows(rows: list[dict[str, float]]) -> list[dict[str, float | str]]:
    scored = []
    for row in rows:
        min_clearance = min(row["forward_clearance_m"], row["left_clearance_m"], row["right_clearance_m"])
        clearance_risk = max(0.0, (3.0 - min_clearance) / 3.0)
        attitude_risk = min((abs(row["roll_deg"]) + abs(row["pitch_deg"])) / 18.0, 1.2)
        altitude_risk = max(0.0, (2.6 - row["altitude_m"]) / 2.6)
        descent_risk = max(0.0, -row["vertical_speed_mps"] / 1.2)
        risk = 0.46 * clearance_risk + 0.22 * attitude_risk + 0.18 * altitude_risk + 0.14 * descent_risk
        level = "critical" if risk >= 0.58 else "warning" if risk >= 0.34 else "safe"
        action = recommend_action(row, min_clearance, level)
        scored.append({
            **row,
            "min_clearance_m": round(min_clearance, 3),
            "safety_risk": round(float(risk), 4),
            "risk_level": level,
            "recommended_action": action,
        })
    return scored


def recommend_action(row: dict[str, float], min_clearance: float, level: str) -> str:
    if level == "safe":
        return "keep_course"
    if row["forward_clearance_m"] == min_clearance:
        return "brake_and_climb"
    if row["left_clearance_m"] == min_clearance:
        return "shift_right"
    if row["right_clearance_m"] == min_clearance:
        return "shift_left"
    return "slow_down"


def summarize(scored: list[dict[str, float | str]]) -> dict[str, object]:
    risk = np.array([float(row["safety_risk"]) for row in scored])
    return {
        "source": "AirSim LiDAR and flight-state log",
        "frames": len(scored),
        "critical_frames": sum(row["risk_level"] == "critical" for row in scored),
        "warning_frames": sum(row["risk_level"] == "warning" for row in scored),
        "avoidance_actions": dict(sorted({str(row["recommended_action"]): 0 for row in scored}.items())),
        "max_risk": round(float(risk.max()), 4),
        "min_clearance_m": round(min(float(row["min_clearance_m"]) for row in scored), 3),
    }


def plot(scored: list[dict[str, float | str]], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    t = np.array([float(row["time_s"]) for row in scored])
    risk = np.array([float(row["safety_risk"]) for row in scored])
    clearance = np.array([float(row["min_clearance_m"]) for row in scored])
    altitude = np.array([float(row["altitude_m"]) for row in scored])
    paths = []

    path = output / "airsim_clearance_risk.png"
    fig, ax1 = plt.subplots(figsize=(8.3, 4.8))
    ax1.plot(t, clearance, marker="o", color="#2f80ed", label="min clearance")
    ax1.axhline(1.5, color="#eb5757", linestyle="--", linewidth=1)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("clearance (m)")
    ax2 = ax1.twinx()
    ax2.plot(t, risk, marker="s", color="#f2994a", label="safety risk")
    ax2.set_ylabel("risk")
    ax1.set_title("AirSim LiDAR clearance and collision risk")
    fig.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    path = output / "airsim_altitude_attitude.png"
    plt.figure(figsize=(7.5, 5))
    plt.scatter(altitude, risk, c=[abs(float(row["roll_deg"])) + abs(float(row["pitch_deg"])) for row in scored], cmap="magma", s=90)
    plt.colorbar(label="|roll| + |pitch|")
    plt.xlabel("altitude (m)")
    plt.ylabel("safety risk")
    plt.title("Altitude-attitude safety relation")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    path = output / "airsim_action_distribution.png"
    actions = [str(row["recommended_action"]) for row in scored]
    names = sorted(set(actions))
    counts = [actions.count(name) for name in names]
    plt.figure(figsize=(7.4, 4.8))
    plt.bar(names, counts, color="#27ae60")
    plt.ylabel("frames")
    plt.title("Recommended avoidance actions")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    return paths


def run(output: Path, data: Path = DATA) -> dict[str, object]:
    scored = score_rows(load_log(data))
    files = plot(scored, output)
    csv_path = output / "airsim_safety_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)
    files.append(csv_path)
    report = summarize(scored)
    actions = [str(row["recommended_action"]) for row in scored]
    report["avoidance_actions"] = {name: actions.count(name) for name in sorted(set(actions))}
    report["generated_files"] = [p.name for p in files]
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/pr_assets/airsim_lidar_safety_monitor"))
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.data), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
