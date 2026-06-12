"""AirSim multirotor IMU vibration and GPS drift monitor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path(__file__).with_name("sample_data") / "airsim_multirotor_imu.csv"


def load_rows(path: Path = DATA) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def analyze(rows: list[dict[str, float]]) -> list[dict[str, float | str]]:
    scored = []
    for r in rows:
        acc_norm = float(np.linalg.norm([r["acc_x"], r["acc_y"], r["acc_z"]]))
        gyro_norm = float(np.linalg.norm([r["gyro_x"], r["gyro_y"], r["gyro_z"]]))
        drift = float(np.linalg.norm([r["gps_x_m"] - r["pos_x_m"], r["gps_y_m"] - r["pos_y_m"], r["gps_z_m"] - r["pos_z_m"]]))
        vibration = abs(acc_norm - 9.81) + 3.0 * gyro_norm
        score = min(1.0, 0.45 * vibration / 1.2 + 0.55 * drift / 3.8)
        action = "relocalize" if score > 0.70 else "smooth_control" if score > 0.40 else "normal"
        scored.append({**r, "acc_norm": round(acc_norm, 4), "gyro_norm": round(gyro_norm, 4), "gps_drift_m": round(drift, 4), "health_score": round(score, 4), "monitor_action": action})
    return scored


def plot(scored: list[dict[str, float | str]], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    p = output / "airsim_runtime_trajectory_drift.png"
    plt.figure(figsize=(8, 5.2))
    plt.plot([r["pos_x_m"] for r in scored], [r["pos_y_m"] for r in scored], marker="o", label="AirSim ground truth")
    plt.plot([r["gps_x_m"] for r in scored], [r["gps_y_m"] for r in scored], marker="s", label="GPS estimate")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("AirSim multirotor trajectory and GPS drift")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(p)
    p = output / "airsim_imu_health_curve.png"
    t = [r["time_s"] for r in scored]
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, [r["health_score"] for r in scored], marker="o", color="#eb5757", label="health score")
    plt.plot(t, [r["gps_drift_m"] for r in scored], marker="s", color="#2f80ed", label="GPS drift (m)")
    plt.xlabel("time (s)")
    plt.title("AirSim IMU vibration and drift monitor")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(p)
    return paths


def run(output: Path) -> dict[str, object]:
    scored = analyze(load_rows())
    files = plot(scored, output)
    csv_path = output / "airsim_imu_drift_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)
    files.append(csv_path)
    report = {
        "source": "AirSim multirotor IMU and GPS telemetry log",
        "records": len(scored),
        "max_gps_drift_m": round(max(float(r["gps_drift_m"]) for r in scored), 4),
        "relocalize_frames": sum(r["monitor_action"] == "relocalize" for r in scored),
        "generated_files": [p.name for p in files],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/pr_assets/airsim_imu_drift_monitor"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
