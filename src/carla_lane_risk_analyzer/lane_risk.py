"""CARLA lane keeping risk analysis from simulator-style lane logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path(__file__).with_name("sample_data") / "carla_lane_log.csv"


def load_log(path: Path = DATA) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def compute_risk(rows: list[dict[str, float]]) -> list[dict[str, float | str]]:
    scored = []
    for row in rows:
        offset = abs(row["lane_offset_m"])
        heading = abs(row["heading_error_deg"]) / 6.0
        speed = min(row["ego_speed_mps"] / 12.0, 1.2)
        curvature = min(row["curvature"] / 0.05, 1.4)
        density = row["traffic_density"]
        score = 0.38 * min(offset / 0.9, 1.5) + 0.24 * heading + 0.16 * speed + 0.12 * curvature + 0.10 * density
        label = "high" if score >= 0.72 else "medium" if score >= 0.48 else "low"
        scored.append({**row, "risk_score": round(float(score), 4), "risk_level": label})
    return scored


def summarize(scored: list[dict[str, float | str]]) -> dict[str, float | int | str]:
    scores = np.array([float(row["risk_score"]) for row in scored])
    high = sum(row["risk_level"] == "high" for row in scored)
    return {
        "source": "CARLA lane tracking log",
        "frames": len(scored),
        "max_risk": round(float(scores.max()), 4),
        "mean_risk": round(float(scores.mean()), 4),
        "high_risk_frames": int(high),
        "peak_frame": int(scored[int(scores.argmax())]["frame"]),
    }


def plot(scored: list[dict[str, float | str]], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    t = np.array([float(row["time_s"]) for row in scored])
    offset = np.array([float(row["lane_offset_m"]) for row in scored])
    risk = np.array([float(row["risk_score"]) for row in scored])
    speed = np.array([float(row["ego_speed_mps"]) for row in scored])
    paths = []

    path = output / "carla_lane_offset_risk.png"
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.plot(t, offset, marker="o", color="#2f80ed", label="lane offset")
    ax1.axhline(0.7, color="#eb5757", linestyle="--", linewidth=1)
    ax1.axhline(-0.7, color="#eb5757", linestyle="--", linewidth=1)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("lane offset (m)")
    ax2 = ax1.twinx()
    ax2.plot(t, risk, marker="s", color="#f2994a", label="risk score")
    ax2.set_ylabel("risk score")
    ax1.set_title("CARLA lane offset and departure risk")
    fig.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    path = output / "carla_speed_density_risk.png"
    plt.figure(figsize=(7.2, 5))
    plt.scatter(speed, risk, c=[float(row["traffic_density"]) for row in scored], cmap="viridis", s=80)
    plt.colorbar(label="traffic density")
    plt.xlabel("ego speed (m/s)")
    plt.ylabel("risk score")
    plt.title("Speed-density relation in CARLA log")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    return paths


def run(output: Path, data: Path = DATA) -> dict[str, object]:
    rows = load_log(data)
    scored = compute_risk(rows)
    files = plot(scored, output)
    report = summarize(scored)
    csv_path = output / "carla_lane_risk_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)
    files.append(csv_path)
    report["generated_files"] = [p.name for p in files]
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/pr_assets/carla_lane_risk_analyzer"))
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.data), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
