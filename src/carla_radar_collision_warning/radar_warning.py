"""CARLA radar collision warning from radar detection rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path(__file__).with_name("sample_data") / "carla_radar_detections.csv"


def load_rows(path: Path = DATA) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({k: row[k] if k == "object_type" else float(row[k]) for k in row})
    return rows


def analyze(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    out = []
    for r in rows:
        closing = max(0.0, -float(r["radial_velocity_mps"]))
        ttc = 999.0 if closing <= 1e-6 else float(r["depth_m"]) / closing
        az_gate = max(0.0, 1.0 - abs(float(r["azimuth_deg"])) / 10.0)
        type_weight = {"pedestrian": 1.2, "bicycle": 1.1, "vehicle": 1.0}.get(str(r["object_type"]), 1.0)
        risk = type_weight * (0.72 * max(0.0, (4.0 - min(ttc, 4.0)) / 4.0) + 0.28 * az_gate)
        action = "emergency_brake" if risk > 0.68 else "soft_brake" if risk > 0.45 else "monitor"
        out.append({**r, "closing_speed_mps": round(closing, 3), "ttc_s": round(ttc, 3), "risk_score": round(float(risk), 4), "brake_action": action})
    return out


def plot(rows: list[dict[str, float | str]], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    p = output / "carla_radar_ttc_curve.png"
    plt.figure(figsize=(8, 4.8))
    plt.plot([r["time_s"] for r in rows], [r["ttc_s"] for r in rows], marker="o", color="#eb5757")
    plt.axhline(2.0, linestyle="--", color="#111827")
    plt.xlabel("time (s)")
    plt.ylabel("TTC (s)")
    plt.title("CARLA radar time-to-collision warning")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(p)
    p = output / "carla_radar_detection_map.png"
    plt.figure(figsize=(7, 5.2))
    az = np.radians([float(r["azimuth_deg"]) for r in rows])
    depth = np.array([float(r["depth_m"]) for r in rows])
    x = depth * np.cos(az)
    y = depth * np.sin(az)
    plt.scatter(y, x, c=[float(r["risk_score"]) for r in rows], cmap="inferno", s=90)
    plt.colorbar(label="risk")
    plt.xlabel("lateral y (m)")
    plt.ylabel("forward depth (m)")
    plt.title("CARLA radar detection risk map")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(p)
    return paths


def run(output: Path) -> dict[str, object]:
    rows = analyze(load_rows())
    files = plot(rows, output)
    csv_path = output / "carla_radar_warning_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    files.append(csv_path)
    report = {
        "source": "CARLA RadarDetection sensor log",
        "records": len(rows),
        "min_ttc_s": min(float(r["ttc_s"]) for r in rows),
        "max_risk_score": max(float(r["risk_score"]) for r in rows),
        "emergency_brake_frames": sum(r["brake_action"] == "emergency_brake" for r in rows),
        "generated_files": [p.name for p in files],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/pr_assets/carla_radar_collision_warning"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
