"""CARLA object-track TTC brake advisor."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path(__file__).with_name("sample_data") / "carla_object_tracks.csv"


def load_tracks(path: Path = DATA) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "frame": float(row["frame"]),
                "time_s": float(row["time_s"]),
                "object_id": row["object_id"],
                "class_name": row["class_name"],
                "relative_x_m": float(row["relative_x_m"]),
                "relative_y_m": float(row["relative_y_m"]),
                "relative_vx_mps": float(row["relative_vx_mps"]),
                "relative_vy_mps": float(row["relative_vy_mps"]),
                "ego_speed_mps": float(row["ego_speed_mps"]),
            })
    return rows


def time_to_collision(x: float, y: float, vx: float, vy: float) -> float:
    closing_speed = -(x * vx + y * vy) / max(math.hypot(x, y), 1e-6)
    if closing_speed <= 0:
        return float("inf")
    return math.hypot(x, y) / closing_speed


def lateral_gate(y: float, class_name: str) -> float:
    width = {"pedestrian": 2.2, "bicycle": 2.0, "vehicle": 1.7}.get(class_name, 1.8)
    return max(0.0, 1.0 - abs(y) / width)


def advise(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    scored = []
    for row in rows:
        ttc = time_to_collision(
            float(row["relative_x_m"]),
            float(row["relative_y_m"]),
            float(row["relative_vx_mps"]),
            float(row["relative_vy_mps"]),
        )
        gate = lateral_gate(float(row["relative_y_m"]), str(row["class_name"]))
        class_weight = {"pedestrian": 1.18, "bicycle": 1.08, "vehicle": 1.0}.get(str(row["class_name"]), 1.0)
        ttc_risk = 0.0 if math.isinf(ttc) else max(0.0, min((4.5 - ttc) / 4.5, 1.4))
        risk = class_weight * (0.72 * ttc_risk + 0.28 * gate)
        if risk >= 0.75:
            action = "emergency_brake"
        elif risk >= 0.45:
            action = "soft_brake"
        elif risk >= 0.25:
            action = "prepare_brake"
        else:
            action = "keep_speed"
        scored.append({
            **row,
            "ttc_s": round(ttc, 3) if not math.isinf(ttc) else 999.0,
            "lane_conflict": round(gate, 4),
            "collision_risk": round(float(risk), 4),
            "brake_action": action,
        })
    return scored


def summarize(scored: list[dict[str, float | str]]) -> dict[str, object]:
    risk = np.array([float(row["collision_risk"]) for row in scored])
    ttc = [float(row["ttc_s"]) for row in scored if float(row["ttc_s"]) < 999]
    return {
        "source": "CARLA object track log",
        "objects": len(set(str(row["object_id"]) for row in scored)),
        "records": len(scored),
        "max_collision_risk": round(float(risk.max()), 4),
        "min_ttc_s": round(min(ttc), 3),
        "emergency_brake_frames": sum(row["brake_action"] == "emergency_brake" for row in scored),
    }


def plot(scored: list[dict[str, float | str]], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in scored:
        grouped[str(row["object_id"])].append(row)
    paths = []

    path = output / "carla_ttc_by_object.png"
    plt.figure(figsize=(8.4, 4.9))
    for object_id, rows in grouped.items():
        rows = sorted(rows, key=lambda r: float(r["time_s"]))
        plt.plot([float(r["time_s"]) for r in rows], [float(r["ttc_s"]) for r in rows], marker="o", label=f"id {object_id}")
    plt.axhline(2.0, color="#eb5757", linestyle="--", label="urgent TTC")
    plt.xlabel("time (s)")
    plt.ylabel("TTC (s)")
    plt.ylim(0, 8)
    plt.title("CARLA time-to-collision by tracked object")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    path = output / "carla_collision_risk_map.png"
    plt.figure(figsize=(7.2, 5.4))
    x = [float(row["relative_x_m"]) for row in scored]
    y = [float(row["relative_y_m"]) for row in scored]
    c = [float(row["collision_risk"]) for row in scored]
    plt.scatter(y, x, c=c, cmap="inferno", s=85)
    plt.colorbar(label="collision risk")
    plt.xlabel("relative y (m)")
    plt.ylabel("relative x ahead (m)")
    plt.title("Object position risk map")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    return paths


def run(output: Path, data: Path = DATA) -> dict[str, object]:
    scored = advise(load_tracks(data))
    files = plot(scored, output)
    csv_path = output / "carla_ttc_advice.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)
    files.append(csv_path)
    report = summarize(scored)
    report["generated_files"] = [p.name for p in files]
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/pr_assets/carla_ttc_brake_advisor"))
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.data), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
