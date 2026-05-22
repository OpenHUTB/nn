"""CARLA intersection queue delay estimator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path(__file__).with_name("sample_data") / "carla_intersection_queue.csv"
PHASE_SCORE = {"red": 1.0, "yellow": 0.55, "green": 0.0}


def load_rows(path: Path = DATA) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({k: (v if k == "signal_phase" else float(v)) for k, v in row.items()})
    return rows


def estimate(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    scored = []
    for row in rows:
        queue_total = float(row["ego_lane_queue"]) + float(row["opposite_lane_queue"])
        pressure = queue_total * (1.0 + PHASE_SCORE[str(row["signal_phase"])]) + 1.6 * float(row["arrival_rate_vpm"])
        discharge_gap = max(0.0, queue_total - float(row["departed_vehicles"]) * 0.35)
        predicted_delay = 0.82 * float(row["mean_wait_s"]) + 1.15 * discharge_gap + 0.35 * pressure
        action = "extend_green" if predicted_delay > 50 else "prepare_green" if predicted_delay > 35 else "keep_cycle"
        scored.append({**row, "queue_total": queue_total, "pressure_score": round(pressure, 3), "predicted_delay_s": round(predicted_delay, 3), "signal_action": action})
    return scored


def plot(scored: list[dict[str, float | str]], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    t = np.array([float(r["time_s"]) for r in scored])
    queue = np.array([float(r["queue_total"]) for r in scored])
    delay = np.array([float(r["predicted_delay_s"]) for r in scored])
    phase = np.array([PHASE_SCORE[str(r["signal_phase"])] for r in scored])
    paths = []

    path = output / "carla_queue_delay_curve.png"
    fig, ax1 = plt.subplots(figsize=(8.3, 4.8))
    ax1.plot(t, queue, marker="o", color="#2f80ed", label="queue")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("queued vehicles")
    ax2 = ax1.twinx()
    ax2.plot(t, delay, marker="s", color="#eb5757", label="predicted delay")
    ax2.set_ylabel("delay (s)")
    ax1.set_title("CARLA intersection queue and predicted delay")
    fig.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    path = output / "carla_signal_pressure.png"
    plt.figure(figsize=(7.5, 4.8))
    plt.scatter(queue, delay, c=phase, cmap="RdYlGn_r", s=90)
    plt.colorbar(label="red/yellow/green phase pressure")
    plt.xlabel("total queue")
    plt.ylabel("predicted delay (s)")
    plt.title("Signal phase pressure map")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    return paths


def run(output: Path, data: Path = DATA) -> dict[str, object]:
    scored = estimate(load_rows(data))
    files = plot(scored, output)
    csv_path = output / "carla_queue_delay_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)
    files.append(csv_path)
    delay = np.array([float(r["predicted_delay_s"]) for r in scored])
    report = {
        "source": "CARLA intersection queue log",
        "records": len(scored),
        "max_predicted_delay_s": round(float(delay.max()), 3),
        "mean_predicted_delay_s": round(float(delay.mean()), 3),
        "extend_green_frames": sum(r["signal_action"] == "extend_green" for r in scored),
        "generated_files": [p.name for p in files],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/pr_assets/carla_queue_delay_estimator"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
