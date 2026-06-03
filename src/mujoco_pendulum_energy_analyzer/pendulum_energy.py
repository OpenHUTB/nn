"""Analyze MuJoCo double-pendulum control energy and stability."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path(__file__).with_name("sample_data") / "mujoco_pendulum_rollout.csv"


def load_rows(path: Path = DATA) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def analyze(rows: list[dict[str, float]]) -> dict[str, object]:
    t = np.array([r["time_s"] for r in rows])
    power = np.array([r["instant_power"] for r in rows])
    err = np.array([abs(r["q_shoulder"]) + abs(r["q_elbow"]) for r in rows])
    energy = float(np.trapezoid(power, t))
    settle_idx = np.where(err < 0.18)[0]
    settle_time = float(t[settle_idx[0]]) if len(settle_idx) else float(t[-1])
    return {
        "source": "MuJoCo double pendulum actuator rollout",
        "records": len(rows),
        "total_energy": round(energy, 6),
        "max_power": round(float(power.max()), 6),
        "final_angle_error": round(float(err[-1]), 6),
        "settle_time_s": round(settle_time, 4),
    }


def plot(rows: list[dict[str, float]], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    t = [r["time_s"] for r in rows]
    paths = []
    p = output / "mujoco_pendulum_angles.png"
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, [r["q_shoulder"] for r in rows], label="shoulder")
    plt.plot(t, [r["q_elbow"] for r in rows], label="elbow")
    plt.xlabel("time (s)")
    plt.ylabel("angle (rad)")
    plt.title("MuJoCo double pendulum joint angles")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(p)
    p = output / "mujoco_pendulum_power.png"
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, [r["instant_power"] for r in rows], color="#eb5757")
    plt.xlabel("time (s)")
    plt.ylabel("instant power")
    plt.title("MuJoCo actuator power consumption")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(p)
    return paths


def run(output: Path) -> dict[str, object]:
    rows = load_rows()
    metrics = analyze(rows)
    files = plot(rows, output)
    metrics["generated_files"] = [p.name for p in files]
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/pr_assets/mujoco_pendulum_energy_analyzer"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
