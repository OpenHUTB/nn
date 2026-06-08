"""Analyze MuJoCo cart-pole recovery rollout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path(__file__).with_name("sample_data") / "mujoco_cartpole_rollout.csv"


def load_rows(path: Path = DATA) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def analyze(rows: list[dict[str, float]]) -> dict[str, object]:
    t = np.array([r["time_s"] for r in rows])
    angle = np.abs(np.array([r["pole_angle_rad"] for r in rows]))
    force = np.abs(np.array([r["control_force"] for r in rows]))
    stable_idx = np.where(angle < 0.06)[0]
    stable_time = float(t[stable_idx[0]]) if len(stable_idx) else float(t[-1])
    effort = float(np.trapezoid(force, t))
    return {
        "source": "MuJoCo cart-pole recovery rollout",
        "records": len(rows),
        "initial_angle_rad": round(float(angle[0]), 6),
        "final_angle_rad": round(float(angle[-1]), 6),
        "max_control_force": round(float(force.max()), 6),
        "control_effort": round(effort, 6),
        "stable_time_s": round(stable_time, 4),
    }


def plot(rows: list[dict[str, float]], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    t = [r["time_s"] for r in rows]
    paths = []
    p = output / "mujoco_cartpole_recovery_curve.png"
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, [r["pole_angle_rad"] for r in rows], label="pole angle")
    plt.plot(t, [r["cart_x_m"] for r in rows], label="cart position")
    plt.xlabel("time (s)")
    plt.title("MuJoCo cart-pole recovery")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(p)
    p = output / "mujoco_cartpole_control_force.png"
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, [r["control_force"] for r in rows], color="#eb5757")
    plt.xlabel("time (s)")
    plt.ylabel("control force")
    plt.title("Cart actuator control force")
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
    parser.add_argument("--output", type=Path, default=Path("docs/pr_assets/mujoco_cartpole_recovery"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
