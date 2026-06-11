"""MuJoCo gait stability scoring from humanoid simulator logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path(__file__).with_name("sample_data") / "mujoco_gait_log.csv"


def load_log(path: Path = DATA) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def score_rows(rows: list[dict[str, float]]) -> list[dict[str, float | str]]:
    scored = []
    prev_contact = None
    for row in rows:
        contact_pair = (int(row["left_foot_contact"]), int(row["right_foot_contact"]))
        support_switch = 0 if prev_contact is None or contact_pair != prev_contact else 1
        prev_contact = contact_pair
        height_risk = max(0.0, (0.95 - row["com_height_m"]) / 0.25)
        pitch_risk = min(abs(row["torso_pitch_deg"]) / 18.0, 1.4)
        speed_risk = max(0.0, (row["com_velocity_mps"] - 0.9) / 0.45)
        asymmetry = abs(row["left_knee_deg"] - row["right_knee_deg"]) / 30.0
        risk = 0.34 * height_risk + 0.28 * pitch_risk + 0.18 * speed_risk + 0.14 * min(asymmetry, 1.2) + 0.06 * support_switch
        state = "fall_warning" if risk >= 0.65 else "unstable" if risk >= 0.40 else "stable"
        scored.append({**row, "gait_risk": round(float(risk), 4), "stability_state": state})
    return scored


def summarize(scored: list[dict[str, float | str]]) -> dict[str, object]:
    risk = np.array([float(row["gait_risk"]) for row in scored])
    return {
        "source": "MuJoCo humanoid gait state log",
        "steps": len(scored),
        "fall_warning_steps": sum(row["stability_state"] == "fall_warning" for row in scored),
        "unstable_steps": sum(row["stability_state"] == "unstable" for row in scored),
        "max_gait_risk": round(float(risk.max()), 4),
        "lowest_com_height_m": round(min(float(row["com_height_m"]) for row in scored), 3),
    }


def plot(scored: list[dict[str, float | str]], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    t = np.array([float(row["time_s"]) for row in scored])
    height = np.array([float(row["com_height_m"]) for row in scored])
    risk = np.array([float(row["gait_risk"]) for row in scored])
    pitch = np.array([float(row["torso_pitch_deg"]) for row in scored])
    paths = []

    path = output / "mujoco_com_height_risk.png"
    fig, ax1 = plt.subplots(figsize=(8.3, 4.8))
    ax1.plot(t, height, marker="o", color="#2f80ed")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("COM height (m)")
    ax2 = ax1.twinx()
    ax2.plot(t, risk, marker="s", color="#eb5757")
    ax2.set_ylabel("gait risk")
    ax1.set_title("MuJoCo COM height and gait stability risk")
    fig.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    path = output / "mujoco_pitch_risk_scatter.png"
    plt.figure(figsize=(7.3, 5))
    plt.scatter(pitch, risk, c=height, cmap="viridis", s=90)
    plt.colorbar(label="COM height")
    plt.xlabel("torso pitch (deg)")
    plt.ylabel("gait risk")
    plt.title("Torso pitch relation to fall warning")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    return paths


def run(output: Path, data: Path = DATA) -> dict[str, object]:
    scored = score_rows(load_log(data))
    files = plot(scored, output)
    csv_path = output / "mujoco_gait_scores.csv"
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
    parser.add_argument("--output", type=Path, default=Path("docs/pr_assets/mujoco_gait_stability_monitor"))
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.data), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
