"""Generate MuJoCo double-pendulum control data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODEL = HERE / "sample_data" / "mujoco_double_pendulum.xml"
OUT = HERE / "sample_data" / "mujoco_pendulum_rollout.csv"


def generate(output: Path = OUT, model_path: Path = MODEL, steps: int = 240) -> Path:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = [0.65, -0.45]
    rows = []
    for i in range(steps):
        target = np.array([0.0, 0.0])
        kp = np.array([2.6, 1.8])
        kd = np.array([0.35, 0.28])
        data.ctrl[:] = -kp * (data.qpos[:] - target) - kd * data.qvel[:]
        mujoco.mj_step(model, data)
        power = float(np.sum(np.abs(data.ctrl[:] * data.qvel[:2])))
        rows.append({
            "time_s": round(float(data.time), 4),
            "q_shoulder": round(float(data.qpos[0]), 6),
            "q_elbow": round(float(data.qpos[1]), 6),
            "v_shoulder": round(float(data.qvel[0]), 6),
            "v_elbow": round(float(data.qvel[1]), 6),
            "ctrl_shoulder": round(float(data.ctrl[0]), 6),
            "ctrl_elbow": round(float(data.ctrl[1]), 6),
            "instant_power": round(power, 6),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    print(f"generated {generate(args.output)}")


if __name__ == "__main__":
    main()
