"""Generate MuJoCo cart-pole recovery rollout."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODEL = HERE / "sample_data" / "mujoco_cartpole.xml"
OUT = HERE / "sample_data" / "mujoco_cartpole_rollout.csv"


def generate(output: Path = OUT, model_path: Path = MODEL, steps: int = 260) -> Path:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = [0.0, 0.24]
    rows = []
    for _ in range(steps):
        x, theta = float(data.qpos[0]), float(data.qpos[1])
        vx, omega = float(data.qvel[0]), float(data.qvel[1])
        ctrl = -1.1 * x - 1.5 * vx - 18.0 * theta - 3.2 * omega
        data.ctrl[0] = max(-8.0, min(8.0, ctrl))
        mujoco.mj_step(model, data)
        rows.append({
            "time_s": round(float(data.time), 4),
            "cart_x_m": round(float(data.qpos[0]), 6),
            "pole_angle_rad": round(float(data.qpos[1]), 6),
            "cart_velocity_mps": round(float(data.qvel[0]), 6),
            "pole_angular_velocity": round(float(data.qvel[1]), 6),
            "control_force": round(float(data.ctrl[0]), 6),
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
