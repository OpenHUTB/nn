"""Generate contact-balance data by stepping the MuJoCo MJCF model.

This script requires the official ``mujoco`` Python package. It loads
``mujoco_quadruped_balance.xml``, steps the simulator, and exports MuJoCo
``sensordata`` plus ``cfrc_ext``-style contact force records.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MJCF = HERE / "sample_data" / "mujoco_quadruped_balance.xml"
OUT = HERE / "sample_data" / "mujoco_sensor_export.csv"


def generate(output: Path = OUT, model_path: Path = MJCF, steps: int = 160, stride: int = 10) -> Path:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    left_front = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_front_foot")
    right_front = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_front_foot")
    left_rear = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_rear_foot")
    right_rear = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_rear_foot")

    data.qpos[:7] = np.array([0.0, 0.0, 0.535, 1.0, 0.0, 0.0, 0.0])
    rows = []
    for step in range(steps):
        if 45 <= step < 75:
            data.xfrc_applied[1, 0] = 38.0
            data.xfrc_applied[1, 1] = 22.0
        elif 115 <= step < 145:
            data.xfrc_applied[1, 0] = -34.0
            data.xfrc_applied[1, 1] = -28.0
        else:
            data.xfrc_applied[:] = 0.0
        mujoco.mj_step(model, data)
        if step % stride == 0:
            qmat = np.zeros(9)
            mujoco.mju_quat2Mat(qmat, data.qpos[3:7])
            roll_rate = float(data.qvel[3])
            rows.append({
                "time_s": round(float(data.time), 3),
                "sensordata_com_x": round(float(data.sensordata[0]), 5),
                "sensordata_com_y": round(float(data.sensordata[1]), 5),
                "sensordata_com_z": round(float(data.sensordata[2]), 5),
                "cfrc_lf_z": round(abs(float(data.sensordata[5])), 5),
                "cfrc_rf_z": round(abs(float(data.sensordata[8])), 5),
                "cfrc_lr_z": round(abs(float(data.sensordata[11])), 5),
                "cfrc_rr_z": round(abs(float(data.sensordata[14])), 5),
                "qvel_roll": round(roll_rate, 5),
                "solver_niter": int(data.solver_niter[0]),
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
    parser.add_argument("--model", type=Path, default=MJCF)
    args = parser.parse_args()
    path = generate(args.output, args.model)
    print(f"generated {path}")


if __name__ == "__main__":
    main()
