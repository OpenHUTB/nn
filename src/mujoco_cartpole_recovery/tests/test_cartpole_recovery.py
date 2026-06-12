from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cartpole_recovery import analyze, load_rows, run
from generate_cartpole_data import generate


def test_mujoco_cartpole_rollout() -> None:
    generate()
    metrics = analyze(load_rows())
    assert metrics["source"] == "MuJoCo cart-pole recovery rollout"
    assert metrics["records"] >= 200


def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        generate()
        metrics = run(Path(tmp))
        assert metrics["max_control_force"] > 1
        assert (Path(tmp) / "mujoco_cartpole_recovery_curve.png").exists()
        assert (Path(tmp) / "mujoco_cartpole_control_force.png").exists()


if __name__ == "__main__":
    test_mujoco_cartpole_rollout()
    test_exports()
    print("mujoco_cartpole_recovery tests passed")
