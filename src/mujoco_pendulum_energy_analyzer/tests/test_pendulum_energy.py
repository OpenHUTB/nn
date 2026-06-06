from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from generate_pendulum_data import generate
from pendulum_energy import analyze, load_rows, run


def test_mujoco_generated_rollout() -> None:
    generate()
    rows = load_rows()
    metrics = analyze(rows)
    assert metrics["source"] == "MuJoCo double pendulum actuator rollout"
    assert metrics["total_energy"] > 0


def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        generate()
        metrics = run(Path(tmp))
        assert metrics["records"] >= 200
        assert (Path(tmp) / "mujoco_pendulum_angles.png").exists()
        assert (Path(tmp) / "mujoco_pendulum_power.png").exists()


if __name__ == "__main__":
    test_mujoco_generated_rollout()
    test_exports()
    print("mujoco_pendulum_energy_analyzer tests passed")
