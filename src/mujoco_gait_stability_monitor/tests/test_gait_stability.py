from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from gait_stability import load_log, run, score_rows


def test_mujoco_log_detects_fall_warning() -> None:
    scored = score_rows(load_log())
    assert any(row["stability_state"] == "fall_warning" for row in scored)
    assert min(float(row["com_height_m"]) for row in scored) < 0.9


def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metrics = run(Path(tmp))
        assert metrics["source"] == "MuJoCo humanoid gait state log"
        assert metrics["fall_warning_steps"] > 0
        assert (Path(tmp) / "mujoco_com_height_risk.png").exists()
        assert (Path(tmp) / "mujoco_pitch_risk_scatter.png").exists()


if __name__ == "__main__":
    test_mujoco_log_detects_fall_warning()
    test_exports()
    print("mujoco_gait_stability_monitor tests passed")
