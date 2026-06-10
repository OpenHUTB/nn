from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from airsim_safety import load_log, run, score_rows


def test_airsim_log_detects_risk() -> None:
    scored = score_rows(load_log())
    assert any(row["risk_level"] == "critical" for row in scored)
    assert any(row["recommended_action"] != "keep_course" for row in scored)
    assert min(float(row["min_clearance_m"]) for row in scored) < 1.2


def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metrics = run(Path(tmp))
        assert metrics["source"] == "AirSim LiDAR and flight-state log"
        assert metrics["critical_frames"] > 0
        assert "brake_and_climb" in metrics["avoidance_actions"]
        assert (Path(tmp) / "airsim_clearance_risk.png").exists()
        assert (Path(tmp) / "airsim_altitude_attitude.png").exists()
        assert (Path(tmp) / "airsim_action_distribution.png").exists()


if __name__ == "__main__":
    test_airsim_log_detects_risk()
    test_exports()
    print("airsim_lidar_safety_monitor tests passed")
