from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from imu_drift import analyze, load_rows, run


def test_detects_drift() -> None:
    scored = analyze(load_rows())
    assert any(r["monitor_action"] == "relocalize" for r in scored)
    assert max(float(r["gps_drift_m"]) for r in scored) > 2.0


def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metrics = run(Path(tmp))
        assert metrics["source"] == "AirSim multirotor IMU and GPS telemetry log"
        assert (Path(tmp) / "airsim_runtime_trajectory_drift.png").exists()
        assert (Path(tmp) / "airsim_imu_health_curve.png").exists()


if __name__ == "__main__":
    test_detects_drift()
    test_exports()
    print("airsim_imu_drift_monitor tests passed")
