from __future__ import annotations
import sys, tempfile
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PROJECT))
from torque_anomaly import detect, load_rows, run
def test_detects_critical() -> None:
    scored = detect(load_rows()); assert any(r["anomaly_level"] == "critical" for r in scored)
def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        m = run(Path(tmp)); assert m["critical_frames"] > 0; assert (Path(tmp)/"mujoco_torque_anomaly_curve.png").exists()
if __name__ == "__main__": test_detects_critical(); test_exports(); print("mujoco_torque_anomaly_detector tests passed")
