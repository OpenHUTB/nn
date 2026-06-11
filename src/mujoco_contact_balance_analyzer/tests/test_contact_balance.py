from __future__ import annotations
import sys, tempfile
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT))
from contact_balance import analyze, load_sensor_rows, parse_mjcf, run, sensor_rows_to_balance_rows
def test_balance_action() -> None:
    sensor_rows=load_sensor_rows()
    assert max(r["cfrc_lf_z"] for r in sensor_rows) > 1.0
    rows=analyze(sensor_rows_to_balance_rows(sensor_rows)); assert any(r["balance_action"]=="recover_posture" for r in rows)
    mjcf=parse_mjcf(); assert mjcf["sensor_count"] >= 5; assert mjcf["contact_geom_count"] == 4
def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        m=run(Path(tmp)); assert m["source"] == "MuJoCo MJCF model stepped by mujoco Python, sensordata force sensors export"; assert m["recover_posture_frames"]>0; assert (Path(tmp)/"mujoco_contact_balance_curve.png").exists(); assert (Path(tmp)/"mujoco_support_polygon_replay.png").exists()
if __name__=="__main__": test_balance_action(); test_exports(); print("mujoco_contact_balance_analyzer tests passed")
