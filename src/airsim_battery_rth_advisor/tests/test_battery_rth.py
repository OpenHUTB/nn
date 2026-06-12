from __future__ import annotations
import sys, tempfile
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from battery_rth import advise, load_rows, run

def test_rth_decision() -> None:
    scored = advise(load_rows())
    assert any(r["rth_action"] == "return_now" for r in scored)
    assert min(float(r["battery_margin"]) for r in scored) < 8

def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metrics = run(Path(tmp))
        assert metrics["source"] == "AirSim battery telemetry log"
        assert (Path(tmp) / "airsim_battery_requirement.png").exists()
        assert (Path(tmp) / "airsim_battery_margin.png").exists()

if __name__ == "__main__":
    test_rth_decision(); test_exports(); print("airsim_battery_rth_advisor tests passed")
