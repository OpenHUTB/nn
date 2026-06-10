from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from radar_warning import analyze, load_rows, run


def test_radar_warning_detects_braking() -> None:
    rows = analyze(load_rows())
    assert any(r["brake_action"] == "emergency_brake" for r in rows)
    assert min(float(r["ttc_s"]) for r in rows) < 2.0


def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metrics = run(Path(tmp))
        assert metrics["source"] == "CARLA RadarDetection sensor log"
        assert (Path(tmp) / "carla_radar_ttc_curve.png").exists()
        assert (Path(tmp) / "carla_radar_detection_map.png").exists()


if __name__ == "__main__":
    test_radar_warning_detects_braking()
    test_exports()
    print("carla_radar_collision_warning tests passed")
