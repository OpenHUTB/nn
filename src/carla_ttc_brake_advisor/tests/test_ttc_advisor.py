from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ttc_advisor import advise, load_tracks, run, time_to_collision


def test_ttc_and_brake_action() -> None:
    assert 1.8 < time_to_collision(10.0, 0.0, -5.0, 0.0) < 2.2
    scored = advise(load_tracks())
    assert any(row["brake_action"] == "emergency_brake" for row in scored)
    assert min(float(row["ttc_s"]) for row in scored) < 2.0


def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metrics = run(Path(tmp))
        assert metrics["source"] == "CARLA object track log"
        assert metrics["emergency_brake_frames"] > 0
        assert (Path(tmp) / "carla_ttc_by_object.png").exists()
        assert (Path(tmp) / "carla_collision_risk_map.png").exists()


if __name__ == "__main__":
    test_ttc_and_brake_action()
    test_exports()
    print("carla_ttc_brake_advisor tests passed")
