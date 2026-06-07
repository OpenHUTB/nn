from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from depth_avoidance import advise, load_rows, run


def test_depth_advisor_turns() -> None:
    rows = advise(load_rows())
    assert any(r["avoidance_action"] == "turn_right" for r in rows)
    assert min(float(r["center_depth_m"]) for r in rows) < 2.5


def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metrics = run(Path(tmp))
        assert metrics["source"] == "AirSim DepthPerspective front camera strip log"
        assert (Path(tmp) / "airsim_depth_runtime_view.png").exists()
        assert (Path(tmp) / "airsim_avoidance_action_curve.png").exists()


if __name__ == "__main__":
    test_depth_advisor_turns()
    test_exports()
    print("airsim_depth_obstacle_avoidance tests passed")
