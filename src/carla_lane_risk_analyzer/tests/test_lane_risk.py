from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from lane_risk import compute_risk, load_log, run


def test_carla_log_has_high_risk_frames() -> None:
    scored = compute_risk(load_log())
    assert len(scored) >= 20
    assert any(row["risk_level"] == "high" for row in scored)


def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metrics = run(Path(tmp))
        assert metrics["source"] == "CARLA lane tracking log"
        assert metrics["high_risk_frames"] > 0
        assert (Path(tmp) / "carla_lane_offset_risk.png").exists()
        assert (Path(tmp) / "carla_speed_density_risk.png").exists()


if __name__ == "__main__":
    test_carla_log_has_high_risk_frames()
    test_exports()
    print("carla_lane_risk_analyzer tests passed")
