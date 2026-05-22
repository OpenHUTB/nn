from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from queue_delay import estimate, load_rows, run


def test_queue_delay_action() -> None:
    scored = estimate(load_rows())
    assert any(r["signal_action"] == "extend_green" for r in scored)
    assert max(float(r["predicted_delay_s"]) for r in scored) > 50


def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metrics = run(Path(tmp))
        assert metrics["source"] == "CARLA intersection queue log"
        assert (Path(tmp) / "carla_queue_delay_curve.png").exists()
        assert (Path(tmp) / "carla_signal_pressure.png").exists()


if __name__ == "__main__":
    test_queue_delay_action()
    test_exports()
    print("carla_queue_delay_estimator tests passed")
