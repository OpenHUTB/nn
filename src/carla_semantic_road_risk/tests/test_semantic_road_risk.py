from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from semantic_road_risk import analyze, load_grid, run


def test_detects_semantic_hazards() -> None:
    metrics = analyze(load_grid())
    assert metrics["source"] == "CARLA SemanticSegmentation camera label grid"
    assert metrics["vehicle_pixels"] > 0
    assert metrics["pedestrian_pixels"] > 0


def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metrics = run(Path(tmp))
        assert metrics["risk_score"] > 0.3
        assert (Path(tmp) / "carla_semantic_camera_view.png").exists()
        assert (Path(tmp) / "carla_semantic_risk_heatmap.png").exists()


if __name__ == "__main__":
    test_detects_semantic_hazards()
    test_exports()
    print("carla_semantic_road_risk tests passed")
