"""Text reporting for the rain sensor fusion demo."""

from __future__ import annotations

from ..fusion.deep_fusion import FusionResult
from ..sensor.synthetic import SensorSnapshot


def _pct(value: float) -> str:
    return f"{100.0 * value:5.1f}%"


def format_result(snapshot: SensorSnapshot, result: FusionResult) -> str:
    """Render a compact CLI report."""

    weights = result.modality_weights
    box = result.box_3d
    return "\n".join(
        [
            f"Scenario: {snapshot.scenario_id}",
            f"Weather: rain={_pct(float(snapshot.weather[0]))}, "
            f"fog={_pct(float(snapshot.weather[1]))}, wet_road={_pct(float(snapshot.weather[2]))}",
            "Deep fusion weights: "
            f"camera={_pct(weights['camera'])}, lidar={_pct(weights['lidar'])}, radar={_pct(weights['radar'])}",
            f"Object confidence: {_pct(result.object_confidence)}",
            f"Risk score: {_pct(result.risk_score)} "
            f"(network={_pct(result.network_risk)}, physical={_pct(result.physical_risk)})",
            f"Recommended action: {result.recommended_action}",
            "3D box: "
            f"x={box['center_x_m']}m, y={box['center_y_m']}m, "
            f"lwh=({box['length_m']}, {box['width_m']}, {box['height_m']})m, "
            f"conf={_pct(box['confidence'])}",
        ]
    )
