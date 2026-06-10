"""Interpretable safety heads on top of fused deep features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..config import FusionConfig
from ..sensor.synthetic import SensorSnapshot


@dataclass(frozen=True)
class BoundingBox3D:
    """Minimal 3D detection output for a single frontal obstacle."""

    center_x_m: float
    center_y_m: float
    center_z_m: float
    length_m: float
    width_m: float
    height_m: float
    confidence: float

    def as_dict(self) -> dict[str, float]:
        return {
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "center_z_m": self.center_z_m,
            "length_m": self.length_m,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "confidence": self.confidence,
        }


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def estimate_physical_risk(snapshot: SensorSnapshot) -> float:
    """Estimate risk from distance, closing speed, rain, and uncertainty."""

    close = float(snapshot.radar[0])
    closing = float(snapshot.radar[1])
    obstacle = float(snapshot.radar[3])
    rain = float(snapshot.weather[0])
    sensor_uncertainty = 1.0 - float(np.mean([snapshot.camera[0], snapshot.lidar[0], snapshot.radar[2]]))
    risk = obstacle * (0.58 * close + 0.30 * closing + 0.12 * rain) + 0.12 * sensor_uncertainty
    return _clip01(risk)


def estimate_object_confidence(
    snapshot: SensorSnapshot,
    weights: Mapping[str, float] | np.ndarray,
) -> float:
    """Fuse objectness cues according to modality weights."""

    if isinstance(weights, Mapping):
        weight_array = np.array(
            [weights["camera"], weights["lidar"], weights["radar"]],
            dtype=np.float32,
        )
    else:
        weight_array = np.asarray(weights, dtype=np.float32)

    cues = np.array(
        [snapshot.camera[4], snapshot.lidar[1], snapshot.radar[3]],
        dtype=np.float32,
    )
    weight_array = weight_array / max(float(weight_array.sum()), 1e-6)
    return _clip01(float(np.dot(weight_array, cues)))


def estimate_box3d(
    snapshot: SensorSnapshot,
    confidence: float,
    *,
    max_distance_m: float,
) -> BoundingBox3D:
    """Build a coarse 3D box from the synthetic fused obstacle state."""

    distance_m = 5.0 + (1.0 - float(snapshot.radar[0])) * (max_distance_m - 5.0)
    width_m = 1.8 + 0.25 * float(snapshot.radar[3])
    length_m = 4.0 + 0.50 * float(snapshot.radar[3])
    height_m = 1.45 + 0.20 * float(snapshot.radar[3])
    return BoundingBox3D(
        center_x_m=round(float(distance_m), 3),
        center_y_m=round(float(snapshot.lateral_offset_m), 3),
        center_z_m=round(height_m / 2.0, 3),
        length_m=round(length_m, 3),
        width_m=round(width_m, 3),
        height_m=round(height_m, 3),
        confidence=round(_clip01(confidence), 3),
    )


def choose_action(risk_score: float, config: FusionConfig) -> str:
    """Return a planning hint from the risk score."""

    if risk_score >= config.brake_threshold:
        return "brake"
    if risk_score >= config.slow_down_threshold:
        return "slow_down"
    return "monitor"
