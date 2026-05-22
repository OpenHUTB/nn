"""Synthetic rain-weather sensor snapshots.

The generator keeps the project runnable without CARLA while preserving the
same multimodal shape expected from a camera, LiDAR, and millimeter-wave radar
fusion stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


SENSOR_NAMES: Tuple[str, str, str] = ("camera", "lidar", "radar")
DEMO_SCENARIOS: Tuple[str, str, str, str] = (
    "clear",
    "light_rain",
    "heavy_rain",
    "foggy_storm",
)

_DEMO_SCENARIO_PARAMS = {
    "clear": dict(
        rain=0.0,
        fog=0.0,
        wet_road=0.0,
        obstacle_distance_m=42.0,
        closing_speed_mps=2.0,
        lateral_offset_m=0.1,
        seed=1,
    ),
    "light_rain": dict(
        rain=0.28,
        fog=0.12,
        wet_road=0.45,
        obstacle_distance_m=31.0,
        closing_speed_mps=4.0,
        lateral_offset_m=-0.2,
        seed=2,
    ),
    "heavy_rain": dict(
        rain=0.82,
        fog=0.42,
        wet_road=0.90,
        obstacle_distance_m=17.0,
        closing_speed_mps=8.0,
        lateral_offset_m=-0.4,
        seed=3,
    ),
    "foggy_storm": dict(
        rain=0.68,
        fog=0.78,
        wet_road=0.85,
        obstacle_distance_m=22.0,
        closing_speed_mps=6.5,
        lateral_offset_m=0.35,
        seed=4,
    ),
}


def _clip01(value: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)


@dataclass(frozen=True)
class SensorSnapshot:
    """One synchronized multimodal frame.

    Camera features:
        visibility, contrast, sharpness, lane visibility, object appearance.
    LiDAR features:
        point density, cluster confidence, range consistency, splash noise.
    Radar features:
        normalized closeness, normalized closing speed, doppler confidence,
        obstacle presence.
    Weather features:
        rain intensity, fog density, wet road level.
    """

    camera: np.ndarray
    lidar: np.ndarray
    radar: np.ndarray
    weather: np.ndarray
    scenario_id: str = "custom"
    obstacle_distance_m: float = 30.0
    closing_speed_mps: float = 0.0
    lateral_offset_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera", _clip01(self.camera))
        object.__setattr__(self, "lidar", _clip01(self.lidar))
        object.__setattr__(self, "radar", _clip01(self.radar))
        object.__setattr__(self, "weather", _clip01(self.weather))
        self._validate()

    def _validate(self) -> None:
        expected = {
            "camera": (5,),
            "lidar": (4,),
            "radar": (4,),
            "weather": (3,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")

    def as_arrays(self) -> Dict[str, np.ndarray]:
        """Return a copy-safe dictionary of feature arrays."""

        return {
            "camera": self.camera.copy(),
            "lidar": self.lidar.copy(),
            "radar": self.radar.copy(),
            "weather": self.weather.copy(),
        }


def make_sensor_snapshot(
    *,
    rain: float,
    fog: float,
    wet_road: float,
    obstacle_distance_m: float,
    closing_speed_mps: float,
    lateral_offset_m: float = 0.0,
    scenario_id: str = "custom",
    seed: int | None = None,
) -> SensorSnapshot:
    """Create a deterministic synthetic multimodal rain scenario."""

    rng = np.random.default_rng(seed)
    rain = float(_clip01(rain))
    fog = float(_clip01(fog))
    wet_road = float(_clip01(wet_road))
    obstacle_distance_m = float(np.clip(obstacle_distance_m, 5.0, 80.0))
    closing_speed_mps = float(np.clip(closing_speed_mps, 0.0, 20.0))

    closeness = 1.0 - (obstacle_distance_m - 5.0) / 75.0
    closing_norm = closing_speed_mps / 20.0

    camera_quality = np.clip(
        1.0 - 0.58 * rain - 0.32 * fog - 0.10 * wet_road
        + rng.normal(0.0, 0.015),
        0.05,
        1.0,
    )
    lidar_quality = np.clip(
        1.0 - 0.34 * rain - 0.12 * fog + rng.normal(0.0, 0.015),
        0.10,
        1.0,
    )
    radar_quality = np.clip(
        0.72 + 0.18 * rain - 0.04 * fog + rng.normal(0.0, 0.012),
        0.20,
        1.0,
    )

    camera = np.array(
        [
            camera_quality,
            np.clip(camera_quality - 0.10 * wet_road, 0.0, 1.0),
            np.clip(camera_quality - 0.22 * rain, 0.0, 1.0),
            np.clip(camera_quality - 0.16 * fog, 0.0, 1.0),
            np.clip((0.25 + 0.75 * closeness) * (0.55 + 0.45 * camera_quality), 0.0, 1.0),
        ],
        dtype=np.float32,
    )

    lidar = np.array(
        [
            lidar_quality,
            np.clip((0.20 + 0.80 * closeness) * (0.68 + 0.32 * lidar_quality), 0.0, 1.0),
            np.clip(lidar_quality - 0.10 * wet_road, 0.0, 1.0),
            np.clip(0.08 + 0.78 * rain + 0.12 * wet_road + rng.normal(0.0, 0.01), 0.0, 1.0),
        ],
        dtype=np.float32,
    )

    radar = np.array(
        [
            closeness,
            closing_norm,
            radar_quality,
            np.clip(0.20 + 0.80 * closeness + 0.12 * closing_norm, 0.0, 1.0),
        ],
        dtype=np.float32,
    )

    weather = np.array([rain, fog, wet_road], dtype=np.float32)
    return SensorSnapshot(
        camera=camera,
        lidar=lidar,
        radar=radar,
        weather=weather,
        scenario_id=scenario_id,
        obstacle_distance_m=obstacle_distance_m,
        closing_speed_mps=closing_speed_mps,
        lateral_offset_m=float(lateral_offset_m),
    )


def make_demo_snapshot(name: str = "heavy_rain") -> SensorSnapshot:
    """Return a named scenario for the CLI and tests."""

    if name not in _DEMO_SCENARIO_PARAMS:
        available = ", ".join(DEMO_SCENARIOS)
        raise ValueError(f"unknown scenario '{name}', choose from: {available}")
    return make_sensor_snapshot(scenario_id=name, **_DEMO_SCENARIO_PARAMS[name])


def _teacher_weights(snapshot: SensorSnapshot) -> np.ndarray:
    rain, fog, _wet_road = snapshot.weather
    camera_q = float(np.mean(snapshot.camera[:4])) * (1.0 - 0.45 * rain - 0.25 * fog)
    lidar_q = (
        0.32 * snapshot.lidar[0]
        + 0.34 * snapshot.lidar[1]
        + 0.24 * snapshot.lidar[2]
        + 0.10 * (1.0 - snapshot.lidar[3])
    ) * (1.0 - 0.20 * rain)
    radar_q = (
        0.25 * snapshot.radar[0]
        + 0.25 * snapshot.radar[1]
        + 0.35 * snapshot.radar[2]
        + 0.15 * snapshot.radar[3]
    ) * (1.0 + 0.25 * rain + 0.08 * fog)
    raw = np.maximum(np.array([camera_q, lidar_q, radar_q], dtype=np.float32), 1e-4)
    return raw / raw.sum()


def generate_training_batch(
    batch_size: int,
    *,
    seed: int | None = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Generate synthetic features and teacher labels for quick DL training."""

    rng = np.random.default_rng(seed)
    snapshots = [
        make_sensor_snapshot(
            rain=float(rng.uniform(0.0, 1.0)),
            fog=float(rng.uniform(0.0, 0.85)),
            wet_road=float(rng.uniform(0.0, 1.0)),
            obstacle_distance_m=float(rng.uniform(6.0, 80.0)),
            closing_speed_mps=float(rng.uniform(0.0, 14.0)),
            lateral_offset_m=float(rng.uniform(-1.2, 1.2)),
            scenario_id="synthetic_train",
            seed=int(rng.integers(0, 1_000_000)),
        )
        for _ in range(batch_size)
    ]

    features = {
        name: np.stack([getattr(snapshot, name) for snapshot in snapshots]).astype(np.float32)
        for name in ("camera", "lidar", "radar", "weather")
    }

    weights = np.stack([_teacher_weights(snapshot) for snapshot in snapshots]).astype(np.float32)
    close = features["radar"][:, 0:1]
    closing = features["radar"][:, 1:2]
    obstacle = features["radar"][:, 3:4]
    rain = features["weather"][:, 0:1]
    uncertainty = 1.0 - np.mean(
        np.concatenate(
            [
                features["camera"][:, 0:1],
                features["lidar"][:, 0:1],
                features["radar"][:, 2:3],
            ],
            axis=1,
        ),
        axis=1,
        keepdims=True,
    )
    risk = np.clip(obstacle * (0.58 * close + 0.30 * closing + 0.12 * rain) + 0.12 * uncertainty, 0.0, 1.0)
    objectness = np.clip(
        weights[:, 0:1] * features["camera"][:, 4:5]
        + weights[:, 1:2] * features["lidar"][:, 1:2]
        + weights[:, 2:3] * features["radar"][:, 3:4],
        0.0,
        1.0,
    )
    labels = {
        "weights": weights,
        "risk": risk.astype(np.float32),
        "objectness": objectness.astype(np.float32),
        "rain": rain.astype(np.float32),
    }
    return features, labels
