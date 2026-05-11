"""Road surface and emergency braking scenario for ABS slip control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RoadSegment:
    name: str
    start_m: float
    end_m: float
    friction_mu: float


@dataclass
class BrakingScenario:
    time: np.ndarray
    initial_speed_mps: float
    gravity: float
    wheel_radius_m: float
    optimal_slip: float
    road_segments: list[RoadSegment]


def make_demo_scenario() -> BrakingScenario:
    """Create a mixed-friction emergency braking scenario."""
    dt = 0.02
    time = np.arange(0.0, 12.0 + dt, dt)
    segments = [
        RoadSegment("dry asphalt", 0.0, 18.0, 0.86),
        RoadSegment("wet asphalt", 18.0, 55.0, 0.50),
        RoadSegment("thin ice", 55.0, 110.0, 0.22),
        RoadSegment("recovered asphalt", 110.0, 260.0, 0.72),
    ]
    return BrakingScenario(
        time=time,
        initial_speed_mps=30.0,
        gravity=9.81,
        wheel_radius_m=0.31,
        optimal_slip=0.18,
        road_segments=segments,
    )


def friction_at_position(position_m: float, scenario: BrakingScenario) -> float:
    """Return the road friction coefficient at the vehicle position."""
    for segment in scenario.road_segments:
        if segment.start_m <= position_m < segment.end_m:
            return segment.friction_mu
    return scenario.road_segments[-1].friction_mu


def friction_profile(distance: np.ndarray, scenario: BrakingScenario) -> np.ndarray:
    """Vectorized friction lookup for plotting."""
    return np.array([friction_at_position(float(value), scenario) for value in distance])
