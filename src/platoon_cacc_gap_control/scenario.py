"""Scenario generation for cooperative adaptive cruise control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlatoonScenario:
    """Parameters and lead-vehicle profile for a platoon simulation."""

    time: np.ndarray
    leader_position: np.ndarray
    leader_speed: np.ndarray
    leader_acceleration: np.ndarray
    vehicle_count: int
    standstill_gap: float
    time_headway: float
    vehicle_length: float
    initial_speed: float
    communication_delay_s: float
    dropout_windows: list[tuple[float, float]]


def leader_acceleration_profile(time: np.ndarray) -> np.ndarray:
    """Create a repeatable leader profile with braking and recovery events."""
    acceleration = np.zeros_like(time)
    acceleration[(time >= 8.0) & (time < 13.0)] = -1.45
    acceleration[(time >= 18.0) & (time < 24.0)] = 1.05
    acceleration[(time >= 34.0) & (time < 39.0)] = -1.10
    acceleration[(time >= 44.0) & (time < 51.0)] = 0.85
    return acceleration


def integrate_leader(time: np.ndarray, initial_speed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate leader speed and position from a planned acceleration signal."""
    dt = float(time[1] - time[0])
    acceleration = leader_acceleration_profile(time)
    speed = np.zeros_like(time)
    position = np.zeros_like(time)
    speed[0] = initial_speed

    for index in range(1, len(time)):
        speed[index] = np.clip(speed[index - 1] + acceleration[index - 1] * dt, 2.0, 26.0)
        position[index] = position[index - 1] + speed[index - 1] * dt + 0.5 * acceleration[index - 1] * dt * dt

    return position, speed, acceleration


def make_demo_scenario() -> PlatoonScenario:
    """Return a deterministic five-car platoon scenario."""
    dt = 0.1
    time = np.arange(0.0, 62.0 + dt, dt)
    initial_speed = 18.0
    leader_position, leader_speed, leader_acceleration = integrate_leader(time, initial_speed)
    return PlatoonScenario(
        time=time,
        leader_position=leader_position,
        leader_speed=leader_speed,
        leader_acceleration=leader_acceleration,
        vehicle_count=5,
        standstill_gap=6.0,
        time_headway=0.85,
        vehicle_length=4.6,
        initial_speed=initial_speed,
        communication_delay_s=0.35,
        dropout_windows=[(27.0, 31.0), (48.0, 50.5)],
    )
