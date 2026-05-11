"""Scenario generation for bus headway control and bunching suppression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BusScenario:
    stop_positions_km: np.ndarray
    passenger_arrival_rate_per_min: np.ndarray
    bus_count: int
    target_headway_min: float
    cruise_speed_kmph: float
    dwell_base_min: float
    dwell_per_passenger_min: float
    boarding_capacity: int
    horizon_min: float
    delay_stop_index: int
    delay_bus_index: int
    delay_min: float


def make_demo_scenario() -> BusScenario:
    """Create a deterministic corridor with uneven passenger demand."""
    stop_positions = np.array([0.0, 0.8, 1.7, 2.6, 3.8, 5.1, 6.4, 7.6, 8.7, 9.6])
    arrival_rate = np.array([1.8, 1.2, 2.4, 1.6, 3.1, 2.6, 1.9, 1.5, 1.1, 0.4])
    return BusScenario(
        stop_positions_km=stop_positions,
        passenger_arrival_rate_per_min=arrival_rate,
        bus_count=8,
        target_headway_min=6.0,
        cruise_speed_kmph=24.0,
        dwell_base_min=0.28,
        dwell_per_passenger_min=0.035,
        boarding_capacity=52,
        horizon_min=78.0,
        delay_stop_index=3,
        delay_bus_index=2,
        delay_min=4.5,
    )


def travel_time_between_stops(scenario: BusScenario) -> np.ndarray:
    distance = np.diff(scenario.stop_positions_km)
    return distance / scenario.cruise_speed_kmph * 60.0
