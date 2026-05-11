"""Scenario generation for elevator group dispatch optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PassengerRequest:
    request_id: int
    arrival_s: int
    origin_floor: int
    destination_floor: int
    passengers: int


@dataclass
class ElevatorScenario:
    requests: list[PassengerRequest]
    elevator_count: int
    floor_count: int
    capacity: int
    seconds_per_floor: float
    door_time_s: float
    horizon_s: int


def _sample_destination(rng: np.random.Generator, origin: int, floor_count: int) -> int:
    if origin == 1:
        return int(rng.choice(np.arange(3, floor_count + 1), p=_upper_floor_probabilities(floor_count - 2)))
    if rng.random() < 0.62:
        return 1
    choices = [floor for floor in range(2, floor_count + 1) if floor != origin]
    return int(rng.choice(choices))


def _upper_floor_probabilities(count: int) -> np.ndarray:
    weights = np.linspace(1.25, 0.75, count)
    return weights / weights.sum()


def make_demo_scenario(seed: int = 31) -> ElevatorScenario:
    """Create a deterministic office-building elevator demand profile."""
    rng = np.random.default_rng(seed)
    horizon_s = 45 * 60
    floor_count = 16

    lobby_peak = rng.normal(8 * 60, 140, size=42)
    inter_floor = rng.normal(22 * 60, 230, size=22)
    down_peak = rng.normal(36 * 60, 170, size=34)
    arrivals = np.clip(np.concatenate([lobby_peak, inter_floor, down_peak]), 0, horizon_s - 120)
    arrivals = np.sort(np.round(arrivals / 5.0) * 5.0).astype(int)

    requests: list[PassengerRequest] = []
    for request_id, arrival in enumerate(arrivals.tolist(), start=1):
        pattern = rng.random()
        if arrival < 14 * 60 or pattern < 0.45:
            origin = 1
        elif arrival > 31 * 60 or pattern < 0.76:
            origin = int(rng.integers(3, floor_count + 1))
        else:
            origin = int(rng.integers(2, floor_count + 1))
        destination = _sample_destination(rng, origin, floor_count)
        group_size = int(rng.choice([1, 1, 1, 2, 2, 3], p=[0.34, 0.20, 0.18, 0.16, 0.08, 0.04]))
        requests.append(
            PassengerRequest(
                request_id=request_id,
                arrival_s=int(arrival),
                origin_floor=origin,
                destination_floor=destination,
                passengers=group_size,
            )
        )

    return ElevatorScenario(
        requests=requests,
        elevator_count=4,
        floor_count=floor_count,
        capacity=10,
        seconds_per_floor=2.8,
        door_time_s=8.0,
        horizon_s=horizon_s,
    )
