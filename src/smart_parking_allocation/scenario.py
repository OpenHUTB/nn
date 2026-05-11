"""Scenario generation for smart parking-space allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParkingSpot:
    spot_id: int
    zone: str
    x_m: float
    y_m: float
    distance_to_exit_m: float
    has_charger: bool
    is_accessible: bool


@dataclass
class VehicleRequest:
    vehicle_id: int
    arrival_min: int
    dwell_min: int
    needs_charger: bool
    needs_accessible: bool
    priority: int


@dataclass
class ParkingScenario:
    spots: list[ParkingSpot]
    requests: list[VehicleRequest]
    horizon_min: int
    time_step_min: int
    entrance_xy: tuple[float, float]
    exit_xy: tuple[float, float]


def _build_spots() -> list[ParkingSpot]:
    spots: list[ParkingSpot] = []
    spot_id = 1
    zone_layout = {
        "A": (18.0, 20.0, 3),
        "B": (58.0, 23.0, 4),
        "C": (98.0, 27.0, 4),
        "D": (138.0, 34.0, 3),
    }
    exit_xy = (158.0, 8.0)
    for zone, (x0, y0, rows) in zone_layout.items():
        for row in range(rows):
            for col in range(6):
                x = x0 + col * 5.2
                y = y0 + row * 7.0
                distance_to_exit = abs(x - exit_xy[0]) + abs(y - exit_xy[1])
                has_charger = zone in {"A", "D"} and col in {0, 1, 2}
                is_accessible = zone == "A" and row == 0 and col in {0, 1, 2}
                spots.append(
                    ParkingSpot(
                        spot_id=spot_id,
                        zone=zone,
                        x_m=x,
                        y_m=y,
                        distance_to_exit_m=distance_to_exit,
                        has_charger=has_charger,
                        is_accessible=is_accessible,
                    )
                )
                spot_id += 1
    return spots


def _arrival_profile(rng: np.random.Generator) -> np.ndarray:
    morning = rng.normal(8.6 * 60, 55, size=34)
    noon = rng.normal(12.4 * 60, 45, size=18)
    evening = rng.normal(18.2 * 60, 62, size=32)
    return np.clip(np.concatenate([morning, noon, evening]), 6 * 60, 22 * 60)


def make_demo_scenario(seed: int = 41) -> ParkingScenario:
    """Create a deterministic day of parking arrivals."""
    rng = np.random.default_rng(seed)
    arrivals = np.sort(np.round(_arrival_profile(rng) / 5.0) * 5.0).astype(int)
    requests: list[VehicleRequest] = []
    for vehicle_id, arrival in enumerate(arrivals.tolist(), start=1):
        dwell = int(rng.integers(45, 210))
        needs_charger = bool(rng.random() < 0.20)
        needs_accessible = bool(rng.random() < 0.07)
        priority = int(rng.choice([1, 2, 3], p=[0.62, 0.28, 0.10]))
        requests.append(
            VehicleRequest(
                vehicle_id=vehicle_id,
                arrival_min=int(arrival),
                dwell_min=dwell,
                needs_charger=needs_charger,
                needs_accessible=needs_accessible,
                priority=priority,
            )
        )

    return ParkingScenario(
        spots=_build_spots(),
        requests=requests,
        horizon_min=24 * 60,
        time_step_min=5,
        entrance_xy=(0.0, 8.0),
        exit_xy=(158.0, 8.0),
    )


def manhattan_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return abs(point_a[0] - point_b[0]) + abs(point_a[1] - point_b[1])
