"""Scenario generation for EV charging station scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChargingRequest:
    vehicle_id: int
    arrival_min: int
    deadline_min: int
    energy_kwh: float
    priority: int


@dataclass
class ChargingScenario:
    requests: list[ChargingRequest]
    charger_count: int
    charger_power_kw: float
    horizon_min: int
    time_step_min: int
    grid_limit_kw: float
    solar_kw: np.ndarray
    base_load_kw: np.ndarray


def _daily_solar_profile(minutes: np.ndarray) -> np.ndarray:
    center = 13.0 * 60.0
    width = 4.4 * 60.0
    profile = 48.0 * np.exp(-0.5 * ((minutes - center) / width) ** 2)
    return np.where((minutes >= 7 * 60) & (minutes <= 18 * 60), profile, 0.0)


def _base_load_profile(minutes: np.ndarray) -> np.ndarray:
    morning_peak = 20.0 * np.exp(-0.5 * ((minutes - 8.2 * 60) / 100.0) ** 2)
    evening_peak = 34.0 * np.exp(-0.5 * ((minutes - 18.3 * 60) / 125.0) ** 2)
    return 42.0 + morning_peak + evening_peak + 5.0 * np.sin(minutes / 1440.0 * 2.0 * np.pi)


def make_demo_scenario(seed: int = 23) -> ChargingScenario:
    """Create a deterministic mixed-arrival charging-station scenario."""
    rng = np.random.default_rng(seed)
    horizon_min = 24 * 60
    time_step_min = 5
    minutes = np.arange(0, horizon_min, time_step_min)

    arrival_clusters = np.concatenate(
        [
            rng.normal(8.0 * 60, 55, size=10),
            rng.normal(12.5 * 60, 70, size=8),
            rng.normal(18.0 * 60, 65, size=12),
        ]
    )
    arrivals = np.clip(np.round(arrival_clusters / time_step_min) * time_step_min, 0, horizon_min - 180).astype(int)
    requests: list[ChargingRequest] = []
    for vehicle_id, arrival in enumerate(sorted(arrivals.tolist()), start=1):
        energy = float(rng.uniform(18.0, 46.0))
        parking_window = int(rng.integers(120, 330))
        deadline = min(horizon_min, int(arrival + parking_window))
        priority = int(rng.choice([1, 2, 3], p=[0.45, 0.40, 0.15]))
        requests.append(
            ChargingRequest(
                vehicle_id=vehicle_id,
                arrival_min=int(arrival),
                deadline_min=int(deadline),
                energy_kwh=round(energy, 2),
                priority=priority,
            )
        )

    return ChargingScenario(
        requests=requests,
        charger_count=6,
        charger_power_kw=22.0,
        horizon_min=horizon_min,
        time_step_min=time_step_min,
        grid_limit_kw=118.0,
        solar_kw=_daily_solar_profile(minutes),
        base_load_kw=_base_load_profile(minutes),
    )
