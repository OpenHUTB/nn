"""Scenario generation for sensor fusion fault diagnosis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SENSORS = ("gps", "radar", "odometry")


@dataclass(frozen=True)
class FaultWindow:
    """A labelled time window for an injected sensor fault."""

    sensor: str
    start: int
    end: int
    fault_type: str

    def contains(self, step: int) -> bool:
        return self.start <= step < self.end


@dataclass
class Scenario:
    """Synthetic vehicle trajectory and noisy sensor measurements."""

    dt: float
    truth: np.ndarray
    measurements: dict[str, np.ndarray]
    faults: tuple[FaultWindow, ...]

    @property
    def steps(self) -> int:
        return int(self.truth.shape[0])


def make_vehicle_trajectory(steps: int = 180, dt: float = 0.1) -> np.ndarray:
    """Create a smooth 2D vehicle trajectory with position and velocity."""
    time = np.arange(steps) * dt
    x = 2.4 * time
    y = 4.0 * np.sin(time / 2.2) + 0.08 * time**2
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    return np.column_stack([x, y, vx, vy])


def make_measurements(truth: np.ndarray, seed: int = 42) -> tuple[dict[str, np.ndarray], tuple[FaultWindow, ...]]:
    """Generate GPS, radar and odometry measurements with injected faults."""
    rng = np.random.default_rng(seed)
    steps = truth.shape[0]

    gps = truth[:, :2] + rng.normal(0.0, 0.9, size=(steps, 2))
    radar = truth[:, :2] + rng.normal(0.0, 0.55, size=(steps, 2))
    odometry = truth[:, :2] + rng.normal(0.0, 0.35, size=(steps, 2))

    faults = (
        FaultWindow("gps", 45, 75, "bias drift"),
        FaultWindow("radar", 95, 118, "spike noise"),
        FaultWindow("odometry", 130, 158, "scale drift"),
    )

    gps[45:75] += np.linspace([0.0, 0.0], [7.0, -5.0], 30)
    radar[95:118] += rng.normal(0.0, 4.0, size=(23, 2))
    drift = np.linspace(1.0, 1.18, 28)
    odometry[130:158] = truth[130:158, :2] * drift[:, None] + rng.normal(0.0, 0.45, size=(28, 2))

    return {"gps": gps, "radar": radar, "odometry": odometry}, faults


def make_demo_scenario(seed: int = 42) -> Scenario:
    """Build the deterministic demo scenario used by the project."""
    dt = 0.1
    truth = make_vehicle_trajectory(dt=dt)
    measurements, faults = make_measurements(truth, seed=seed)
    return Scenario(dt=dt, truth=truth, measurements=measurements, faults=faults)


def fault_labels(scenario: Scenario) -> dict[str, np.ndarray]:
    """Return per-sensor binary labels for fault intervals."""
    labels = {sensor: np.zeros(scenario.steps, dtype=int) for sensor in SENSORS}
    for fault in scenario.faults:
        labels[fault.sensor][fault.start : fault.end] = 1
    return labels
