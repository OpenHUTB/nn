"""Scenario definition for energy-aware drone delivery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

Cell = tuple[int, int]


@dataclass(frozen=True)
class DeliveryTask:
    task_id: str
    pickup: Cell
    dropoff: Cell
    payload_kg: float
    priority: int = 1


@dataclass(frozen=True)
class ChargingStation:
    station_id: str
    cell: Cell
    charge_rate: float = 18.0


@dataclass(frozen=True)
class DroneSpec:
    start: Cell
    battery_capacity: float = 100.0
    reserve_ratio: float = 0.15
    base_cost: float = 1.0
    payload_cost: float = 0.28
    wind_cost: float = 0.45

    @property
    def usable_energy(self) -> float:
        return self.battery_capacity * (1.0 - self.reserve_ratio)


@dataclass(frozen=True)
class DeliveryScenario:
    grid: np.ndarray
    drone: DroneSpec
    tasks: tuple[DeliveryTask, ...]
    stations: tuple[ChargingStation, ...]
    wind_u: np.ndarray
    wind_v: np.ndarray

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid, dtype=np.uint8)
        object.__setattr__(self, "grid", grid)
        if grid.ndim != 2:
            raise ValueError("grid must be a 2D array")
        if self.wind_u.shape != grid.shape or self.wind_v.shape != grid.shape:
            raise ValueError("wind field must match grid shape")
        self._validate_free(self.drone.start, "drone start")
        for task in self.tasks:
            self._validate_free(task.pickup, f"task {task.task_id} pickup")
            self._validate_free(task.dropoff, f"task {task.task_id} dropoff")
        for station in self.stations:
            self._validate_free(station.cell, f"station {station.station_id}")

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    def in_bounds(self, cell: Cell) -> bool:
        row, col = cell
        return 0 <= row < self.height and 0 <= col < self.width

    def is_blocked(self, cell: Cell) -> bool:
        row, col = cell
        return bool(self.grid[row, col])

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and not self.is_blocked(cell)

    def neighbors(self, cell: Cell) -> Iterator[Cell]:
        row, col = cell
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (row + d_row, col + d_col)
            if self.is_free(nxt):
                yield nxt

    def _validate_free(self, cell: Cell, name: str) -> None:
        if not self.in_bounds(cell):
            raise ValueError(f"{name} {cell} is outside map")
        if self.is_blocked(cell):
            raise ValueError(f"{name} {cell} is blocked")


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def make_demo_scenario() -> DeliveryScenario:
    """Create a deterministic city-like map with no-fly zones and wind."""
    height, width = 32, 42
    grid = np.zeros((height, width), dtype=np.uint8)

    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    # No-fly blocks representing tall buildings and restricted areas.
    grid[5:13, 7:12] = 1
    grid[16:26, 8:13] = 1
    grid[4:10, 21:28] = 1
    grid[14:22, 20:25] = 1
    grid[20:28, 31:36] = 1
    grid[11:16, 32:38] = 1

    # Open corridors.
    grid[13:16, 1:-1] = 0
    grid[26:29, 1:-1] = 0
    grid[1:-1, 16:19] = 0
    grid[1:-1, 28:31] = 0

    rows, cols = np.indices((height, width))
    wind_u = 0.7 * np.sin(cols / 5.0) + 0.25 * np.cos(rows / 4.0)
    wind_v = 0.5 * np.cos(rows / 6.0) - 0.2 * np.sin(cols / 7.0)

    drone = DroneSpec(start=(28, 3), battery_capacity=130.0)
    tasks = (
        DeliveryTask("D1", pickup=(27, 6), dropoff=(3, 34), payload_kg=1.2, priority=3),
        DeliveryTask("D2", pickup=(25, 17), dropoff=(6, 18), payload_kg=0.8, priority=2),
        DeliveryTask("D3", pickup=(28, 32), dropoff=(12, 5), payload_kg=1.6, priority=2),
        DeliveryTask("D4", pickup=(14, 29), dropoff=(27, 38), payload_kg=0.6, priority=1),
    )
    stations = (
        ChargingStation("C1", (27, 17)),
        ChargingStation("C2", (14, 18)),
        ChargingStation("C3", (13, 30)),
        ChargingStation("C4", (4, 5)),
    )
    return DeliveryScenario(grid, drone, tasks, stations, wind_u, wind_v)

