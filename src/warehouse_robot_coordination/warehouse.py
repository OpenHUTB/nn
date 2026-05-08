"""Warehouse map and task definitions for multi-robot coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np

Cell = tuple[int, int]


@dataclass(frozen=True)
class Task:
    """A pickup-and-delivery task in the warehouse."""

    task_id: str
    pickup: Cell
    dropoff: Cell
    priority: int = 1


@dataclass(frozen=True)
class Robot:
    """A warehouse robot with a start position and battery budget."""

    robot_id: str
    start: Cell
    battery: int = 120


@dataclass(frozen=True)
class WarehouseMap:
    """A rectangular warehouse occupancy grid.

    The grid uses 0 for free cells and 1 for shelves/walls. Coordinates use
    (row, column) order to match NumPy indexing.
    """

    occupancy: np.ndarray
    robots: tuple[Robot, ...]
    tasks: tuple[Task, ...]

    def __post_init__(self) -> None:
        grid = np.asarray(self.occupancy, dtype=np.uint8)
        object.__setattr__(self, "occupancy", grid)
        if grid.ndim != 2:
            raise ValueError("occupancy must be a 2D array")
        for robot in self.robots:
            self._validate_free(robot.start, f"robot {robot.robot_id} start")
        for task in self.tasks:
            self._validate_free(task.pickup, f"task {task.task_id} pickup")
            self._validate_free(task.dropoff, f"task {task.task_id} dropoff")

    @property
    def height(self) -> int:
        return int(self.occupancy.shape[0])

    @property
    def width(self) -> int:
        return int(self.occupancy.shape[1])

    def in_bounds(self, cell: Cell) -> bool:
        row, col = cell
        return 0 <= row < self.height and 0 <= col < self.width

    def is_obstacle(self, cell: Cell) -> bool:
        row, col = cell
        return bool(self.occupancy[row, col])

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and not self.is_obstacle(cell)

    def neighbors(self, cell: Cell) -> Iterator[Cell]:
        row, col = cell
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (row + d_row, col + d_col)
            if self.is_free(nxt):
                yield nxt

    def _validate_free(self, cell: Cell, name: str) -> None:
        if not self.in_bounds(cell):
            raise ValueError(f"{name} {cell} is outside the map")
        if self.is_obstacle(cell):
            raise ValueError(f"{name} {cell} cannot be on a shelf or wall")


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def make_demo_warehouse() -> WarehouseMap:
    """Create a deterministic warehouse with shelf blocks and four robots."""
    grid = np.zeros((26, 38), dtype=np.uint8)

    # Outer walls.
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    # Shelf blocks with horizontal aisles between them.
    for col_start in (5, 11, 17, 23, 29):
        grid[4:10, col_start : col_start + 3] = 1
        grid[14:21, col_start : col_start + 3] = 1

    # Cross aisles and loading area keep the map connected.
    grid[11:14, 1:-1] = 0
    grid[21:24, 1:-1] = 0
    grid[1:4, 1:10] = 0
    grid[1:4, 28:37] = 0

    robots = (
        Robot("R1", (22, 3), battery=120),
        Robot("R2", (22, 8), battery=120),
        Robot("R3", (22, 30), battery=120),
        Robot("R4", (3, 34), battery=120),
    )

    tasks = (
        Task("T1", pickup=(5, 4), dropoff=(2, 3), priority=3),
        Task("T2", pickup=(8, 15), dropoff=(2, 32), priority=2),
        Task("T3", pickup=(17, 16), dropoff=(23, 34), priority=1),
        Task("T4", pickup=(19, 28), dropoff=(23, 3), priority=2),
        Task("T5", pickup=(6, 34), dropoff=(2, 29), priority=3),
        Task("T6", pickup=(15, 10), dropoff=(22, 18), priority=1),
    )

    return WarehouseMap(grid, robots, tasks)


def ordered_unique(cells: Sequence[Cell]) -> list[Cell]:
    seen: set[Cell] = set()
    result: list[Cell] = []
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            result.append(cell)
    return result
