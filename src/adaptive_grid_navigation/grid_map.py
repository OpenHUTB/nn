"""Grid map utilities for adaptive grid navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np

Cell = tuple[int, int]


@dataclass(frozen=True)
class GridMap:
    """A 2D occupancy grid.

    The grid uses 0 for free cells and 1 for obstacles. Coordinates are
    represented as (row, column), matching NumPy indexing.
    """

    occupancy: np.ndarray
    start: Cell
    goal: Cell

    def __post_init__(self) -> None:
        grid = np.asarray(self.occupancy, dtype=np.uint8)
        object.__setattr__(self, "occupancy", grid)
        if grid.ndim != 2:
            raise ValueError("occupancy must be a 2D array")
        for name, cell in (("start", self.start), ("goal", self.goal)):
            if not self.in_bounds(cell):
                raise ValueError(f"{name} cell {cell} is outside the map")
            if self.is_obstacle(cell):
                raise ValueError(f"{name} cell {cell} cannot be an obstacle")

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

    def is_free(self, cell: Cell, extra_obstacles: Iterable[Cell] | None = None) -> bool:
        if not self.in_bounds(cell) or self.is_obstacle(cell):
            return False
        if extra_obstacles is not None and cell in set(extra_obstacles):
            return False
        return True

    def neighbors(self, cell: Cell, diagonal: bool = False) -> Iterator[Cell]:
        steps: Sequence[Cell]
        if diagonal:
            steps = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        else:
            steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
        row, col = cell
        for d_row, d_col in steps:
            nxt = (row + d_row, col + d_col)
            if self.in_bounds(nxt) and not self.is_obstacle(nxt):
                yield nxt

    def with_obstacles(self, obstacles: Iterable[Cell]) -> "GridMap":
        grid = self.occupancy.copy()
        for row, col in obstacles:
            if self.in_bounds((row, col)) and (row, col) not in {self.start, self.goal}:
                grid[row, col] = 1
        return GridMap(grid, self.start, self.goal)


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def path_length(path: Sequence[Cell]) -> int:
    return max(0, len(path) - 1)


def make_demo_map() -> GridMap:
    """Build a deterministic demo map with corridors and narrow passages."""
    grid = np.zeros((28, 36), dtype=np.uint8)

    grid[4:24, 7] = 1
    grid[4, 7:28] = 1
    grid[10:25, 16] = 1
    grid[20, 16:32] = 1
    grid[6:18, 26] = 1
    grid[14, 3:12] = 1
    grid[23, 4:13] = 1
    grid[2:8, 32] = 1

    # Doorways keep the map solvable while forcing non-trivial detours.
    for cell in [(12, 7), (4, 14), (17, 16), (20, 24), (11, 26), (14, 9), (23, 9)]:
        grid[cell] = 0

    start = (25, 2)
    goal = (2, 33)
    grid[start] = 0
    grid[goal] = 0
    return GridMap(grid, start, goal)


def make_random_map(width: int = 40, height: int = 30, obstacle_ratio: float = 0.22, seed: int = 7) -> GridMap:
    """Generate a random map and carve a guaranteed rough corridor."""
    if not 0 <= obstacle_ratio < 0.7:
        raise ValueError("obstacle_ratio must be in [0, 0.7)")
    rng = np.random.default_rng(seed)
    grid = (rng.random((height, width)) < obstacle_ratio).astype(np.uint8)
    start = (height - 3, 2)
    goal = (2, width - 3)

    row, col = start
    grid[row, col] = 0
    while col < goal[1]:
        grid[row, col] = 0
        col += 1
    while row > goal[0]:
        grid[row, col] = 0
        row -= 1
    grid[goal] = 0
    return GridMap(grid, start, goal)
