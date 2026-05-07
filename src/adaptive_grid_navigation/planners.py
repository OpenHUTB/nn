"""Path planners for grid based robot navigation."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from typing import Callable, Iterable

from grid_map import Cell, GridMap, manhattan, path_length


@dataclass
class PlanResult:
    """Result returned by a planner."""

    planner: str
    path: list[Cell]
    expanded: list[Cell] = field(default_factory=list)
    cost: float = 0.0
    success: bool = False

    @property
    def path_length(self) -> int:
        return path_length(self.path)

    @property
    def expanded_count(self) -> int:
        return len(self.expanded)


def reconstruct_path(came_from: dict[Cell, Cell], start: Cell, goal: Cell) -> list[Cell]:
    if goal not in came_from and goal != start:
        return []
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


class BasePlanner:
    name = "base"

    def __init__(self, diagonal: bool = False) -> None:
        self.diagonal = diagonal

    def plan(self, grid_map: GridMap, blocked: Iterable[Cell] | None = None) -> PlanResult:
        raise NotImplementedError

    def _is_allowed(self, grid_map: GridMap, cell: Cell, blocked: set[Cell]) -> bool:
        return grid_map.in_bounds(cell) and not grid_map.is_obstacle(cell) and cell not in blocked


class DijkstraPlanner(BasePlanner):
    name = "Dijkstra"

    def plan(self, grid_map: GridMap, blocked: Iterable[Cell] | None = None) -> PlanResult:
        blocked_set = set(blocked or [])
        start, goal = grid_map.start, grid_map.goal
        queue: list[tuple[float, int, Cell]] = []
        tie = count()
        heappush(queue, (0.0, next(tie), start))
        came_from: dict[Cell, Cell] = {}
        cost_so_far = {start: 0.0}
        expanded: list[Cell] = []

        while queue:
            _, _, current = heappop(queue)
            if current in expanded:
                continue
            expanded.append(current)
            if current == goal:
                break

            for nxt in grid_map.neighbors(current, diagonal=self.diagonal):
                if not self._is_allowed(grid_map, nxt, blocked_set):
                    continue
                new_cost = cost_so_far[current] + 1.0
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    came_from[nxt] = current
                    heappush(queue, (new_cost, next(tie), nxt))

        path = reconstruct_path(came_from, start, goal)
        return PlanResult(self.name, path, expanded, cost_so_far.get(goal, float("inf")), bool(path))


class AStarPlanner(BasePlanner):
    name = "A*"

    def __init__(self, diagonal: bool = False, heuristic: Callable[[Cell, Cell], float] = manhattan) -> None:
        super().__init__(diagonal=diagonal)
        self.heuristic = heuristic

    def plan(self, grid_map: GridMap, blocked: Iterable[Cell] | None = None) -> PlanResult:
        blocked_set = set(blocked or [])
        start, goal = grid_map.start, grid_map.goal
        queue: list[tuple[float, int, Cell]] = []
        tie = count()
        heappush(queue, (0.0, next(tie), start))
        came_from: dict[Cell, Cell] = {}
        cost_so_far = {start: 0.0}
        expanded: list[Cell] = []

        while queue:
            _, _, current = heappop(queue)
            if current in expanded:
                continue
            expanded.append(current)
            if current == goal:
                break

            for nxt in grid_map.neighbors(current, diagonal=self.diagonal):
                if not self._is_allowed(grid_map, nxt, blocked_set):
                    continue
                new_cost = cost_so_far[current] + 1.0
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self.heuristic(nxt, goal)
                    came_from[nxt] = current
                    heappush(queue, (priority, next(tie), nxt))

        path = reconstruct_path(came_from, start, goal)
        return PlanResult(self.name, path, expanded, cost_so_far.get(goal, float("inf")), bool(path))


class GreedyBestFirstPlanner(BasePlanner):
    name = "Greedy best-first"

    def plan(self, grid_map: GridMap, blocked: Iterable[Cell] | None = None) -> PlanResult:
        blocked_set = set(blocked or [])
        start, goal = grid_map.start, grid_map.goal
        queue: list[tuple[float, int, Cell]] = []
        tie = count()
        heappush(queue, (manhattan(start, goal), next(tie), start))
        came_from: dict[Cell, Cell] = {}
        visited = {start}
        expanded: list[Cell] = []

        while queue:
            _, _, current = heappop(queue)
            expanded.append(current)
            if current == goal:
                break

            for nxt in grid_map.neighbors(current, diagonal=self.diagonal):
                if nxt in visited or not self._is_allowed(grid_map, nxt, blocked_set):
                    continue
                visited.add(nxt)
                came_from[nxt] = current
                heappush(queue, (manhattan(nxt, goal), next(tie), nxt))

        path = reconstruct_path(came_from, start, goal)
        return PlanResult(self.name, path, expanded, float(path_length(path)), bool(path))


def default_planners() -> list[BasePlanner]:
    return [DijkstraPlanner(), AStarPlanner(), GreedyBestFirstPlanner()]
