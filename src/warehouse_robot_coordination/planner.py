"""A* path planning and task assignment for warehouse robots."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count

from warehouse import Cell, Robot, Task, WarehouseMap, manhattan


@dataclass
class RouteSegment:
    """One route segment, usually robot->pickup or pickup->dropoff."""

    label: str
    path: list[Cell]

    @property
    def length(self) -> int:
        return max(0, len(self.path) - 1)


@dataclass
class RobotPlan:
    """Task and path plan for a single robot."""

    robot: Robot
    tasks: list[Task] = field(default_factory=list)
    segments: list[RouteSegment] = field(default_factory=list)

    @property
    def route(self) -> list[Cell]:
        cells: list[Cell] = []
        for segment in self.segments:
            if not cells:
                cells.extend(segment.path)
            else:
                cells.extend(segment.path[1:])
        return cells or [self.robot.start]

    @property
    def route_length(self) -> int:
        return max(0, len(self.route) - 1)

    @property
    def task_count(self) -> int:
        return len(self.tasks)


@dataclass
class AssignmentResult:
    plans: list[RobotPlan]

    @property
    def total_distance(self) -> int:
        return sum(plan.route_length for plan in self.plans)

    @property
    def makespan(self) -> int:
        return max((plan.route_length for plan in self.plans), default=0)


def reconstruct(came_from: dict[Cell, Cell], start: Cell, goal: Cell) -> list[Cell]:
    if goal != start and goal not in came_from:
        return []
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_path(warehouse: WarehouseMap, start: Cell, goal: Cell) -> list[Cell]:
    """Plan a shortest path in the warehouse grid."""
    queue: list[tuple[int, int, Cell]] = []
    tie = count()
    heappush(queue, (0, next(tie), start))
    came_from: dict[Cell, Cell] = {}
    cost_so_far = {start: 0}
    visited: set[Cell] = set()

    while queue:
        _, _, current = heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            break

        for nxt in warehouse.neighbors(current):
            new_cost = cost_so_far[current] + 1
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + manhattan(nxt, goal)
                came_from[nxt] = current
                heappush(queue, (priority, next(tie), nxt))

    return reconstruct(came_from, start, goal)


def route_for_task(warehouse: WarehouseMap, start: Cell, task: Task) -> tuple[list[RouteSegment], Cell, int]:
    to_pickup = astar_path(warehouse, start, task.pickup)
    to_dropoff = astar_path(warehouse, task.pickup, task.dropoff)
    if not to_pickup or not to_dropoff:
        raise ValueError(f"task {task.task_id} is not reachable")
    segments = [
        RouteSegment(f"to {task.task_id} pickup", to_pickup),
        RouteSegment(f"to {task.task_id} dropoff", to_dropoff),
    ]
    total = segments[0].length + segments[1].length
    return segments, task.dropoff, total


def greedy_assign_tasks(warehouse: WarehouseMap) -> AssignmentResult:
    """Assign tasks using priority-aware nearest feasible robot selection."""
    plans = [RobotPlan(robot) for robot in warehouse.robots]
    robot_positions = {plan.robot.robot_id: plan.robot.start for plan in plans}
    remaining = sorted(warehouse.tasks, key=lambda task: (-task.priority, task.task_id))

    for task in remaining:
        candidates: list[tuple[float, int, RobotPlan, list[RouteSegment], Cell]] = []
        for plan in plans:
            start = robot_positions[plan.robot.robot_id]
            segments, end, cost = route_for_task(warehouse, start, task)
            load_penalty = plan.task_count * 2.5
            score = cost + load_penalty - task.priority * 0.8
            candidates.append((score, cost, plan, segments, end))

        _, _, selected, segments, end = min(candidates, key=lambda item: (item[0], item[1], item[2].robot.robot_id))
        selected.tasks.append(task)
        selected.segments.extend(segments)
        robot_positions[selected.robot.robot_id] = end

    return AssignmentResult(plans)


def baseline_single_robot_plan(warehouse: WarehouseMap) -> RobotPlan:
    """A simple baseline where the first robot performs every task."""
    robot = warehouse.robots[0]
    plan = RobotPlan(robot)
    current = robot.start
    for task in sorted(warehouse.tasks, key=lambda item: item.task_id):
        segments, current, _ = route_for_task(warehouse, current, task)
        plan.tasks.append(task)
        plan.segments.extend(segments)
    return plan
