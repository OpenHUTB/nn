from __future__ import annotations
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count, chain
from typing import List, Tuple, Dict

from warehouse import Cell, Robot, Task, WarehouseMap, manhattan


@dataclass
class RouteSegment:
    """One route segment, usually robot -> pickup or pickup -> dropoff."""
    label: str
    path: List[Cell]

    @property
    def length(self) -> int:
        return max(0, len(self.path) - 1)


@dataclass
class RobotPlan:
    """Task and path plan for a single robot."""
    robot: Robot
    tasks: List[Task] = field(default_factory=list)
    segments: List[RouteSegment] = field(default_factory=list)

    @property
    def route(self) -> List[Cell]:
        """Concatenate all segments into a full route."""
        return list(chain.from_iterable(
            seg.path if i == 0 else seg.path[1:]
            for i, seg in enumerate(self.segments)
        )) or [self.robot.start]

    @property
    def route_length(self) -> int:
        return max(0, len(self.route) - 1)

    @property
    def task_count(self) -> int:
        return len(self.tasks)


@dataclass
class AssignmentResult:
    plans: List[RobotPlan]

    @property
    def total_distance(self) -> int:
        return sum(plan.route_length for plan in self.plans)

    @property
    def makespan(self) -> int:
        return max((plan.route_length for plan in self.plans), default=0)


# -------------------
# Path planning
# -------------------

def reconstruct(came_from: Dict[Cell, Cell], start: Cell, goal: Cell) -> List[Cell]:
    """Reconstruct path from came_from dict."""
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from.get(current, start)
    path.append(start)
    path.reverse()
    return path


def astar_path(warehouse: WarehouseMap, start: Cell, goal: Cell) -> List[Cell]:
    """Plan a shortest path in the warehouse grid using A*."""
    queue: List[Tuple[int, int, Cell]] = []
    tie = count()
    heappush(queue, (0, next(tie), start))
    came_from: Dict[Cell, Cell] = {}
    cost_so_far: Dict[Cell, int] = {start: 0}
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


# -------------------
# Task routing
# -------------------

def route_for_task(warehouse: WarehouseMap, start: Cell, task: Task) -> Tuple[List[RouteSegment], Cell, int]:
    """Compute route segments for a single task."""
    to_pickup = astar_path(warehouse, start, task.pickup)
    to_dropoff = astar_path(warehouse, task.pickup, task.dropoff)

    if not to_pickup or not to_dropoff:
        raise ValueError(f"Task {task.task_id} is not reachable")

    segments = [
        RouteSegment(f"to {task.task_id} pickup", to_pickup),
        RouteSegment(f"to {task.task_id} dropoff", to_dropoff),
    ]
    total_length = sum(seg.length for seg in segments)
    return segments, task.dropoff, total_length


# -------------------
# Greedy assignment
# -------------------

def compute_task_score(plan: RobotPlan, cost: int, task: Task) -> float:
    """Compute score for assigning a task to a robot plan."""
    load_penalty = plan.task_count * 2.5
    return cost + load_penalty - task.priority * 0.8


def greedy_assign_tasks(warehouse: WarehouseMap) -> AssignmentResult:
    """Assign tasks using priority-aware nearest feasible robot selection."""
    plans: List[RobotPlan] = [RobotPlan(robot) for robot in warehouse.robots]
    robot_positions: Dict[int, Cell] = {plan.robot.robot_id: plan.robot.start for plan in plans}
    remaining_tasks = sorted(warehouse.tasks, key=lambda t: (-t.priority, t.task_id))

    for task in remaining_tasks:
        candidates: List[Tuple[float, int, RobotPlan, List[RouteSegment], Cell]] = []

        for plan in plans:
            start = robot_positions[plan.robot.robot_id]
            segments, end, cost = route_for_task(warehouse, start, task)
            score = compute_task_score(plan, cost, task)
            candidates.append((score, cost, plan, segments, end))

        # select robot with minimum score, tie-break by cost then robot ID
        _, _, selected_plan, segments, end = min(candidates, key=lambda x: (x[0], x[1], x[2].robot.robot_id))
        selected_plan.tasks.append(task)
        selected_plan.segments.extend(segments)
        robot_positions[selected_plan.robot.robot_id] = end

    return AssignmentResult(plans)


# -------------------
# Baseline
# -------------------

def baseline_single_robot_plan(warehouse: WarehouseMap) -> RobotPlan:
    """Assign all tasks to the first robot as a simple baseline."""
    robot = warehouse.robots[0]
    plan = RobotPlan(robot)
    current = robot.start
    for task in sorted(warehouse.tasks, key=lambda t: t.task_id):
        segments, current, _ = route_for_task(warehouse, current, task)
        plan.tasks.append(task)
        plan.segments.extend(segments)
    return plan