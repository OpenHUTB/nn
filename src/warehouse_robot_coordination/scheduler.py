"""Time-expanded conflict detection and avoidance for multi-robot routes."""

from __future__ import annotations

from dataclasses import dataclass

from planner import AssignmentResult
from warehouse import Cell


@dataclass(frozen=True)
class TimedCell:
    robot_id: str
    time: int
    cell: Cell


@dataclass(frozen=True)
class Conflict:
    time: int
    robot_a: str
    robot_b: str
    cell: Cell
    conflict_type: str


@dataclass
class Schedule:
    trajectories: dict[str, list[Cell]]
    conflicts_before: list[Conflict]
    conflicts_after: list[Conflict]
    inserted_waits: int

    @property
    def makespan(self) -> int:
        return max((len(path) - 1 for path in self.trajectories.values()), default=0)


def position_at(path: list[Cell], time: int) -> Cell | None:
    """Return a robot position only while it is active in the schedule."""
    if 0 <= time < len(path):
        return path[time]
    return None


def pad_position(path: list[Cell], time: int) -> Cell:
    """Return a display position, keeping completed robots at their final cell."""
    if time < len(path):
        return path[time]
    return path[-1]


def detect_conflicts(trajectories: dict[str, list[Cell]]) -> list[Conflict]:
    """Detect vertex and edge-swap conflicts in active time-indexed paths.

    Finished robots are treated as having left the pickup/drop-off bay, which
    prevents a completed robot from blocking a shared loading location forever.
    """
    conflicts: list[Conflict] = []
    if not trajectories:
        return conflicts
    max_time = max(len(path) for path in trajectories.values())
    robot_ids = sorted(trajectories)

    for time in range(max_time):
        occupied: dict[Cell, str] = {}
        for robot_id in robot_ids:
            cell = position_at(trajectories[robot_id], time)
            if cell is None:
                continue
            if cell in occupied:
                conflicts.append(Conflict(time, occupied[cell], robot_id, cell, "vertex"))
            else:
                occupied[cell] = robot_id

        if time == 0:
            continue
        for idx, robot_a in enumerate(robot_ids):
            a_prev = position_at(trajectories[robot_a], time - 1)
            a_now = position_at(trajectories[robot_a], time)
            if a_prev is None or a_now is None:
                continue
            for robot_b in robot_ids[idx + 1 :]:
                b_prev = position_at(trajectories[robot_b], time - 1)
                b_now = position_at(trajectories[robot_b], time)
                if b_prev is None or b_now is None:
                    continue
                if a_prev == b_now and a_now == b_prev and a_now != b_now:
                    conflicts.append(Conflict(time, robot_a, robot_b, a_now, "edge-swap"))
    return conflicts


def build_trajectories(result: AssignmentResult) -> dict[str, list[Cell]]:
    return {plan.robot.robot_id: plan.route for plan in result.plans}


def insert_wait(path: list[Cell], time: int) -> list[Cell]:
    if time <= 0:
        return [path[0]] + path
    wait_index = min(time - 1, len(path) - 1)
    return path[: wait_index + 1] + [path[wait_index]] + path[wait_index + 1 :]


def resolve_conflicts(result: AssignmentResult, max_iterations: int = 120) -> Schedule:
    """Resolve conflicts by adding wait actions to one robot at a time.

    The demo warehouse contains narrow one-cell aisles. For opposite-flow
    conflicts in such aisles, waiting at the route start is more stable than
    waiting inside the bottleneck.
    """
    trajectories = build_trajectories(result)
    conflicts_before = detect_conflicts(trajectories)
    waits = 0

    for _ in range(max_iterations):
        conflicts = detect_conflicts(trajectories)
        if not conflicts:
            break
        conflict = conflicts[0]
        delayed = choose_robot_to_delay(conflict)
        wait_time = choose_wait_time(conflict)
        trajectories[delayed] = insert_wait(trajectories[delayed], wait_time)
        waits += 1

    return Schedule(
        trajectories=trajectories,
        conflicts_before=conflicts_before,
        conflicts_after=detect_conflicts(trajectories),
        inserted_waits=waits,
    )


def choose_robot_to_delay(conflict: Conflict) -> str:
    """Delay the first robot in the conflict so the other robot gets priority."""
    return conflict.robot_a


def choose_wait_time(conflict: Conflict) -> int:
    """Insert waits at the route start so robots yield before entering bottlenecks."""
    return 0


def trajectory_loads(trajectories: dict[str, list[Cell]]) -> dict[str, int]:
    return {robot_id: max(0, len(path) - 1) for robot_id, path in trajectories.items()}
