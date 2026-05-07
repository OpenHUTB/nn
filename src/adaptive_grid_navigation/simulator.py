"""Dynamic replanning simulation for grid navigation."""

from __future__ import annotations

from dataclasses import dataclass, field

from grid_map import Cell, GridMap, manhattan
from planners import AStarPlanner, PlanResult


@dataclass
class MovingObstacle:
    """A deterministic obstacle that moves along a cyclic route."""

    route: list[Cell]
    offset: int = 0

    def position_at(self, step: int) -> Cell:
        if not self.route:
            raise ValueError("moving obstacle route cannot be empty")
        return self.route[(step + self.offset) % len(self.route)]


@dataclass
class SimulationFrame:
    step: int
    robot: Cell
    goal: Cell
    dynamic_obstacles: list[Cell]
    planned_path: list[Cell]
    replanned: bool = False


@dataclass
class SimulationResult:
    frames: list[SimulationFrame] = field(default_factory=list)
    reached_goal: bool = False
    replans: int = 0
    travelled: int = 0

    @property
    def final_position(self) -> Cell | None:
        return self.frames[-1].robot if self.frames else None


def make_demo_obstacles(grid_map: GridMap) -> list[MovingObstacle]:
    """Create moving obstacles that cross the likely shortest route."""
    return [
        MovingObstacle([(18, col) for col in range(11, 25)] + [(18, col) for col in range(25, 10, -1)]),
        MovingObstacle([(row, 22) for row in range(6, 18)] + [(row, 22) for row in range(18, 5, -1)], offset=5),
        MovingObstacle([(9, col) for col in range(18, 31)] + [(9, col) for col in range(31, 17, -1)], offset=9),
    ]


def obstacle_positions(obstacles: list[MovingObstacle], step: int) -> list[Cell]:
    return [obstacle.position_at(step) for obstacle in obstacles]


def run_dynamic_replanning(
    grid_map: GridMap,
    obstacles: list[MovingObstacle] | None = None,
    max_steps: int = 120,
) -> SimulationResult:
    """Move a robot toward the goal and replan when dynamic obstacles block it."""
    if obstacles is None:
        obstacles = make_demo_obstacles(grid_map)

    robot = grid_map.start
    planner = AStarPlanner()
    frames: list[SimulationFrame] = []
    replans = 0
    travelled = 0
    current_path: list[Cell] = []

    for step in range(max_steps):
        blocked = set(obstacle_positions(obstacles, step))
        local_map = GridMap(grid_map.occupancy, robot, grid_map.goal)
        needs_replan = not current_path or len(current_path) < 2 or current_path[1] in blocked

        if needs_replan:
            plan: PlanResult = planner.plan(local_map, blocked=blocked)
            current_path = plan.path
            replans += 1
        else:
            plan = PlanResult(planner.name, current_path, [], float(len(current_path) - 1), True)

        frames.append(
            SimulationFrame(
                step=step,
                robot=robot,
                goal=grid_map.goal,
                dynamic_obstacles=sorted(blocked),
                planned_path=current_path.copy(),
                replanned=needs_replan,
            )
        )

        if robot == grid_map.goal:
            return SimulationResult(frames, True, replans, travelled)
        if not current_path or len(current_path) < 2:
            # No feasible move at this step; stay and try again on the next tick.
            continue

        next_cell = current_path[1]
        if next_cell not in blocked and manhattan(next_cell, grid_map.goal) <= manhattan(robot, grid_map.goal) + 1:
            robot = next_cell
            travelled += 1
            current_path = current_path[1:]

    return SimulationResult(frames, robot == grid_map.goal, replans, travelled)
