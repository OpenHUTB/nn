"""Tests for warehouse robot coordination."""

from pathlib import Path
import sys

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from planner import astar_path, baseline_single_robot_plan, greedy_assign_tasks
from scheduler import detect_conflicts, resolve_conflicts
from warehouse import make_demo_warehouse


def test_demo_warehouse_is_valid():
    warehouse = make_demo_warehouse()

    assert warehouse.height > 0
    assert warehouse.width > 0
    assert len(warehouse.robots) == 4
    assert len(warehouse.tasks) == 6
    assert all(warehouse.is_free(robot.start) for robot in warehouse.robots)


def test_astar_reaches_task_points():
    warehouse = make_demo_warehouse()
    robot = warehouse.robots[0]
    task = warehouse.tasks[0]

    path = astar_path(warehouse, robot.start, task.pickup)

    assert path[0] == robot.start
    assert path[-1] == task.pickup
    assert len(path) > 1


def test_greedy_assignment_assigns_every_task_once():
    warehouse = make_demo_warehouse()
    result = greedy_assign_tasks(warehouse)
    assigned = [task.task_id for plan in result.plans for task in plan.tasks]

    assert sorted(assigned) == sorted(task.task_id for task in warehouse.tasks)
    assert result.total_distance > 0
    assert result.makespan > 0


def test_multi_robot_plan_improves_makespan_against_single_robot_baseline():
    warehouse = make_demo_warehouse()
    result = greedy_assign_tasks(warehouse)
    baseline = baseline_single_robot_plan(warehouse)

    assert result.makespan < baseline.route_length


def test_conflict_resolution_removes_detected_conflicts():
    warehouse = make_demo_warehouse()
    result = greedy_assign_tasks(warehouse)
    schedule = resolve_conflicts(result)

    assert schedule.conflicts_after == []
    assert schedule.makespan >= result.makespan


def test_detect_conflicts_catches_vertex_conflict():
    trajectories = {
        "R1": [(1, 1), (1, 2), (1, 3)],
        "R2": [(2, 2), (1, 2), (0, 2)],
    }
    conflicts = detect_conflicts(trajectories)

    assert any(conflict.conflict_type == "vertex" for conflict in conflicts)
