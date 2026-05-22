"""Tests for energy-aware drone delivery."""

from pathlib import Path
import sys

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from planner import astar_energy_path, battery_trace, movement_energy, naive_distance_plan, plan_mission
from scenario import make_demo_scenario


def test_demo_scenario_is_valid():
    scenario = make_demo_scenario()

    assert scenario.height > 0
    assert scenario.width > 0
    assert len(scenario.tasks) == 4
    assert len(scenario.stations) == 4
    assert scenario.is_free(scenario.drone.start)


def test_energy_path_reaches_goal():
    scenario = make_demo_scenario()
    task = scenario.tasks[0]
    path, energy = astar_energy_path(scenario, scenario.drone.start, task.pickup, payload_kg=0.0)

    assert path[0] == scenario.drone.start
    assert path[-1] == task.pickup
    assert energy > 0


def test_payload_increases_energy_cost():
    scenario = make_demo_scenario()
    current = scenario.drone.start
    nxt = next(scenario.neighbors(current))

    light = movement_energy(scenario, current, nxt, payload_kg=0.0)
    heavy = movement_energy(scenario, current, nxt, payload_kg=2.0)

    assert heavy > light


def test_plan_mission_finishes_all_tasks():
    scenario = make_demo_scenario()
    plan = plan_mission(scenario)

    assert plan.feasible
    assert sorted(plan.task_order) == sorted(task.task_id for task in scenario.tasks)
    assert plan.total_energy > 0
    assert plan.total_steps > 0


def test_energy_aware_plan_respects_reserve_better_than_naive_order():
    scenario = make_demo_scenario()
    smart = plan_mission(scenario)
    naive = naive_distance_plan(scenario)

    smart_min = min(battery_trace(scenario, smart))
    naive_min = min(battery_trace(scenario, naive))

    assert smart_min >= scenario.drone.battery_capacity * scenario.drone.reserve_ratio
    assert smart_min > naive_min
