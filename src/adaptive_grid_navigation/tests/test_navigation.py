"""Basic tests for adaptive grid navigation."""

from pathlib import Path
import sys

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from grid_map import GridMap, make_demo_map
from planners import AStarPlanner, DijkstraPlanner, GreedyBestFirstPlanner
from simulator import run_dynamic_replanning


def test_demo_map_is_valid():
    grid_map = make_demo_map()

    assert grid_map.height > 0
    assert grid_map.width > 0
    assert grid_map.is_free(grid_map.start)
    assert grid_map.is_free(grid_map.goal)


def test_astar_finds_path_on_demo_map():
    grid_map = make_demo_map()
    result = AStarPlanner().plan(grid_map)

    assert result.success
    assert result.path[0] == grid_map.start
    assert result.path[-1] == grid_map.goal
    assert result.path_length > 0


def test_astar_expands_no_more_than_dijkstra_on_demo_map():
    grid_map = make_demo_map()
    astar = AStarPlanner().plan(grid_map)
    dijkstra = DijkstraPlanner().plan(grid_map)

    assert astar.success and dijkstra.success
    assert astar.path_length == dijkstra.path_length
    assert astar.expanded_count <= dijkstra.expanded_count


def test_greedy_best_first_returns_valid_path_when_successful():
    grid_map = make_demo_map()
    result = GreedyBestFirstPlanner().plan(grid_map)

    assert result.success
    assert result.path[0] == grid_map.start
    assert result.path[-1] == grid_map.goal


def test_dynamic_replanning_produces_frames():
    grid_map = make_demo_map()
    result = run_dynamic_replanning(grid_map, max_steps=60)

    assert result.frames
    assert result.replans > 0
    assert result.travelled > 0


def test_invalid_start_on_obstacle_is_rejected():
    grid_map = make_demo_map()
    grid = grid_map.occupancy.copy()
    grid[grid_map.start] = 1

    try:
        GridMap(grid, grid_map.start, grid_map.goal)
    except ValueError as exc:
        assert "start" in str(exc)
    else:
        raise AssertionError("GridMap should reject obstacle start")
