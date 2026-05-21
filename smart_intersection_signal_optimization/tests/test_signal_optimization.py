"""Tests for smart intersection signal optimization."""

from pathlib import Path
import sys

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from optimizer import genes_to_plan, optimize_signal_plan
from scenario import SignalPlan, make_demo_scenario, webster_baseline
from simulator import simulate_plan


def test_demo_scenario_has_four_phases():
    scenario = make_demo_scenario()

    assert len(scenario.approaches) == 4
    assert scenario.horizon_seconds > 0
    assert all(approach.demand_vph > 0 for approach in scenario.approaches)


def test_signal_plan_normalization_preserves_cycle_budget():
    scenario = make_demo_scenario()
    plan = SignalPlan(
        cycle_length=100,
        greens={
            "north_south_through": 50,
            "east_west_through": 20,
            "north_south_left": 12,
            "east_west_left": 8,
        },
    ).normalized(scenario.approaches)

    assert sum(plan.greens.values()) == plan.cycle_length - plan.lost_time
    assert min(plan.greens.values()) >= 8


def test_webster_baseline_can_be_simulated():
    scenario = make_demo_scenario()
    result = simulate_plan(scenario, webster_baseline(scenario))

    assert result.average_delay > 0
    assert result.max_queue > 0
    assert 0 < result.throughput <= 1.0


def test_gene_conversion_creates_valid_plan():
    scenario = make_demo_scenario()
    plan = genes_to_plan(scenario, [90, 0.4, 0.3, 0.2, 0.1])

    assert 60 <= plan.cycle_length <= 150
    assert sum(plan.greens.values()) == plan.cycle_length - plan.lost_time


def test_optimization_improves_or_matches_baseline_score():
    scenario = make_demo_scenario()
    result = optimize_signal_plan(scenario, population_size=12, generations=8, seed=3)

    assert result.optimized.score <= result.baseline.score
    assert result.optimized.plan.cycle_length >= 60
    assert len(result.history.generation) == 8
