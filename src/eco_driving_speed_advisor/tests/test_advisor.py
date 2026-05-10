"""Tests for eco-driving speed advisor."""

from pathlib import Path
import sys

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from advisor import plan_eco_speed, signal_compliance, simulate_cruise_baseline
from scenario import make_demo_scenario


def test_demo_scenario_has_signalized_corridor():
    scenario = make_demo_scenario()

    assert scenario.length_m > 0
    assert len(scenario.lights) == 4
    assert scenario.vehicle.max_speed > scenario.vehicle.min_speed


def test_eco_plan_reaches_corridor_end():
    scenario = make_demo_scenario()
    eco = plan_eco_speed(scenario)

    assert eco.position_m[-1] >= scenario.length_m
    assert eco.time_s[-1] > 0
    assert eco.energy > 0


def test_eco_plan_passes_all_signals_on_green():
    scenario = make_demo_scenario()
    eco = plan_eco_speed(scenario)
    compliance = signal_compliance(scenario, eco)

    assert all(compliance.values())


def test_baseline_has_at_least_one_stop():
    scenario = make_demo_scenario()
    baseline = simulate_cruise_baseline(scenario)

    assert baseline.stops >= 1


def test_eco_plan_reduces_energy_against_baseline():
    scenario = make_demo_scenario()
    eco = plan_eco_speed(scenario)
    baseline = simulate_cruise_baseline(scenario)

    assert eco.energy < baseline.energy
