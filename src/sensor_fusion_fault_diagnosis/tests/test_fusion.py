"""Tests for sensor fusion fault diagnosis."""

from pathlib import Path
import sys

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from fusion import diagnosis_scores, run_fusion
from scenario import SENSORS, fault_labels, make_demo_scenario


def test_demo_scenario_contains_all_sensors():
    scenario = make_demo_scenario()

    assert scenario.steps > 0
    assert set(scenario.measurements) == set(SENSORS)
    assert all(values.shape == (scenario.steps, 2) for values in scenario.measurements.values())
    assert len(scenario.faults) == 3


def test_fusion_outputs_have_expected_shape():
    scenario = make_demo_scenario()
    result = run_fusion(scenario)

    assert result.estimates.shape == scenario.truth.shape
    assert result.naive_estimates.shape == (scenario.steps, 2)
    assert result.rmse_robust > 0


def test_robust_fusion_improves_rmse_against_naive_average():
    scenario = make_demo_scenario()
    result = run_fusion(scenario)

    assert result.rmse_robust < result.rmse_naive


def test_fault_diagnosis_detects_each_injected_sensor_fault():
    scenario = make_demo_scenario()
    result = run_fusion(scenario)
    scores = diagnosis_scores(result, fault_labels(scenario))

    for sensor in SENSORS:
        assert scores[sensor]["true_positive"] > 0
        assert scores[sensor]["recall"] > 0.2


def test_fault_labels_match_scenario_length():
    scenario = make_demo_scenario()
    labels = fault_labels(scenario)

    assert set(labels) == set(SENSORS)
    assert all(len(label) == scenario.steps for label in labels.values())
