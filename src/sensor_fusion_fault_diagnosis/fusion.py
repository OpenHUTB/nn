"""Robust Kalman fusion and residual-based fault diagnosis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario import SENSORS, Scenario


@dataclass
class FusionResult:
    """Outputs from the robust fusion pipeline."""

    estimates: np.ndarray
    naive_estimates: np.ndarray
    residuals: dict[str, np.ndarray]
    fault_flags: dict[str, np.ndarray]
    weights: dict[str, np.ndarray]
    rmse_robust: float
    rmse_naive: float


SENSOR_NOISE = {
    "gps": 1.25,
    "radar": 0.75,
    "odometry": 0.45,
}


def transition_matrix(dt: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def run_fusion(scenario: Scenario, threshold: float = 3.0) -> FusionResult:
    """Fuse sensor positions while down-weighting residual outliers."""
    steps = scenario.steps
    dt = scenario.dt
    f = transition_matrix(dt)
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    q = np.diag([0.03, 0.03, 0.2, 0.2])

    state = np.array([*scenario.measurements["odometry"][0], 0.0, 0.0], dtype=float)
    cov = np.eye(4) * 3.0

    estimates = np.zeros((steps, 4))
    naive_estimates = np.zeros((steps, 2))
    residuals = {sensor: np.zeros(steps) for sensor in SENSORS}
    fault_flags = {sensor: np.zeros(steps, dtype=int) for sensor in SENSORS}
    weights = {sensor: np.zeros(steps) for sensor in SENSORS}

    for step in range(steps):
        state_pred = f @ state
        cov_pred = f @ cov @ f.T + q
        pred_pos = h @ state_pred

        robust_sum = np.zeros(2)
        weight_sum = 0.0
        naive_sum = np.zeros(2)

        for sensor in SENSORS:
            measurement = scenario.measurements[sensor][step]
            residual = float(np.linalg.norm(measurement - pred_pos))
            residuals[sensor][step] = residual
            sigma = SENSOR_NOISE[sensor]
            normalized = residual / sigma
            is_fault = normalized > threshold
            fault_flags[sensor][step] = int(is_fault)

            base_weight = 1.0 / (sigma * sigma)
            robust_weight = base_weight * (0.12 if is_fault else 1.0)
            weights[sensor][step] = robust_weight
            robust_sum += robust_weight * measurement
            weight_sum += robust_weight
            naive_sum += measurement

        robust_measurement = robust_sum / weight_sum
        naive_estimates[step] = naive_sum / len(SENSORS)

        r_scale = 1.0 / max(weight_sum, 1e-6)
        r = np.eye(2) * r_scale
        innovation = robust_measurement - pred_pos
        s = h @ cov_pred @ h.T + r
        kalman_gain = cov_pred @ h.T @ np.linalg.inv(s)
        state = state_pred + kalman_gain @ innovation
        cov = (np.eye(4) - kalman_gain @ h) @ cov_pred
        estimates[step] = state

    robust_rmse = position_rmse(estimates[:, :2], scenario.truth[:, :2])
    naive_rmse = position_rmse(naive_estimates, scenario.truth[:, :2])
    return FusionResult(estimates, naive_estimates, residuals, fault_flags, weights, robust_rmse, naive_rmse)


def position_rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    error = estimate - truth
    return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))


def diagnosis_scores(result: FusionResult, labels: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Compute simple precision/recall scores for each sensor."""
    scores: dict[str, dict[str, float]] = {}
    for sensor in SENSORS:
        pred = result.fault_flags[sensor].astype(bool)
        truth = labels[sensor].astype(bool)
        tp = int(np.sum(pred & truth))
        fp = int(np.sum(pred & ~truth))
        fn = int(np.sum(~pred & truth))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores[sensor] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
        }
    return scores
