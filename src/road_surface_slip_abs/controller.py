"""Emergency braking simulation with locked braking and adaptive ABS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario import BrakingScenario, friction_at_position


@dataclass
class BrakingResult:
    controller_name: str
    time: np.ndarray
    position_m: np.ndarray
    speed_mps: np.ndarray
    slip_ratio: np.ndarray
    brake_command: np.ndarray
    true_mu: np.ndarray
    estimated_mu: np.ndarray
    deceleration_mps2: np.ndarray
    stopping_distance_m: float
    stopping_time_s: float
    max_slip_ratio: float
    mean_abs_slip_error: float
    comfort_cost: float
    low_mu_detection_time_s: float | None


def tire_grip_factor(slip_ratio: float) -> float:
    """Approximate normalized tire force from longitudinal slip."""
    slip = float(np.clip(slip_ratio, 0.0, 1.0))
    if slip <= 0.18:
        return 0.55 + 0.45 * slip / 0.18
    return max(0.42, 1.0 - 0.70 * (slip - 0.18) / 0.82)


def _target_slip(controller_name: str, scenario: BrakingScenario, estimated_mu: float) -> float:
    if controller_name == "locked_brake":
        return 0.94
    if controller_name == "adaptive_abs":
        low_mu_bias = 0.04 * max(0.0, 0.45 - estimated_mu)
        return float(np.clip(scenario.optimal_slip - low_mu_bias, 0.12, 0.20))
    raise ValueError(f"unknown controller: {controller_name}")


def simulate_braking(scenario: BrakingScenario, controller_name: str) -> BrakingResult:
    """Simulate emergency braking over a mixed-friction road."""
    time = scenario.time
    dt = float(time[1] - time[0])
    n = len(time)

    position = np.zeros(n)
    speed = np.zeros(n)
    slip = np.zeros(n)
    brake_command = np.zeros(n)
    true_mu = np.zeros(n)
    estimated_mu = np.zeros(n)
    deceleration = np.zeros(n)

    speed[0] = scenario.initial_speed_mps
    estimated_mu[0] = 0.80
    stopped_index = n - 1

    for step in range(1, n):
        true_mu[step - 1] = friction_at_position(float(position[step - 1]), scenario)
        target = _target_slip(controller_name, scenario, float(estimated_mu[step - 1]))

        response = 2.4 if controller_name == "locked_brake" else 18.0
        surface_jump = abs(true_mu[step - 1] - true_mu[max(step - 2, 0)])
        slip_overshoot = 0.10 * surface_jump if controller_name == "adaptive_abs" else 0.0
        slip[step] = np.clip(slip[step - 1] + response * dt * (target + slip_overshoot - slip[step - 1]), 0.0, 0.99)
        brake_command[step] = np.clip(slip[step] / 0.20, 0.0, 1.0)

        grip = tire_grip_factor(float(slip[step]))
        deceleration[step] = true_mu[step - 1] * scenario.gravity * grip
        next_speed = max(0.0, speed[step - 1] - deceleration[step] * dt)
        average_speed = 0.5 * (speed[step - 1] + next_speed)
        speed[step] = next_speed
        position[step] = position[step - 1] + average_speed * dt

        measured_mu = deceleration[step] / max(scenario.gravity * grip, 1e-6)
        alpha = 0.08 if controller_name == "locked_brake" else 0.18
        estimated_mu[step] = (1.0 - alpha) * estimated_mu[step - 1] + alpha * measured_mu

        if next_speed <= 0.05:
            stopped_index = step
            position[step + 1 :] = position[step]
            speed[step + 1 :] = 0.0
            slip[step + 1 :] = slip[step]
            brake_command[step + 1 :] = brake_command[step]
            true_mu[step:] = friction_at_position(float(position[step]), scenario)
            estimated_mu[step + 1 :] = estimated_mu[step]
            deceleration[step + 1 :] = 0.0
            break

    true_mu[-1] = friction_at_position(float(position[-1]), scenario)
    slip_error = np.abs(slip[: stopped_index + 1] - scenario.optimal_slip)
    comfort_cost = float(np.mean(np.diff(deceleration[: stopped_index + 1], prepend=0.0) ** 2))

    low_mu_time = None
    low_mu_indices = np.where(estimated_mu[: stopped_index + 1] < 0.35)[0]
    if len(low_mu_indices) > 0:
        low_mu_time = float(time[int(low_mu_indices[0])])

    return BrakingResult(
        controller_name=controller_name,
        time=time,
        position_m=position,
        speed_mps=speed,
        slip_ratio=slip,
        brake_command=brake_command,
        true_mu=true_mu,
        estimated_mu=estimated_mu,
        deceleration_mps2=deceleration,
        stopping_distance_m=float(position[stopped_index]),
        stopping_time_s=float(time[stopped_index]),
        max_slip_ratio=float(np.max(slip[: stopped_index + 1])),
        mean_abs_slip_error=float(np.mean(slip_error)),
        comfort_cost=comfort_cost,
        low_mu_detection_time_s=low_mu_time,
    )


def compare_controllers(scenario: BrakingScenario) -> tuple[BrakingResult, BrakingResult]:
    baseline = simulate_braking(scenario, "locked_brake")
    optimized = simulate_braking(scenario, "adaptive_abs")
    return baseline, optimized
