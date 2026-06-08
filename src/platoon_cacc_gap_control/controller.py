"""Baseline ACC and communication-aware CACC platoon simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario import PlatoonScenario


@dataclass
class SimulationResult:
    """State history and safety metrics for one controller."""

    controller_name: str
    time: np.ndarray
    positions: np.ndarray
    speeds: np.ndarray
    accelerations: np.ndarray
    gaps: np.ndarray
    desired_gaps: np.ndarray
    gap_errors: np.ndarray
    min_gap: float
    mean_abs_gap_error: float
    max_abs_gap_error: float
    speed_std_amplification: float
    comfort_cost: float
    emergency_brake_count: int


def _is_dropout(time_value: float, windows: list[tuple[float, float]]) -> bool:
    return any(start <= time_value <= end for start, end in windows)


def _leader_accel_with_delay(scenario: PlatoonScenario, current_index: int) -> float:
    delay_steps = int(round(scenario.communication_delay_s / (scenario.time[1] - scenario.time[0])))
    delayed_index = max(0, current_index - delay_steps)
    return float(scenario.leader_acceleration[delayed_index])


def desired_gap(speed: float, scenario: PlatoonScenario) -> float:
    return scenario.standstill_gap + scenario.time_headway * max(speed, 0.0)


def simulate_platoon(scenario: PlatoonScenario, controller_name: str) -> SimulationResult:
    """Run either a human-like ACC baseline or a feed-forward CACC controller."""
    if controller_name not in {"acc_baseline", "cacc_feedforward"}:
        raise ValueError(f"unknown controller: {controller_name}")

    time = scenario.time
    dt = float(time[1] - time[0])
    n_steps = len(time)
    n_vehicles = scenario.vehicle_count

    positions = np.zeros((n_steps, n_vehicles))
    speeds = np.zeros((n_steps, n_vehicles))
    accelerations = np.zeros((n_steps, n_vehicles))

    spacing = desired_gap(scenario.initial_speed, scenario) + scenario.vehicle_length
    for vehicle in range(n_vehicles):
        positions[0, vehicle] = scenario.leader_position[0] - vehicle * spacing
        speeds[0, vehicle] = scenario.initial_speed

    positions[:, 0] = scenario.leader_position
    speeds[:, 0] = scenario.leader_speed
    accelerations[:, 0] = scenario.leader_acceleration

    kp_gap = 0.18 if controller_name == "acc_baseline" else 0.23
    kd_speed = 0.55 if controller_name == "acc_baseline" else 0.75
    feedforward_gain = 0.0 if controller_name == "acc_baseline" else 0.72

    for step in range(1, n_steps):
        for vehicle in range(1, n_vehicles):
            previous_position = positions[step - 1, vehicle - 1]
            previous_speed = speeds[step - 1, vehicle - 1]
            own_position = positions[step - 1, vehicle]
            own_speed = speeds[step - 1, vehicle]

            gap = previous_position - own_position - scenario.vehicle_length
            target_gap = desired_gap(own_speed, scenario)
            gap_error = gap - target_gap
            relative_speed = previous_speed - own_speed

            feedforward = 0.0
            if controller_name == "cacc_feedforward" and not _is_dropout(float(time[step]), scenario.dropout_windows):
                feedforward = feedforward_gain * _leader_accel_with_delay(scenario, step)

            command = kp_gap * gap_error + kd_speed * relative_speed + feedforward
            if gap < scenario.standstill_gap * 0.72:
                command -= 2.0

            accelerations[step - 1, vehicle] = float(np.clip(command, -4.2, 2.2))
            speeds[step, vehicle] = np.clip(own_speed + accelerations[step - 1, vehicle] * dt, 0.0, 28.0)
            positions[step, vehicle] = own_position + own_speed * dt + 0.5 * accelerations[step - 1, vehicle] * dt * dt

    accelerations[-1, 1:] = accelerations[-2, 1:]

    gaps = positions[:, :-1] - positions[:, 1:] - scenario.vehicle_length
    desired_gaps = scenario.standstill_gap + scenario.time_headway * speeds[:, 1:]
    gap_errors = gaps - desired_gaps
    follower_speed_std = float(np.mean(np.std(speeds[:, 1:], axis=0)))
    leader_speed_std = float(np.std(speeds[:, 0]))

    return SimulationResult(
        controller_name=controller_name,
        time=time,
        positions=positions,
        speeds=speeds,
        accelerations=accelerations,
        gaps=gaps,
        desired_gaps=desired_gaps,
        gap_errors=gap_errors,
        min_gap=float(np.min(gaps)),
        mean_abs_gap_error=float(np.mean(np.abs(gap_errors))),
        max_abs_gap_error=float(np.max(np.abs(gap_errors))),
        speed_std_amplification=float(follower_speed_std / max(leader_speed_std, 1e-6)),
        comfort_cost=float(np.mean(accelerations[:, 1:] ** 2)),
        emergency_brake_count=int(np.sum(accelerations[:, 1:] < -3.0)),
    )


def compare_controllers(scenario: PlatoonScenario) -> tuple[SimulationResult, SimulationResult]:
    """Run baseline and optimized controller for the same scenario."""
    baseline = simulate_platoon(scenario, "acc_baseline")
    optimized = simulate_platoon(scenario, "cacc_feedforward")
    return baseline, optimized
