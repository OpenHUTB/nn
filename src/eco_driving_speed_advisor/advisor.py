"""Eco-driving speed planning and baseline simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scenario import CorridorScenario, TrafficLight


@dataclass
class Trajectory:
    name: str
    time_s: np.ndarray
    position_m: np.ndarray
    speed_mps: np.ndarray
    acceleration_mps2: np.ndarray
    stops: int
    delay_s: float
    energy: float
    pass_times: dict[str, float] = field(default_factory=dict)


def target_arrival_for_light(scenario: CorridorScenario, light: TrafficLight, current_pos: float, current_time: float) -> float:
    """Find a feasible arrival time inside a green window."""
    distance = max(0.0, light.position_m - current_pos)
    earliest = current_time + distance / scenario.vehicle.max_speed
    latest = current_time + distance / scenario.vehicle.min_speed
    probe = earliest
    for _ in range(8):
        green_start, green_end = light.next_green_window(probe)
        arrival = max(earliest, green_start + 1.0)
        if arrival <= min(latest, green_end - 1.0):
            return arrival
        probe = green_end + 0.1
    return earliest


def plan_eco_speed(scenario: CorridorScenario) -> Trajectory:
    """Plan a smooth speed profile that tries to pass signals during green."""
    anchors = [(0.0, 0.0)]
    current_pos = 0.0
    current_time = 0.0
    for light in scenario.lights:
        arrival = target_arrival_for_light(scenario, light, current_pos, current_time)
        anchors.append((arrival, light.position_m))
        current_pos = light.position_m
        current_time = arrival
    remaining = scenario.length_m - current_pos
    finish_time = current_time + remaining / scenario.vehicle.cruise_speed
    anchors.append((finish_time, scenario.length_m))
    return sample_profile("eco speed advisor", anchors, scenario)


def simulate_cruise_baseline(scenario: CorridorScenario) -> Trajectory:
    """Simulate fixed-speed cruise with red-light stops."""
    dt = scenario.dt
    time_values = [0.0]
    pos_values = [0.0]
    speed_values = [scenario.vehicle.cruise_speed]
    pass_times: dict[str, float] = {}
    stops = 0
    delay = 0.0
    current_time = 0.0
    current_pos = 0.0

    light_index = 0
    while current_pos < scenario.length_m - 1e-6:
        speed = scenario.vehicle.cruise_speed
        next_pos = min(scenario.length_m, current_pos + speed * dt)
        if light_index < len(scenario.lights):
            light = scenario.lights[light_index]
            if current_pos < light.position_m <= next_pos:
                arrival_time = current_time + (light.position_m - current_pos) / speed
                if not light.is_green(arrival_time):
                    green_start, _ = light.next_green_window(arrival_time)
                    wait = green_start - arrival_time
                    stops += 1
                    delay += wait
                    time_values.extend([arrival_time, green_start])
                    pos_values.extend([light.position_m, light.position_m])
                    speed_values.extend([0.0, 0.0])
                    current_time = green_start
                    current_pos = light.position_m
                else:
                    current_time = arrival_time
                    current_pos = light.position_m
                pass_times[light.light_id] = current_time
                light_index += 1
            else:
                current_time += dt
                current_pos = next_pos
        else:
            current_time += dt
            current_pos = next_pos
        if time_values[-1] != current_time or pos_values[-1] != current_pos:
            time_values.append(current_time)
            pos_values.append(current_pos)
            speed_values.append(speed)

    time = np.asarray(time_values)
    position = np.asarray(pos_values)
    speed = np.asarray(speed_values)
    acceleration = np.gradient(speed, time, edge_order=1) if len(time) > 1 else np.zeros_like(speed)
    return Trajectory(
        "fixed cruise baseline",
        time,
        position,
        speed,
        acceleration,
        stops,
        delay,
        estimate_energy(speed, acceleration, scenario.dt),
        pass_times,
    )


def sample_profile(name: str, anchors: list[tuple[float, float]], scenario: CorridorScenario) -> Trajectory:
    dt = scenario.dt
    finish = anchors[-1][0]
    time = np.arange(0.0, finish + dt, dt)
    anchor_t = np.asarray([item[0] for item in anchors])
    anchor_x = np.asarray([item[1] for item in anchors])
    position = np.interp(time, anchor_t, anchor_x)
    speed = np.gradient(position, dt)
    speed = np.clip(speed, scenario.vehicle.min_speed, scenario.vehicle.max_speed)
    acceleration = np.gradient(speed, dt)
    pass_times = {}
    for light in scenario.lights:
        index = int(np.argmin(np.abs(position - light.position_m)))
        pass_times[light.light_id] = float(time[index])
    return Trajectory(name, time, position, speed, acceleration, stops=0, delay_s=0.0, energy=estimate_energy(speed, acceleration, dt), pass_times=pass_times)


def estimate_energy(speed: np.ndarray, acceleration: np.ndarray, dt: float) -> float:
    """A lightweight traction energy proxy for comparison."""
    rolling = 0.012 * speed**2
    cruise_loss = 0.18 * speed
    accel_cost = 5.0 * np.maximum(acceleration, 0.0) ** 2
    idle_loss = 1.8 * (speed < 0.5)
    jerk = np.gradient(acceleration, dt)
    comfort = 0.08 * jerk**2
    return float(np.sum((rolling + cruise_loss + accel_cost + idle_loss + comfort) * dt))


def signal_compliance(scenario: CorridorScenario, trajectory: Trajectory) -> dict[str, bool]:
    return {
        light.light_id: light.is_green(trajectory.pass_times.get(light.light_id, 0.0))
        for light in scenario.lights
    }
