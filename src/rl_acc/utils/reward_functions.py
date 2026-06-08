import numpy as np


def calculate_reward(ego_speed, lead_speed, distance, acceleration, target_speed, safety_distance):
    reward = 0.0

    speed_error = abs(ego_speed - target_speed)
    speed_reward = -0.1 * speed_error
    if speed_error < 1.0:
        speed_reward += 0.5

    desired_distance = safety_distance + ego_speed * 1.5
    distance_error = abs(distance - desired_distance)

    if distance < 0:
        distance_reward = -100.0
    elif distance < safety_distance:
        distance_reward = -5.0 - 0.5 * (safety_distance - distance)
    elif distance > 150:
        distance_reward = -2.0
    else:
        distance_reward = 1.0 - 0.01 * distance_error

    acceleration_penalty = -0.5 * abs(acceleration)

    relative_speed = lead_speed - ego_speed
    relative_speed_reward = -0.1 * abs(relative_speed)

    efficiency_reward = 0.05 * ego_speed

    if distance > safety_distance + ego_speed * 0.5:
        collision_avoidance_reward = 0.5
    else:
        collision_avoidance_reward = 0.0

    reward = (
        speed_reward +
        distance_reward +
        acceleration_penalty +
        relative_speed_reward +
        efficiency_reward +
        collision_avoidance_reward
    )

    if ego_speed < 0:
        reward -= 10.0

    if acceleration > 3.0 or acceleration < -4.0:
        reward -= 2.0

    return reward