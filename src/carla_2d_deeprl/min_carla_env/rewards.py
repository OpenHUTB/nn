import math
import carla
import numpy as np


class RewardCalculator:
    def __init__(self, weights=None):
        self.weights = weights or {
            "lane_center": 1.0,
            "speed": 0.5,
            "collision": -50.0,
            "lane_invasion": -20.0,
            "stuck": -10.0,
            "progress": 0.1,
            "steering_smoothness": -0.05,
        }
        self._prev_steer = 0.0
        self._prev_location = None
        self._cumulative_progress = 0.0

    def reset(self):
        self._prev_steer = 0.0
        self._prev_location = None
        self._cumulative_progress = 0.0

    def lane_center_reward(self, world_map, location):
        wp = world_map.get_waypoint(location, carla.LaneType.Driving)
        wp_location = wp.transform.location
        dist = math.sqrt(
            (wp_location.x - location.x) ** 2 +
            (wp_location.y - location.y) ** 2 +
            (wp_location.z - location.z) ** 2
        )
        if dist < 0.5:
            return 0.5
        else:
            return -np.exp(dist)

    def speed_reward(self, kmh, target_speed=30.0):
        if kmh < 1.0:
            return -0.5
        diff = abs(kmh - target_speed)
        if diff < 5.0:
            return 0.3
        elif diff < 10.0:
            return 0.0
        else:
            return -0.1 * (diff / 10.0)

    def collision_penalty(self, has_collision):
        return 1.0 if has_collision else 0.0

    def lane_invasion_penalty(self, crossed_types):
        for t in crossed_types:
            if t in (carla.LaneMarkingType.Solid, carla.LaneMarkingType.NONE):
                return 1.0
        return 0.0

    def stuck_penalty(self, is_stuck):
        return 1.0 if is_stuck else 0.0

    def progress_reward(self, location):
        if self._prev_location is None:
            self._prev_location = location
            return 0.0
        dist = math.sqrt(
            (location.x - self._prev_location.x) ** 2 +
            (location.y - self._prev_location.y) ** 2
        )
        self._prev_location = location
        self._cumulative_progress += dist
        return dist

    def steering_smoothness_penalty(self, steer):
        penalty = abs(steer - self._prev_steer)
        self._prev_steer = steer
        return penalty

    def compute(self, world_map, location, kmh, has_collision,
                crossed_types, is_stuck, steer):
        reward = 0.0
        info = {}

        r_lane = self.lane_center_reward(world_map, location)
        reward += self.weights["lane_center"] * r_lane
        info["r_lane_center"] = r_lane

        r_speed = self.speed_reward(kmh)
        reward += self.weights["speed"] * r_speed
        info["r_speed"] = r_speed

        if has_collision:
            c = self.collision_penalty(has_collision)
            reward += self.weights["collision"] * c
            info["r_collision"] = c

        if crossed_types:
            li = self.lane_invasion_penalty(crossed_types)
            if li > 0:
                reward += self.weights["lane_invasion"] * li
                info["r_lane_invasion"] = li

        if is_stuck:
            s = self.stuck_penalty(is_stuck)
            reward += self.weights["stuck"] * s
            info["r_stuck"] = s

        r_progress = self.progress_reward(location)
        reward += self.weights["progress"] * r_progress
        info["r_progress"] = r_progress

        r_steer = self.steering_smoothness_penalty(steer)
        reward += self.weights["steering_smoothness"] * r_steer
        info["r_steering_smoothness"] = r_steer

        info["total"] = reward
        return reward, info