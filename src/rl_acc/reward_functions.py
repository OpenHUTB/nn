import numpy as np
from config import TARGET_SPEED, SAFETY_DISTANCE, MAX_ACCELERATION, MAX_DECELERATION

class RewardFunction:
    def __init__(self):
        self.speed_weight = 1.0
        self.distance_weight = 1.5
        self.comfort_weight = 0.5
        self.efficiency_weight = 0.3
        
    def calculate(self, ego_speed, front_speed, distance, acceleration):
        rewards = {}
        
        speed_reward = self._speed_tracking_reward(ego_speed)
        rewards['speed'] = speed_reward
        
        distance_reward = self._distance_control_reward(ego_speed, distance)
        rewards['distance'] = distance_reward
        
        comfort_reward = self._comfort_reward(acceleration)
        rewards['comfort'] = comfort_reward
        
        efficiency_reward = self._efficiency_reward(acceleration)
        rewards['efficiency'] = efficiency_reward
        
        total_reward = (
            self.speed_weight * speed_reward +
            self.distance_weight * distance_reward +
            self.comfort_weight * comfort_reward +
            self.efficiency_weight * efficiency_reward
        )
        
        return total_reward, rewards
    
    def _speed_tracking_reward(self, ego_speed):
        speed_error = abs(ego_speed - TARGET_SPEED)
        reward = np.exp(-speed_error / 5.0) - 0.5
        return reward
    
    def _distance_control_reward(self, ego_speed, distance):
        safe_distance = SAFETY_DISTANCE + ego_speed * 1.5
        distance_error = abs(distance - safe_distance)
        
        if distance < 5:
            return -100.0
        elif distance < SAFETY_DISTANCE:
            return -50.0 * (SAFETY_DISTANCE - distance) / SAFETY_DISTANCE
        
        max_error = 100
        normalized_error = min(distance_error / max_error, 1.0)
        reward = np.exp(-normalized_error * 3.0) - 0.3
        return reward
    
    def _comfort_reward(self, acceleration):
        jerk_penalty = abs(acceleration)
        max_jerk = abs(MAX_ACCELERATION - MAX_DECELERATION)
        normalized_jerk = jerk_penalty / max_jerk
        reward = np.exp(-normalized_jerk * 2.0) - 0.5
        return reward
    
    def _efficiency_reward(self, acceleration):
        if acceleration > 0:
            efficiency = -acceleration * 0.1
        elif acceleration < -0.5:
            efficiency = -abs(acceleration) * 0.05
        else:
            efficiency = 0.1
        return efficiency
