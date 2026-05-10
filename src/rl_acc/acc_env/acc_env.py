import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    TARGET_SPEED, SAFETY_DISTANCE, MAX_ACCELERATION,
    MAX_DECELERATION, DT, EPISODE_LENGTH, MAX_SPEED, MIN_SPEED
)
from utils.reward_functions import calculate_reward


class ACCEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.target_speed = TARGET_SPEED
        self.safety_distance = SAFETY_DISTANCE
        self.max_acceleration = MAX_ACCELERATION
        self.max_deceleration = MAX_DECELERATION
        self.dt = DT
        self.episode_length = EPISODE_LENGTH
        self.max_speed = MAX_SPEED
        self.min_speed = MIN_SPEED

        self.action_space = spaces.Box(
            low=np.array([self.max_deceleration], dtype=np.float32),
            high=np.array([self.max_acceleration], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -10.0, 0.0], dtype=np.float32),
            high=np.array([self.max_speed, self.max_speed * 2, 200.0, 10.0, self.max_speed], dtype=np.float32),
            dtype=np.float32
        )

        self.state = None
        self.current_step = 0
        self.history = {
            'ego_speed': [], 'lead_speed': [], 'distance': [],
            'acceleration': [], 'reward': []
        }

    def _get_observation(self):
        ego_speed, lead_speed, distance = self.state
        relative_speed = lead_speed - ego_speed
        return np.array([
            ego_speed, lead_speed, distance, relative_speed, self.target_speed
        ], dtype=np.float32)

    def _is_done(self):
        ego_speed, lead_speed, distance = self.state
        if distance < 0:
            return True
        if distance > 200:
            return True
        if ego_speed < 0.1 and self.current_step > 100:
            return True
        if self.current_step >= self.episode_length:
            return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.ego_speed = np.random.uniform(self.target_speed * 0.8, self.target_speed * 1.2)
        self.lead_speed = np.random.uniform(self.target_speed * 0.8, self.target_speed * 1.2)
        self.distance = np.random.uniform(self.safety_distance * 1.5, self.safety_distance * 3)

        self.state = np.array([self.ego_speed, self.lead_speed, self.distance])
        self.current_step = 0
        self.history = {
            'ego_speed': [], 'lead_speed': [], 'distance': [],
            'acceleration': [], 'reward': []
        }
        self._lead_acceleration = 0.0

        return self._get_observation(), {}

    def step(self, action):
        acceleration = np.clip(action[0], self.max_deceleration, self.max_acceleration)

        ego_speed, lead_speed, distance = self.state

        new_ego_speed = ego_speed + acceleration * self.dt
        new_ego_speed = np.clip(new_ego_speed, self.min_speed, self.max_speed)

        if self.current_step % 100 == 0:
            behavior = np.random.choice(['constant', 'accelerate', 'decelerate', 'random'], p=[0.4, 0.2, 0.2, 0.2])
            if behavior == 'constant':
                self._lead_acceleration = 0.0
            elif behavior == 'accelerate':
                self._lead_acceleration = np.random.uniform(0.5, 1.5)
            elif behavior == 'decelerate':
                self._lead_acceleration = np.random.uniform(-2.0, -0.5)
            else:
                self._lead_acceleration = np.random.uniform(-1.0, 1.0)

        new_lead_speed = lead_speed + self._lead_acceleration * self.dt
        new_lead_speed = np.clip(new_lead_speed, 0.0, self.max_speed * 1.5)

        new_distance = distance + (lead_speed - new_ego_speed) * self.dt

        self.state = np.array([new_ego_speed, new_lead_speed, new_distance])

        reward = calculate_reward(
            new_ego_speed, new_lead_speed, new_distance,
            acceleration, self.target_speed, self.safety_distance
        )

        self.history['ego_speed'].append(new_ego_speed)
        self.history['lead_speed'].append(new_lead_speed)
        self.history['distance'].append(new_distance)
        self.history['acceleration'].append(acceleration)
        self.history['reward'].append(reward)

        self.current_step += 1

        observation = self._get_observation()
        done = self._is_done()

        info = {
            'step': self.current_step,
            'collision': new_distance < 0,
            'ego_speed': new_ego_speed,
            'lead_speed': new_lead_speed,
            'distance': new_distance
        }

        return observation, reward, done, False, info

    def render(self):
        if self.current_step % 100 == 0:
            ego_speed, lead_speed, distance = self.state
            print(f"Step: {self.current_step} | Ego: {ego_speed:.1f}m/s | Lead: {lead_speed:.1f}m/s | Dist: {distance:.1f}m")

    def close(self):
        pass