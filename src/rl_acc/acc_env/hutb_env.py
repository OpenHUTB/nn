import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class HutbACCEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        self.observation_space = spaces.Box(
            low=np.array([MIN_SPEED, MIN_SPEED, 0, -MAX_SPEED, MIN_SPEED]),
            high=np.array([MAX_SPEED, MAX_SPEED, 200, MAX_SPEED, MAX_SPEED]),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([MAX_DECELERATION]),
            high=np.array([MAX_ACCELERATION]),
            dtype=np.float32
        )
        
        self.simulator = None
        self._init_simulator()
        
        self.state = None
        self.steps = 0
        
    def _init_simulator(self):
        try:
            from hutb_simulator import HutbSimulator
            self.simulator = HutbSimulator()
            self.use_hutb = True
        except ImportError:
            self.use_hutb = False
            print("HUTB simulator not found, using simple simulation")
    
    def _get_observation(self):
        if self.use_hutb and self.simulator:
            ego_speed = self.simulator.get_ego_speed()
            front_speed = self.simulator.get_front_vehicle_speed()
            distance = self.simulator.get_distance_to_front()
            rel_speed = front_speed - ego_speed
            target_speed = TARGET_SPEED
        else:
            ego_speed = self.state[0]
            front_speed = self.state[1]
            distance = self.state[2]
            rel_speed = front_speed - ego_speed
            target_speed = TARGET_SPEED
        
        return np.array([ego_speed, front_speed, distance, rel_speed, target_speed], dtype=np.float32)
    
    def _calculate_reward(self, action):
        ego_speed = self.state[0]
        front_speed = self.state[1]
        distance = self.state[2]
        acceleration = action[0]
        
        speed_error = abs(ego_speed - TARGET_SPEED)
        speed_reward = -0.5 * speed_error
        
        safe_distance = SAFETY_DISTANCE + ego_speed * 1.5
        distance_error = abs(distance - safe_distance)
        distance_reward = -0.3 * distance_error if distance > 5 else -100
        
        comfort_reward = -0.1 * abs(acceleration)
        
        collision_penalty = -500 if distance < 3 else 0
        
        return speed_reward + distance_reward + comfort_reward + collision_penalty
    
    def step(self, action):
        self.steps += 1
        
        acceleration = np.clip(action[0], MAX_DECELERATION, MAX_ACCELERATION)
        
        if self.use_hutb and self.simulator:
            self.simulator.set_acceleration(acceleration)
            self.simulator.step(DT)
            
            ego_speed = self.simulator.get_ego_speed()
            front_speed = self.simulator.get_front_vehicle_speed()
            distance = self.simulator.get_distance_to_front()
        else:
            ego_speed = self.state[0] + acceleration * DT
            ego_speed = np.clip(ego_speed, MIN_SPEED, MAX_SPEED)
            
            front_accel = np.random.uniform(-1, 1)
            front_speed = self.state[1] + front_accel * DT
            front_speed = np.clip(front_speed, MIN_SPEED, MAX_SPEED)
            
            distance = self.state[2] + (self.state[1] - self.state[0]) * DT
            distance = max(distance, 0)
        
        self.state = np.array([ego_speed, front_speed, distance])
        
        reward = self._calculate_reward(action)
        
        terminated = distance < 3 or self.steps >= EPISODE_LENGTH
        truncated = False
        
        observation = self._get_observation()
        info = {
            'ego_speed': ego_speed,
            'front_speed': front_speed,
            'distance': distance,
            'acceleration': acceleration,
            'reward': reward
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.use_hutb and self.simulator:
            self.simulator.reset()
            ego_speed = self.simulator.get_ego_speed()
            front_speed = self.simulator.get_front_vehicle_speed()
            distance = self.simulator.get_distance_to_front()
        else:
            ego_speed = np.random.uniform(15, 25)
            front_speed = np.random.uniform(15, 25)
            distance = np.random.uniform(20, 50)
        
        self.state = np.array([ego_speed, front_speed, distance])
        self.steps = 0
        
        return self._get_observation(), {}
    
    def render(self):
        if self.render_mode == 'human':
            print(f"Step: {self.steps:4d} | "
                  f"Ego: {self.state[0]:.2f} m/s | "
                  f"Front: {self.state[1]:.2f} m/s | "
                  f"Dist: {self.state[2]:.2f} m")
        return None
    
    def close(self):
        if self.simulator:
            self.simulator.close()
