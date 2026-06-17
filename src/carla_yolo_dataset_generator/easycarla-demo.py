"""
Author: SilverWings
GitHub: https://github.com/silverwingsbot

This script provides a minimal demo to interact with the EasyCarla-RL environment.
It follows the standard Gym interface (reset, step) and demonstrates basic environment usage.

"""

import gym
import easycarla
import carla
import random
import numpy as np

# Configure environment parameters
params = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'dt': 0.1,  # time interval between two frames
    'ego_vehicle_filter': 'vehicle.tesla.model3',  # filter for defining ego vehicle
    'surrounding_vehicle_spawned_randomly': True, # Whether surrounding vehicles are spawned randomly (True) or set manually (False)
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypoints': 12,  # maximum number of waypoints
    'visualize_waypoints': True,  # Whether to visualize waypoints (default: True)
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'view_mode' : 'top',  # 'top' for bird's-eye view, 'follow' for third-person view
    'traffic': 'off',  # 'on' for normal traffic lights, 'off' for always green and frozen
    'lidar_max_range': 50.0,  # Maximum LIDAR perception range (meters)
    'max_nearby_vehicles': 5,  # Maximum number of nearby vehicles to observe
}

# Create the environment
env = gym.make('carla-v0', params=params)
obs = env.reset()

# Define a simple action policy
def get_action(env, obs):
    """Randomly choose either a simple manual action or an autopilot action."""
    p = random.random()
    if p < 0.5:
        # Use autopilot (Expert mode)
        env.ego.set_autopilot(True)
        control = env.ego.get_control()
        action = [control.throttle, control.steer, control.brake]
    else:
        # Use random action (Novice mode)
        env.ego.set_autopilot(False)
        throttle = random.uniform(0.0, 1.0)
        steer = random.uniform(-0.6, 0.6)
        brake = random.uniform(0.0, 0.3)
        action = [throttle, steer, brake]
    return action

# Interact with the environment
for episode in range(5):  # Run 5 episodes
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = get_action(env, obs)
        next_obs, reward, cost, done, info = env.step(action)

        print(f"Step: {env.time_step}, Reward: {reward:.2f}, Cost: {cost:.2f}, Done: {done}")

        obs = next_obs
        total_reward += reward

    print(f"Episode {episode} finished. Total reward: {total_reward:.2f}")

env.close()






