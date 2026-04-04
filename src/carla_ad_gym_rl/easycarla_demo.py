"""
该脚本提供了一个与 EasyCarla-RL 环境交互的最小示例。
它遵循标准的 Gym 接口（reset、step），并演示了环境的基本使用方法。
"""

import gym
import easycarla
import carla
import random
import numpy as np

# 配置环境参数
params = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'dt': 0.1,  # 两帧之间的时间间隔
    'ego_vehicle_filter': 'vehicle.tesla.model3',  # 用于定义自车的车辆过滤器
    'surrounding_vehicle_spawned_randomly': True, # 周围车辆是否随机生成（True）或手动设置（False）
    'port': 2000,  # 连接端口
    'town': 'Town03',  # 要模拟的城市场景
    'max_time_episode': 1000,  # 每个 episode 的最大时间步数
    'max_waypoints': 12,  # 最大路点数量
    'visualize_waypoints': True,  # 是否可视化路点（默认：True）
    'desired_speed': 8,  # 期望速度（米/秒）
    'max_ego_spawn_times': 200,  # 自车生成的最大尝试次数
    'view_mode' : 'top',  # 'top' 表示鸟瞰视角，'follow' 表示第三人称跟随视角
    'traffic': 'off',  # 'on' 表示正常交通灯，'off' 表示始终绿灯并冻结
    'lidar_max_range': 50.0,  # 激光雷达最大感知范围（米）
    'max_nearby_vehicles': 5,  # 可观测的附近车辆最大数量
}

# 创建环境
env = gym.make('carla-v0', params=params)
obs = env.reset()

# 定义一个简单的动作策略
def get_action(env, obs):
    env.ego.set_autopilot(True)
    control = env.ego.get_control()
    return [control.throttle, control.steer, control.brake]

# 与环境交互
for episode in range(5):  # 运行 5 个 episode
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