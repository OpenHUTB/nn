import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print("开始导入模块...")

from config import Config
from environment import RobotNavigationEnv
from dqn_agent import DQNAgent
from visualization import NavigationVisualizer

print("模块导入完成")

config = Config()
config.EPISODES = 100
config.PLOT_INTERVAL = 20

os.makedirs(config.RESULT_DIR, exist_ok=True)

env = RobotNavigationEnv()
agent = DQNAgent(config.STATE_SIZE, config.ACTION_SIZE)
visualizer = NavigationVisualizer()

all_rewards = []
all_distances = []
all_lengths = []

print("开始训练...")
print("=" * 50)

for episode in range(config.EPISODES):
    state = env.reset()
    total_reward = 0
    distance_history = []
    
    visualizer.init_plot()
    visualizer.draw_obstacles(env.get_obstacles())
    visualizer.draw_target(env.get_target_pose()[0], env.get_target_pose()[1])
    visualizer.reset_path()
    
    for step in range(config.MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done, distance = env.step(action)
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        distance_history.append(distance)
        
        if episode % config.PLOT_INTERVAL == 0 and config.VISUALIZE:
            x, y, theta = env.get_robot_pose()
            lidar_data = next_state[:config.LIDAR_ANGLES] * config.LIDAR_RANGE
            visualizer.draw_robot(x, y, theta)
            visualizer.draw_path(x, y)
            visualizer.draw_lidar(x, y, theta, lidar_data)
        
        if done:
            break
    
    agent.replay(config.BATCH_SIZE)
    
    all_rewards.append(total_reward)
    all_distances.append(distance_history[-1] if distance_history else config.MAP_WIDTH)
    all_lengths.append(step + 1)
    
    if episode % 10 == 0:
        avg_reward = np.mean(all_rewards[-10:])
        avg_distance = np.mean(all_distances[-10:])
        print("Episode %4d | Reward: %6.2f | Avg: %6.2f | Dist: %.2f | Steps: %3d" % 
              (episode, total_reward, avg_reward, distance_history[-1], step + 1))
    
    if episode % config.PLOT_INTERVAL == 0 and config.VISUALIZE:
        visualizer.save_figure('navigation_episode_%d.png' % episode)
        print("  -> 保存导航截图")

print("=" * 50)
print("训练完成！")

agent.save_model()
print("模型已保存")

visualizer.plot_training_history(all_rewards, all_distances, all_lengths)
print("训练历史图已保存")

print("所有结果已保存到", config.RESULT_DIR)