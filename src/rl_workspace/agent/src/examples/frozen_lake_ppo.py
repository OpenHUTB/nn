"""
FrozenLake PPO算法训练器

使用Proximal Policy Optimization算法训练
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.simple_ppo import create_ppo_agent
from rl.env_tools import EpisodeStatistics
from rl.visualizer import TrainingVisualizer, PerformanceAnalyzer


def train_ppo(env: gym.Env, epochs: int = 3000, alpha: float = 0.01, 
              gamma: float = 0.99, log_interval: int = 500) -> tuple:
    """训练PPO算法"""
    print(f"\n使用 PPO 算法开始训练...")
    print(f"状态空间: {env.observation_space.n}, 动作空间: {env.action_space.n}")
    
    agent = create_ppo_agent(
        state_dim=env.observation_space.n,
        action_dim=env.action_space.n,
        learning_rate=alpha,
        gamma=gamma
    )
    
    stats = EpisodeStatistics()
    rewards = []
    
    for episode in range(epochs):
        state = env.reset()[0]
        done = False
        episode_reward = 0
        states_list, actions_list, rewards_list, dones_list = [], [], [], []
        
        while not done:
            action, _ = agent.select_action(state, training=True)
            next_state, reward, done, trunc, info = env.step(action)
            
            states_list.append(state)
            actions_list.append(action)
            rewards_list.append(reward)
            dones_list.append(done)
            episode_reward += reward
            state = next_state
        
        agent.update(states_list, actions_list, rewards_list, dones_list)
        
        rewards.append(episode_reward)
        stats.update(episode_reward)
        
        if episode % log_interval == 0 and episode > 0:
            mean_reward = stats.get_mean_reward()
            print(f"Episode {episode:6d}/{epochs}: 平均奖励={mean_reward:6.2f}")
    
    env.close()
    
    final_reward = np.mean(rewards[-500:])
    print(f"\n训练完成! 最终平均奖励={final_reward:.2f}")
    
    return agent, rewards


def main():
    parser = argparse.ArgumentParser(description='FrozenLake PPO训练器')
    parser.add_argument('--epochs', type=int, default=3000, help='训练轮数')
    parser.add_argument('--map-size', type=int, default=4, help='地图大小')
    parser.add_argument('--alpha', type=float, default=0.01, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--log-interval', type=int, default=500, help='日志间隔')
    parser.add_argument('--test', action='store_true', help='测试模式')
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"FrozenLake PPO 训练器")
    print(f"{'='*60}")
    print(f"训练轮数: {args.epochs}")
    print(f"地图大小: {args.map_size}x{args.map_size}")
    print(f"学习率: {args.alpha}")
    print(f"{'='*60}")
    
    env = gym.make(
        "FrozenLake-v1",
        desc=generate_random_map(size=args.map_size),
        is_slippery=False,
        render_mode="ansi"
    )
    
    agent, rewards = train_ppo(
        env,
        epochs=args.epochs,
        alpha=args.alpha,
        gamma=args.gamma,
        log_interval=args.log_interval
    )
    
    TrainingVisualizer.plot_training_curve(
        rewards,
        title="FrozenLake PPO 训练曲线",
        save_path="plots/frozen_lake_training_ppo.png",
        show=False
    )
    
    PerformanceAnalyzer.plot_success_rate(
        rewards,
        window_size=100,
        title="FrozenLake PPO 成功率变化",
        save_path="plots/frozen_lake_success_rate_ppo.png",
        show=False
    )
    
    print("训练可视化已保存到 plots/ 目录")


if __name__ == "__main__":
    main()
