"""
FrozenLake Q-Learning 训练器

使用 Q-Learning 算法训练 FrozenLake 环境
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.algorithms import QLearner
from rl.visualizer import TrainingVisualizer, PerformanceAnalyzer


def train(epochs=10000, map_size=4, alpha=0.8, gamma=0.9, test=False, model_path='frozen_lake_q_table.npy'):
    """训练或测试 FrozenLake 环境"""
    
    print(f"{'='*60}")
    print(f"FrozenLake Q-Learning 训练器")
    print(f"{'='*60}")
    print(f"训练轮数: {epochs}")
    print(f"地图大小: {map_size}x{map_size}")
    print(f"学习率: {alpha}")
    print(f"折扣因子: {gamma}")
    print(f"{'='*60}")
    
    # 创建环境
    env = gym.make(
        "FrozenLake-v1",
        desc=generate_random_map(size=map_size),
        is_slippery=False,
        render_mode="ansi"
    )
    
    if test:
        # 测试模式
        q_table = np.load(model_path)
        print(f"加载模型: {model_path}")
        test_env(env, q_table, episodes=100)
    else:
        # 训练模式
        learner = QLearner(
            states=env.observation_space.n,
            actions=env.action_space.n,
            alpha=alpha,
            gamma=gamma
        )
        
        rewards = []
        for episode in range(epochs):
            state = env.reset()[0]
            done = False
            episode_reward = 0
            
            # 初始动作
            action = learner.get_next_action_without_Q_table_update(state)
            
            while not done:
                next_state, reward, done, trunc, info = env.step(action)
                episode_reward += reward
                
                # 更新 Q 表
                action = learner.get_next_action_with_Q_table_update(next_state, reward)
            
            # 衰减探索率
            learner.decay_rar(episode)
            rewards.append(episode_reward)
            
            if episode % 1000 == 0 and episode > 0:
                avg_reward = np.mean(rewards[-1000:])
                success_rate = np.sum([r > 0 for r in rewards[-1000:]]) / 1000 * 100
                print(f"Episode {episode}: 平均奖励={avg_reward:.2f}, 成功率={success_rate:.1f}%")
        
        final_reward = np.mean(rewards[-1000:])
        print(f"\n训练完成! 最终平均奖励: {final_reward:.2f}")
        
        # 保存模型
        np.save(model_path, learner.Q)
        print(f"Q 表已保存: {model_path}")
        
        # 可视化
        TrainingVisualizer.plot_training_curve(
            rewards,
            title="FrozenLake Q-Learning 训练曲线",
            save_path="plots/frozen_lake_training_q_learning.png",
            show=False
        )
        
        PerformanceAnalyzer.plot_success_rate(
            rewards,
            window_size=100,
            title="FrozenLake Q-Learning 成功率变化",
            save_path="plots/frozen_lake_policy_q_learning.png",
            show=False
        )
        
        print("可视化已保存到 plots/ 目录")
    
    env.close()


def test_env(env, q_table, episodes=100):
    """测试训练好的智能体"""
    total_rewards = []
    success_count = 0
    
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        episode_reward = 0
        
        while not done:
            # 选择最优动作
            action = np.argmax(q_table[state])
            next_state, reward, done, trunc, info = env.step(action)
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        if episode_reward > 0:
            success_count += 1
    
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / episodes * 100
    print(f"\n测试结果 ({episodes} 轮):")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"成功率: {success_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='FrozenLake Q-Learning 训练器')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='训练轮数')
    parser.add_argument('--map-size', type=int, default=4,
                        help='地图大小')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='学习率')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='折扣因子')
    parser.add_argument('--test', action='store_true',
                        help='测试模式')
    parser.add_argument('--model', type=str, default='frozen_lake_q_table.npy',
                        help='模型路径')
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        map_size=args.map_size,
        alpha=args.alpha,
        gamma=args.gamma,
        test=args.test,
        model_path=args.model
    )


if __name__ == "__main__":
    main()
