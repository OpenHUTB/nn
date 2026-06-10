"""
CartPole Q-Learning训练器

使用Q-Learning算法训练CartPole平衡控制
"""

import argparse
import numpy as np
import gymnasium as gym
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.multi_env import CartPoleTrainer
from rl.visualizer import TrainingVisualizer


def train_cartpole(episodes: int = 10000, alpha: float = 0.1, gamma: float = 0.99,
                  epsilon: float = 1.0, epsilon_decay: float = 0.995,
                  epsilon_min: float = 0.01):
    """训练CartPole"""
    print(f"{'='*60}")
    print(f"CartPole Q-Learning 训练器")
    print(f"{'='*60}")
    print(f"训练轮数: {episodes}")
    print(f"学习率: {alpha}")
    print(f"折扣因子: {gamma}")
    print(f"{'='*60}")
    
    trainer = CartPoleTrainer()
    env = trainer.create_env()
    
    state_bins = (6, 6, 6, 6)
    q_table = np.zeros(state_bins + (env.action_space.n,))
    
    rewards, avg_reward = trainer.train(
        env, q_table, episodes, alpha, gamma, epsilon, epsilon_decay, epsilon_min
    )
    
    print(f"\n训练完成! 最终平均奖励: {avg_reward:.1f}")
    
    mean, std = trainer.evaluate(env, q_table, episodes=100)
    print(f"评估结果: 平均奖励={mean:.1f}±{std:.1f}")
    
    env.close()
    
    TrainingVisualizer.plot_training_curve(
        rewards,
        title="CartPole Q-Learning 训练曲线",
        save_path="plots/cartpole_training_q_learning.png",
        show=False
    )
    
    print("训练曲线已保存到 plots/cartpole_training_q_learning.png")
    
    np.save("cartpole_q_table.npy", q_table)
    print("Q表已保存到 cartpole_q_table.npy")
    
    return q_table, rewards


def test_cartpole(model_path: str = "cartpole_q_table.npy", episodes: int = 10):
    """测试CartPole"""
    print(f"\n测试CartPole ({episodes}轮)...")
    
    trainer = CartPoleTrainer()
    env = trainer.create_env()
    
    q_table = np.load(model_path)
    
    mean, std = trainer.evaluate(env, q_table, episodes=episodes)
    print(f"测试结果: 平均奖励={mean:.1f}±{std:.1f}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='CartPole Q-Learning训练器')
    parser.add_argument('--episodes', type=int, default=10000, help='训练轮数')
    parser.add_argument('--alpha', type=float, default=0.1, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--epsilon', type=float, default=1.0, help='初始探索率')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='探索率衰减')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='最小探索率')
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--model', type=str, default='cartpole_q_table.npy', help='模型路径')
    
    args = parser.parse_args()
    
    if args.test:
        test_cartpole(args.model)
    else:
        train_cartpole(
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min
        )


if __name__ == "__main__":
    main()
