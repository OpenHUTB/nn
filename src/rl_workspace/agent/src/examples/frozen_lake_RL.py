import argparse
import os
import time
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

import QLearner as ql


def try_environment(env: gym.Env, steps: int = 15) -> None:
    """测试环境的随机动作"""
    for step in range(steps):
        print(env.render())
        action = env.action_space.sample()
        observation, reward, done, trunc, info = env.step(action)
        time.sleep(0.2)
        os.system("cls" if os.name == "nt" else "clear")
        if done:
            env.reset()

    env.close()


def train(
    env: gym.Env,
    epochs: int = 10000,
    alpha: float = 0.8,
    gamma: float = 0.9,
    radr: float = 0.001,
    log_interval: int = 1000,
    verbose: bool = False
) -> np.ndarray:
    """
    训练Q-Learning代理
    
    参数:
        env: Gymnasium环境
        epochs: 训练轮数
        alpha: 学习率
        gamma: 折扣因子
        radr: 探索率衰减率
        log_interval: 日志输出间隔
        verbose: 是否输出详细日志
    
    返回:
        训练好的Q表
    """
    rewards = []
    success_count = 0
    
    learner = ql.QLearner(
        states=env.observation_space.n,
        actions=env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        radr=radr
    )
    
    for episode in range(epochs):
        if verbose:
            print(f"{'='*50}\n运行 episode: {episode + 1}/{epochs}\n{'='*50}")
        
        state = env.reset()[0]
        done = False
        total_rewards = 0
        
        action = learner.get_next_action_without_Q_table_update(state)
        
        while not done:
            new_state, reward, done, trunc, info = env.step(action)
            action = learner.get_next_action_with_Q_table_update(new_state, reward)
            total_rewards += reward
        
        learner.decay_rar(episode)
        rewards.append(total_rewards)
        
        if total_rewards > 0:
            success_count += 1
        
        if episode % log_interval == 0 and episode > 0:
            avg_reward = np.mean(rewards[-log_interval:]) * 100
            success_rate = np.sum([r > 0 for r in rewards[-log_interval:]]) / log_interval * 100
            print(f"Episode {episode:6d}/{epochs}: 平均奖励={avg_reward:6.2f}%, 成功率={success_rate:5.1f}%, 探索率={learner.rar:.4f}")
    
    env.close()
    
    final_success_rate = success_count / epochs * 100
    print(f"\n训练完成! 总成功率={final_success_rate:.2f}%")
    
    return learner.Q


def check_performance(env: gym.Env, q_table: np.ndarray, render: bool = True) -> float:
    """
    测试训练好的Q表性能
    
    参数:
        env: Gymnasium环境
        q_table: 训练好的Q表
        render: 是否渲染环境
    
    返回:
        获得的总奖励
    """
    state = env.reset()[0]
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 100:
        if render:
            print(env.render())
            time.sleep(0.3)
            os.system("cls" if os.name == "nt" else "clear")
        
        action = int(np.argmax(q_table[state, :]))
        state, reward, done, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
    
    env.close()
    print(f"\n测试完成! 获得奖励: {total_reward}")
    return total_reward


def plot_training_curve(rewards: list, save_path: Optional[str] = None) -> None:
    """绘制训练曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='每轮奖励', alpha=0.5)
    
    window_size = 100
    running_mean = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(rewards)), running_mean, 
             label=f'{window_size}轮滑动平均', color='red')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Q-Learning训练曲线')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"训练曲线已保存到: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='FrozenLake Q-Learning训练器')
    
    parser.add_argument('--epochs', type=int, default=10000, help='训练轮数')
    parser.add_argument('--map-size', type=int, default=4, help='地图大小')
    parser.add_argument('--alpha', type=float, default=0.8, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.9, help='折扣因子')
    parser.add_argument('--radr', type=float, default=0.001, help='探索率衰减率')
    parser.add_argument('--log-interval', type=int, default=1000, help='日志输出间隔')
    parser.add_argument('--test', action='store_true', help='测试已训练的模型')
    parser.add_argument('--render', action='store_true', help='渲染测试过程')
    parser.add_argument('--plot', action='store_true', help='绘制训练曲线')
    parser.add_argument('--save-path', type=str, default='frozen_lake_q_table.npy', 
                        help='Q表保存路径')
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"FrozenLake Q-Learning 训练器")
    print(f"{'='*60}")
    print(f"配置参数:")
    print(f"  训练轮数: {args.epochs}")
    print(f"  地图大小: {args.map_size}x{args.map_size}")
    print(f"  学习率(alpha): {args.alpha}")
    print(f"  折扣因子(gamma): {args.gamma}")
    print(f"  探索率衰减率(radr): {args.radr}")
    print(f"{'='*60}")
    
    env = gym.make(
        "FrozenLake-v1",
        desc=generate_random_map(size=args.map_size),
        is_slippery=False,
        render_mode="ansi" if args.render else "ansi",
        max_episode_steps=1000,
    )
    
    if not args.test:
        print(f"\n开始训练...")
        q_table = train(
            env=env,
            epochs=args.epochs,
            alpha=args.alpha,
            gamma=args.gamma,
            radr=args.radr,
            log_interval=args.log_interval
        )
        
        np.save(args.save_path, q_table)
        print(f"\nQ表已保存到: {args.save_path}")
    
    else:
        if os.path.exists(args.save_path):
            q_table = np.load(args.save_path)
            print(f"已加载Q表: {args.save_path}")
            check_performance(env, q_table, render=args.render)
        else:
            print(f"错误: Q表文件不存在 - {args.save_path}")


if __name__ == "__main__":
    main()
