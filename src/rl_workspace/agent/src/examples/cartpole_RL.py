import argparse
import os
import time
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import QLearner as ql


def create_bins(num_bins_per_obs: int = 10) -> np.ndarray:
    """创建状态空间的离散化区间"""
    bins_cart_position = np.linspace(-4.8, 4.8, num_bins_per_obs)
    bins_cart_velocity = np.linspace(-5, 5, num_bins_per_obs)
    bins_pole_angle = np.linspace(-0.418, 0.418, num_bins_per_obs)
    bins_pole_angular_velocity = np.linspace(-5, 5, num_bins_per_obs)

    bins = np.array([
        bins_cart_position,
        bins_cart_velocity,
        bins_pole_angle,
        bins_pole_angular_velocity,
    ])

    return bins


def custom_reward(done: bool, points: int, reward: float) -> float:
    """自定义奖励函数：如果提前结束则给予惩罚"""
    if done and points < 500:
        return -600
    return reward


def discretize_observation(observations: np.ndarray, bins: np.ndarray) -> tuple:
    """将连续状态离散化"""
    binned_observations = []
    for i, observation in enumerate(observations):
        discretized_observation = np.digitize(observation, bins[i])
        discretized_observation = min(bins.shape[1] - 1, discretized_observation)
        binned_observations.append(discretized_observation)
    return tuple(binned_observations)


def train(
    env: gym.Env,
    bins: np.ndarray,
    num_bins: int,
    epochs: int = 50000,
    alpha: float = 0.1,
    gamma: float = 0.995,
    log_interval: int = 1000,
    verbose: bool = False
) -> np.ndarray:
    """
    训练CartPole Q-Learning代理
    
    参数:
        env: Gymnasium环境
        bins: 离散化区间
        num_bins: 每个维度的区间数
        epochs: 训练轮数
        alpha: 学习率
        gamma: 折扣因子
        log_interval: 日志输出间隔
        verbose: 是否输出详细日志
    
    返回:
        训练好的Q表
    """
    points_log = []
    mean_points_log = []
    
    learner = ql.QLearner(
        states=(num_bins, num_bins, num_bins, num_bins),
        actions=env.action_space.n,
        alpha=alpha,
        gamma=gamma,
    )
    
    for epoch in range(epochs):
        if verbose and epoch % 1000 == 0:
            print(f"{'='*50}\n运行 episode: {epoch + 1}/{epochs}\n{'='*50}")
        
        initial_state = env.reset()[0]
        discretized_state = discretize_observation(initial_state, bins)
        done = False
        points = 0
        
        action = learner.get_next_action_without_Q_table_update(discretized_state)
        
        while not done:
            new_state, reward, done, trunc, info = env.step(action)
            discretized_state = discretize_observation(new_state, bins)
            reward = custom_reward(done, points, reward)
            action = learner.get_next_action_with_Q_table_update(discretized_state, reward)
            points += 1
        
        learner.decay_rar(epoch)
        points_log.append(points)
        
        running_mean = round(np.mean(points_log[-30:]), 2)
        mean_points_log.append(running_mean)
        
        if epoch % log_interval == 0 and epoch > 0:
            print(f"Episode {epoch:6d}/{epochs}: 平均得分={running_mean:6.2f}")
    
    env.close()
    return learner.Q


def check_performance(
    env: gym.Env,
    bins: np.ndarray,
    q_table: np.ndarray,
    use_q_table: bool = True,
    render: bool = True
) -> int:
    """
    测试训练好的Q表性能
    
    参数:
        env: Gymnasium环境
        bins: 离散化区间
        q_table: 训练好的Q表
        use_q_table: 是否使用Q表
        render: 是否渲染环境
    
    返回:
        获得的总分数
    """
    total_reward = 0
    done = False
    state = env.reset()[0]
    
    for steps in range(500):
        if render:
            env.render()
        
        if use_q_table:
            discrete_state = discretize_observation(state, bins)
            action = int(np.argmax(q_table[discrete_state]))
        else:
            action = env.action_space.sample()
        
        state, reward, done, trunc, info = env.step(action)
        total_reward += 1
        
        if done:
            break
    
    env.close()
    print(f"\n测试完成! 获得分数: {total_reward}")
    return total_reward


def plot_training_curve(
    points_log: list,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """绘制训练曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(points_log, label='每轮得分', alpha=0.5)
    
    window_size = 30
    running_mean = np.convolve(points_log, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(points_log)), running_mean, 
             label=f'{window_size}轮滑动平均', color='red')
    
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('CartPole Q-Learning训练曲线')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"训练曲线已保存到: {save_path}")
    
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='CartPole Q-Learning训练器')
    
    parser.add_argument('--epochs', type=int, default=50000, help='训练轮数')
    parser.add_argument('--num-bins', type=int, default=50, help='每个状态维度的区间数')
    parser.add_argument('--alpha', type=float, default=0.1, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.995, help='折扣因子')
    parser.add_argument('--log-interval', type=int, default=1000, help='日志输出间隔')
    parser.add_argument('--training', action='store_true', help='进行训练')
    parser.add_argument('--testing', action='store_true', help='进行测试')
    parser.add_argument('--render', action='store_true', help='渲染环境')
    parser.add_argument('--save-path', type=str, default='cartpole_q_table.npy', 
                        help='Q表保存路径')
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"CartPole Q-Learning 训练器")
    print(f"{'='*60}")
    print(f"配置参数:")
    print(f"  训练轮数: {args.epochs}")
    print(f"  状态区间数: {args.num_bins}")
    print(f"  学习率(alpha): {args.alpha}")
    print(f"  折扣因子(gamma): {args.gamma}")
    print(f"{'='*60}")
    
    BINS = create_bins(args.num_bins)
    
    if args.training:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        print("\n开始训练...")
        q_table = train(
            env=env,
            bins=BINS,
            num_bins=args.num_bins,
            epochs=args.epochs,
            alpha=args.alpha,
            gamma=args.gamma,
            log_interval=args.log_interval
        )
        
        np.save(args.save_path, q_table)
        print(f"\nQ表已保存到: {args.save_path}")
    
    if args.testing:
        render_mode = "human" if args.render else "rgb_array"
        env = gym.make("CartPole-v1", render_mode=render_mode)
        
        if os.path.exists(args.save_path):
            q_table = np.load(args.save_path)
            print(f"已加载Q表: {args.save_path}")
            check_performance(env, BINS, q_table, use_q_table=True, render=args.render)
        else:
            print(f"错误: Q表文件不存在 - {args.save_path}")


if __name__ == "__main__":
    main()
