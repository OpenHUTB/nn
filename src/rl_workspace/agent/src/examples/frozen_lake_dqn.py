"""
FrozenLake DQN 训练器

使用深度Q网络(DQN)训练智能体在冰冻湖环境中找到最优路径
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.deep_learning import DQNLearner, DDQNLearner
from rl.config import get_config_parser, FrozenLakeConfig
from rl.visualizer import TrainingVisualizer, QTableVisualizer, PerformanceAnalyzer
from rl.advanced_trainer import (
    ProgressBar, EarlyStopping, MetricsTracker, 
    CheckpointManager, TrainingMonitor
)


def train_with_monitor(
    env: gym.Env,
    config: FrozenLakeConfig,
    algorithm: str = 'dqn'
) -> tuple:
    """
    带监控的高级训练
    
    参数:
        env: Gymnasium环境
        config: 训练配置
        algorithm: 算法选择 ('dqn' 或 'ddqn')
    
    返回:
        (训练好的模型, 奖励记录)
    """
    rewards = []
    success_count = 0
    
    if algorithm == 'dqn':
        learner = DQNLearner(
            state_dim=env.observation_space.n,
            action_dim=env.action_space.n,
            gamma=config.gamma,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            buffer_capacity=10000,
            batch_size=32
        )
    else:
        learner = DDQNLearner(
            state_dim=env.observation_space.n,
            action_dim=env.action_space.n,
            gamma=config.gamma,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            buffer_capacity=10000,
            batch_size=32
        )
    
    progress_bar = ProgressBar(config.epochs, prefix=f"{algorithm.upper()}训练")
    early_stopping = EarlyStopping(patience=200, mode='max')
    metrics_tracker = MetricsTracker(window_size=100)
    checkpoint_manager = CheckpointManager(save_dir="checkpoints", max_keep=3)
    monitor = TrainingMonitor()
    
    monitor.start()
    
    print(f"\n使用 {algorithm.upper()} 算法开始训练...")
    print(f"状态空间: {env.observation_space.n}, 动作空间: {env.action_space.n}")
    print(f"探索率从 1.0 衰减到 0.01")
    
    for episode in range(config.epochs):
        state = env.reset()[0]
        done = False
        episode_reward = 0
        
        while not done:
            action = learner.get_action(np.array([state]), training=True)
            next_state, reward, done, trunc, info = env.step(action)
            
            learner.store_transition(state, action, reward, next_state, done)
            learner.train_step()
            
            state = next_state
            episode_reward += reward
        
        rewards.append(episode_reward)
        if episode_reward > 0:
            success_count += 1
        
        metrics_tracker.update({
            'reward': episode_reward,
            'epsilon': learner.epsilon,
            'success': float(episode_reward > 0)
        })
        
        if episode % 10 == 0:
            mean_reward = metrics_tracker.get_mean('reward')
            success_rate = metrics_tracker.get_mean('success') * 100
            
            monitor.log(episode, {
                'mean_reward': mean_reward,
                'success_rate': success_rate,
                'epsilon': learner.epsilon
            })
            
            metrics = {
                '奖励': mean_reward,
                '成功率': success_rate,
                'ε': learner.epsilon
            }
            progress_bar.update(episode + 1, metrics)
            
            if early_stopping(success_rate, episode):
                print(f"\n早停触发! 在第 {episode} 轮停止训练")
                break
            
            if episode % config.log_interval == 0:
                checkpoint_manager.save(
                    learner.q_network,
                    episode,
                    {'success_rate': success_rate, 'mean_reward': mean_reward}
                )
    
    progress_bar.close()
    monitor.print_summary()
    
    final_success_rate = success_count / config.epochs * 100
    print(f"\n训练完成! 总成功率={final_success_rate:.2f}%")
    
    env.close()
    
    return learner, rewards


def test_agent(
    env: gym.Env,
    learner,
    episodes: int = 10,
    render: bool = True
):
    """测试训练好的智能体"""
    print(f"\n测试智能体 ({episodes} 轮)...")
    
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        episode_reward = 0
        
        while not done:
            if render:
                print(env.render())
            
            action = learner.get_action(np.array([state]), training=False)
            state, reward, done, trunc, info = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        
        if render:
            print(f"Episode {episode + 1}: 奖励 = {episode_reward}")
    
    env.close()
    
    mean_reward = np.mean(total_rewards)
    success_rate = sum(1 for r in total_rewards if r > 0) / episodes * 100
    
    print(f"\n测试结果:")
    print(f"  平均奖励: {mean_reward:.2f}")
    print(f"  成功率: {success_rate:.1f}%")


def main():
    parser = get_config_parser('FrozenLake')
    parser.add_argument('--algorithm', type=str, default='dqn', 
                       choices=['q_learning', 'sarsa', 'dqn', 'ddqn'],
                       help='算法选择')
    parser.add_argument('--no-early-stop', action='store_true', help='禁用早停')
    parser.add_argument('--checkpoint-interval', type=int, default=1000, 
                       help='检查点保存间隔')
    
    args = parser.parse_args()
    
    config = FrozenLakeConfig(vars(args))
    
    print(f"{'='*60}")
    print(f"FrozenLake {config.algorithm.upper()} 训练器 (高级版)")
    print(f"{'='*60}")
    print(f"配置参数:")
    print(f"  训练轮数: {config.epochs}")
    print(f"  地图大小: {config.map_size}x{config.map_size}")
    print(f"  学习率(alpha): {config.alpha}")
    print(f"  折扣因子(gamma): {config.gamma}")
    print(f"  使用算法: {config.algorithm.upper()}")
    print(f"  早停: {'禁用' if config.test else ('启用' if not args.no_early_stop else '禁用')}")
    print(f"{'='*60}")
    
    env = gym.make(
        "FrozenLake-v1",
        desc=generate_random_map(size=config.map_size),
        is_slippery=False,
        render_mode="ansi",
        max_episode_steps=1000,
    )
    
    if not config.test:
        learner, rewards = train_with_monitor(env, config, config.algorithm)
        
        model_path = f"{config.algorithm}_model.npy"
        learner.save_model(model_path)
        print(f"\n模型已保存到: {model_path}")
        
        TrainingVisualizer.plot_training_curve(
            rewards,
            title=f"FrozenLake {config.algorithm.upper()} 训练曲线",
            save_path=f"plots/frozen_lake_training_{config.algorithm}.png",
            show=False
        )
        
        PerformanceAnalyzer.plot_success_rate(
            rewards,
            window_size=100,
            title=f"FrozenLake {config.algorithm.upper()} 成功率变化",
            save_path=f"plots/frozen_lake_success_rate_{config.algorithm}.png",
            show=False
        )
        
        print("训练可视化已保存到 plots/ 目录")
    
    else:
        model_path = f"{config.algorithm}_model.npy"
        if os.path.exists(model_path):
            if config.algorithm in ['dqn', 'ddqn']:
                learner = DDQNLearner if config.algorithm == 'ddqn' else DQNLearner
                temp_learner = learner(
                    state_dim=env.observation_space.n,
                    action_dim=env.action_space.n
                )
                temp_learner.load_model(model_path)
                test_agent(env, temp_learner, render=config.render)
            else:
                print(f"测试模式暂不支持 {config.algorithm} 算法")
        else:
            print(f"错误: 模型文件不存在 - {model_path}")


if __name__ == "__main__":
    main()
