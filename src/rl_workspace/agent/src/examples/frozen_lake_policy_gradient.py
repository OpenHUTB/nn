"""
FrozenLake 策略梯度算法训练器

使用REINFORCE、Actor-Critic、A2C等策略梯度算法训练
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.policy_gradient import REINFORCEAgent, ActorCriticAgent, A2CAgent
from rl.visualizer import TrainingVisualizer, PerformanceAnalyzer
from rl.experiment_manager import ExperimentTracker, ExperimentConfig


def train_policy_gradient(
    env: gym.Env,
    config: argparse.Namespace,
    algorithm: str = 'reinforce'
) -> tuple:
    """
    训练策略梯度算法
    """
    rewards = []
    success_count = 0
    
    if algorithm == 'reinforce':
        agent = REINFORCEAgent(
            state_dim=env.observation_space.n,
            action_dim=env.action_space.n,
            learning_rate=config.alpha,
            gamma=config.gamma
        )
    elif algorithm == 'actor_critic':
        agent = ActorCriticAgent(
            state_dim=env.observation_space.n,
            action_dim=env.action_space.n,
            learning_rate=config.alpha,
            gamma=config.gamma
        )
    elif algorithm == 'a2c':
        agent = A2CAgent(
            state_dim=env.observation_space.n,
            action_dim=env.action_space.n,
            learning_rate=config.alpha,
            gamma=config.gamma
        )
    else:
        raise ValueError(f"未知算法: {algorithm}")
    
    tracker = ExperimentTracker(experiment_dir=f"experiments/{algorithm}")
    tracker.start_experiment(ExperimentConfig(
        algorithm=algorithm,
        env_name="FrozenLake-v1",
        hyperparameters={'alpha': config.alpha, 'gamma': config.gamma, 'epochs': config.epochs}
    ))
    
    print(f"\n使用 {algorithm.upper()} 算法开始训练...")
    print(f"状态空间: {env.observation_space.n}, 动作空间: {env.action_space.n}")
    
    for episode in range(config.epochs):
        state = env.reset()[0]
        done = False
        episode_reward = 0
        
        if algorithm == 'reinforce':
            trajectory = []
            while not done:
                action, log_prob = agent.select_action(np.array([state]))
                next_state, reward, done, trunc, info = env.step(action)
                trajectory.append((state, action, reward))
                episode_reward += reward
                state = next_state
            agent.update(trajectory)
        
        elif algorithm == 'actor_critic':
            action, _ = agent.select_action(np.array([state]))
            while not done:
                next_state, reward, done, trunc, info = env.step(action)
                advantage = agent.compute_advantage(state, reward, next_state, done)
                agent.update(state, action, advantage)
                episode_reward += reward
                state = next_state
                if not done:
                    action, _ = agent.select_action(np.array([state]))
                    
        elif algorithm == 'a2c':
            # 激活 A2C 专用的多步轨迹同步收集流
            ep_states, ep_actions, ep_rewards, ep_dones = [], [], [], []
            while not done:
                action, _ = agent.select_action(np.array([state]))
                next_state, reward, done, trunc, info = env.step(action)
                
                ep_states.append(state)
                ep_actions.append(action)
                ep_rewards.append(reward)
                ep_dones.append(done)
                
                episode_reward += reward
                state = next_state
            
            # 完整 Episode 结束后单次触发批量 GAE 更新
            agent.update(ep_states, ep_actions, ep_rewards, ep_dones)
        
        rewards.append(episode_reward)
        if episode_reward > 0:
            success_count += 1
        
        if episode % config.log_interval == 0 and episode > 0:
            avg_reward = np.mean(rewards[-config.log_interval:]) * 100
            success_rate = np.sum([r > 0 for r in rewards[-config.log_interval:]]) / config.log_interval * 100
            print(f"Episode {episode:6d}/{config.epochs}: 平均奖励={avg_reward:6.2f}%, 成功率={success_rate:5.1f}%")
            
            tracker.log_metrics({
                'reward': episode_reward,
                'success_rate': success_rate,
                'mean_reward': avg_reward
            }, step=episode)
    
    tracker.end_experiment(success=True)
    env.close()
    
    final_success_rate = success_count / config.epochs * 100
    print(f"\n训练完成! 总成功率={final_success_rate:.2f}%")
    
    return agent, rewards


def main():
    parser = argparse.ArgumentParser(description='FrozenLake策略梯度算法训练器')
    parser.add_argument('--epochs', type=int, default=5000, help='训练轮数')
    parser.add_argument('--map-size', type=int, default=4, help='地图大小')
    parser.add_argument('--alpha', type=float, default=0.01, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--log-interval', type=int, default=1000, help='日志输出间隔')
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--render', action='store_true', help='渲染环境')
    parser.add_argument('--algorithm', type=str, 
                       choices=['reinforce', 'actor_critic', 'a2c'],
                       default='reinforce',
                       help='策略梯度算法选择')
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"FrozenLake {args.algorithm.upper()} 训练器")
    print(f"{'='*60}")
    print(f"配置参数:")
    print(f"  训练轮数: {args.epochs}")
    print(f"  地图大小: {args.map_size}x{args.map_size}")
    print(f"  学习率(alpha): {args.alpha}")
    print(f"  折扣因子(gamma): {args.gamma}")
    print(f"  使用算法: {args.algorithm.upper()}")
    print(f"{'='*60}")
    
    env = gym.make(
        "FrozenLake-v1",
        desc=generate_random_map(size=args.map_size),
        is_slippery=False,
        render_mode="ansi",
        max_episode_steps=1000,
    )
    
    if not args.test:
        agent, rewards = train_policy_gradient(env, args, args.algorithm)
        
        TrainingVisualizer.plot_training_curve(
            rewards,
            title=f"FrozenLake {args.algorithm.upper()} 训练曲线",
            save_path=f"plots/frozen_lake_training_{args.algorithm}.png",
            show=False
        )
        
        PerformanceAnalyzer.plot_success_rate(
            rewards,
            window_size=100,
            title=f"FrozenLake {args.algorithm.upper()} 成功率变化",
            save_path=f"plots/frozen_lake_success_rate_{args.algorithm}.png",
            show=False
        )
        
        print("训练可视化已保存到 plots/ 目录")
    
    else:
        print(f"测试模式暂不支持")


if __name__ == "__main__":
    main()