"""
完整算法对比工具

对比Q-Learning、SARSA、DQN、DDQN、REINFORCE、Actor-Critic、A2C等所有算法
"""

import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.algorithms import QLearner, SARSALearner
from rl.deep_learning import DQNLearner, DDQNLearner
from rl.policy_gradient import REINFORCEAgent, ActorCriticAgent, A2CAgent
from rl.visualizer import TrainingVisualizer


class ComprehensiveBenchmark:
    """综合算法基准测试"""
    
    def __init__(self, map_size: int = 4):
        self.map_size = map_size
        self.results = {}
    
    def _create_env(self):
        """创建环境"""
        return gym.make(
            "FrozenLake-v1",
            desc=generate_random_map(size=self.map_size),
            is_slippery=False,
            render_mode="ansi"
        )
    
    def benchmark_q_learning(self, epochs: int = 3000) -> dict:
        """Q-Learning基准测试"""
        print(f"\n{'='*50}")
        print("测试 Q-Learning 算法")
        print(f"{'='*50}")
        
        env = self._create_env()
        learner = QLearner(
            states=env.observation_space.n,
            actions=env.action_space.n,
            alpha=0.8, gamma=0.9, radr=0.001
        )
        
        rewards = []
        start_time = time.time()
        
        for episode in range(epochs):
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
            
            if episode % 1000 == 0 and episode > 0:
                success_rate = np.mean(rewards[-1000:]) * 100
                print(f"Episode {episode:5d}: 成功率={success_rate:.1f}%")
        
        env.close()
        training_time = time.time() - start_time
        
        return {
            'algorithm': 'Q-Learning',
            'rewards': rewards,
            'final_success_rate': np.mean(rewards[-1000:]) * 100,
            'training_time': training_time
        }
    
    def benchmark_sarsa(self, epochs: int = 3000) -> dict:
        """SARSA基准测试"""
        print(f"\n{'='*50}")
        print("测试 SARSA 算法")
        print(f"{'='*50}")
        
        env = self._create_env()
        learner = SARSALearner(
            states=env.observation_space.n,
            actions=env.action_space.n,
            alpha=0.8, gamma=0.9, radr=0.001
        )
        
        rewards = []
        start_time = time.time()
        
        for episode in range(epochs):
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
            
            if episode % 1000 == 0 and episode > 0:
                success_rate = np.mean(rewards[-1000:]) * 100
                print(f"Episode {episode:5d}: 成功率={success_rate:.1f}%")
        
        env.close()
        training_time = time.time() - start_time
        
        return {
            'algorithm': 'SARSA',
            'rewards': rewards,
            'final_success_rate': np.mean(rewards[-1000:]) * 100,
            'training_time': training_time
        }
    
    def benchmark_dqn(self, epochs: int = 3000) -> dict:
        """DQN基准测试"""
        print(f"\n{'='*50}")
        print("测试 DQN 算法")
        print(f"{'='*50}")
        
        env = self._create_env()
        learner = DQNLearner(
            state_dim=env.observation_space.n,
            action_dim=env.action_space.n,
            gamma=0.9, epsilon_decay=0.995
        )
        
        rewards = []
        start_time = time.time()
        
        for episode in range(epochs):
            state = env.reset()[0]
            done = False
            total_rewards = 0
            
            while not done:
                action = learner.get_action(np.array([state]), training=True)
                new_state, reward, done, trunc, info = env.step(action)
                learner.store_transition(state, action, reward, new_state, done)
                learner.train_step()
                state = new_state
                total_rewards += reward
            
            rewards.append(total_rewards)
            
            if episode % 1000 == 0 and episode > 0:
                success_rate = np.mean(rewards[-1000:]) * 100
                print(f"Episode {episode:5d}: 成功率={success_rate:.1f}%")
        
        env.close()
        training_time = time.time() - start_time
        
        return {
            'algorithm': 'DQN',
            'rewards': rewards,
            'final_success_rate': np.mean(rewards[-1000:]) * 100,
            'training_time': training_time
        }
    
    def benchmark_reinforce(self, epochs: int = 3000) -> dict:
        """REINFORCE基准测试"""
        print(f"\n{'='*50}")
        print("测试 REINFORCE 算法")
        print(f"{'='*50}")
        
        env = self._create_env()
        agent = REINFORCEAgent(
            state_dim=env.observation_space.n,
            action_dim=env.action_space.n,
            learning_rate=0.01, gamma=0.99
        )
        
        rewards = []
        start_time = time.time()
        
        for episode in range(epochs):
            state = env.reset()[0]
            done = False
            total_rewards = 0
            trajectory = []
            
            while not done:
                action, _ = agent.select_action(np.array([state]))
                new_state, reward, done, trunc, info = env.step(action)
                trajectory.append((state, action, reward))
                state = new_state
                total_rewards += reward
            
            agent.update(trajectory)
            rewards.append(total_rewards)
            
            if episode % 1000 == 0 and episode > 0:
                success_rate = np.mean(rewards[-1000:]) * 100
                print(f"Episode {episode:5d}: 成功率={success_rate:.1f}%")
        
        env.close()
        training_time = time.time() - start_time
        
        return {
            'algorithm': 'REINFORCE',
            'rewards': rewards,
            'final_success_rate': np.mean(rewards[-1000:]) * 100,
            'training_time': training_time
        }
    
    def benchmark_actor_critic(self, epochs: int = 3000) -> dict:
        """Actor-Critic基准测试"""
        print(f"\n{'='*50}")
        print("测试 Actor-Critic 算法")
        print(f"{'='*50}")
        
        env = self._create_env()
        agent = ActorCriticAgent(
            state_dim=env.observation_space.n,
            action_dim=env.action_space.n,
            learning_rate=0.001, gamma=0.99
        )
        
        rewards = []
        start_time = time.time()
        
        for episode in range(epochs):
            state = env.reset()[0]
            done = False
            total_rewards = 0
            
            action, _ = agent.select_action(np.array([state]))
            while not done:
                new_state, reward, done, trunc, info = env.step(action)
                advantage = agent.compute_advantage(state, reward, next_state, done)
                agent.update(state, action, advantage)
                state = new_state
                total_rewards += reward
                
                if not done:
                    action, _ = agent.select_action(np.array([state]))
            
            rewards.append(total_rewards)
            
            if episode % 1000 == 0 and episode > 0:
                success_rate = np.mean(rewards[-1000:]) * 100
                print(f"Episode {episode:5d}: 成功率={success_rate:.1f}%")
        
        env.close()
        training_time = time.time() - start_time
        
        return {
            'algorithm': 'Actor-Critic',
            'rewards': rewards,
            'final_success_rate': np.mean(rewards[-1000:]) * 100,
            'training_time': training_time
        }
    
    def benchmark_a2c(self, epochs: int = 3000) -> dict:
        """A2C基准测试（全新集成，支持批量GAE更新）"""
        print(f"\n{'='*50}")
        print("测试 A2C 算法")
        print(f"{'='*50}")
        
        env = self._create_env()
        agent = A2CAgent(
            state_dim=env.observation_space.n,
            action_dim=env.action_space.n,
            learning_rate=0.001, gamma=0.99
        )
        
        rewards = []
        start_time = time.time()
        
        for episode in range(epochs):
            state = env.reset()[0]
            done = False
            total_rewards = 0
            
            ep_states, ep_actions, ep_rewards, ep_dones = [], [], [], []
            while not done:
                action, _ = agent.select_action(np.array([state]))
                new_state, reward, done, trunc, info = env.step(action)
                
                ep_states.append(state)
                ep_actions.append(action)
                ep_rewards.append(reward)
                ep_dones.append(done)
                
                state = new_state
                total_rewards += reward
                
            agent.update(ep_states, ep_actions, ep_rewards, ep_dones)
            rewards.append(total_rewards)
            
            if episode % 1000 == 0 and episode > 0:
                success_rate = np.mean(rewards[-1000:]) * 100
                print(f"Episode {episode:5d}: 成功率={success_rate:.1f}%")
        
        env.close()
        training_time = time.time() - start_time
        
        return {
            'algorithm': 'A2C',
            'rewards': rewards,
            'final_success_rate': np.mean(rewards[-1000:]) * 100,
            'training_time': training_time
        }
    
    def run_all(self, epochs: int = 3000, algorithms: list = None):
        """运行所有算法基准测试"""
        if algorithms is None:
            # 默认启动列表扩增 a2c
            algorithms = ['q_learning', 'sarsa', 'dqn', 'reinforce', 'actor_critic', 'a2c']
        
        algorithm_map = {
            'q_learning': self.benchmark_q_learning,
            'sarsa': self.benchmark_sarsa,
            'dqn': self.benchmark_dqn,
            'reinforce': self.benchmark_reinforce,
            'actor_critic': self.benchmark_actor_critic,
            'a2c': self.benchmark_a2c
        }
        
        print(f"\n{'='*60}")
        print("开始综合算法基准测试")
        print(f"{'='*60}")
        print(f"环境: FrozenLake-v1")
        print(f"地图大小: {self.map_size}x{self.map_size}")
        print(f"训练轮数: {epochs}")
        print(f"测试算法: {len(algorithms)}个")
        
        for algo in algorithms:
            if algo in algorithm_map:
                result = algorithm_map[algo](epochs)
                self.results[result['algorithm']] = result
        
        self.print_summary()
        self.plot_comparison()
        
        return self.results
    
    def print_summary(self):
        """打印测试摘要"""
        print(f"\n{'='*60}")
        print("算法性能对比摘要")
        print(f"{'='*60}")
        
        print(f"\n{'算法':<20} {'最终成功率':<15} {'训练时间':<15}")
        print("-" * 60)
        
        for algo_name, result in self.results.items():
            print(f"{algo_name:<20} {result['final_success_rate']:>10.2f}%     {result['training_time']:>10.2f}s")
        
        best_algo = max(self.results.items(), key=lambda x: x[1]['final_success_rate'])
        print(f"\n🏆 最佳算法: {best_algo[0]} (成功率: {best_algo[1]['final_success_rate']:.2f}%)")
    
    def plot_comparison(self):
        """绘制算法对比图"""
        curves = {}
        
        for algo_name, result in self.results.items():
            window_size = 100
            if len(result['rewards']) >= window_size:
                smoothed = np.convolve(
                    result['rewards'],
                    np.ones(window_size)/window_size,
                    mode='valid'
                )
                curves[f"{algo_name}"] = list(smoothed)
        
        TrainingVisualizer.plot_multiple_curves(
            curves,
            title="Reinforcement Learning Algorithms Comparison",
            xlabel="Episode",
            ylabel="Success Rate (%)",
            window_size=100,
            save_path="plots/algorithm_comparison.png",
            show=False
        )
        
        print("\n对比图已保存到: plots/algorithm_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='强化学习综合算法基准测试')
    parser.add_argument('--epochs', type=int, default=3000, help='训练轮数')
    parser.add_argument('--map-size', type=int, default=4, help='地图大小')
    parser.add_argument('--algorithms', nargs='+',
                       choices=['q_learning', 'sarsa', 'dqn', 'reinforce', 'actor_critic', 'a2c'],
                       default=['q_learning', 'sarsa', 'a2c'],
                       help='要测试的算法')
    
    args = parser.parse_args()
    
    benchmark = ComprehensiveBenchmark(map_size=args.map_size)
    results = benchmark.run_all(epochs=args.epochs, algorithms=args.algorithms)
    
    print(f"\n基准测试完成!")


if __name__ == "__main__":
    main()