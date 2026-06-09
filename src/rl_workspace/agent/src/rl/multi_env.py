"""
多环境支持模块

支持CartPole、Acrobot、MountainCar等经典强化学习环境
"""

import numpy as np
from typing import Tuple, Optional
import gymnasium as gym


class CartPoleTrainer:
    """CartPole环境训练器"""
    
    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def create_env(self):
        """创建CartPole环境"""
        return gym.make("CartPole-v1", render_mode="rgb_array")
    
    def discretize_state(self, state: np.ndarray, bins: Tuple = (6, 6, 6, 6)) -> Tuple:
        """离散化连续状态"""
        state_bounds = [
            (-2.4, 2.4),
            (-3.0, 3.0),
            (-0.21, 0.21),
            (-3.0, 3.0)
        ]
        
        discrete_state = []
        for i, (low, high) in enumerate(state_bounds):
            if i >= len(state):
                discrete_state.append(0)
            elif state[i] < low:
                discrete_state.append(0)
            elif state[i] >= high:
                discrete_state.append(bins[i] - 1)
            else:
                discrete_state.append(int((state[i] - low) / (high - low) * bins[i]))
        
        return tuple(discrete_state)
    
    def train(self, env, q_table: np.ndarray, episodes: int = 10000,
              alpha: float = 0.1, gamma: float = 0.99,
              epsilon: float = 1.0, epsilon_decay: float = 0.995,
              epsilon_min: float = 0.01) -> Tuple[list, float]:
        """训练Q-Learning"""
        rewards = []
        
        for episode in range(episodes):
            state = env.reset()[0]
            discrete_state = self.discretize_state(state)
            done = False
            episode_reward = 0
            
            while not done:
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(q_table[discrete_state]))
                
                next_state, reward, done, trunc, info = env.step(action)
                next_discrete = self.discretize_state(next_state)
                
                q_table[discrete_state][action] = q_table[discrete_state][action] + \
                    alpha * (reward + gamma * np.max(q_table[next_discrete]) - 
                            q_table[discrete_state][action])
                
                discrete_state = next_discrete
                episode_reward += reward
            
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            rewards.append(episode_reward)
            
            if episode % 1000 == 0:
                avg_reward = np.mean(rewards[-1000:])
                print(f"Episode {episode}: 平均奖励={avg_reward:.1f}")
        
        return rewards, np.mean(rewards[-100:])
    
    def evaluate(self, env, q_table: np.ndarray, episodes: int = 100) -> Tuple[float, float]:
        """评估策略"""
        episode_rewards = []
        
        for _ in range(episodes):
            state = env.reset()[0]
            discrete_state = self.discretize_state(state)
            done = False
            episode_reward = 0
            
            while not done:
                action = int(np.argmax(q_table[discrete_state]))
                next_state, reward, done, trunc, info = env.step(action)
                next_discrete = self.discretize_state(next_state)
                discrete_state = next_discrete
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards)


class MountainCarTrainer:
    """MountainCar环境训练器"""
    
    def __init__(self):
        self.position_min = -1.2
        self.position_max = 0.6
        self.velocity_min = -0.07
        self.velocity_max = 0.07
    
    def create_env(self):
        """创建MountainCar环境"""
        return gym.make("MountainCar-v0", render_mode="rgb_array")
    
    def discretize_state(self, state: np.ndarray, 
                        position_bins: int = 20, velocity_bins: int = 20) -> Tuple:
        """离散化状态"""
        position, velocity = state
        
        position_idx = int((position - self.position_min) / 
                          (self.position_max - self.position_min) * position_bins)
        velocity_idx = int((velocity - self.velocity_min) / 
                          (self.velocity_max - self.velocity_min) * velocity_bins)
        
        position_idx = max(0, min(position_bins - 1, position_idx))
        velocity_idx = max(0, min(velocity_bins - 1, velocity_idx))
        
        return (position_idx, velocity_idx)
    
    def train(self, env, q_table: np.ndarray, episodes: int = 10000,
              alpha: float = 0.1, gamma: float = 0.99,
              epsilon: float = 1.0, epsilon_decay: float = 0.995,
              epsilon_min: float = 0.01) -> Tuple[list, float]:
        """训练"""
        rewards = []
        successes = 0
        
        for episode in range(episodes):
            state = env.reset()[0]
            discrete_state = self.discretize_state(state)
            done = False
            episode_reward = 0
            
            while not done:
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(q_table[discrete_state]))
                
                next_state, reward, done, trunc, info = env.step(action)
                next_discrete = self.discretize_state(next_state)
                
                reward = -1 if not done else 100
                
                q_table[discrete_state][action] = q_table[discrete_state][action] + \
                    alpha * (reward + gamma * np.max(q_table[next_discrete]) - 
                            q_table[discrete_state][action])
                
                discrete_state = next_discrete
                episode_reward += reward
            
            if episode_reward > -200:
                successes += 1
            
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            rewards.append(episode_reward)
            
            if episode % 1000 == 0:
                success_rate = successes / (episode + 1) * 100
                print(f"Episode {episode}: 成功率={success_rate:.1f}%")
        
        return rewards, successes / episodes * 100
    
    def evaluate(self, env, q_table: np.ndarray, episodes: int = 100) -> Tuple[float, float]:
        """评估"""
        successes = 0
        episode_rewards = []
        
        for _ in range(episodes):
            state = env.reset()[0]
            discrete_state = self.discretize_state(state)
            done = False
            episode_reward = 0
            
            while not done:
                action = int(np.argmax(q_table[discrete_state]))
                next_state, reward, done, trunc, info = env.step(action)
                next_discrete = self.discretize_state(next_state)
                discrete_state = next_discrete
                episode_reward += reward
            
            if episode_reward > -200:
                successes += 1
            episode_rewards.append(episode_reward)
        
        return successes / episodes * 100, np.mean(episode_rewards)


class AcrobotTrainer:
    """Acrobot环境训练器"""
    
    def __init__(self):
        self.theta1_bins = 10
        self.theta2_bins = 10
        self.theta1_vel_bins = 10
        self.theta2_vel_bins = 10
    
    def create_env(self):
        """创建Acrobot环境"""
        return gym.make("Acrobot-v1", render_mode="rgb_array")
    
    def discretize_state(self, state: np.ndarray) -> Tuple:
        """离散化状态"""
        cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta1_dot, theta2_dot = state
        
        theta1 = np.arctan2(sin_theta1, cos_theta1)
        theta2 = np.arctan2(sin_theta2, cos_theta2)
        
        theta1_bins = np.linspace(-np.pi, np.pi, self.theta1_bins)
        theta2_bins = np.linspace(-np.pi, np.pi, self.theta2_bins)
        theta1_vel_bins = np.linspace(-4 * np.pi, 4 * np.pi, self.theta1_vel_bins)
        theta2_vel_bins = np.linspace(-9 * np.pi, 9 * np.pi, self.theta2_vel_bins)
        
        theta1_idx = np.digitize(theta1, theta1_bins) - 1
        theta2_idx = np.digitize(theta2, theta2_bins) - 1
        theta1_dot_idx = np.digitize(theta1_dot, theta1_vel_bins) - 1
        theta2_dot_idx = np.digitize(theta2_dot, theta2_vel_bins) - 1
        
        theta1_idx = max(0, min(self.theta1_bins - 1, theta1_idx))
        theta2_idx = max(0, min(self.theta2_bins - 1, theta2_idx))
        theta1_dot_idx = max(0, min(self.theta1_vel_bins - 1, theta1_dot_idx))
        theta2_dot_idx = max(0, min(self.theta2_vel_bins - 1, theta2_dot_idx))
        
        return (theta1_idx, theta2_idx, theta1_dot_idx, theta2_dot_idx)
    
    def train(self, env, q_table: np.ndarray, episodes: int = 50000,
              alpha: float = 0.1, gamma: float = 0.99,
              epsilon: float = 1.0, epsilon_decay: float = 0.995,
              epsilon_min: float = 0.01) -> Tuple[list, float]:
        """训练"""
        rewards = []
        successes = 0
        
        for episode in range(episodes):
            state = env.reset()[0]
            discrete_state = self.discretize_state(state)
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < 500:
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(q_table[discrete_state]))
                
                next_state, reward, done, trunc, info = env.step(action)
                next_discrete = self.discretize_state(next_state)
                
                reward = -1 if not done else 0
                
                q_table[discrete_state][action] = q_table[discrete_state][action] + \
                    alpha * (reward + gamma * np.max(q_table[next_discrete]) - 
                            q_table[discrete_state][action])
                
                discrete_state = next_discrete
                episode_reward += reward
                steps += 1
            
            if steps < 500:
                successes += 1
            
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            rewards.append(-episode_reward)
            
            if episode % 5000 == 0:
                success_rate = successes / (episode + 1) * 100
                print(f"Episode {episode}: 成功率={success_rate:.1f}%")
        
        return rewards, successes / episodes * 100


def create_trainer(env_name: str):
    """创建对应环境的训练器"""
    trainers = {
        'CartPole-v1': CartPoleTrainer,
        'CartPole-v0': CartPoleTrainer,
        'MountainCar-v0': MountainCarTrainer,
        'MountainCar-v1': MountainCarTrainer,
        'Acrobot-v1': AcrobotTrainer,
        'Acrobot-v2': AcrobotTrainer
    }
    
    if env_name not in trainers:
        raise ValueError(f"不支持的环境: {env_name}")
    
    return trainers[env_name]()
