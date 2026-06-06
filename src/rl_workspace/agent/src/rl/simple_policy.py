"""
简化版策略梯度算法模块

使用softmax策略的REINFORCE和Actor-Critic算法
"""

import numpy as np
from typing import Tuple, List


class SoftmaxPolicy:
    """Softmax策略"""
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weights = np.zeros((state_dim, action_dim))
    
    def get_action(self, state: int) -> Tuple[int, float]:
        """根据softmax策略选择动作"""
        logits = np.dot(state if isinstance(state, np.ndarray) else self._one_hot(state), self.weights)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        action = np.random.choice(self.action_dim, p=probs)
        return action, probs[action]
    
    def _one_hot(self, state: int) -> np.ndarray:
        """One-hot编码"""
        vec = np.zeros(self.state_dim)
        vec[state] = 1
        return vec
    
    def update(self, states: List[int], actions: List[int], advantages: np.ndarray, lr: float):
        """策略梯度更新"""
        for s, a, adv in zip(states, actions, advantages):
            state_vec = self._one_hot(s) if isinstance(s, int) else s
            logits = np.dot(state_vec, self.weights)
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            grad = np.zeros_like(self.weights)
            for action in range(self.action_dim):
                if action == a:
                    grad[:, action] = state_vec * probs[action] * (1 - probs[action])
                else:
                    grad[:, action] = -state_vec * probs[action] * probs[a]
            
            self.weights += lr * adv * grad


class REINFORCEAgent:
    """简化版REINFORCE算法"""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.1, gamma: float = 0.99):
        self.policy = SoftmaxPolicy(state_dim, action_dim)
        self.gamma = gamma
        self.lr = learning_rate
    
    def select_action(self, state: int) -> Tuple[int, float]:
        """选择动作"""
        return self.policy.get_action(state)
    
    def update(self, trajectory: List[tuple]):
        """更新策略网络"""
        states = [t[0] for t in trajectory]
        actions = [t[1] for t in trajectory]
        rewards = [t[2] for t in trajectory]
        
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        self.policy.update(states, actions, returns, self.lr)


class SimpleActorCritic:
    """简化版Actor-Critic"""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.1, gamma: float = 0.99):
        self.actor = SoftmaxPolicy(state_dim, action_dim)
        self.critic_weights = np.zeros(state_dim)
        self.gamma = gamma
        self.actor_lr = learning_rate
        self.critic_lr = learning_rate * 0.5
    
    def select_action(self, state: int) -> Tuple[int, float]:
        """选择动作"""
        return self.actor.get_action(state)
    
    def compute_td_error(self, state: int, reward: float, next_state: int, done: bool) -> float:
        """计算TD误差"""
        current_value = np.dot(self._one_hot(state), self.critic_weights)
        next_value = np.dot(self._one_hot(next_state), self.critic_weights) if not done else 0
        return reward + self.gamma * next_value - current_value
    
    def _one_hot(self, state: int) -> np.ndarray:
        """One-hot编码"""
        vec = np.zeros(len(self.critic_weights))
        vec[state] = 1
        return vec
    
    def update(self, state: int, action: int, td_error: float):
        """更新Actor和Critic"""
        state_vec = self._one_hot(state)
        
        self.critic_weights += self.critic_lr * td_error * state_vec
        
        logits = np.dot(state_vec, self.actor.weights)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        grad = np.zeros_like(self.actor.weights)
        for a in range(self.actor.action_dim):
            if a == action:
                grad[:, a] = state_vec * probs[a] * (1 - probs[a])
            else:
                grad[:, a] = -state_vec * probs[a] * probs[action]
        
        self.actor.weights += self.actor_lr * td_error * grad


def create_agent(algorithm: str, state_dim: int, action_dim: int, 
                learning_rate: float = 0.1, gamma: float = 0.99):
    """创建策略梯度代理"""
    if algorithm == 'reinforce':
        return REINFORCEAgent(state_dim, action_dim, learning_rate, gamma)
    elif algorithm == 'actor_critic':
        return SimpleActorCritic(state_dim, action_dim, learning_rate, gamma)
    else:
        raise ValueError(f"未知算法: {algorithm}")
