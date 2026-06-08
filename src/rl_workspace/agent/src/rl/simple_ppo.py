"""
简化版PPO算法

使用线性近似和softmax策略的PPO实现
"""

import numpy as np
from typing import Tuple, List


class SimplePPOAgent:
    """简化版PPO智能体"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        actor_lr: float = 0.01,
        critic_lr: float = 0.1
    ):
        self.actor_weights = np.zeros((state_dim, action_dim))
        self.critic_weights = np.zeros(state_dim)
        self.gamma = gamma
        self.epsilon = epsilon
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.old_log_probs = None
    
    def _preprocess_state(self, state) -> np.ndarray:
        """预处理状态"""
        if isinstance(state, (int, np.integer)):
            x = np.zeros(len(self.critic_weights))
            x[int(state)] = 1.0
            return x
        elif isinstance(state, np.ndarray):
            if state.ndim == 0 or state.size == 1:
                x = np.zeros(len(self.critic_weights))
                x[int(np.ravel(state)[0])] = 1.0
                return x
            return state.flatten()
        return np.array(state).flatten()
    
    def _softmax_policy(self, state: np.ndarray) -> np.ndarray:
        """Softmax策略"""
        logits = np.dot(state, self.actor_weights)
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / (np.sum(exp_logits) + 1e-8)
    
    def _get_log_prob(self, state: np.ndarray, action: int) -> float:
        """计算对数概率"""
        probs = self._softmax_policy(state)
        return float(np.log(probs[action] + 1e-8))
    
    def select_action(self, state, training: bool = True) -> Tuple[int, float]:
        """选择动作"""
        state_vec = self._preprocess_state(state)
        probs = self._softmax_policy(state_vec)
        
        if training:
            action = np.random.choice(len(probs), p=probs)
        else:
            action = np.argmax(probs)
        
        return int(action), float(probs[action])
    
    def compute_gae(self, rewards: List[float], states: List[np.ndarray],
                   dones: List[bool], lambda_: float = 0.95):
        """计算GAE"""
        values = [np.dot(self._preprocess_state(s), self.critic_weights) for s in states]
        
        advantages = np.zeros(len(rewards))
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * lambda_ * (1 - dones[t]) * last_advantage
        
        returns = advantages + np.array(values)
        return advantages, returns
    
    def update(self, states: List, actions: List, rewards: List, dones: List):
        """PPO更新"""
        states_vec = [self._preprocess_state(s) for s in states]
        
        old_log_probs = np.array([self._get_log_prob(s, a) for s, a in zip(states_vec, actions)])
        
        advantages, returns = self.compute_gae(rewards, states_vec, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(4):
            for i in range(len(states)):
                state = states_vec[i]
                action = int(actions[i])
                advantage = advantages[i]
                return_target = returns[i]
                
                probs = self._softmax_policy(state)
                new_log_prob = np.log(probs[action] + 1e-8)
                old_log_prob = old_log_probs[i]
                
                ratio = np.exp(new_log_prob - old_log_prob)
                
                surr1 = ratio * advantage
                surr2 = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                policy_loss = -np.minimum(surr1, surr2)
                
                grad_log = np.zeros_like(self.actor_weights)
                for a in range(len(probs)):
                    if a == action:
                        grad_log[:, a] = state * probs[a] * (1 - probs[a])
                    else:
                        grad_log[:, a] = -state * probs[a] * probs[action]
                
                self.actor_weights += self.actor_lr * policy_loss * grad_log
                
                value_error = return_target - np.dot(state, self.critic_weights)
                self.critic_weights += self.critic_lr * value_error * state


def create_ppo_agent(state_dim: int, action_dim: int, 
                    learning_rate: float = 0.01, gamma: float = 0.99):
    """创建PPO智能体"""
    return SimplePPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma
    )
