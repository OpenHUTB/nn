"""
高级强化学习算法模块

包含PPO、SAC、DDPG等高级RL算法实现
"""

import numpy as np
from typing import Tuple, List, Optional


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) 算法
    
    PPO通过限制策略更新幅度来保证训练的稳定性
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.001
    ):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.old_policy = ActorNetwork(state_dim, action_dim)
        self.old_policy.weights = self._copy_weights(self.actor.weights)
    
    def _copy_weights(self, weights: dict) -> dict:
        """复制权重"""
        return {k: [w.copy() for w in v] if isinstance(v, list) else v 
                for k, v in weights.items()}
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """选择动作"""
        return self.actor.get_action(state, training)
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   dones: List[bool], gamma: float = 0.99, lambda_: float = 0.95):
        """计算广义优势估计"""
        advantages = np.zeros(len(rewards))
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + gamma * lambda_ * (1 - dones[t]) * last_advantage
        
        returns = advantages + np.array(values)
        return advantages, returns
    
    def update(self, states: np.ndarray, actions: np.ndarray, 
              rewards: List[float], dones: List[bool]):
        """PPO更新"""
        values = [self.critic.forward(s) for s in states]
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(10):
            for i in range(len(states)):
                state = states[i]
                action = int(actions[i])
                advantage = advantages[i]
                return_target = returns[i]
                
                old_log_prob = self.old_policy.get_log_prob(state, action)
                new_log_prob = self.actor.get_log_prob(state, action)
                
                ratio = np.exp(new_log_prob - old_log_prob)
                
                surr1 = ratio * advantage
                surr2 = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                policy_loss = -np.minimum(surr1, surr2)
                
                value_loss = self.value_coef * (self.critic.forward(state) - return_target) ** 2
                
                self.actor.update([state], [action], [advantage], self.actor_lr)
                self.critic.update([state], [return_target], self.critic_lr)
        
        self.old_policy.weights = self._copy_weights(self.actor.weights)


class ActorNetwork:
    """Actor网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple = (64, 64)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weights = self._init_weights(hidden_sizes)
    
    def _init_weights(self, hidden_sizes: Tuple) -> dict:
        """初始化权重"""
        layer_sizes = [self.state_dim] + list(hidden_sizes) + [self.action_dim]
        weights = {'W': [], 'b': []}
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            weights['W'].append(w)
            weights['b'].append(b)
        return weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-8)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def forward(self, state) -> np.ndarray:
        x = self._preprocess_state(state)
        for i in range(len(self.weights['W']) - 1):
            x = np.dot(x, self.weights['W'][i]) + self.weights['b'][i]
            x = self._relu(x)
        x = np.dot(x, self.weights['W'][-1]) + self.weights['b'][-1]
        return self._softmax(x).flatten()
    
    def _preprocess_state(self, state) -> np.ndarray:
        if isinstance(state, (int, np.integer)):
            x = np.zeros((1, self.state_dim))
            x[0, int(state)] = 1.0
        elif isinstance(state, np.ndarray):
            if state.ndim == 0 or state.size == 1:
                x = np.zeros((1, self.state_dim))
                x[0, int(np.ravel(state)[0])] = 1.0
            else:
                x = state.flatten().reshape(1, -1)
        else:
            x = np.array(state).flatten().reshape(1, -1)
        return x
    
    def get_action(self, state, training: bool = True) -> Tuple[int, float]:
        probs = self.forward(state)
        if training:
            action = np.random.choice(self.action_dim, p=probs)
        else:
            action = np.argmax(probs)
        return int(action), float(probs[action])
    
    def get_log_prob(self, state, action: int) -> float:
        probs = self.forward(state)
        return float(np.log(probs[action] + 1e-8))
    
    def update(self, states, actions, advantages, lr: float):
        for i in range(len(states)):
            state = states[i]
            action = int(actions[i])
            advantage = float(advantages[i])
            
            probs = self.forward(state)
            grad_log_prob = np.zeros(self.action_dim)
            for a in range(self.action_dim):
                if a == action:
                    grad_log_prob[a] = probs[a] * (1 - probs[a])
                else:
                    grad_log_prob[a] = -probs[a] * probs[action]
            
            delta = advantage * grad_log_prob
            x = self._preprocess_state(state)
            
            for i in range(len(self.weights['W'])):
                self.weights['W'][i] += lr * np.dot(x.T, delta.reshape(1, -1))
                self.weights['b'][i] += lr * delta.reshape(1, -1)
                
                if i < len(self.weights['W']) - 1:
                    x = self._relu(np.dot(x, self.weights['W'][i]) + self.weights['b'][i])


class CriticNetwork:
    """Critic网络"""
    
    def __init__(self, state_dim: int, hidden_sizes: Tuple = (64, 64)):
        self.state_dim = state_dim
        self.weights = self._init_weights(hidden_sizes)
    
    def _init_weights(self, hidden_sizes: Tuple) -> dict:
        layer_sizes = [self.state_dim] + list(hidden_sizes) + [1]
        weights = {'W': [], 'b': []}
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            weights['W'].append(w)
            weights['b'].append(b)
        return weights
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _preprocess_state(self, state) -> np.ndarray:
        if isinstance(state, (int, np.integer)):
            x = np.zeros((1, self.state_dim))
            x[0, int(state)] = 1.0
        elif isinstance(state, np.ndarray):
            if state.ndim == 0 or state.size == 1:
                x = np.zeros((1, self.state_dim))
                x[0, int(np.ravel(state)[0])] = 1.0
            else:
                x = state.flatten().reshape(1, -1)
        else:
            x = np.array(state).flatten().reshape(1, -1)
        return x
    
    def forward(self, state) -> float:
        x = self._preprocess_state(state)
        for i in range(len(self.weights['W']) - 1):
            x = np.dot(x, self.weights['W'][i]) + self.weights['b'][i]
            x = self._relu(x)
        x = np.dot(x, self.weights['W'][-1]) + self.weights['b'][-1]
        return float(x.flatten()[0])
    
    def update(self, states, targets, lr: float):
        for i in range(len(states)):
            state = states[i]
            target = float(targets[i])
            
            current = self.forward(state)
            error = target - current
            
            x = self._preprocess_state(state)
            for j in range(len(self.weights['W'])):
                grad = error * x if j == len(self.weights['W']) - 1 else error
                self.weights['W'][j] += lr * np.dot(x.T, grad.reshape(1, -1))
                self.weights['b'][j] += lr * grad.reshape(1, -1)
                
                if j < len(self.weights['W']) - 1:
                    x = self._relu(np.dot(x, self.weights['W'][j]) + self.weights['b'][j])


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = []
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def sample(self, batch_size: int) -> Tuple:
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in batch:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) 算法
    
    适用于连续动作空间的off-policy算法
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bound: float = 1.0,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.005
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        
        self.actor = DeterministicActor(state_dim, action_dim, action_bound)
        self.critic = QNetwork(state_dim, action_dim)
        self.target_actor = DeterministicActor(state_dim, action_dim, action_bound)
        self.target_critic = QNetwork(state_dim, action_dim)
        self._update_targetNetworks(tau=1.0)
        
        self.replay_buffer = ReplayBuffer()
        self.noise = OUNoise(action_dim)
    
    def _update_targetNetworks(self, tau: float = 0.005):
        """软更新目标网络"""
        self.target_actor.weights = self._soft_update(self.actor.weights, self.target_actor.weights, tau)
        self.target_critic.weights = self._soft_update(self.critic.weights, self.target_critic.weights, tau)
    
    def _soft_update(self, source: dict, target: dict, tau: float) -> dict:
        """软更新"""
        result = {}
        for key in source:
            if isinstance(source[key], list):
                result[key] = [tau * s + (1 - tau) * t for s, t in zip(source[key], target[key])]
            else:
                result[key] = tau * source[key] + (1 - tau) * target[key]
        return result
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        action = self.actor.forward(state)
        if training:
            action = action + self.noise.sample()
        return np.clip(action, -self.action_bound, self.action_bound)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self, batch_size: int = 64):
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        next_actions = self.target_actor.forward(next_states)
        target_q = self.target_critic.forward(next_states, next_actions)
        
        q_targets = rewards + self.gamma * target_q * (1 - dones)
        
        self.critic.update(states, actions, q_targets)
        
        actor_actions = self.actor.forward(states)
        actor_loss = -self.critic.forward(states, actor_actions).mean()
        
        self.actor.update(states, actor_loss)
        
        self._update_targetNetworks(self.tau)


class DeterministicActor:
    """确定性Actor网络"""
    
    def __init__(self, state_dim: int, action_dim: int, action_bound: float = 1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.weights = self._init_weights((400, 300))
    
    def _init_weights(self, hidden_sizes: Tuple) -> dict:
        layer_sizes = [self.state_dim] + list(hidden_sizes) + [self.action_dim]
        weights = {'W': [], 'b': []}
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            weights['W'].append(w)
            weights['b'].append(b)
        return weights
    
    def _relu(self, x): return np.maximum(0, x)
    def _tanh(self, x): return np.tanh(x)
    
    def forward(self, state) -> np.ndarray:
        x = state.flatten().reshape(1, -1)
        for i in range(len(self.weights['W']) - 1):
            x = np.dot(x, self.weights['W'][i]) + self.weights['b'][i]
            x = self._relu(x)
        x = np.dot(x, self.weights['W'][-1]) + self.weights['b'][-1]
        return self._tanh(x).flatten() * self.action_bound
    
    def update(self, states, loss, lr: float = 0.0001):
        for i in range(len(states)):
            x = states[i].flatten().reshape(1, -1)
            activations = [x]
            for j in range(len(self.weights['W']) - 1):
                x = self._relu(np.dot(x, self.weights['W'][j]) + self.weights['b'][j])
                activations.append(x)
            
            for j in range(len(self.weights['W']) - 1, -1, -1):
                grad_w = -loss * activations[j].T
                grad_b = -loss
                self.weights['W'][j] += lr * grad_w
                self.weights['b'][j] += lr * grad_b


class QNetwork:
    """Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weights = self._init_weights((400, 300))
    
    def _init_weights(self, hidden_sizes: Tuple) -> dict:
        layer_sizes = [self.state_dim + self.action_dim] + list(hidden_sizes) + [1]
        weights = {'W': [], 'b': []}
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            weights['W'].append(w)
            weights['b'].append(b)
        return weights
    
    def _relu(self, x): return np.maximum(0, x)
    
    def forward(self, state, action) -> float:
        x = np.concatenate([state.flatten(), action.flatten()]).reshape(1, -1)
        for i in range(len(self.weights['W']) - 1):
            x = np.dot(x, self.weights['W'][i]) + self.weights['b'][i]
            x = self._relu(x)
        x = np.dot(x, self.weights['W'][-1]) + self.weights['b'][-1]
        return float(x.flatten()[0])
    
    def update(self, states, actions, q_targets, lr: float = 0.001):
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            q_target = float(q_targets[i])
            
            current_q = self.forward(state, action)
            error = q_target - current_q
            
            x = np.concatenate([state.flatten(), action.flatten()]).reshape(1, -1)
            for j in range(len(self.weights['W'])):
                self.weights['W'][j] += lr * error * x.T
                self.weights['b'][j] += lr * error
                if j < len(self.weights['W']) - 1:
                    x = self._relu(np.dot(x, self.weights['W'][j]) + self.weights['b'][j])


class OUNoise:
    """Ornstein-Uhlenbeck噪声"""
    
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(action_dim)
    
    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state
