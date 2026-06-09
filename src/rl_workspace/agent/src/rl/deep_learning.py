"""
深度强化学习算法模块

包含DQN、DDQN等深度RL算法的实现
"""

import numpy as np
from typing import Optional, Tuple
import random
from collections import deque


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def __len__(self):
        return len(self.buffer)


class DQNLearner:
    """
    Deep Q-Network (DQN) 强化学习代理
    
    DQN使用深度神经网络来近似Q函数，结合经验回放和目标网络来稳定训练。
    
    核心特性:
        - 经验回放 (Experience Replay)
        - 目标网络 (Target Network)
        - 离线学习 (Off-policy)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step = 0
        
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self._update_target_network()
    
    def _build_network(self) -> dict:
        """构建Q网络（使用简单的线性层模拟，实际应用中应用PyTorch/TensorFlow）"""
        layer_sizes = [self.state_dim, 64, 64, self.action_dim]
        weights = []
        biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            weights.append(w)
            biases.append(b)
        
        return {'weights': weights, 'biases': biases}
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _forward(self, x: np.ndarray, network: dict) -> np.ndarray:
        """前向传播"""
        x = x.reshape(1, -1)
        for i, (w, b) in enumerate(zip(network['weights'], network['biases'])):
            x = np.dot(x, w) + b
            if i < len(network['weights']) - 1:
                x = self._relu(x)
        return x.flatten()
    
    def _update_network(self, states: np.ndarray, target_q: np.ndarray):
        """更新Q网络（简化版，实际应用应使用梯度下降）"""
        x = states.reshape(-1, self.state_dim)
        activations = [x]
        
        for i, (w, b) in enumerate(zip(self.q_network['weights'], self.q_network['biases'])):
            z = np.dot(activations[-1], w) + b
            if i < len(self.q_network['weights']) - 1:
                activations.append(self._relu(z))
            else:
                activations.append(z)
        
        output = activations[-1]
        error = output - target_q
        learning_rate = 0.001
        
        for i in range(len(self.q_network['weights']) - 1, -1, -1):
            grad = np.dot(activations[i].T, error) / len(states)
            self.q_network['weights'][i] -= learning_rate * grad
            self.q_network['biases'][i] -= learning_rate * np.mean(error, axis=0)
            
            if i > 0:
                error = np.dot(error, self.q_network['weights'][i].T)
                error = error * (activations[i-1] > 0)
    
    def _update_target_network(self):
        """复制Q网络到目标网络"""
        self.target_network = {
            'weights': [w.copy() for w in self.q_network['weights']],
            'biases': [b.copy() for b in self.q_network['biases']]
        }
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """根据ε-贪心策略选择动作"""
        if training and np.random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        q_values = self._forward(state, self.q_network)
        return int(np.argmax(q_values))
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转换到经验回放缓冲区"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """执行一次训练步骤"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        q_values = np.array([self._forward(s, self.q_network) for s in states])
        next_q_values = np.array([self._forward(s, self.target_network) for s in next_states])
        
        target_q = q_values.copy()
        for i in range(len(actions)):
            action = actions[i]
            reward = rewards[i]
            done = dones[i]
            
            if done:
                target_q[i, action] = reward
            else:
                target_q[i, action] = reward + self.gamma * np.max(next_q_values[i])
        
        self._update_network(states, target_q)
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_step += 1
        
        if self.train_step % self.target_update_freq == 0:
            self._update_target_network()
        
        return float(np.mean(np.abs(target_q - q_values)))
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_dict = {
            'q_network': self.q_network,
            'target_network': self.target_network,
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }
        np.save(filepath, model_dict, allow_pickle=True)
    
    def load_model(self, filepath: str):
        """加载模型"""
        model_dict = np.load(filepath, allow_pickle=True).item()
        self.q_network = model_dict['q_network']
        self.target_network = model_dict['target_network']
        self.epsilon = model_dict['epsilon']
        self.train_step = model_dict['train_step']


class DDQNLearner(DQNLearner):
    """
    Double DQN (DDQN) 强化学习代理
    
    DDQN通过解耦动作选择和动作评估来减少Q值的过估计问题。
    
    改进:
        - 使用在线网络选择动作
        - 使用目标网络评估动作价值
    """
    
    def train_step(self) -> Optional[float]:
        """执行一次DDQN训练步骤"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        q_values = np.array([self._forward(s, self.q_network) for s in states])
        next_q_online = np.array([self._forward(s, self.q_network) for s in next_states])
        next_q_target = np.array([self._forward(s, self.target_network) for s in next_states])
        
        target_q = q_values.copy()
        for i in range(len(actions)):
            action = actions[i]
            reward = rewards[i]
            done = dones[i]
            
            if done:
                target_q[i, action] = reward
            else:
                best_action = np.argmax(next_q_online[i])
                target_q[i, action] = reward + self.gamma * next_q_target[i, best_action]
        
        self._update_network(states, target_q)
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_step += 1
        
        if self.train_step % self.target_update_freq == 0:
            self._update_target_network()
        
        return float(np.mean(np.abs(target_q - q_values)))
