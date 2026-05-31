"""
策略梯度算法模块

包含REINFORCE、Actor-Critic、A2C等策略梯度算法实现
"""

import numpy as np
from typing import Tuple


class PolicyNetwork:
    """策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple = (32, 16)):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = hidden_sizes
        self.weights = self._initialize_weights()
    
    def _initialize_weights(self) -> dict:
        """初始化网络权重"""
        layer_sizes = [self.state_dim] + list(self.hidden_sizes) + [self.action_dim]
        weights = {'W': [], 'b': []}
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            weights['W'].append(w)
            weights['b'].append(b)
        
        return weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-8)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _preprocess_state(self, state) -> np.ndarray:
        """预处理状态"""
        if isinstance(state, (int, np.integer)):
            x = np.zeros((1, self.state_dim))
            x[0, int(state)] = 1.0
        elif isinstance(state, np.ndarray):
            if state.ndim == 0 or state.size == 1:
                x = np.zeros((1, self.state_dim))
                x[0, int(np.ravel(state)[0])] = 1.0
            elif state.shape[-1] != self.state_dim:
                x = np.zeros((1, self.state_dim))
                x[0, int(np.ravel(state)[0])] = 1.0
            else:
                x = state.flatten().reshape(1, -1)
        else:
            x = np.array(state).flatten().reshape(1, -1)
        
        return x
    
    def forward(self, state) -> np.ndarray:
        """前向传播"""
        x = self._preprocess_state(state)
        
        for i in range(len(self.weights['W']) - 1):
            x = np.dot(x, self.weights['W'][i]) + self.weights['b'][i]
            x = self._relu(x)
        
        x = np.dot(x, self.weights['W'][-1]) + self.weights['b'][-1]
        return self._softmax(x).flatten()
    
    def get_action(self, state) -> Tuple[int, float]:
        """采样动作"""
        probs = self.forward(state)
        action = np.random.choice(self.action_dim, p=probs)
        return int(action), float(probs[action])
    
    def update(self, states, actions, advantages, learning_rate: float = 0.001):
        """策略梯度更新（修正激活值索引对齐）"""
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
            
            x = self._preprocess_state(state).flatten()
            activations = [x]
            
            for j in range(len(self.weights['W']) - 1):
                x = np.dot(x, self.weights['W'][j]) + self.weights['b'][j].flatten()
                x = self._relu(x)
                activations.append(x)
            
            for j in range(len(self.weights['W']) - 1, -1, -1):
                if j == len(self.weights['W']) - 1:
                    layer_error = delta
                else:
                    layer_error = np.dot(layer_error, self.weights['W'][j+1].T)
                    # 👇 终极修复：将 activations[j] 修改为 activations[j+1]，完美匹配当前隐藏层的维度
                    layer_error = layer_error * (activations[j+1] > 0)
                
                dw = np.outer(activations[j], layer_error)
                db = layer_error.reshape(1, -1)
                
                self.weights['W'][j] += learning_rate * dw
                self.weights['b'][j] += learning_rate * db


class ValueNetwork:
    """价值网络"""
    
    def __init__(self, state_dim: int, hidden_sizes: Tuple = (32, 16)):
        self.state_dim = int(state_dim)
        self.hidden_sizes = hidden_sizes
        self.weights = self._initialize_weights()
    
    def _initialize_weights(self) -> dict:
        """初始化网络权重"""
        layer_sizes = [self.state_dim] + list(self.hidden_sizes) + [1]
        weights = {'W': [], 'b': []}
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            weights['W'].append(w)
            weights['b'].append(b)
        
        return weights
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _preprocess_state(self, state) -> np.ndarray:
        """预处理状态"""
        if isinstance(state, (int, np.integer)):
            x = np.zeros((1, self.state_dim))
            x[0, int(state)] = 1.0
        elif isinstance(state, np.ndarray):
            if state.ndim == 0 or state.size == 1:
                x = np.zeros((1, self.state_dim))
                x[0, int(np.ravel(state)[0])] = 1.0
            elif state.shape[-1] != self.state_dim:
                x = np.zeros((1, self.state_dim))
                x[0, int(np.ravel(state)[0])] = 1.0
            else:
                x = state.flatten().reshape(1, -1)
        else:
            x = np.array(state).flatten().reshape(1, -1)
        
        return x
    
    def forward(self, state) -> float:
        """前向传播"""
        x = self._preprocess_state(state)
        
        for i in range(len(self.weights['W']) - 1):
            x = np.dot(x, self.weights['W'][i]) + self.weights['b'][i]
            x = self._relu(x)
        
        x = np.dot(x, self.weights['W'][-1]) + self.weights['b'][-1]
        return float(x.flatten()[0])
    
    def update(self, states, targets, learning_rate: float = 0.001):
        """价值网络更新（修正激活值索引对齐）"""
        for i in range(len(states)):
            state = states[i]
            target = float(targets[i])
            
            current_value = self.forward(state)
            error = target - current_value
            
            x = self._preprocess_state(state).flatten()
            activations = [x]
            
            for j in range(len(self.weights['W']) - 1):
                x = np.dot(x, self.weights['W'][j]) + self.weights['b'][j].flatten()
                x = self._relu(x)
                activations.append(x)
            
            layer_error = np.array([error])
            
            for j in range(len(self.weights['W']) - 1, -1, -1):
                if j == len(self.weights['W']) - 1:
                    pass
                else:
                    layer_error = np.dot(layer_error, self.weights['W'][j+1].T)
                    # 👇 终极修复：同样将 activations[j] 修改为 activations[j+1]
                    layer_error = layer_error * (activations[j+1] > 0)
                
                dw = np.outer(activations[j], layer_error)
                db = layer_error.reshape(1, -1)
                
                self.weights['W'][j] += learning_rate * dw
                self.weights['b'][j] += learning_rate * db


class REINFORCEAgent:
    """REINFORCE算法"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.01,
        gamma: float = 0.99
    ):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.gamma = gamma
        self.lr = learning_rate
    
    def select_action(self, state) -> Tuple[int, float]:
        """选择动作"""
        return self.policy.get_action(state)
    
    def update(self, trajectory: list):
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


class ActorCriticAgent:
    """Actor-Critic算法"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        actor_lr: float = 0.001,
        critic_lr: float = 0.01
    ):
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = ValueNetwork(state_dim)
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
    
    def select_action(self, state) -> Tuple[int, float]:
        """选择动作"""
        return self.actor.get_action(state)
    
    def compute_advantage(self, state, reward, next_state, done: bool) -> float:
        """计算TD误差"""
        current_value = self.critic.forward(state)
        next_value = self.critic.forward(next_state) if not done else 0
        
        td_target = reward + self.gamma * next_value
        td_error = td_target - current_value
        
        return float(td_error)
    
    def update(self, state, action: int, advantage: float):
        """更新Actor和Critic"""
        self.actor.update([state], [action], [advantage], self.actor_lr)
        
        td_target = advantage + self.critic.forward(state)
        self.critic.update([state], [td_target], self.critic_lr)


class A2CAgent:
    """A2C算法"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        n_steps: int = 5,
        actor_lr: float = 0.0007,
        critic_lr: float = 0.001
    ):
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = ValueNetwork(state_dim)
        self.gamma = gamma
        self.n_steps = n_steps
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
    
    def select_action(self, state) -> Tuple[int, float]:
        """选择动作"""
        return self.actor.get_action(state)
    
    def compute_gae(self, rewards, values, dones, gamma: float = 0.99, lambda_: float = 0.95):
        """计算广义优势估计"""
        advantages = np.zeros_like(rewards, dtype=float)
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
    
    def update(self, states, actions, rewards, dones):
        """批量更新"""
        values = np.array([self.critic.forward(s) for s in states], dtype=float)
        
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.actor.update(states, actions, advantages, self.actor_lr)
        self.critic.update(states, returns, self.critic_lr)