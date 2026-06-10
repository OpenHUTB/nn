"""
强化学习核心算法模块

包含Q-Learning和SARSA等经典RL算法的实现
"""

from typing import Union, Tuple, Optional
import numpy as np


class BaseLearner:
    """强化学习代理基类"""
    
    def __init__(
        self,
        states: Union[int, Tuple[int, ...]],
        actions: int,
        alpha: float = 0.8,
        gamma: float = 0.9,
        rar: float = 1.0,
        min_rar: float = 0.01,
        max_rar: float = 1.0,
        radr: float = 0.00005,
    ):
        self.states = states
        self.actions = actions
        self.lr = alpha
        self.dr = gamma
        self.rar = rar
        self.min_rar = min_rar
        self.max_rar = max_rar
        self.radr = radr
        
        self._validate_parameters()
        self.Q = self._initialize_q_table()
        
        self.s: Optional[Union[int, Tuple[int, ...]]] = None
        self.a: Optional[int] = None
    
    def _validate_parameters(self) -> None:
        """验证输入参数的有效性"""
        if not isinstance(self.states, (int, np.integer, tuple)):
            raise TypeError("states必须是int或tuple类型")
        
        if isinstance(self.states, tuple):
            for dim in self.states:
                if not isinstance(dim, (int, np.integer)):
                    raise TypeError("tuple中的每个元素必须是int类型")
        
        if not isinstance(self.actions, (int, np.integer)):
            raise TypeError("actions必须是int类型")
        
        if not (0 <= self.lr <= 1):
            raise ValueError("学习率alpha必须在[0, 1]范围内")
        
        if not (0 <= self.dr <= 1):
            raise ValueError("折扣因子gamma必须在[0, 1]范围内")
    
    def _initialize_q_table(self) -> np.ndarray:
        """初始化Q表"""
        shape = []
        if isinstance(self.states, (int, np.integer)):
            shape.append(int(self.states))
        elif isinstance(self.states, tuple):
            shape.extend(list(self.states))
        
        shape.append(self.actions)
        return np.zeros(shape=shape, dtype=np.float64)
    
    def get_action(self, state: Union[int, Tuple[int, ...]]) -> int:
        """根据ε-贪心策略选择动作"""
        if np.random.random() > self.rar:
            return int(np.argmax(self.Q[state]))
        else:
            return int(np.random.choice(self.actions))
    
    def decay_rar(self, epoch: int) -> None:
        """指数衰减探索率"""
        self.rar = self.min_rar + (self.max_rar - self.min_rar) * np.exp(-self.radr * epoch)
    
    def save_q_table(self, filepath: str) -> None:
        """保存Q表到文件"""
        np.save(filepath, self.Q)
    
    @classmethod
    def load_q_table(cls, filepath: str, actions: int) -> 'BaseLearner':
        """从文件加载Q表"""
        q_table = np.load(filepath)
        states_shape = q_table.shape[:-1]
        if len(states_shape) == 1:
            states = int(states_shape[0])
        else:
            states = tuple(states_shape)
        
        learner = cls(states=states, actions=actions)
        learner.Q = q_table
        return learner


class QLearner(BaseLearner):
    """
    Q-Learning 强化学习代理
    
    Q-Learning是一种无模型强化学习算法，通过学习动作价值函数(Q函数)
    来指导智能体在环境中选择最优动作。
    
    核心公式:
        Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
    """
    
    def get_next_action_without_Q_table_update(self, state: Union[int, Tuple[int, ...]]) -> int:
        """获取下一个动作但不更新Q表"""
        self.s = state
        action = self.get_action(state)
        self.a = action
        return action
    
    def get_next_action_with_Q_table_update(self, new_state: Union[int, Tuple[int, ...]], reward: float) -> int:
        """获取下一个动作并更新Q表"""
        if self.s is None or self.a is None:
            raise RuntimeError("必须先调用get_next_action_without_Q_table_update初始化状态")
        
        self.update_Q(self.s, self.a, new_state, reward)
        self.s = new_state
        self.a = self.get_action(new_state)
        return self.a
    
    def update_Q(self, last_state: Union[int, Tuple[int, ...]], last_action: int, 
                 new_state: Union[int, Tuple[int, ...]], reward: float) -> None:
        """Q值更新公式"""
        next_optimal_q_value = float(np.max(self.Q[new_state]))
        
        if isinstance(self.states, (int, np.integer)):
            old_q_value = float(self.Q[last_state, last_action])
            self.Q[last_state, last_action] = old_q_value + self.lr * (
                reward + self.dr * next_optimal_q_value - old_q_value
            )
        elif isinstance(self.states, tuple):
            old_q_value = float(self.Q[last_state + (last_action,)])
            self.Q[last_state + (last_action,)] = old_q_value + self.lr * (
                reward + self.dr * next_optimal_q_value - old_q_value
            )


class SARSALearner(BaseLearner):
    """
    SARSA 强化学习代理
    
    SARSA(State-Action-Reward-State-Action)是一种在线学习算法，
    使用当前策略选择下一个动作来更新Q值。
    
    核心公式:
        Q(s, a) = Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]
    
    与Q-Learning的区别:
        - Q-Learning: 使用max(Q(s', a'))，是off-policy算法
        - SARSA: 使用Q(s', a')，是on-policy算法
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_a: Optional[int] = None
    
    def get_next_action_without_Q_table_update(self, state: Union[int, Tuple[int, ...]]) -> int:
        """获取下一个动作但不更新Q表"""
        self.s = state
        action = self.get_action(state)
        self.a = action
        return action
    
    def get_next_action_with_Q_table_update(self, new_state: Union[int, Tuple[int, ...]], reward: float) -> int:
        """获取下一个动作并更新Q表(SARSA特有)"""
        if self.s is None or self.a is None:
            raise RuntimeError("必须先调用get_next_action_without_Q_table_update初始化状态")
        
        next_action = self.get_action(new_state)
        self.update_Q(self.s, self.a, new_state, next_action, reward)
        self.s = new_state
        self.a = next_action
        return next_action
    
    def update_Q(self, last_state: Union[int, Tuple[int, ...]], last_action: int,
                 new_state: Union[int, Tuple[int, ...]], next_action: int, reward: float) -> None:
        """SARSA Q值更新公式"""
        next_q_value = float(self.Q[new_state, next_action]) if isinstance(self.states, (int, np.integer)) \
                       else float(self.Q[new_state + (next_action,)])
        
        if isinstance(self.states, (int, np.integer)):
            old_q_value = float(self.Q[last_state, last_action])
            self.Q[last_state, last_action] = old_q_value + self.lr * (
                reward + self.dr * next_q_value - old_q_value
            )
        elif isinstance(self.states, tuple):
            old_q_value = float(self.Q[last_state + (last_action,)])
            self.Q[last_state + (last_action,)] = old_q_value + self.lr * (
                reward + self.dr * next_q_value - old_q_value
            )
