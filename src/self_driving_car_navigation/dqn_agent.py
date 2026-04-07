import torch
import torch.nn as nn
import numpy as np
from .pruning import prune_model
from .quantization import quantize_model
class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = config['agent']['epsilon_decay']
        self.epsilon_min = config['agent']['epsilon_min']
        self.learning_rate = config['train']['learning_rate']
        self.model = self._build_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model(torch.FloatTensor(state))
        return np.argmax(q_values.detach().numpy())

    def replay(self, batch_size):
        minibatch = np.random.choice(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model(torch.FloatTensor(next_state)).detach().numpy())
            target_f = self.model(torch.FloatTensor(state))
            target_f[action] = target
            self.model.fit(torch.FloatTensor(state), target_f.unsqueeze(0), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_reward(self, current_position, target_position, road_position, done):
        #计算奖励
        #:param current_position: 当前机器人位置 (x, y)
        #:param target_position: 目标位置 (x, y)
        #:param road_position: 机器人在道路上的位置
        #:param done: 是否完成任务
        #:return: 奖励值        
        distance_to_target = np.linalg.norm(current_position - target_position)
        distance_to_road = np.linalg.norm(current_position - road_position)

        if done:
            return 100  # 到达目标的奖励
        elif distance_to_target < 1.0:  # 靠近目标位置
            return 10  # 靠近目标的奖励
        elif distance_to_road > 1.0:  # 远离道路
            return -5  # 远离道路的惩罚
        elif distance_to_target < 5.0:  # 在某个距离范围内
            return 1  # 轻微奖励
        else:
            return -1  # 远离目标的惩罚

    def get_state(self, position, orientation, target_position, road_position):
        
        #获取状态
        #:param position: 机器人当前位置 (x, y)
        #:param orientation: 机器人朝向 (angle)
        #:param target_position: 目标位置 (x, y)
        #:param road_position: 机器人在道路上的位置
       # :return: 状态数组
        state = np.array([
            position[0],  # 当前 x 坐标
            position[1],  # 当前 y 坐标
            orientation,   # 当前朝向
            target_position[0],  # 目标 x 坐标
            target_position[1],  # 目标 y 坐标
            road_position[0],    # 道路 x 坐标
            road_position[1]     # 道路 y 坐标
        ])
        return state
