# 强化学习 (RL)

## 概述

强化学习（Reinforcement Learning, RL）是机器学习的一个分支，智能体通过与环境交互来学习最优策略，以最大化累积奖励。

## 主要特点

- **试错学习**：通过尝试不同的动作来发现最优策略
- **延迟奖励**：当前动作的奖励可能在未来才显现
- **探索与利用**：平衡探索新策略和利用已知最优策略

## 基本概念

### 1. 智能体 (Agent)

- 学习和决策的主体
- 根据当前状态选择动作

### 2. 环境 (Environment)

- 智能体交互的外部系统
- 提供状态和奖励反馈

### 3. 状态 (State)

- 环境的当前情况
- 智能体决策的依据

### 4. 动作 (Action)

- 智能体可以执行的操作
- 影响环境状态转移

### 5. 奖励 (Reward)

- 环境对动作的即时反馈
- 指导智能体学习方向

## 主要算法

### 值函数方法

- Q-Learning
- SARSA
- Deep Q-Network (DQN)

### 策略梯度方法

- REINFORCE
- Actor-Critic
- Proximal Policy Optimization (PPO)

## 应用领域

- 游戏AI（围棋、电子游戏）
- 机器人控制
- 自动驾驶
- 推荐系统
- 资源调度
  
## 代码示例

使用 PyTorch 实现 DQN 网络：

```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    """深度Q网络"""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)
```

## 在 Carla 中的应用

- **车道保持**：以偏离车道线距离作为负奖励
- **避障**：碰撞时给予大额负奖励，安全行驶给予正奖励
- **速度控制**：以实际速度与目标速度差值作为奖励信号

## 参考资料

- [PyTorch 强化学习教程](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [OpenAI Gym 文档](https://gymnasium.farama.org/)
