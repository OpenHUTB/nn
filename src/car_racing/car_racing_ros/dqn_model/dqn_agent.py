"""
================================================================================
DQN 智能体实现
================================================================================
继承自 BaseAgent，只包含 DQN 特有的更新逻辑。

DQN (Deep Q-Network) 算法原理:
================================================================================
核心思想: 用深度神经网络来逼近Q值函数

Q-Learning 更新公式:
    Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))
    
其中:
- Q(s, a): 状态s下动作a的Q值
- α: 学习率
- r: 当前奖励
- γ: 折扣因子
- max(Q(s', a')): 下一状态的最大Q值

DQN的改进:
1. 使用深度神经网络逼近Q函数
2. 经验回放: 打破样本时间相关性
3. 目标网络: 提供稳定的学习目标

损失函数:
    Loss = MSE(Q(s, a), r + γ * max(Q_target(s', a')))
================================================================================
"""
import torch
import numpy as np
from base_agent import BaseAgent, BaseDQNNetwork, SkipFrame, plot_rewards


class DQNAgent(BaseAgent):
    """
    标准 DQN 智能体
    
    特点:
    - 使用固定的目标网络更新（每5000步同步一次）
    - 周期性同步策略网络到目标网络
    
    与 DoubleDQN 的区别:
    - DQN: 使用目标网络选择和评估动作
    - DoubleDQN: 使用策略网络选择，目标网络评估（减少Q值过估计）
    """
    
    def _build_networks(self):
        """构建策略网络和目标网络"""
        self.policy_net = BaseDQNNetwork(self.state_shape, self.action_n).float()
        self.frozen_net = BaseDQNNetwork(self.state_shape, self.action_n).float()
        self.frozen_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net = self.policy_net.to(self.device)
        self.frozen_net = self.frozen_net.to(self.device)
    
    def update_net(self, batch_size):
        """
        DQN 核心更新逻辑
        
        算法步骤:
        1. 从回放缓冲区采样一批经验
        2. 计算当前Q值 (使用策略网络)
        3. 计算目标Q值 (使用目标网络)
        4. 最小化两者之间的差距
        
        目标Q值计算:
        - 如果 episode 结束: target = r
        - 如果未结束: target = r + γ * max(Q_target(s', a'))
        """
        self.n_updates += 1
        states, actions, rewards, new_states, terminateds = self.get_samples(batch_size)
        
        # -------------------------------------------------------------------------
        # 步骤1: 计算当前Q值
        # -------------------------------------------------------------------------
        # 从策略网络获取所有动作的Q值
        # actions 是采样时执行的动作索引
        # np.arange(batch_size) 创建 [0, 1, 2, ..., batch_size-1]
        current_q = self.policy_net(states)[np.arange(batch_size), actions]
        
        # -------------------------------------------------------------------------
        # 步骤2: 计算目标Q值
        # -------------------------------------------------------------------------
        with torch.no_grad():
            # 使用目标网络计算下一状态所有动作的Q值
            # .max(1)[0] 选取最大Q值
            # target = r + γ * max(Q_target(s', a')) * (1 - done)
            # 乘以 (1-done) 是为了在 episode 结束时不让未来Q值影响当前决策
            target_q = rewards + (1 - terminateds.float()) * self.gamma * \
                      self.frozen_net(new_states).max(1)[0]
        
        # -------------------------------------------------------------------------
        # 步骤3: 计算损失并更新
        # -------------------------------------------------------------------------
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()               # 反向传播
        self.optimizer.step()        # 更新参数
        
        # -------------------------------------------------------------------------
        # 步骤4: 定期同步目标网络
        # -------------------------------------------------------------------------
        if self.n_updates % self.hyperparameters.get('target_update', 5000) == 0:
            self.sync_target_net()
        
        return current_q.mean().item(), loss.item()


# ============================================================================
# 兼容性别名 (为了与旧代码兼容)
# ============================================================================
Agent = DQNAgent
plot_reward = plot_rewards
