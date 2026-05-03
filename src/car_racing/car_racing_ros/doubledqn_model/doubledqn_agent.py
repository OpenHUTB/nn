"""
================================================================================
Double DQN 智能体实现
================================================================================
继承自 BaseAgent，包含 DoubleDQN 特有的更新逻辑。

Double DQN 算法原理:
================================================================================
解决的问题: 标准DQN中Q值过估计 (Overestimation) 的问题

为什么会有过估计?
- DQN使用 max(Q_target(s', a')) 作为目标
- 由于函数逼近的误差，Q值可能会被高估
- 这会导致学习不稳定甚至发散

解决方案:
- 使用两个网络分离"选择动作"和"评估动作"
- 策略网络负责选择: a' = argmax(Q_policy(s', a'))
- 目标网络负责评估: Q_target(s', a')

这样可以减少过估计，因为选择和评估由不同的网络完成。

其他改进:
- Soft Update (软更新): 平滑地更新目标网络，比硬更新更稳定
- 梯度裁剪: 防止梯度爆炸
- 学习率调度: 自动调整学习率
================================================================================
"""
import torch
import yaml
from pathlib import Path
from torch import nn
from dqn_model.base_agent import BaseAgent, BaseDQNNetwork, SkipFrame, plot_rewards


class DoubleDQNAgent(BaseAgent):
    """
    Double DQN 智能体
    
    相对于标准DQN的改进:
    1. Double DQN: 减少Q值过估计
    2. Soft Update: 更稳定的目标网络更新
    3. 梯度裁剪: 防止训练不稳定
    4. 学习率调度: 自动学习率衰减
    """
    
    def __init__(self, state_space_shape, action_n, config_path=None,
                 load_state=False, load_model=None, hyperparameter_overrides=None, **kwargs):
        """
        初始化 DoubleDQN 智能体
        
        特殊处理:
        - 加载并合并 DQN 和 DoubleDQN 的配置文件
        - 设置 DoubleDQN 特有的超参数
        - 初始化学习率调度器
        """
        # 默认配置文件路径
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'configs' / 'double_dqn.yaml'
        
        config_path = Path(config_path)
        
        # -------------------------------------------------------------------------
        # 加载并合并配置文件
        # -------------------------------------------------------------------------
        # 基础 DQN 配置
        base_config_path = config_path.parent / 'dqn.yaml'
        with open(base_config_path) as f:
            base_config = yaml.safe_load(f)
        
        # DoubleDQN 特有配置
        with open(config_path) as f:
            ddqn_config = yaml.safe_load(f)
        
        # 合并配置: 先加载基础配置，再用DoubleDQN配置覆盖
        merged_config = {'hyperparameters': base_config.get('hyperparameters', {})}
        merged_config['hyperparameters'].update(ddqn_config.get('hyperparameters', {}))
        
        # DoubleDQN 特有超参数默认值
        merged_config['hyperparameters'].setdefault('tau', 0.005)           # 软更新系数
        merged_config['hyperparameters'].setdefault('update_target_every', 10000)  # 更新间隔
        merged_config['hyperparameters'].setdefault('max_grad_norm', 10.0)   # 梯度裁剪阈值
        
        self.config = merged_config
        
        # 提取 DoubleDQN 特有参数
        self.tau = merged_config['hyperparameters']['tau']                    # 软更新系数
        self.update_target_every = merged_config['hyperparameters']['update_target_every']
        self.max_grad_norm = merged_config['hyperparameters']['max_grad_norm']
        
        # 调用父类初始化
        super().__init__(
            state_space_shape=state_space_shape,
            action_n=action_n,
            config=self.config,
            config_path=None,
            load_state=load_state,
            load_model=load_model,
            hyperparameter_overrides=hyperparameter_overrides
        )
        
        # -------------------------------------------------------------------------
        # 初始化学习率调度器
        # -------------------------------------------------------------------------
        scheduler_type = self.hyperparameters.get('scheduler_type', 'step')
        if scheduler_type == 'step':
            # 步进调度: 每N步将学习率乘以 gamma
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.hyperparameters.get('scheduler_step', 10000),
                gamma=self.hyperparameters.get('scheduler_gamma', 0.9)
            )
        else:
            self.scheduler = None
    
    def _build_networks(self):
        """构建策略网络和目标网络"""
        dueling = bool(self.hyperparameters.get('dueling', True))
        self.policy_net = BaseDQNNetwork(self.state_shape, self.action_n, dueling=dueling).float()
        self.frozen_net = BaseDQNNetwork(self.state_shape, self.action_n, dueling=dueling).float()
        self.frozen_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net = self.policy_net.to(self.device)
        self.frozen_net = self.frozen_net.to(self.device)
    
    def update_net(self, batch_size):
        """
        Double DQN 核心更新逻辑
        
        与标准DQN的区别:
        1. 使用策略网络选择下一状态的最优动作
        2. 使用目标网络评估该动作的Q值
        
        算法步骤:
        1. 从回放缓冲区采样
        2. 计算当前Q值
        3. Double DQN 目标计算:
           - 用策略网络选择动作: a_selected = argmax(Q_policy(s', :))
           - 用目标网络评估: Q_target(s', a_selected)
        4. 应用软更新到目标网络
        """
        self.n_updates += 1
        states, actions, rewards, new_states, terminateds = self.get_samples(batch_size)

        if self.use_amp:
            from torch.cuda.amp import autocast
            with torch.no_grad(), autocast():
                next_actions = self.policy_net(new_states).argmax(1, keepdim=True)
                next_q = self.frozen_net(new_states).gather(1, next_actions)
                target_q = rewards.unsqueeze(1) + (1 - terminateds.float().unsqueeze(1)) * self.gamma * next_q

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
                loss = self.loss_fn(current_q, target_q)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_actions = self.policy_net(new_states).argmax(1, keepdim=True)
                next_q = self.frozen_net(new_states).gather(1, next_actions)
                target_q = rewards.unsqueeze(1) + (1 - terminateds.float().unsqueeze(1)) * self.gamma * next_q
            loss = self.loss_fn(current_q, target_q)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # -------------------------------------------------------------------------
        # 步骤4: 更新学习率
        # -------------------------------------------------------------------------
        if self.scheduler:
            self.scheduler.step()
        
        # -------------------------------------------------------------------------
        # 步骤5: 软更新目标网络
        # -------------------------------------------------------------------------
        # 软更新公式: Q_target = τ * Q_policy + (1 - τ) * Q_target
        # 其中 τ 是很小的值 (如 0.005)
        # 这比硬更新 (每N步完全复制) 更平滑稳定
        if self.n_updates % self.update_target_every == 0:
            for target_param, policy_param in zip(
                self.frozen_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
                )
        
        return current_q.mean().item(), float(loss.item())
    
    def save(self, save_dir, filename):
        """
        保存模型 - 包含调度器状态
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{filename}.pt")
        
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'frozen_net_state_dict': self.frozen_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epsilon': self.epsilon,
            'n_updates': self.n_updates,
            'config': self.config
        }, model_path)
        print(f"模型已保存到: {model_path}")
    
    def load(self, path):
        """
        加载模型 - 恢复调度器状态
        """
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.frozen_net.load_state_dict(checkpoint['frozen_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.n_updates = checkpoint['n_updates']
        
        # 恢复学习率调度器状态
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"模型已从 {path} 加载")
    
    def get_current_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


# ============================================================================
# 兼容性别名
# ============================================================================
Agent = DoubleDQNAgent
plot_reward = plot_rewards
