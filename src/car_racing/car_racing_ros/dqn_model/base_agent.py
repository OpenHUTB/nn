"""
================================================================================
深度强化学习智能体基类
================================================================================
包含 DQN 和 DoubleDQN 智能体共享的公共功能。

核心概念说明:
- Q值 (Q-Value): 表示在某个状态下采取某个动作的预期长期收益
- 策略网络 (Policy Net): 负责选择动作的网络，不断更新
- 目标网络 (Target Net): 提供稳定的目标Q值，定期从策略网络同步
- 经验回放 (Replay Buffer): 存储历史经验，用于打破样本时间相关性
- ε-greedy: 以ε概率随机探索，1-ε概率选择最优动作
================================================================================
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import warnings
import torch
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
import gymnasium as gym

# 检测是否在 Jupyter Notebook 环境中
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


# ================================================================================
# 环境预处理 Wrapper
# ================================================================================

class SkipFrame(gym.Wrapper):
    """
    跳帧 wrapper - 加速训练
    
    原理: 连续执行相同的动作N步，只保留最后的状态和累积奖励
    作用: 
    1. 加快训练速度（减少环境交互次数）
    2. 使智能体学会处理时间序列信息
    
    参数:
        env: gym 环境
        skip: 跳过的帧数
    """
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """执行动作skip次，累积奖励"""
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return state, total_reward, terminated, truncated, info


# ================================================================================
# 神经网络架构
# ================================================================================

class BaseDQNNetwork(nn.Module):
    """
    基础CNN网络架构 - 用于处理CarRacing的图像输入
    
    网络结构:
    ┌─────────────────────────────────────────────────────────┐
    │  输入: (batch, 4, 84, 84)                                │
    │  4通道是因为堆叠了连续4帧灰度图像，提供时序信息            │
    │                                                          │
    │  Conv2d(4→16, 8×8, stride=4)  →  ReLU                  │
    │  Conv2d(16→32, 4×4, stride=2) →  ReLU                   │
    │  Flatten → (batch, 2592)                                │
    │  Linear(2592→256) → ReLU                               │
    │  Linear(256→action_n)  → 5个动作的Q值                    │
    └─────────────────────────────────────────────────────────┘
    
    参数:
        in_dim: 输入维度，通常是 (通道数, 高度, 宽度)
        out_dim: 输出维度，即动作空间大小（CarRacing是5）
    """
    def __init__(self, in_dim, out_dim, dueling: bool = False):
        super().__init__()
        channel_n, height, width = in_dim

        # 检查输入尺寸是否符合预期
        if height != 84 or width != 84:
            raise ValueError(f"网络要求输入尺寸为 (84, 84)，但收到 ({height}, {width})")

        self.dueling = bool(dueling)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=channel_n, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        if self.dueling:
            self.advantage = nn.Sequential(
                nn.Linear(2592, 256),
                nn.ReLU(),
                nn.Linear(256, out_dim),
            )
            self.value = nn.Sequential(
                nn.Linear(2592, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(2592, 256),
                nn.ReLU(),
                nn.Linear(256, out_dim),
            )

    def forward(self, x):
        """前向传播"""
        feats = self.features(x)
        if self.dueling:
            advantage = self.advantage(feats)
            value = self.value(feats)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.head(feats)


# ================================================================================
# 智能体基类
# ================================================================================

class BaseAgent:
    """
    DQN系列智能体的基类
    
    提供以下公共功能:
    - 神经网络管理（策略网络、目标网络）
    - 经验回放缓冲区的管理
    - ε-greedy 动作选择
    - 模型保存和加载
    - 训练日志记录
    
    子类需要实现:
    - update_net(): 核心更新逻辑（Q-Learning算法）
    - _build_networks(): 网络构建逻辑
    """
    
    def __init__(self, state_space_shape, action_n, config=None, config_path=None,
                 load_state=False, load_model=None, hyperparameter_overrides=None):
        """
        初始化智能体
        
        参数:
            state_space_shape: 状态空间的形状，如 (4, 84, 84)
            action_n: 动作空间的大小，CarRacing是5
            config: 配置字典，如果为None则从文件加载
            config_path: 配置文件路径
            load_state: 是否加载已保存的模型
            load_model: 模型文件名
        """
        self.state_shape = state_space_shape
        self.action_n = action_n
        
        # 1. 加载配置文件
        if config is None:
            if config_path is None:
                config_path = Path(__file__).parent.parent / 'configs' / 'dqn.yaml'
            elif isinstance(config_path, str):
                config_path = Path(config_path)
            
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
        
        # 2. 提取超参数
        self.hyperparameters = self.config.get('hyperparameters', {})
        if hyperparameter_overrides:
            self.hyperparameters = dict(self.hyperparameters)
            self.hyperparameters.update(hyperparameter_overrides)
        
        # 折扣因子 gamma: 未来奖励的重要性，越接近1越重视长期收益
        self.gamma = self.hyperparameters.get('gamma', 0.99)
        
        # 探索率 epsilon: 随机动作的概率，开始时为1（完全探索）
        self.epsilon = self.hyperparameters.get('epsilon_start', 1.0)
        
        # epsilon 衰减: 每次更新后探索率乘以这个值，逐渐减少探索
        self.epsilon_decay = self.hyperparameters.get('epsilon_decay', 0.9999)
        
        # epsilon 最小值: 探索率不会低于这个值
        self.epsilon_min = self.hyperparameters.get('epsilon_min', 0.05)
        
        # 3. 设置设备（GPU或CPU）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cuda_available = torch.cuda.is_available()
        self.device = "cuda" if cuda_available else "cpu"
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.normalize_obs = bool(self.hyperparameters.get('normalize_obs', True))
        self.use_amp = bool(self.hyperparameters.get('amp', True)) and self.device == "cuda"
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # 4. 构建神经网络
        self.policy_net = None
        self.frozen_net = None
        self._build_networks()
        
        # 5. 优化器和损失函数
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.hyperparameters.get('lr', 0.0001)
        )
        self.loss_fn = nn.SmoothL1Loss()
        
        # 6. 经验回放缓冲区
        self.buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                self.hyperparameters.get('buffer_size', 100000),
                device=torch.device("cpu")
            )
        )
        
        # 7. 训练统计
        self.act_taken = 0
        self.n_updates = 0
        
        # 8. 设置保存路径
        repo = Path(__file__).resolve().parents[1]
        self.save_dir = str(repo / "training" / "saved_models")
        self.log_dir = str(repo / "training" / "logs")
        
        # 9. 加载已保存的模型
        if load_state and load_model:
            self.load(os.path.join(self.save_dir, load_model))

    def _build_networks(self):
        """
        构建策略网络和目标网络
        
        策略网络 (Policy Net): 负责选择动作，不断更新
        目标网络 (Target Net): 提供稳定的目标Q值
        """
        dueling = bool(self.hyperparameters.get('dueling', True))
        self.policy_net = BaseDQNNetwork(self.state_shape, self.action_n, dueling=dueling).float()
        self.frozen_net = BaseDQNNetwork(self.state_shape, self.action_n, dueling=dueling).float()
        self.frozen_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net = self.policy_net.to(self.device)
        self.frozen_net = self.frozen_net.to(self.device)

    def store(self, state, action, reward, new_state, terminated):
        """
        将经验存储到回放缓冲区
        
        每条经验包含:
        - state: 当前状态
        - action: 执行的动作
        - reward: 获得的奖励
        - new_state: 下一个状态
        - terminated: 是否结束
        """
        state_arr = np.asarray(state)
        new_state_arr = np.asarray(new_state)
        self.buffer.add(TensorDict({
            "state": torch.as_tensor(state_arr),
            "action": torch.as_tensor(action, dtype=torch.int64),
            "reward": torch.as_tensor(reward, dtype=torch.float32),
            "new_state": torch.as_tensor(new_state_arr),
            "terminated": torch.as_tensor(terminated, dtype=torch.bool),
        }, batch_size=[]))

    def get_samples(self, batch_size):
        """从回放缓冲区随机采样一批经验"""
        batch = self.buffer.sample(batch_size).to(self.device)
        states_t = batch.get('state')
        new_states_t = batch.get('new_state')
        if self.normalize_obs and states_t.dtype == torch.uint8:
            states = states_t.to(dtype=torch.float32).mul_(1.0 / 255.0)
        else:
            states = states_t.to(dtype=torch.float32)
        if self.normalize_obs and new_states_t.dtype == torch.uint8:
            new_states = new_states_t.to(dtype=torch.float32).mul_(1.0 / 255.0)
        else:
            new_states = new_states_t.to(dtype=torch.float32)
        actions = batch.get('action').to(dtype=torch.int64).view(-1)
        rewards = batch.get('reward').to(dtype=torch.float32).view(-1)
        terminateds = batch.get('terminated').to(dtype=torch.bool).view(-1)
        return states, actions, rewards, new_states, terminateds

    def take_action(self, state):
        """
        使用 ε-greedy 策略选择动作
        
        ε-greedy 策略:
        - 以概率 ε 选择随机动作（探索）
        - 以概率 1-ε 选择当前最优动作（利用）
        """
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_n)
        else:
            state_arr = np.asarray(state)
            state_tensor = torch.as_tensor(state_arr, device=self.device)
            if state_tensor.dtype != torch.float32:
                state_tensor = state_tensor.to(dtype=torch.float32)
            if self.normalize_obs and state_arr.dtype == np.uint8:
                state_tensor = state_tensor.mul_(1.0 / 255.0)
            state_tensor = state_tensor.unsqueeze(0)
            with torch.inference_mode():
                action_idx = int(self.policy_net(state_tensor).argmax(dim=1).item())
        
        if self.epsilon != 0:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min
            
        self.act_taken += 1
        return action_idx

    def update_net(self, batch_size):
        """
        更新神经网络 - 子类必须实现
        
        这是Q-Learning的核心
        """
        raise NotImplementedError

    def sync_target_net(self):
        """同步目标网络 - 将策略网络的权重复制到目标网络"""
        self.frozen_net.load_state_dict(self.policy_net.state_dict())

    def save(self, save_dir, filename):
        """保存模型到文件"""
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{filename}.pt")
        
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'frozen_net_state_dict': self.frozen_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'n_updates': self.n_updates,
            'config': self.config
        }, model_path)
        print(f"模型已保存到: {model_path}")

    def load(self, path):
        """从文件加载模型"""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.frozen_net.load_state_dict(checkpoint['frozen_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.n_updates = checkpoint['n_updates']
        print(f"模型已从 {path} 加载")

    def write_log(self, date_list, time_list, reward_list, length_list,
                  loss_list, epsilon_list, log_filename='log.csv', extra_rows=None):
        """将训练日志写入CSV文件"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        rows = [
            ['date'] + date_list,
            ['time'] + time_list,
            ['reward'] + reward_list,
            ['length'] + length_list,
            ['loss'] + loss_list,
            ['epsilon'] + epsilon_list
        ]
        if extra_rows:
            for key, values in extra_rows.items():
                rows.append([str(key)] + list(values))
        with open(os.path.join(self.log_dir, log_filename), 'w') as f:
            csv.writer(f).writerows(rows)


# ================================================================================
# 可视化工具
# ================================================================================

def plot_rewards(episode_num, reward_list, n_steps):
    """
    绘制训练奖励曲线
    
    功能:
    - 显示每个episode的原始奖励
    - 计算并显示50期移动平均线
    """
    plt.figure(1)
    rewards_tensor = torch.tensor(reward_list, dtype=torch.float)
    
    if len(rewards_tensor) >= 11:
        eval_reward = rewards_tensor[-10:]
        mean_reward = round(torch.mean(eval_reward).item(), 2)
        std_reward = round(torch.std(eval_reward).item(), 2)
        plt.clf()
        plt.title(f'Episode #{episode_num}: {n_steps} steps, '
                  f'reward {mean_reward}±{std_reward}')
    else:
        plt.clf()
        plt.title('Training...')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_tensor.numpy())
    
    if len(rewards_tensor) >= 50:
        reward_f = rewards_tensor[:50]
        means = rewards_tensor.unfold(0, 50, 1).mean(1)
        means = torch.cat((torch.ones(49) * torch.mean(reward_f), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)
