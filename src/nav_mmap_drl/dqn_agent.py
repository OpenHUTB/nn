import os
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from .pruning import prune_model
from .quantization import quantize_model


class DQNAgent:
    def __init__(self, state_shape, action_size, config):
        """
        初始化DQN智能体（含Target Network，适配CARLA图像输入）

        Args:
            state_shape: 图像形状 (128, 128, 3)
            action_size: 动作维度（4：前进/左转/右转/后退）
            config: 配置字典
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── 超参数（从config读取，提供默认值） ──
        agent_cfg = config.get('agent', {})
        train_cfg = config.get('train', {})

        self.memory = deque(maxlen=agent_cfg.get('memory_capacity', 10000))
        self.gamma = agent_cfg.get('gamma', 0.95)
        self.epsilon = agent_cfg.get('epsilon_start', 1.0)
        self.epsilon_decay = agent_cfg['epsilon_decay']
        self.epsilon_min = agent_cfg['epsilon_min']
        self.learning_rate = train_cfg['learning_rate']
        self.batch_size = train_cfg.get('batch_size', 64)

        # Target Network 更新频率
        self.target_update_freq = train_cfg.get('target_update_freq', 100)
        self.train_step_counter = 0

        # ── 构建双网络（Online + Target） ──
        self.model = self._build_model().to(self.device)           # Online Q-network
        self.target_model = self._build_model().to(self.device)    # Target Q-network
        self.update_target_model()  # 初始同步

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # 模型保存路径
        self.model_dir = config.get('model_dir', './models')
        os.makedirs(self.model_dir, exist_ok=True)

        self._validate_model_dim()

    def _build_model(self):
        """构建适配128×128×3图像的CNN模型（自适应池化，无需手动计算卷积维度）"""
        return nn.Sequential(
            # 卷积层1：提取低级视觉特征 (3,128,128) → (32,31,31)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=1),
            nn.ReLU(),
            # 卷积层2：提取中级特征 (32,31,31) → (64,15,15)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 卷积层3：提取高级特征 (64,15,15) → (64,15,15)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 自适应池化：强制输出8×8，彻底避免维度计算错误 (64,15,15) → (64,8,8)
            nn.AdaptiveAvgPool2d((8, 8)),
            # 展平特征图（64*8*8=4096，固定维度）
            nn.Flatten(),
            # 全连接层：映射到动作空间
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_size)  # 输出4个动作的Q值
        )

    def _validate_model_dim(self):
        """模型维度校验（可选，用于调试）"""
        try:
            import torch
            dummy_input = torch.randn(1, 3, self.state_shape[0], self.state_shape[1]).to(self.device)
            with torch.no_grad():
                dummy_output = self.model(dummy_input)
            print(f"模型维度校验通过 | 输入: {dummy_input.shape} | 输出: {dummy_output.shape}")
        except Exception as e:
            raise ValueError(f"模型维度校验失败: {e}")

    # ── 网络管理 ──────────────────────────────────────────

    def update_target_model(self):
        """将 Online Network 的权重复制到 Target Network（硬更新）"""
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, filename: str = "dqn_model.pth"):
        """保存模型权重"""
        path = os.path.join(self.model_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step_counter,
        }, path)
        print(f"模型已保存至: {path}")

    def load_model(self, filename: str = "dqn_model.pth"):
        """加载模型权重"""
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            return False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.train_step_counter = checkpoint.get('train_step', 0)
        print(f"模型已从 {path} 加载 (epsilon={self.epsilon:.4f}, step={self.train_step_counter})")
        return True

    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放池（标准化数据格式）"""
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """ε-贪心策略选择动作（适配图像输入）"""
        # 探索阶段：随机选动作
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        # 利用阶段：模型预测最优动作
        # 维度转换：HWC(128,128,3) → CHW(3,128,128) + batch维度 + 归一化
        state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        # 模型推理（无梯度）
        with torch.no_grad():
            q_values = self.model(state_tensor)
        # 返回Q值最大的动作
        return np.argmax(q_values.cpu().detach().numpy()[0])

    def replay(self, batch_size=None):
        """批量经验回放（使用Target Network稳定训练）"""
        batch_size = batch_size or self.batch_size
        if len(self.memory) < batch_size:
            return

        # 随机采样
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])

        # 维度转换：HWC → CHW + 归一化
        states_tensor = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device) / 255.0
        next_states_tensor = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(self.device) / 255.0
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # 当前 Q 值（Online Network）
        current_q = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # 目标 Q 值（使用 Target Network 计算，减少估计偏差）
        with torch.no_grad():
            next_q = self.target_model(next_states_tensor).max(1)[0]
            target_q = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)

        # 梯度更新
        self.optimizer.zero_grad()
        loss = self.loss_fn(current_q, target_q)
        loss.backward()
        self.optimizer.step()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 定期同步 Target Network
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_model()

    def calculate_reward(self, current_position, target_position, road_position, done):
        """奖励函数（适配CARLA环境）"""
        distance_to_target = np.linalg.norm(current_position - target_position)
        distance_to_road = np.linalg.norm(current_position - road_position)

        if done:
            return 100.0
        elif distance_to_target < 1.0:
            return 10.0
        elif distance_to_road > 1.0:
            return -5.0
        elif distance_to_target < 5.0:
            return 1.0
        else:
            return -1.0

    def get_state(self, position, orientation, target_position, road_position):
        """备用：低维状态提取（实际训练用图像）"""
        state = np.array([
            position[0], position[1], orientation,
            target_position[0], target_position[1],
            road_position[0], road_position[1]
        ], dtype=np.float32)
        return state
