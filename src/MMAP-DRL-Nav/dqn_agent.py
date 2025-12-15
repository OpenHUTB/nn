import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque  # 高效内存操作
from .pruning import prune_model
from .quantization import quantize_model

class DQNAgent:
    def __init__(self, state_shape, action_size, config):
        """
        初始化DQN智能体（适配CARLA图像输入，修复卷积维度计算错误）
        :param state_shape: 图像形状 (128, 128, 3)
        :param action_size: 动作维度（4：前进/左转/右转/后退）
        :param config: 配置字典
        """
        self.state_shape = state_shape  # (128, 128, 3)
        self.action_size = action_size
        self.memory = deque(maxlen=config.get('agent', {}).get('memory_capacity', 10000))  # 经验池
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = config['agent']['epsilon_decay']  # 探索率衰减
        self.epsilon_min = config['agent']['epsilon_min']      # 最小探索率
        self.learning_rate = config['train']['learning_rate']  # 学习率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备
        
        # 构建CNN模型（自适应池化解决维度计算错误）并移到指定设备
        self.model = self._build_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 优化器
        self.loss_fn = nn.MSELoss()  # 损失函数

        # 模型输入输出校验（新增：验证维度匹配）
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
        """验证模型输入输出维度是否匹配（新增）"""
        try:
            # 构建测试输入：(batch, 3, H, W)
            dummy_input = torch.randn(1, 3, self.state_shape[0], self.state_shape[1]).to(self.device)
            with torch.no_grad():
                dummy_output = self.model(dummy_input)
            print(f"✅ 模型维度校验通过 | 输入维度：{dummy_input.shape} | 输出维度：{dummy_output.shape}")
        except Exception as e:
            raise ValueError(f"❌ 模型维度校验失败：{e}")

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

    def replay(self, batch_size):
        """批量经验回放（GPU加速，替换逐样本更新）"""
        if len(self.memory) < batch_size:
            return
        
        # 随机采样批次经验
        minibatch = random.sample(self.memory, batch_size)
        # 拆分批次数据
        states = np.array([exp[0] for exp in minibatch])  # (batch, 128, 128, 3)
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])  # (batch, 128, 128, 3)
        dones = np.array([exp[4] for exp in minibatch])

        # 维度转换：HWC → CHW + 移到设备 + 归一化
        states_tensor = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device) / 255.0  # (batch, 3, 128, 128)
        next_states_tensor = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(self.device) / 255.0
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # 计算当前Q值（仅选执行动作的Q值）
        current_q = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值（Bellman方程）
        with torch.no_grad():
            next_q = self.model(next_states_tensor).max(1)[0]  # 下一个状态的最大Q值
            target_q = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)  # 目标Q值

        # 梯度下降更新
        self.optimizer.zero_grad()
        loss = self.loss_fn(current_q, target_q)
        loss.backward()
        self.optimizer.step()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
