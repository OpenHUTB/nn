import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# CNN 图像编码器 - 优化以更好地提取转向和路径特征
class CNNEncoder(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, 2, stride=1)
        self.flatten = nn.Flatten()

        # 计算卷积后的特征维度
        # 输入 (4,84,84)
        # conv1(8,s4): (84-8)/4+1=20 → (32,20,20)
        # conv2(4,s2): (20-4)/2+1=9  → (64,9,9)
        # conv3(3,s1): (9-3)/1+1=7   → (64,7,7)
        # conv4(2,s1): (7-2)/1+1=6   → (128,6,6)
        self.fc = nn.Linear(128 * 6 * 6, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return x

# Actor 策略网络 - 优化转向动作输出
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, use_cnn=True):
        super().__init__()
        self.use_cnn = use_cnn
        if use_cnn:
            self.encoder = CNNEncoder(input_channels=4)
            self.fc1 = nn.Linear(512, 384)
        else:
            self.fc1 = nn.Linear(state_dim[0], 384)

        self.fc2 = nn.Linear(384, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, x):
        if self.use_cnn:
            x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))

        # 确保输出范围正确
        output = self.max_action * x
        # 确保油门和刹车非负
        output[..., 1] = torch.clamp(output[..., 1], 0.0, self.max_action[1] if isinstance(self.max_action, (list, tuple, np.ndarray)) else self.max_action)
        output[..., 2] = torch.clamp(output[..., 2], 0.0, self.max_action[2] if isinstance(self.max_action, (list, tuple, np.ndarray)) else self.max_action)
        return output

# Critic 价值网络 - 增强对转向动作的估值
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, use_cnn=True):
        super().__init__()
        self.use_cnn = use_cnn
        if use_cnn:
            self.encoder = CNNEncoder(input_channels=4)
            self.fc1 = nn.Linear(512 + action_dim, 384)
        else:
            self.fc1 = nn.Linear(state_dim[0] + action_dim, 384)

        self.fc2 = nn.Linear(384, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x, action):
        if self.use_cnn:
            x = self.encoder(x)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)