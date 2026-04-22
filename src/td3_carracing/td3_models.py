import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN 图像编码器
class CNNEncoder(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.flatten = nn.Flatten()
        # 计算卷积后的特征维度
        # 输入 (4,42,42) -> conv1: (32,9,9) -> conv2: (64,3,3) -> conv3: (64,1,1) -> 展平: 64
        self.fc = nn.Linear(64, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return x

# Actor 策略网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, use_cnn=True):
        super().__init__()
        self.use_cnn = use_cnn
        if use_cnn:
            self.encoder = CNNEncoder(input_channels=4)
            self.fc1 = nn.Linear(256, 256)
        else:
            self.fc1 = nn.Linear(state_dim[0], 256)

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        if self.use_cnn:
            x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        # 输出前再次限制转向动作
        x[:, 0] = x[:, 0] * 0.8  # 转向输出额外衰减20%
        return self.max_action * x

# Critic 价值网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, use_cnn=True):
        super().__init__()
        self.use_cnn = use_cnn
        if use_cnn:
            self.encoder = CNNEncoder(input_channels=4)
            self.fc1 = nn.Linear(256 + action_dim, 256)
        else:
            self.fc1 = nn.Linear(state_dim[0] + action_dim, 256)

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, action):
        if self.use_cnn:
            x = self.encoder(x)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)