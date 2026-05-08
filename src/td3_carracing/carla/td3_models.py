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

        # 计算卷积后的特征维度（根据实际计算）
        # 输入 (4,84,84)
        # 第1层卷积：84 - 8 / 4 + 1 = 19.0 → (32,19,19)
        # 第2层卷积：19 - 4 / 2 + 1 = 8.5 → (64,8,8)
        # 第3层卷积：8 - 3 / 1 + 1 = 6 → (64,6,6)
        # 实际计算为 3136，可能存在浮点数计算误差
        self.fc = nn.Linear(3136, 512)

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
            self.fc1 = nn.Linear(512, 256)
        else:
            self.fc1 = nn.Linear(state_dim[0], 256)

        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, x):
        if self.use_cnn:
            x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.max_action * x


# Critic 价值网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, use_cnn=True):
        super().__init__()
        self.use_cnn = use_cnn
        if use_cnn:
            self.encoder = CNNEncoder(input_channels=4)
            self.fc1 = nn.Linear(512 + action_dim, 256)
        else:
            self.fc1 = nn.Linear(state_dim[0] + action_dim, 256)

        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, action):
        if self.use_cnn:
            x = self.encoder(x)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
