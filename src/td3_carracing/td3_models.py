import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN 图像编码器（修复BatchNorm问题）
class CNNEncoder(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()

        # 计算正确的展平维度
        self._init_weights()

    def _init_weights(self):
        # 用一个虚拟输入计算展平维度
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 42, 42)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            self.flatten_dim = x.view(1, -1).shape[1]

        self.fc = nn.Linear(self.flatten_dim, 256)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, training=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        if training:
            x = self.dropout(x)
        return x


# Actor 策略网络（改进版，使用更平滑的激活）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, use_cnn=True):
        super().__init__()
        self.use_cnn = use_cnn
        if use_cnn:
            self.encoder = CNNEncoder(input_channels=4)
            self.fc1 = nn.Linear(256, 512)
        else:
            self.fc1 = nn.Linear(state_dim[0], 512)

        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

        # 添加LayerNorm使训练更稳定（LayerNorm不需要batch维度）
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(256)

    def forward(self, x):
        if self.use_cnn:
            x = self.encoder(x, training=self.training)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))

        # 使用tanh但添加温度参数使输出更平滑
        x = torch.tanh(self.fc3(x) * 0.8)  # 降低温度使动作变化更平滑
        return self.max_action * x


# Critic 价值网络（改进版）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, use_cnn=True):
        super().__init__()
        self.use_cnn = use_cnn
        if use_cnn:
            self.encoder = CNNEncoder(input_channels=4)
            self.fc1 = nn.Linear(256 + action_dim, 512)
        else:
            self.fc1 = nn.Linear(state_dim[0] + action_dim, 512)

        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        # 添加LayerNorm
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(256)

    def forward(self, x, action):
        if self.use_cnn:
            x = self.encoder(x, training=self.training)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)