import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Q网络 - 标准全连接网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        # 使用更深的网络结构和层归一化
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.bn2 = nn.LayerNorm(hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """使用Xavier初始化权重，有助于训练稳定性"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        # 偏置初始化为较小的正值
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.constant_(self.fc2.bias, 0.01)
        nn.init.constant_(self.fc3.bias, 0.01)
        nn.init.constant_(self.fc4.bias, 0.01)

    def forward(self, x):
        # 使用LeakyReLU激活函数，避免神经元死亡
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        return self.fc4(x)


class DuelingQNetwork(nn.Module):
    """Dueling DQN 网络 (可选升级)"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingQNetwork, self).__init__()
        # 共享层
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # 价值流
        self.value_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value_out = nn.Linear(hidden_dim, 1)

        # 优势流
        self.advantage_fc = nn.Linear(hidden_dim, hidden_dim)
        self.advantage_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        value = F.relu(self.value_fc(x))
        value = self.value_out(value)

        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage)

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
