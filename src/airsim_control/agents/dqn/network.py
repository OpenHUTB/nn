import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Q网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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
