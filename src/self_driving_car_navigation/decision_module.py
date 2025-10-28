import torch
import torch.nn as nn

class DecisionModule(nn.Module):
    def __init__(self):
        super(DecisionModule, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出：转向角和油门
        )
        self.value_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 值函数
        )

    def forward(self, fused_features):
        aggregated_features = fused_features.mean(dim=1)
        policy = self.policy_net(fused_features)
        value = self.value_net(fused_features)
        return policy, value
