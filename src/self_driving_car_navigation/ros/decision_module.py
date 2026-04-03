# scripts/models/decision_module.py
import torch

class DecisionModule:
    def __init__(self, 
                 input_feature_dim=128,  # 输入感知特征维度（与感知模块输出一致）
                 output_cmd_dim=2):       # 输出控制指令维度（线速度x + 角速度z）
        """初始化决策模块"""
        self.feature_dim = input_feature_dim
        self.cmd_dim = output_cmd_dim
        
        # 示例：初始化一个简单的决策层（可替换为你的实际模型，如DRL策略网络）
        self.decision_layer = torch.nn.Sequential(
            torch.nn.Linear(input_feature_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_cmd_dim)
        )

        # 控制指令范围（示例：线速度0~2m/s，角速度-1~1rad/s）
        self.max_linear_vel = 2.0
        self.min_linear_vel = 0.0
        self.max_angular_vel = 1.0
        self.min_angular_vel = -1.0

    def get_control_cmd(self, fused_feature):
        """
        接收感知特征，输出控制指令
        参数：
            fused_feature: torch.Tensor - 感知融合特征，shape=(B, 128)
        返回：
            control_cmd: dict - 控制指令（线速度x、角速度z）
        """
        # 1. 决策层推理
        raw_cmd = self.decision_layer(fused_feature)  # (B, 2)

        # 2. 裁剪指令到合理范围
        linear_vel_x = torch.clamp(raw_cmd[:, 0], self.min_linear_vel, self.max_linear_vel).item()
        angular_vel_z = torch.clamp(raw_cmd[:, 1], self.min_angular_vel, self.max_angular_vel).item()

        # 3. 包装为字典（便于ROS消息转换）
        control_cmd = {
            "linear_x": linear_vel_x,
            "angular_z": angular_vel_z
        }
        return control_cmd