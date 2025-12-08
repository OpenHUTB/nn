# 1. 导入模块（放在最开头）
import torch
import time
from models.perception_module import PerceptionModule
from models.attention_module import CrossDomainAttention
from models.decision_module import DecisionModule
from models.dqn_agent import DQNAgent
from envs.carla_environment import CarlaEnvironment
import carla

# 2. 定义 IntegratedSystem 类
class IntegratedSystem:
    def __init__(self, device='cpu'):
        self.device = device
        self.perception = PerceptionModule().to(self.device)
        # 补充 input_dims 参数（与感知模块输出维度匹配）
        self.attention = CrossDomainAttention(
            num_blocks=6,
            input_dims=[256, 256, 6, 256, 256]
        ).to(self.device)
        self.decision = DecisionModule().to(self.device)

    def forward(self, image, lidar_data, imu_data):
        scene_info, segmentation, odometry, obstacles, boundary = self.perception(imu_data, image, lidar_data)
        fused_features = self.attention(scene_info, segmentation, odometry, obstacles, boundary)
        policy, value = self.decision(fused_features)
        return policy, value

# 3. 定义 run_simulation 函数
def run_simulation():
    env = CarlaEnvironment()  # 初始化CARLA环境
    system = IntegratedSystem(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    for _ in range(100):  # 运行100步仿真
        # 生成随机传感器数据（实际中应从env获取真实数据）
        image = torch.randn(3, 256, 256).unsqueeze(0).to(system.device)  
        lidar_data = torch.randn(1, 256, 256).unsqueeze(0).to(system.device)
        imu_data = torch.randn(1, 6).to(system.device)

        # 前向传播得到策略
        policy, value = system.forward(image, lidar_data, imu_data)
        
        # 转换为CARLA控制信号（限制范围避免异常）
        throttle = float(torch.clamp(policy[0][0], 0, 1))  # 油门范围[0,1]
        steer = float(torch.clamp(policy[0][1], -1, 1))    # 转向范围[-1,1]
        control = carla.VehicleControl(throttle=throttle, steer=steer)
        env.vehicle.apply_control(control)

        time.sleep(0.1)  # 模拟时间间隔

# 4. 程序入口（放在最后）
if __name__ == "__main__":
    run_simulation()