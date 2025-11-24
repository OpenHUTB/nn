import torch
import time  # 补充time模块导入，用于sleep延迟
from models.perception_module import PerceptionModule
from models.attention_module import CrossDomainAttention
from models.decision_module import DecisionModule
from models.dqn_agent import DQNAgent
from envs.carla_environment import CarlaEnvironment  # 类名是CarlaEnvironment
import carla

class IntegratedSystem:
    def __init__(self, device='cpu'):
        self.device = device
        self.perception = PerceptionModule().to(self.device)
        self.attention = CrossDomainAttention(num_blocks=6).to(self.device)
        self.decision = DecisionModule().to(self.device)

    def forward(self, image, lidar_data, imu_data):
        # 注意：确保perception的输入参数顺序与定义一致
        scene_info, segmentation, odometry, obstacles, boundary = self.perception(imu_data, image, lidar_data)
        fused_features = self.attention(scene_info, segmentation, odometry, obstacles, boundary)
        policy, value = self.decision(fused_features)
        return policy, value

def run_simulation():
    # 实例化CARLA环境（使用正确的类名）
    env = CarlaEnvironment()  
    # 确保环境初始化成功（例如连接CARLA服务器、生成车辆等）
    # 建议添加检查：if not env.initialized: raise Exception("CARLA环境初始化失败")
    
    # 初始化集成系统
    system = IntegratedSystem(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    try:  # 使用try-finally确保环境正确关闭
        for _ in range(100):  # 运行100个模拟步骤
            # 注意：实际中应从env获取真实数据，而非随机生成
            image = torch.randn(3, 256, 256).unsqueeze(0).to(system.device)  
            lidar_data = torch.randn(1, 256, 256).unsqueeze(0).to(system.device)
            imu_data = torch.randn(1, 6).to(system.device)

            # 推理得到策略
            policy, value = system.forward(image, lidar_data, imu_data)
            
            # 将策略转换为CARLA控制信号（确保policy的维度正确）
            # 假设policy[0][0]是油门，policy[0][1]是转向角
            control = carla.VehicleControl(
                throttle=float(policy[0][0].clamp(0, 1)),  # 油门范围[0,1]
                steer=float(policy[0][1].clamp(-1, 1))     # 转向角范围[-1,1]
            )
            env.vehicle.apply_control(control)  # 应用控制信号

            time.sleep(0.1)  # 模拟物理时间步长
    finally:
        # 确保仿真结束后清理环境（关闭连接、销毁 Actors等）
        env.cleanup()  # 假设CarlaEnvironment类有cleanup方法

if __name__ == "__main__":
    run_simulation()
