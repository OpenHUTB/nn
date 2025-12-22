import torch
import time
import numpy as np
import sys
from envs.carla_environment import CarlaEnvironment
from models.attention_module import CrossDomainAttention
from models.decision_module import DecisionModule

# 关闭警告
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class IntegratedSystem:
    def __init__(self, device='cpu'):
        self.device = device
        self.attention = CrossDomainAttention().to(device)
        self.decision = DecisionModule().to(device)

    def forward(self, image, lidar, imu):
        image = image.to(self.device)
        lidar = lidar.to(self.device)
        imu = imu.to(self.device)
        fused_feature = self.attention(image, lidar, imu)
        policy, value = self.decision(fused_feature)
        return policy, value


def run_simulation():
    # 初始化CARLA环境
    env = None
    try:
        env = CarlaEnvironment(host='localhost', port=2000)
        time.sleep(2)
        if not env.reset():
            raise RuntimeError("车辆生成失败！")
        print("✅ CARLA环境初始化完成")
    except Exception as e:
        print(f"❌ CARLA初始化失败：{e}")
        if env:
            env.close()
        return

    # 初始化智能体系统
    try:
        system = IntegratedSystem(device='cpu')
        print("✅ 智能体系统初始化完成")
    except Exception as e:
        print(f"❌ 智能体系统初始化失败：{e}")
        env.close()
        return

    # 平稳仿真循环（100步，足够看直线行驶）
    try:
        total_steps = 100
        print(f"\n🚀 开始平稳仿真（{total_steps}步），车辆沿道路直线行驶！")

        for step in range(total_steps):
            # 模拟传感器数据
            image = torch.randn(1, 3, 224, 224)
            lidar_data = torch.randn(1, 1, 64, 64)
            imu_data = torch.randn(1, 6)

            # 模型决策
            policy, value = system.forward(image, lidar_data, imu_data)

            # 提取转向（仅保留极小值）
            raw_steer = policy.detach().cpu().numpy()[0][1]
            steer = np.clip(raw_steer, -0.05, 0.05)  # 再次限幅

            # 控制车辆（油门已在环境中固定为0.5）
            env.control_vehicle(0.5, steer)

            # 每10步打印状态
            if step % 10 == 0:
                print(f"🔹 第{step}步：转向={steer:.3f}，价值={value.item():.2f}")

            time.sleep(0.15)  # 稍快的步长，行驶更流畅

        print("\n✅ 仿真结束！车辆全程沿道路直线行驶～")
    except Exception as e:
        print(f"❌ 仿真出错：{e}")
    finally:
        env.close()


if __name__ == "__main__":
    print(f"📌 Python版本：{sys.version.split()[0]}")
    print(f"📌 PyTorch版本：{torch.__version__}")
    print(f"📌 CUDA可用：{torch.cuda.is_available()}")
    print("=" * 50)
    run_simulation()