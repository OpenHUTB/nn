import torch
import time
import numpy as np
from envs.carla_environment import CarlaEnvironment
from models.attention_module import CrossDomainAttention
from models.decision_module import DecisionModule

# 关闭Lazy modules警告
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class IntegratedSystem:
    def __init__(self, device='cpu'):
        self.device = device
        # 初始化注意力模块和决策模块
        self.attention = CrossDomainAttention().to(device)
        self.decision = DecisionModule().to(device)

    def forward(self, image, lidar, imu):
        """前向传播：融合特征并输出决策"""
        # 数据移到指定设备
        image = image.to(self.device)
        lidar = lidar.to(self.device)
        imu = imu.to(self.device)

        # 特征融合
        fused_feature = self.attention(image, lidar, imu)
        # 决策输出
        policy, value = self.decision(fused_feature)
        return policy, value


def run_simulation():
    # 1. 初始化CARLA环境
    env = None
    try:
        env = CarlaEnvironment(host='localhost', port=2000)
        time.sleep(2)  # 等待模拟器加载
        if not env.reset():
            raise RuntimeError("车辆生成失败！")
        print("✅ CARLA环境初始化完成，车辆已生成")
    except Exception as e:
        print(f"❌ CARLA初始化失败：{e}")
        if env:
            env.close()
        return

    # 2. 初始化智能体系统
    try:
        system = IntegratedSystem(device='cpu')
        print("✅ 智能体系统初始化完成")
    except Exception as e:
        print(f"❌ 智能体系统初始化失败：{e}")
        env.close()
        return

    # 3. 持续仿真循环（运行100步，足够看到车辆行驶）
    try:
        total_steps = 100  # 延长到100步，车辆行驶更久
        print(f"\n🚀 开始仿真（{total_steps}步），请查看CARLA窗口！")

        for step in range(total_steps):
            # 模拟传感器数据
            image = torch.randn(1, 3, 224, 224)
            lidar_data = torch.randn(1, 1, 64, 64)
            imu_data = torch.randn(1, 6)

            # 前向计算
            policy, value = system.forward(image, lidar_data, imu_data)

            # 固定油门0.6（车辆明显行驶），小幅转向
            throttle = 0.6
            steer = np.clip(policy.detach().cpu().numpy()[0][1], -0.1, 0.1)

            # 控制车辆
            env.control_vehicle(throttle, steer)

            # 每10步打印状态
            if step % 10 == 0:
                print(f"🔹 第{step}步：油门={throttle:.2f}，转向={steer:.2f}，价值={value.item():.2f}")

            time.sleep(0.1)  # 仿真步长

        print("\n✅ 持续仿真结束！")
    except Exception as e:
        print(f"❌ 仿真出错：{e}")
    finally:
        # 4. 清理环境
        env.close()
        print("✅ 仿真结束，CARLA环境已清理")


if __name__ == "__main__":
    # 修复：替换不存在的 torch.version.python，改用 sys 模块获取Python版本
    import sys

    print(f"📌 Python版本：{sys.version.split()[0]}")
    print(f"📌 PyTorch版本：{torch.__version__}")
    print(f"📌 CUDA可用：{torch.cuda.is_available()}")
    print("=" * 50)

    # 运行仿真
    run_simulation()