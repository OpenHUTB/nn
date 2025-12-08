import torch
import time
import sys
import math
from models.perception_module import PerceptionModule
from models.attention_module import CrossDomainAttention
from models.decision_module import DecisionModule
from _agent.carla_environment import CarlaEnvironment
import carla

print("="*60)
print(f"[启动时间] {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[Python解释器] {sys.executable}")
print(f"[虚拟环境] {sys.prefix}")
print("="*60)
sys.stdout.flush()

class IntegratedSystem:
    def __init__(self, device='cpu'):
        print("\n[模型初始化] 开始加载感知、注意力和决策模块...")
        sys.stdout.flush()
        self.device = device
        try:
            self.perception = PerceptionModule().to(self.device)
            self.attention = CrossDomainAttention(num_blocks=6).to(self.device)
            self.decision = DecisionModule().to(self.device)
            print("[模型初始化] 所有模块加载完成")
            sys.stdout.flush()
        except Exception as e:
            print(f"[模型初始化失败] {str(e)}")
            sys.stdout.flush()
            raise

    def forward(self, image, lidar_data, imu_data):
        scene_info, segmentation, odometry, obstacles, boundary = self.perception(imu_data, image, lidar_data)
        fused_features = self.attention(scene_info, segmentation, odometry, obstacles, boundary)
        policy, value = self.decision(fused_features)
        return torch.mean(policy, dim=1), value

def run_simulation():
    env = None
    try:
        print("\n[CARLA连接] 开始创建环境...")
        sys.stdout.flush()
        env = CarlaEnvironment()
        
        print("\n[环境重置] 开始生成车辆和传感器...")
        sys.stdout.flush()
        env.reset()
        
        if not env.vehicle or not env.vehicle.is_alive:
            raise RuntimeError("车辆生成失败！请重启CARLA或更换场景（如Town03）")
        print(f"[车辆状态] 生成成功（ID: {env.vehicle.id}）")
        sys.stdout.flush()

        # 获取CARLA可视化镜头控制器
        spectator = env.world.get_spectator()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n[设备信息] 模型运行在 {device} 上")
        sys.stdout.flush()
        system = IntegratedSystem(device=device)


        # 新增：转向平滑参数（减少晃动）
        steer_buffer = []  # 存储最近3步转向值
        smooth_window = 3  # 滑动平均窗口
        last_steer = 0.0  # 上一步转向值
        max_steer_delta = 0.03  # 转向最大变化量（限制高频波动）

        print("\n[仿真开始] 共运行100步（车辆将明显移动）...")

        sys.stdout.flush()
        for step in range(100):
            # 获取摄像头观测
            observation = env.get_observation()
            if observation is None or observation.size == 0:
                print(f"[警告] 第{step+1}步未获取到图像数据")
                sys.stdout.flush()

            # 转换为模型输入格式
            image = torch.from_numpy(observation.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            lidar_data = torch.randn(1, 256, 256).unsqueeze(0).to(device)
            imu_data = torch.randn(1, 6).to(device)

            # 模型推理
            policy, _ = system.forward(image, lidar_data, imu_data)


            # 关键调整1：放大油门值（确保车辆明显移动，最低0.4）
            raw_throttle = float(policy[0][0].clamp(-0.3, 0.8))  # 原始油门
            throttle = max(0.4, min(1.0, raw_throttle * 5))  # 放大5倍，最低0.4

            # 关键调整2：平滑转向（减少轮胎晃动）
            raw_steer = float(policy[0][1].clamp(-0.5, 0.5))  # 原始转向
            # 滑动平均+限制变化速率
            steer_buffer.append(raw_steer)
            if len(steer_buffer) > smooth_window:
                steer_buffer.pop(0)
            smooth_steer = sum(steer_buffer) / len(steer_buffer)  # 平均
            # 限制与上一步的差值
            delta = smooth_steer - last_steer
            delta = max(-max_steer_delta, min(max_steer_delta, delta))
            final_steer = last_steer + delta
            final_steer = max(-1.0, min(1.0, final_steer))  # 最终限制
            last_steer = final_steer  # 更新历史值


            # 执行控制指令
            control = carla.VehicleControl(throttle=throttle, steer=final_steer)
            env.vehicle.apply_control(control)


            # 打印详细控制参数

            print(f"[步骤 {step+1}/100] 油门: {throttle:.2f} | 转向: {final_steer:.2f}")
            sys.stdout.flush()
            time.sleep(0.1)  # 控制仿真速度

        print("\n[仿真结束] 已完成100步运行")
        sys.stdout.flush()

    except Exception as e:
        print(f"\n[仿真错误] {str(e)}")
        sys.stdout.flush()
    finally:
        if env is not None:
            print("\n[资源清理] 正在销毁车辆和传感器...")
            sys.stdout.flush()
            env.close()
        print("\n[程序退出] 所有操作已完成")
        sys.stdout.flush()

if __name__ == "__main__":
    run_simulation()