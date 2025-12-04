import torch
import time
import sys
import math
import carla
from models.perception_module import PerceptionModule
from models.attention_module import CrossDomainAttention
from models.decision_module import DecisionModule
from _agent.carla_environment import CarlaEnvironment

print("="*60)
print(f"[启动时间] {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[Python解释器] {sys.executable}")
print("="*60)
sys.stdout.flush()

class IntegratedSystem:
    def __init__(self, device='cpu'):
        print("\n[模型初始化] 加载感知、注意力和决策模块...")
        self.device = device
        self.perception = PerceptionModule().to(device)
        self.attention = CrossDomainAttention(num_blocks=6).to(device)
        self.decision = DecisionModule().to(device)
        print("[模型初始化] 完成")

    def forward(self, image, lidar_data, imu_data):
        scene_info, segmentation, odometry, obstacles, boundary = self.perception(imu_data, image, lidar_data)
        fused_features = self.attention(scene_info, segmentation, odometry, obstacles, boundary)
        policy, value = self.decision(fused_features)
        return torch.mean(policy, dim=1), value

def apply_urgent_avoidance(obstacle_distances):
    """分级避障+逃生策略"""
    # 1. 前方碰撞危险（≤1.5米）→ 紧急刹车
    if obstacle_distances['front'] <= 1.5:
        return (0.0, 0.0, 1.0)
    
    # 2. 侧方碰撞危险（≤1.2米）→ 微调避让
    if obstacle_distances['left'] <= 1.2 or obstacle_distances['right'] <= 1.2:
        if obstacle_distances['left'] > obstacle_distances['right']:
            return (0.1, 0.3, 0.0)  # 左微调
        else:
            return (0.1, -0.3, 0.0)  # 右微调
    
    # 3. 前方近距离（≤3米）→ 转向绕开
    if obstacle_distances['front'] <= 3.0:
        if obstacle_distances['left'] > obstacle_distances['right']:
            return (0.2, 0.4, 0.0)  # 左转向
        else:
            return (0.2, -0.4, 0.0)  # 右转向
    
    # 4. 后方过近（≤1.5米）→ 后退逃生
    if obstacle_distances['rear'] <= 1.5:
        if obstacle_distances['left'] > 2.0:
            return (-0.2, 0.3, 0.0)  # 左转向+后退
        elif obstacle_distances['right'] > 2.0:
            return (-0.2, -0.3, 0.0)  # 右转向+后退
        else:
            return (-0.2, 0.0, 0.0)  # 直接后退
    
    return None  # 无紧急情况

def run_simulation():
    env = None
    try:
        print("\n[CARLA连接] 创建环境...")
        env = CarlaEnvironment()
        
        print("\n[环境重置] 生成车辆和传感器...")
        env.reset()
        
        if not env.vehicle or not env.vehicle.is_alive:
            raise RuntimeError("车辆生成失败，请检查CARLA是否正常运行")
        print(f"[车辆状态] 生成成功（ID: {env.vehicle.id}）")

        # 镜头跟随
        spectator = env.world.get_spectator()

        # 模型设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n[设备信息] 模型运行在 {device} 上")
        system = IntegratedSystem(device=device)

        # 控制参数
        max_forward_throttle = 0.5
        max_backward_throttle = -0.3
        max_steer = 0.6
        last_steer = 0.0
        steer_smooth_factor = 0.2

        print("\n[仿真开始] 共运行100步...")
        sys.stdout.flush()

        for step in range(200):
            # 获取传感器数据SZ
            observation = env.get_observation()
            image = observation['image']
            lidar_distances = observation['lidar_distances']
            imu_data = observation['imu']

            # 障碍物检测
            obstacle_distances = env.get_obstacle_directions(lidar_distances)
            print(f"\n[障碍物] 前{obstacle_distances['front']:.1f}m | 后{obstacle_distances['rear']:.1f}m | "
                  f"左{obstacle_distances['left']:.1f}m | 右{obstacle_distances['right']:.1f}m")
            sys.stdout.flush()

            # 避障策略
            avoid_action = apply_urgent_avoidance(obstacle_distances)
            if avoid_action is not None:
                throttle, steer, brake = avoid_action
                print(f"[策略执行] 油门={throttle:.2f}, 转向={steer:.2f}, 刹车={brake:.2f}")
            else:
                # 模型控制
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                lidar_tensor = torch.from_numpy(lidar_distances).unsqueeze(0).unsqueeze(0).float().to(device)
                imu_tensor = torch.from_numpy(imu_data).unsqueeze(0).float().to(device)

                with torch.no_grad():
                    policy, _ = system.forward(image_tensor, lidar_tensor, imu_tensor)

                throttle = float(policy[0][0].clamp(max_backward_throttle, max_forward_throttle))
                raw_steer = float(policy[0][1].clamp(-max_steer, max_steer))
                brake = 0.0

                # 平滑转向
                steer = last_steer * (1 - steer_smooth_factor) + raw_steer * steer_smooth_factor
                last_steer = steer

            # 执行控制
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=False
            )
            env.vehicle.apply_control(control)

            # 镜头跟随设置
            vehicle_transform = env.vehicle.get_transform()
            cam_loc = vehicle_transform.transform(carla.Location(x=-5.0, z=2.0))
            dir_x = vehicle_transform.location.x - cam_loc.x
            dir_y = vehicle_transform.location.y - cam_loc.y
            yaw = math.atan2(dir_y, dir_x) * 180 / math.pi
            spectator.set_transform(carla.Transform(cam_loc, carla.Rotation(pitch=-10, yaw=yaw)))

            print(f"[步骤 {step+1}/00] 油门: {throttle:.2f} | 转向: {steer:.2f} | 刹车: {brake:.2f}")
            sys.stdout.flush()
            time.sleep(0.1)

        print("\n[仿真结束]")
        sys.stdout.flush()

    except Exception as e:
        print(f"\n[仿真错误] {str(e)}")
        sys.stdout.flush()
    finally:
        if env is not None:
            print("\n[资源清理] 销毁资源...")
            env.close()
        print("\n[程序退出]")

if __name__ == "__main__":
    run_simulation()