import torch
import time
import carla  # CARLA官方Python API
import numpy as np


# 感知模块（保持原逻辑）
class PerceptionModule(torch.nn.Module):
    def forward(self, imu_data, image, lidar_data):
        batch_size = image.shape[0]
        scene_info = torch.randn(batch_size, 128).to(image.device)
        segmentation = torch.randn(batch_size, 64, 256, 256).to(image.device)
        odometry = torch.randn(batch_size, 32).to(image.device)
        obstacles = torch.randn(batch_size, 64).to(image.device)
        boundary = torch.randn(batch_size, 32).to(image.device)
        return scene_info, segmentation, odometry, obstacles, boundary


# 跨域注意力模块（保持原逻辑）
class CrossDomainAttention(torch.nn.Module):
    def __init__(self, num_blocks=6):
        super().__init__()
        self.num_blocks = num_blocks
        input_dim = 128 + 64 * 256 * 256 + 32 + 64 + 32
        self.fc = torch.nn.Linear(input_dim, 256)

    def forward(self, scene_info, segmentation, odometry, obstacles, boundary):
        seg_flat = segmentation.flatten(1)
        all_features = torch.cat([scene_info, seg_flat, odometry, obstacles, boundary], dim=1)
        fused = self.fc(all_features)
        return fused


# 决策模块（道路约束，适配0.9.11）
class DecisionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.steer_fc = torch.nn.Linear(256, 1)  # 转向输出
        self.throttle_fc = torch.nn.Linear(256, 1)  # 油门输出

    def forward(self, fused_features, target_steer):
        # 转向：向目标转向角靠拢，范围[-1,1]
        steer = torch.nn.functional.tanh(self.steer_fc(fused_features) + target_steer)
        # 油门：限制在[0.2, 0.5]，避免过快/过慢
        throttle = torch.nn.functional.sigmoid(self.throttle_fc(fused_features)) * 0.3 + 0.2

        policy = torch.cat([throttle, steer], dim=1)
        value = torch.randn(fused_features.shape[0], 1)
        return policy, value


# CARLA环境类（移除get_navigation，适配0.9.11）
class CarlaEnvironment:
    def __init__(self):
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.vehicle = None
        self.spectator = None
        self.collision_sensor = None  # 碰撞传感器
        self.collision_occurred = False  # 碰撞标记
        self._connect_carla()
        self._spawn_vehicle()
        self._init_collision_sensor()  # 初始化碰撞检测
        self._set_vehicle_view()

    def _connect_carla(self):
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(15.0)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            self.spectator = self.world.get_spectator()
            print("✅ CARLA服务器连接成功！")
        except Exception as e:
            raise RuntimeError(
                f"❌ 连接CARLA失败！请确认：\n1. CarlaUE4.exe已启动\n2. 端口2000未被占用\n错误详情：{e}"
            )

    def _spawn_vehicle(self):
        try:
            # 清理残留车辆/传感器
            for actor in self.world.get_actors().filter('*vehicle*'):
                actor.destroy()
            for actor in self.world.get_actors().filter('*sensor*'):
                actor.destroy()

            vehicle_bp = self.blueprint_library.filter('model3')[0]
            spawn_points = self.world.get_map().get_spawn_points()
            # 选开阔直道的生成点（减少初始碰撞）
            spawn_point = spawn_points[20] if len(spawn_points) >= 20 else spawn_points[0]
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            # 初始化车辆状态：刹车、空挡
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0, gear=1))
            print(f"✅ 车辆生成成功！生成点位置：x={spawn_point.location.x:.1f}, y={spawn_point.location.y:.1f}")
        except Exception as e:
            raise RuntimeError(f"❌ 车辆生成失败：{e}")

    def _init_collision_sensor(self):
        """初始化碰撞传感器（0.9.11兼容）"""
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        # 碰撞回调函数：撞障后标记并减速
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        print("✅ 碰撞传感器初始化完成")

    def _on_collision(self, event):
        """碰撞发生时的处理"""
        if not self.collision_occurred:
            self.collision_occurred = True
            print("⚠️ 检测到碰撞！立即减速并调整方向")
            # 撞障后先刹车
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
            time.sleep(0.5)

    def get_target_steer(self):
        """
        适配0.9.11的道路转向逻辑：
        1. 用get_waypoint获取当前道路点
        2. 计算朝向路点的转向角（无navigation接口的替代方案）
        """
        if self.collision_occurred:
            # 撞障后反向微调，避开障碍物
            self.collision_occurred = False
            return torch.tensor([[0.3]], dtype=torch.float32)  # 小幅向右调

        # 核心：0.9.11兼容的路点获取方式
        vehicle_location = self.vehicle.get_transform().location
        # project_to_road=True：将车辆位置投影到最近的道路上
        current_waypoint = self.world.get_map().get_waypoint(
            vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        # 获取前方8米的路点（0.9.11支持next()接口）
        next_waypoint = current_waypoint.next(8.0)[0]

        # 计算车辆到下一个路点的转向误差
        vehicle_transform = self.vehicle.get_transform()
        # 车辆当前前进方向的向量
        vehicle_forward = vehicle_transform.get_forward_vector()
        # 车辆到下一个路点的方向向量
        direction_to_next = next_waypoint.transform.location - vehicle_location
        # 归一化向量
        vehicle_forward = np.array([vehicle_forward.x, vehicle_forward.y])
        direction_to_next = np.array([direction_to_next.x, direction_to_next.y])
        vehicle_forward = vehicle_forward / np.linalg.norm(vehicle_forward)
        direction_to_next = direction_to_next / np.linalg.norm(direction_to_next)

        # 计算夹角（转向误差），归一化到[-1,1]
        dot_product = np.dot(vehicle_forward, direction_to_next)
        cross_product = np.cross(vehicle_forward, direction_to_next)
        steer_error = np.arcsin(cross_product) / np.pi  # 弧度转[-0.5,0.5]，再放大到[-1,1]
        steer_error = np.clip(steer_error * 2, -1.0, 1.0)

        return torch.tensor([[steer_error]], dtype=torch.float32)

    def _set_vehicle_view(self):
        if self.vehicle and self.spectator:
            transform = self.vehicle.get_transform()
            spectator_transform = carla.Transform(
                transform.location + carla.Location(x=-5, z=2),
                transform.rotation
            )
            self.spectator.set_transform(spectator_transform)
            print("✅ 视角已切换到车辆后方！")
            print("   🎮 WASD：移动视角 | 鼠标右键+拖动：旋转视角 | 滚轮：缩放 | P：快速定位到车辆")

    def cleanup(self):
        try:
            if self.collision_sensor:
                self.collision_sensor.destroy()
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                time.sleep(0.5)
                self.vehicle.destroy()
            print("✅ 资源已清理")
        except Exception as e:
            print(f"⚠️ 清理资源时警告：{e}")


# 集成系统（适配新的决策模块）
class IntegratedSystem:
    def __init__(self, device='cpu'):
        self.device = device
        self.perception = PerceptionModule().to(self.device)
        self.attention = CrossDomainAttention(num_blocks=6).to(self.device)
        self.decision = DecisionModule().to(self.device)

    def forward(self, image, lidar_data, imu_data, target_steer):
        """新增target_steer参数，传递道路约束"""
        scene_info, segmentation, odometry, obstacles, boundary = self.perception(imu_data, image, lidar_data)
        fused_features = self.attention(scene_info, segmentation, odometry, obstacles, boundary)
        policy, value = self.decision(fused_features, target_steer.to(self.device))
        return policy, value


# 主仿真函数（道路行驶逻辑）
def run_simulation():
    env = None
    try:
        print("📢 运行前请确认：CarlaUE4.exe已启动（版本0.9.11）")
        time.sleep(2)

        env = CarlaEnvironment()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ 使用计算设备: {device}")
        system = IntegratedSystem(device=device)

        # 延长仿真步数到500步（约10秒）
        print("\n🚗 开始沿道路行驶仿真，共500步...")
        for step in range(500):
            # 模拟传感器输入
            image = torch.randn(1, 3, 256, 256).to(device)
            lidar_data = torch.randn(1, 1, 256, 256).to(device)
            imu_data = torch.randn(1, 6).to(device)

            # 获取道路约束的目标转向角（0.9.11兼容）
            target_steer = env.get_target_steer()

            # 前向推理（传递目标转向角）
            policy, value = system.forward(image, lidar_data, imu_data, target_steer)

            # 解析策略并应用（限制范围）
            throttle = float(policy[0][0])
            steer = float(policy[0][1])
            # 最终控制：碰撞时刹车，否则正常行驶
            if env.collision_occurred:
                control = carla.VehicleControl(throttle=0.0, steer=steer, brake=0.5)
            else:
                control = carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0)

            env.vehicle.apply_control(control)

            # 每20步打印状态
            if (step + 1) % 20 == 0:
                vehicle_loc = env.vehicle.get_transform().location
                print(
                    f"步骤 {step + 1}/500 | 油门={throttle:.2f}, 转向={steer:.2f} | 位置：x={vehicle_loc.x:.1f}, y={vehicle_loc.y:.1f}")

            time.sleep(0.02)

        print("\n✅ 道路行驶仿真完成！")

    except Exception as e:
        print(f"\n❌ 仿真过程中出错: {e}")
    finally:
        if env is not None:
            env.cleanup()
        print("\n🔚 仿真结束，所有资源已清理")


if __name__ == "__main__":
    run_simulation()