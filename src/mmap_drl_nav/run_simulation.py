import torch
import time
import carla  # CARLA官方Python API
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import warnings

# 忽略CARLA API的冗余警告
warnings.filterwarnings("ignore", category=UserWarning, module="carla")


# ===================== 配置常量 =====================
@dataclass(frozen=True)
class Config:
    """全局配置常量，集中管理参数便于修改"""
    CARLA_HOST: str = "localhost"
    CARLA_PORT: int = 2000
    CARLA_TIMEOUT: float = 15.0
    SIMULATION_STEPS: int = 500
    STEP_SLEEP: float = 0.02
    VEHICLE_MODEL: str = "model3"
    SPAWN_POINT_INDEX: int = 20
    WAYPOINT_DISTANCE: float = 8.0
    THROTTLE_MIN: float = 0.2
    THROTTLE_MAX: float = 0.5
    COLLISION_BRAKE_DURATION: float = 0.5
    SPECTATOR_OFFSET: Tuple[float, float, float] = (-5.0, 0.0, 2.0)


# ===================== 感知与决策模块 =====================
class PerceptionModule(torch.nn.Module):
    """感知模块：模拟多传感器数据处理"""

    def forward(self, imu_data: torch.Tensor, image: torch.Tensor, lidar_data: torch.Tensor) -> Tuple[
        torch.Tensor, ...]:
        """
        前向传播：生成模拟的感知输出
        Args:
            imu_data: IMU数据 (batch, 6)
            image: 图像数据 (batch, 3, H, W)
            lidar_data: LiDAR数据 (batch, 1, H, W)
        Returns:
            scene_info, segmentation, odometry, obstacles, boundary
        """
        batch_size = image.shape[0]
        device = image.device

        # 统一使用device参数，避免重复调用image.device
        scene_info = torch.randn(batch_size, 128, device=device)
        segmentation = torch.randn(batch_size, 64, 256, 256, device=device)
        odometry = torch.randn(batch_size, 32, device=device)
        obstacles = torch.randn(batch_size, 64, device=device)
        boundary = torch.randn(batch_size, 32, device=device)

        return scene_info, segmentation, odometry, obstacles, boundary


class CrossDomainAttention(torch.nn.Module):
    """跨域注意力模块：融合多模态感知特征"""

    def __init__(self, num_blocks: int = 6):
        super().__init__()
        self.num_blocks = num_blocks
        # 预计算输入维度，提高可读性
        input_dim = 128 + (64 * 256 * 256) + 32 + 64 + 32
        self.fc = torch.nn.Linear(input_dim, 256)

    def forward(self, scene_info: torch.Tensor, segmentation: torch.Tensor,
                odometry: torch.Tensor, obstacles: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        """融合多模态特征"""
        seg_flat = segmentation.flatten(1)
        all_features = torch.cat([scene_info, seg_flat, odometry, obstacles, boundary], dim=1)
        fused = self.fc(all_features)
        return fused


class DecisionModule(torch.nn.Module):
    """决策模块：基于融合特征输出车辆控制指令"""

    def __init__(self):
        super().__init__()
        self.steer_fc = torch.nn.Linear(256, 1)  # 转向输出
        self.throttle_fc = torch.nn.Linear(256, 1)  # 油门输出

    def forward(self, fused_features: torch.Tensor, target_steer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输出控制策略
        Args:
            fused_features: 融合特征 (batch, 256)
            target_steer: 目标转向角 (batch, 1)
        Returns:
            policy: 控制策略 [throttle, steer] (batch, 2)
            value: 价值估计 (batch, 1)
        """
        # 转向控制：向目标转向角靠拢，范围[-1,1]
        steer = torch.tanh(self.steer_fc(fused_features) + target_steer)
        # 油门控制：限制在[0.2, 0.5]
        throttle = torch.sigmoid(self.throttle_fc(fused_features)) * (
                    Config.THROTTLE_MAX - Config.THROTTLE_MIN) + Config.THROTTLE_MIN

        policy = torch.cat([throttle, steer], dim=1)
        value = torch.randn(fused_features.shape[0], 1, device=fused_features.device)
        return policy, value


# ===================== CARLA环境管理 =====================
class CarlaEnvironment:
    """CARLA环境管理器：负责CARLA连接、车辆生成、传感器管理"""

    def __init__(self):
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.blueprint_library: Optional[carla.BlueprintLibrary] = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.spectator: Optional[carla.Actor] = None
        self.collision_sensor: Optional[carla.Sensor] = None
        self.collision_occurred: bool = False  # 碰撞标记

        self._connect_carla()
        self._cleanup_actors()  # 独立的清理函数
        self._spawn_vehicle()
        self._init_collision_sensor()
        self._set_vehicle_view()

    def _connect_carla(self) -> None:
        """连接CARLA服务器"""
        try:
            self.client = carla.Client(Config.CARLA_HOST, Config.CARLA_PORT)
            self.client.set_timeout(Config.CARLA_TIMEOUT)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            self.spectator = self.world.get_spectator()
            print("✅ CARLA服务器连接成功！")
        except Exception as e:
            raise RuntimeError(
                f"❌ 连接CARLA失败！请确认：\n1. CarlaUE4.exe已启动（版本0.9.11）\n2. 端口{Config.CARLA_PORT}未被占用\n错误详情：{e}"
            )

    def _cleanup_actors(self) -> None:
        """清理残留的车辆和传感器，避免资源泄漏"""
        try:
            # 按类型批量清理，提高效率
            actor_filters = ["*vehicle*", "*sensor*"]
            for filter_str in actor_filters:
                for actor in self.world.get_actors().filter(filter_str):
                    if actor.is_alive:
                        actor.destroy()
            print("✅ 残留Actor清理完成")
        except Exception as e:
            print(f"⚠️ 清理Actor时警告：{e}")

    def _spawn_vehicle(self) -> None:
        """生成车辆并初始化状态"""
        try:
            vehicle_bp = self.blueprint_library.filter(Config.VEHICLE_MODEL)[0]
            spawn_points = self.world.get_map().get_spawn_points()

            # 安全选择生成点
            spawn_idx = Config.SPAWN_POINT_INDEX if len(spawn_points) >= Config.SPAWN_POINT_INDEX else 0
            spawn_point = spawn_points[spawn_idx]

            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            # 初始化车辆状态：刹车、空挡
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0, gear=1))
            print(f"✅ 车辆生成成功！生成点位置：x={spawn_point.location.x:.1f}, y={spawn_point.location.y:.1f}")
        except Exception as e:
            raise RuntimeError(f"❌ 车辆生成失败：{e}")

    def _init_collision_sensor(self) -> None:
        """初始化碰撞传感器"""
        try:
            collision_bp = self.blueprint_library.find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=self.vehicle
            )
            self.collision_sensor.listen(self._on_collision)
            print("✅ 碰撞传感器初始化完成")
        except Exception as e:
            raise RuntimeError(f"❌ 碰撞传感器初始化失败：{e}")

    def _on_collision(self, event: carla.CollisionEvent) -> None:
        """碰撞回调函数：处理碰撞事件"""
        if not self.collision_occurred:
            self.collision_occurred = True
            print(f"⚠️ 检测到碰撞！碰撞对象：{event.other_actor.type_id}")
            # 撞障后紧急刹车
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
            time.sleep(Config.COLLISION_BRAKE_DURATION)

    def get_target_steer(self) -> torch.Tensor:
        """
        计算目标转向角（适配CARLA 0.9.11）
        Returns:
            target_steer: 目标转向角 (1, 1)
        """
        if self.collision_occurred:
            # 撞障后反向微调
            self.collision_occurred = False
            return torch.tensor([[0.3]], dtype=torch.float32)

        # 获取当前车辆位置和路点
        vehicle_location = self.vehicle.get_transform().location
        current_waypoint = self.world.get_map().get_waypoint(
            vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        # 获取前方路点（处理空列表情况）
        next_waypoints = current_waypoint.next(Config.WAYPOINT_DISTANCE)
        if not next_waypoints:
            return torch.tensor([[0.0]], dtype=torch.float32)
        next_waypoint = next_waypoints[0]

        # 计算转向误差
        vehicle_forward = self.vehicle.get_transform().get_forward_vector()
        direction_to_next = next_waypoint.transform.location - vehicle_location

        # 向量归一化（添加防除零保护）
        vehicle_forward = np.array([vehicle_forward.x, vehicle_forward.y])
        direction_to_next = np.array([direction_to_next.x, direction_to_next.y])

        norm_forward = np.linalg.norm(vehicle_forward)
        norm_next = np.linalg.norm(direction_to_next)

        if norm_forward < 1e-6 or norm_next < 1e-6:
            return torch.tensor([[0.0]], dtype=torch.float32)

        vehicle_forward = vehicle_forward / norm_forward
        direction_to_next = direction_to_next / norm_next

        # 计算夹角并归一化到[-1,1]
        dot_product = np.dot(vehicle_forward, direction_to_next)
        cross_product = np.cross(vehicle_forward, direction_to_next)
        steer_error = np.arcsin(cross_product) / np.pi  # 弧度转[-0.5,0.5]
        steer_error = np.clip(steer_error * 2, -1.0, 1.0)

        return torch.tensor([[steer_error]], dtype=torch.float32)

    def _set_vehicle_view(self) -> None:
        """设置观众视角到车辆后方"""
        if not (self.vehicle and self.spectator):
            return

        transform = self.vehicle.get_transform()
        spectator_transform = carla.Transform(
            transform.location + carla.Location(*Config.SPECTATOR_OFFSET),
            transform.rotation
        )
        self.spectator.set_transform(spectator_transform)
        print("✅ 视角已切换到车辆后方！")
        print("   🎮 WASD：移动视角 | 鼠标右键+拖动：旋转视角 | 滚轮：缩放 | P：快速定位到车辆")

    def cleanup(self) -> None:
        """清理所有资源"""
        try:
            # 先停止车辆
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                time.sleep(0.5)

            # 销毁传感器和车辆
            if self.collision_sensor and self.collision_sensor.is_alive:
                self.collision_sensor.destroy()
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.destroy()

            print("✅ 资源已清理")
        except Exception as e:
            print(f"⚠️ 清理资源时警告：{e}")


# ===================== 集成系统 =====================
class IntegratedSystem:
    """集成系统：感知-融合-决策全流程"""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.perception = PerceptionModule().to(device)
        self.attention = CrossDomainAttention(num_blocks=6).to(device)
        self.decision = DecisionModule().to(device)

    def forward(self, image: torch.Tensor, lidar_data: torch.Tensor,
                imu_data: torch.Tensor, target_steer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向推理全流程
        Args:
            image: 图像数据 (1, 3, 256, 256)
            lidar_data: LiDAR数据 (1, 1, 256, 256)
            imu_data: IMU数据 (1, 6)
            target_steer: 目标转向角 (1, 1)
        Returns:
            policy: 控制策略 [throttle, steer] (1, 2)
            value: 价值估计 (1, 1)
        """
        # 感知处理
        scene_info, segmentation, odometry, obstacles, boundary = self.perception(imu_data, image, lidar_data)
        # 特征融合
        fused_features = self.attention(scene_info, segmentation, odometry, obstacles, boundary)
        # 决策输出
        policy, value = self.decision(fused_features, target_steer.to(self.device))
        return policy, value


# ===================== 主函数 =====================
def run_simulation() -> None:
    """运行CARLA自动驾驶仿真"""
    env = None
    try:
        print(f"📢 运行前请确认：CarlaUE4.exe已启动（版本0.9.11），端口{Config.CARLA_PORT}未被占用")
        time.sleep(2)

        # 初始化环境和系统
        env = CarlaEnvironment()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ 使用计算设备: {device}")
        system = IntegratedSystem(device=device)

        # 运行仿真
        print(f"\n🚗 开始沿道路行驶仿真，共{Config.SIMULATION_STEPS}步...")
        for step in range(Config.SIMULATION_STEPS):
            # 模拟传感器输入
            image = torch.randn(1, 3, 256, 256, device=device)
            lidar_data = torch.randn(1, 1, 256, 256, device=device)
            imu_data = torch.randn(1, 6, device=device)

            # 获取目标转向角
            target_steer = env.get_target_steer()

            # 前向推理
            policy, _ = system.forward(image, lidar_data, imu_data, target_steer)

            # 解析并应用控制指令
            throttle = float(policy[0][0].cpu().item())  # 移到CPU避免设备不匹配
            steer = float(policy[0][1].cpu().item())

            if env.collision_occurred:
                control = carla.VehicleControl(throttle=0.0, steer=steer, brake=0.5)
            else:
                control = carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0)

            env.vehicle.apply_control(control)

            # 定期打印状态
            if (step + 1) % 20 == 0:
                vehicle_loc = env.vehicle.get_transform().location
                print(
                    f"步骤 {step + 1}/{Config.SIMULATION_STEPS} | 油门={throttle:.2f}, 转向={steer:.2f} | 位置：x={vehicle_loc.x:.1f}, y={vehicle_loc.y:.1f}")

            time.sleep(Config.STEP_SLEEP)

        print("\n✅ 道路行驶仿真完成！")

    except Exception as e:
        print(f"\n❌ 仿真过程中出错: {e}")
        raise  # 重新抛出异常便于调试
    finally:
        if env is not None:
            env.cleanup()
        print("\n🔚 仿真结束，所有资源已清理")


if __name__ == "__main__":
    run_simulation()