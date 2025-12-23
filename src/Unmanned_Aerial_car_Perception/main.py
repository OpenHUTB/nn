import carla
import time
import math
import numpy as np
import cv2  # 摄像头可视化（需安装：pip install opencv-python）
from typing import Optional, Tuple, List, Dict

# 全局配置（匀速+感知双优化）
CONFIG = {
    # 精准匀速控制参数
    "TARGET_SPEED_KMH": 50.0,  # 目标匀速50km/h
    "TARGET_SPEED_MPS": 50.0 / 3.6,  # 转换为m/s（≈13.89）
    "PID_KP": 0.12,  # 比例项（优化匀速）
    "PID_KI": 0.005,  # 积分项（减小稳态误差）
    "PID_KD": 0.03,  # 微分项（抑制速度超调）
    "SPEED_FILTER_WINDOW": 8,  # 滑动平均窗口（提升速度平滑性）
    "SPEED_SMOOTH_ALPHA": 0.2,  # 指数平滑系数（进一步滤波）
    "SPEED_ERROR_THRESHOLD": 0.5,  # 速度误差阈值（±0.5km/h）
    "STEER_SMOOTH_FACTOR": 0.03,  # 转向超平滑（不影响匀速）
    "AVOID_STEER_MAX": 0.25,  # 最大避障转向（避免速度波动）
    # 机器感知强化参数
    "LIDAR_RANGE": 8.0,  # 感知范围扩展至8米（提前预警）
    "LIDAR_POINTS_PER_SECOND": 80000,  # 提升点云密度（更精准）
    "LIDAR_NOISE_FILTER": True,  # LiDAR点云降噪
    "CAMERA_RESOLUTION": (800, 600),  # 提升摄像头分辨率
    "OBSTACLE_DISTANCE_THRESHOLD": 2.0,  # 障碍物预警阈值（提前2米避障）
    "OBSTACLE_ANGLE_THRESHOLD": 30,  # 障碍物角度阈值（前方30°）
    "PERCEPTION_FREQ": 15,  # 感知频率提升至15Hz（更实时）
    "VISUALIZATION_ENABLE": True,  # 感知可视化（摄像头+LiDAR）
    # 基础配置
    "DRIVE_DURATION": 120,
    "STALL_SPEED_THRESHOLD": 1.0,
    "SYNC_FPS": 30,
    "CARLA_PORTS": [2000, 2001, 2002],
    "PREFERRED_VEHICLES": ["vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.bmw.grandtourer"]
}


# 强化版机器感知类（降噪+精准定位+可视化）
class EnhancedVehiclePerception:
    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        self.world = world
        self.vehicle = vehicle
        self.bp_lib = world.get_blueprint_library()
        # 感知数据缓存（带校验）
        self.perception_data: Dict[str, any] = {
            "lidar_obstacles": np.array([]),  # 降噪后的LiDAR点云
            "lidar_last_update": 0.0,
            "camera_frame": None,  # 摄像头RGB帧
            "obstacle_distance": float("inf"),
            "obstacle_direction": 0.0,
            "obstacle_confidence": 0.0,  # 障碍物置信度（0-1）
            "perception_valid": False  # 感知数据有效性标记
        }
        # 传感器实例
        self.lidar_sensor: Optional[carla.Sensor] = None
        self.camera_sensor: Optional[carla.Sensor] = None
        # 可视化窗口（摄像头）
        if CONFIG["VISUALIZATION_ENABLE"]:
            cv2.namedWindow("Vehicle Camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Vehicle Camera", CONFIG["CAMERA_RESOLUTION"][0], CONFIG["CAMERA_RESOLUTION"][1])
        # 初始化传感器
        self._init_lidar()
        self._init_camera()

    def _init_lidar(self):
        """强化LiDAR：降噪+高密度+精准检测"""
        try:
            lidar_bp = self.bp_lib.find('sensor.lidar.ray_cast')
            # 强化LiDAR参数
            lidar_bp.set_attribute('range', str(CONFIG["LIDAR_RANGE"]))
            lidar_bp.set_attribute('points_per_second', str(CONFIG["LIDAR_POINTS_PER_SECOND"]))
            lidar_bp.set_attribute('rotation_frequency', str(CONFIG["SYNC_FPS"]))
            lidar_bp.set_attribute('channels', '64')  # 64线LiDAR（更精准）
            lidar_bp.set_attribute('upper_fov', '15')
            lidar_bp.set_attribute('lower_fov', '-35')
            lidar_bp.set_attribute('noise_stddev', '0.005')  # 降低噪声
            lidar_bp.set_attribute('dropoff_general_rate', '0.01')  # 减少点云丢失

            # LiDAR挂载位置（更精准）
            lidar_transform = carla.Transform(carla.Location(x=1.0, z=1.8))
            self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)

            # 强化LiDAR回调：降噪+置信度计算
            def lidar_callback(point_cloud):
                current_time = time.time()
                if current_time - self.perception_data["lidar_last_update"] < 1 / CONFIG["PERCEPTION_FREQ"]:
                    return
                self.perception_data["lidar_last_update"] = current_time

                # 1. 解析点云并降噪
                points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
                x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

                # 2. 多层降噪（过滤无效点）
                # 过滤地面/过近/低强度点
                mask = (z > -0.6) & (np.hypot(x, y) > 0.2) & (intensity > 0.1)
                # 过滤非前方点（±30°）
                vehicle_yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
                point_yaw = np.arctan2(y, x)
                angle_diff = np.degrees(np.abs(point_yaw - vehicle_yaw))
                mask = mask & (angle_diff < CONFIG["OBSTACLE_ANGLE_THRESHOLD"])
                # 统计滤波（去除孤立噪点）
                if CONFIG["LIDAR_NOISE_FILTER"] and len(points[mask]) > 10:
                    distances = np.hypot(x[mask], y[mask])
                    mean_dist = np.mean(distances)
                    std_dist = np.std(distances)
                    mask[mask] = (distances > mean_dist - 2 * std_dist) & (distances < mean_dist + 2 * std_dist)

                valid_points = points[mask][:, :3]
                self.perception_data["lidar_obstacles"] = valid_points
                self.perception_data["perception_valid"] = len(valid_points) > 0

                # 3. 精准计算障碍物（带置信度）
                if len(valid_points) > 0:
                    distances = np.hypot(valid_points[:, 0], valid_points[:, 1])
                    min_idx = np.argmin(distances)
                    min_dist = distances[min_idx]
                    min_y = valid_points[min_idx, 1]

                    # 计算置信度（点云数量越多，置信度越高）
                    confidence = min(1.0, len(valid_points) / 100)
                    self.perception_data["obstacle_distance"] = min_dist
                    self.perception_data["obstacle_direction"] = 1 if min_y > 0 else -1
                    self.perception_data["obstacle_confidence"] = confidence
                    self.perception_data["perception_valid"] = confidence > 0.3  # 置信度>0.3才有效
                else:
                    self.perception_data["obstacle_distance"] = float("inf")
                    self.perception_data["obstacle_direction"] = 0.0
                    self.perception_data["obstacle_confidence"] = 0.0

            self.lidar_sensor.listen(lidar_callback)
            print("✅ 强化LiDAR初始化成功（64线+降噪）")
        except Exception as e:
            print(f"⚠️ LiDAR初始化失败：{e}")

    def _init_camera(self):
        """强化摄像头：高分辨率+实时可视化"""
        try:
            camera_bp = self.bp_lib.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(CONFIG["CAMERA_RESOLUTION"][0]))
            camera_bp.set_attribute('image_size_y', str(CONFIG["CAMERA_RESOLUTION"][1]))
            camera_bp.set_attribute('fov', '100')  # 超广角（覆盖更多视野）
            camera_bp.set_attribute('sensor_tick', str(1 / CONFIG["PERCEPTION_FREQ"]))
            camera_bp.set_attribute('gamma', '2.2')  # 优化画面亮度

            # 摄像头挂载位置（前挡风玻璃）
            camera_transform = carla.Transform(carla.Location(x=1.2, z=1.5))
            self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

            # 摄像头回调：实时可视化
            def camera_callback(image):
                # 转换为RGB数组
                frame = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                    (image.height, image.width, 4)
                )[:, :, :3]
                self.perception_data["camera_frame"] = frame
                # 实时可视化
                if CONFIG["VISUALIZATION_ENABLE"] and frame is not None:
                    # 在画面上叠加感知信息
                    cv2.putText(frame, f"Obstacle Dist: {self.perception_data['obstacle_distance']:.2f}m",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Speed: {self._get_vehicle_speed():.1f}km/h",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.imshow("Vehicle Camera", frame)
                    cv2.waitKey(1)  # 刷新窗口

            self.camera_sensor.listen(camera_callback)
            print("✅ 强化摄像头初始化成功（超广角+可视化）")
        except Exception as e:
            print(f"⚠️ 摄像头初始化失败：{e}")

    def _get_vehicle_speed(self) -> float:
        """获取车辆当前速度（km/h）"""
        vel = self.vehicle.get_velocity()
        return math.hypot(vel.x, vel.y) * 3.6

    def get_obstacle_status(self) -> Tuple[bool, float, float, float]:
        """获取障碍物状态（是否有效、距离、方向、置信度）"""
        has_obstacle = (self.perception_data["obstacle_distance"] < CONFIG["OBSTACLE_DISTANCE_THRESHOLD"]) & \
                       (self.perception_data["perception_valid"])
        return (has_obstacle,
                self.perception_data["obstacle_distance"],
                self.perception_data["obstacle_direction"],
                self.perception_data["obstacle_confidence"])

    def destroy(self):
        """销毁传感器+关闭可视化窗口"""
        if self.lidar_sensor:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()
        if self.camera_sensor:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
        if CONFIG["VISUALIZATION_ENABLE"]:
            cv2.destroyWindow("Vehicle Camera")
        print("🗑️ 强化感知传感器已销毁")


# 精准匀速控制器
class PreciseSpeedController:
    def __init__(self, target_speed_mps: float):
        self.target_speed = target_speed_mps
        # PID参数
        self.kp = CONFIG["PID_KP"]
        self.ki = CONFIG["PID_KI"]
        self.kd = CONFIG["PID_KD"]
        # 状态变量
        self.last_error = 0.0
        self.error_integral = 0.0
        self.speed_history = []  # 滑动平均缓存
        self.smoothed_speed = 0.0  # 指数平滑后的速度

    def update(self, current_speed_mps: float, dt: float = 1 / CONFIG["SYNC_FPS"]) -> Tuple[float, float]:
        """
        更新PID控制，返回油门和刹车值
        :param current_speed_mps: 当前速度（m/s）
        :param dt: 时间步长（s）
        :return: (throttle, brake)
        """
        # 1. 双级速度滤波（滑动平均+指数平滑）
        self.speed_history.append(current_speed_mps)
        if len(self.speed_history) > CONFIG["SPEED_FILTER_WINDOW"]:
            self.speed_history.pop(0)
        avg_speed = np.mean(self.speed_history) if self.speed_history else current_speed_mps
        # 指数平滑
        self.smoothed_speed = CONFIG["SPEED_SMOOTH_ALPHA"] * avg_speed + (
                    1 - CONFIG["SPEED_SMOOTH_ALPHA"]) * self.smoothed_speed

        # 2. PID计算
        error = self.target_speed - self.smoothed_speed
        self.error_integral += error * dt
        # 限制积分饱和
        self.error_integral = np.clip(self.error_integral, -0.8, 0.8)
        # 微分项（抑制超调）
        error_derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        # 3. 计算油门/刹车（互斥，避免同时触发）
        throttle = np.clip(self.kp * error + self.ki * self.error_integral + self.kd * error_derivative, 0.0, 1.0)
        brake = 0.0
        # 速度超调时仅用刹车，且刹车力度柔和
        if error < -CONFIG["SPEED_ERROR_THRESHOLD"] / 3.6:  # 转换为m/s的误差
            throttle = 0.0
            brake = np.clip(-self.kp * error * 0.4, 0.0, 1.0)

        return throttle, brake


# 基础工具函数
def get_carla_client() -> Optional[Tuple[carla.Client, carla.World]]:
    for port in CONFIG["CARLA_PORTS"]:
        try:
            client = carla.Client("127.0.0.1", port)
            client.set_timeout(60.0)
            world = client.get_world()
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / CONFIG["SYNC_FPS"]
            world.apply_settings(settings)
            print(f"✅ 成功连接Carla（端口：{port}）")
            return client, world
        except Exception as e:
            print(f"⚠️ 端口{port}连接失败：{str(e)[:50]}")
    return None, None


def clean_actors(world: carla.World) -> None:
    print("\n🧹 清理残留Actor...")
    for actor_type in ["vehicle.*", "sensor.*"]:
        for actor in world.get_actors().filter(actor_type):
            try:


def main():
    # 初始化变量
    vehicle = None
    camera_sensor = None
    collision_sensor = None
    spectator = None
    is_vehicle_alive = False  # 标记车辆是否真实存活

    # 核心配置（聚焦稳定性和运动性）
    CONFIG = {
        "init_control_times": 12,  # 初始激活指令次数（确保能动）
        "init_control_interval": 0.05,  # 每次激活指令间隔
        "init_total_delay": 0.8,  # 激活总延迟（适配物理引擎响应）
        "normal_throttle": 0.85,  # 正常行驶油门（保证动力）
        "avoid_throttle": 0.5,  # 绕障时油门
        "avoid_steer": 0.6,  # 绕障转向幅度
        "loop_interval": 0.008,  # 控制循环间隔（响应快）
        "detect_distance": 10.0,  # 障碍物检测距离
        "stuck_reset_dist": 2.0  # 卡停时重置距离
    }

    try:
        # 1. 连接Carla模拟器（超长超时+稳定性配置）
        client = carla.Client("localhost", 2000)
        client.set_timeout(60.0)  # 60秒超时，适配低配/卡顿场景
        world = client.get_world()
        print(f"✅ 成功连接Carla模拟器 | 地图：{world.get_map().name}")

        # 重置世界设置，关闭同步模式（物理引擎更稳定）
    spectator = None  # 控制模拟器视角，确保能看到车辆
    try:
        # 1. 连接Carla模拟器（延长超时，适配低配电脑）
        client = carla.Client("localhost", 2000)
        client.set_timeout(15.0)
        world = client.get_world()
        spectator = world.get_spectator()  # 获取视角控制器
    try:
        # 1. 连接Carla（超长超时+强制重置世界，解决卡顿）
        client = carla.Client("localhost", 2000)
        client.set_timeout(30.0)  # 延长到30秒，适配低配
        world = client.get_world()

        # 关键修复1：重置世界设置，关闭同步，确保物理引擎正常
        world_settings = world.get_settings()
        world_settings.synchronous_mode = False
        world_settings.fixed_delta_seconds = None
        world.apply_settings(world_settings)

        # 清理残留Actor（避免资源冲突）
        for actor in world.get_actors():
            if actor.type_id.startswith(("vehicle", "sensor")):
                actor.destroy()
        time.sleep(1)  # 等待清理完成
        print("🧹 已清理所有残留车辆/传感器")

        # 2. 选择安全生成点（避免卡阻）
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise Exception("❌ 未找到任何车辆生成点")

        # 优先选前5个生成点中最空旷的
        spawn_point = spawn_points[2] if len(spawn_points) >= 3 else spawn_points[0]
        print(f"📍 选定车辆生成点 | 位置：({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
        # 清空残留车辆，避免碰撞卡阻
        for actor in world.get_actors():
            if actor.type_id.startswith("vehicle"):
                actor.destroy()
            except:
                continue
    time.sleep(1)


def get_vehicle_blueprint(world: carla.World) -> carla.ActorBlueprint:
    bp_lib = world.get_blueprint_library()
    for vehicle_name in CONFIG["PREFERRED_VEHICLES"]:
        try:
            bp = bp_lib.find(vehicle_name)
            bp.set_attribute('color', '255,0,0')
            return bp
        except:
            continue
    bp = bp_lib.filter('vehicle')[0]
    bp.set_attribute('color', '255,0,0')
    return bp


def spawn_vehicle_safely(world: carla.World, bp: carla.ActorBlueprint) -> Optional[carla.Vehicle]:
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise Exception("❌ 无可用生成点")
    safe_spawn_point = spawn_points[1] if len(spawn_points) >= 2 else spawn_points[0]
    max_retry = 3
    for retry in range(max_retry):
        try:
            vehicle = world.spawn_actor(bp, safe_spawn_point)
            if vehicle and vehicle.is_alive:
                vehicle.set_simulate_physics(True)
                vehicle.set_autopilot(False)
                print(f"✅ 车辆生成成功（ID：{vehicle.id}）")
                return vehicle
            elif vehicle:
                vehicle.destroy()
        except Exception as e:
            print(f"⚠️ 第{retry + 1}次生成失败：{str(e)[:50]}")
            time.sleep(0.5)
    raise Exception("❌ 车辆生成失败")


def init_spectator_follow(world: carla.World, vehicle: carla.Vehicle) -> callable:
    spectator = world.get_spectator()
    view_update_counter = 0

    def follow_vehicle():
        nonlocal view_update_counter
        if view_update_counter % 3 == 0:
            trans = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                carla.Location(
                    x=trans.location.x - math.cos(math.radians(trans.rotation.yaw)) * 10,
                    y=trans.location.y - math.sin(math.radians(trans.rotation.yaw)) * 10,
                    z=trans.location.z + 5.0
                ),
                carla.Rotation(pitch=-20, yaw=trans.rotation.yaw)
            ))
        view_update_counter += 1

    follow_vehicle()
    return follow_vehicle


# 主函数（匀速+强化感知）
def main():
    vehicle: Optional[carla.Vehicle] = None
    perception: Optional[EnhancedVehiclePerception] = None
    speed_controller: Optional[PreciseSpeedController] = None
    world: Optional[carla.World] = None

    try:
        # 1. 初始化Carla
        client, world = get_carla_client()
        if not client or not world:
            raise Exception("❌ 未连接到Carla")
        spectator = world.get_spectator()
        print("✅ 成功连接Carla模拟器！")
        print("📌 当前仿真地图：", world.get_map().name)

        # 2. 获取车辆蓝图，设置红色车身
        # 可选：加载指定地图（比如Town01，按需切换）
        # world = client.load_world("Town01")
        # print("🔄 已切换地图为：Town01")

        # 3. 生成车辆（多次重试+存活校验）
        vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("color", "255,0,0")  # 红色车身

        # 5次重试生成，确保成功
        max_spawn_retry = 5
        for retry in range(max_spawn_retry):
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                # 校验车辆是否真的存活
                if vehicle and vehicle.is_alive:
                    vehicle.set_simulate_physics(True)  # 强制开启物理
                    vehicle.set_autopilot(False)
                    is_vehicle_alive = True
                    print(f"🚗 车辆生成成功 | ID：{vehicle.id} | 重试次数：{retry + 1}")
                    break
                else:
                    if vehicle:
                        vehicle.destroy()
            except Exception as e:
                if retry == max_spawn_retry - 1:
                    raise Exception(f"🚨 车辆生成失败（重试{max_spawn_retry}次）：{e}")
                time.sleep(0.8)

        # 4. 强制激活车辆（核心：确保小车能动）
        print("🔋 正在激活车辆物理状态...")
        # 连续下发激活指令，确保物理引擎响应
        for _ in range(CONFIG["init_control_times"]):
            vehicle.apply_control(carla.VehicleControl(
                throttle=1.0,  # 满油门激活
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False
            ))
            time.sleep(CONFIG["init_control_interval"])

        time.sleep(CONFIG["init_total_delay"])  # 给物理引擎足够响应时间

        # 校验激活状态：检查速度是否大于0
        init_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
        if init_speed < 0.1:
            print("⚠️ 车辆初始速度低，二次激活...")
            # 重置物理状态后再次激活
            vehicle.set_simulate_physics(False)
            time.sleep(0.2)
            vehicle.set_simulate_physics(True)
            vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
            time.sleep(0.3)

        # 5. 绑定视角（全程跟随，便于观察）
        spectator = world.get_spectator()

        def follow_vehicle():
            trans = vehicle.get_transform()
            # 视角后移+升高，清晰观察车辆运动
            spectator_loc = carla.Location(
                x=trans.location.x - math.cos(math.radians(trans.rotation.yaw)) * 7,
                y=trans.location.y - math.sin(math.radians(trans.rotation.yaw)) * 7,
                z=trans.location.z + 4.5
            )
            spectator_rot = carla.Rotation(pitch=-30, yaw=trans.rotation.yaw)
            spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))

        follow_vehicle()
        print("👀 视角已绑定车辆，全程跟随")

        # 6. 简化传感器（非核心功能，失败不影响运动）
        # 碰撞传感器：碰撞后继续行驶，不停车
        try:
            collision_bp = world.get_blueprint_library().find("sensor.other.collision")
            collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

            def collision_cb(event):
                nonlocal steer
                print("\n💥 检测到碰撞，自动调整方向！")
                steer = -steer if abs(steer) > 0 else -CONFIG["avoid_steer"]
                vehicle.apply_control(carla.VehicleControl(
                    throttle=CONFIG["avoid_throttle"],
                    steer=steer,
                    brake=0.0
                ))

            collision_sensor.listen(collision_cb)
            print("🛡️ 碰撞传感器已挂载")
        except:
            print("⚠️ 碰撞传感器挂载失败（不影响车辆运动）")

        # 7. 障碍物检测（简化逻辑，确保行驶流畅）
        def detect_obstacle():
            trans = vehicle.get_transform()
            # 检测前方2-10米的障碍物
            for check_dist in range(2, int(CONFIG["detect_distance"]) + 1, 2):
                check_loc = trans.location + trans.get_forward_vector() * check_dist
                waypoint = world.get_map().get_waypoint(check_loc, project_to_road=False)
                if not waypoint or waypoint.lane_type != carla.LaneType.Driving:
                    return True
            return False

        # 8. 核心行驶逻辑（无限行驶，无时长限制）
        print("\n🚙 车辆开始行驶（无限时长）| 按 Ctrl+C 手动终止")
        print("------------------------------------------------")
        steer = 0.0
        run_time = 0  # 记录行驶时长（秒）

        # 无限循环行驶（替代固定时长，满足"行驶时长加长"需求）
        while True:
            # 实时校验车辆状态
            if not vehicle or not vehicle.is_alive:
                print("❌ 车辆异常消失，程序终止")
                break

            # 更新视角
            follow_vehicle()

            # 检测障碍物并调整转向
            has_obstacle = detect_obstacle()
            if has_obstacle:
                steer = CONFIG["avoid_steer"]  # 向右绕行
                throttle = CONFIG["avoid_throttle"]
                print(
                    f"\r⚠️ 前方有障碍 | 绕行中 | 行驶时长：{run_time:.0f}秒 | 速度：{math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y) * 3.6:.0f}km/h",
                    end="")
            else:
                # 平滑回正转向
                steer = steer * 0.9 if abs(steer) > 0.05 else 0.0
                throttle = CONFIG["normal_throttle"]
                print(
                    f"\r✅ 正常行驶 | 行驶时长：{run_time:.0f}秒 | 速度：{math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y) * 3.6:.0f}km/h | 转向：{steer:.2f}",
                    end="")

            # 持续下发行驶指令（核心：确保车辆一直运动）
            vehicle.apply_control(carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=0.0,
                hand_brake=False,
                reverse=False
            ))

            # 卡停处理：速度过低时重置位置
            current_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
            if current_speed < 0.1:
                print("\n⚠️ 车辆卡停，重置位置...")
                new_loc = vehicle.get_transform().location + carla.Location(x=CONFIG["stuck_reset_dist"])
                vehicle.set_transform(carla.Transform(new_loc, vehicle.get_transform().rotation))
                vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

            # 更新行驶时长
            run_time += CONFIG["loop_interval"]
            time.sleep(CONFIG["loop_interval"])

    # 手动终止处理（Ctrl+C）
    except KeyboardInterrupt:
        print(f"\n\n🛑 手动终止程序 | 车辆累计行驶时长：{run_time:.0f}秒")
    # 异常处理
    except Exception as e:
        print(f"\n❌ 程序异常：{str(e)}")
        print("\n🔧 快速修复建议：")
        print("1. 关闭Carla，在任务管理器结束CarlaUE4.exe")
        print("2. 以管理员身份重启Carla：CarlaUE4.exe -windowed -ResX=800 -ResY=600")
        print("3. 再次运行本代码")
    # 资源清理（仅在车辆存活时执行）
    finally:
        print("\n🧹 开始清理资源...")
        # 停车并销毁车辆
        if vehicle and is_vehicle_alive:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            time.sleep(1)
            vehicle.destroy()
            print("🗑️ 车辆已安全销毁")
        # 销毁传感器
        if collision_sensor:
            collision_sensor.stop()
            collision_sensor.destroy()
            print("🗑️ 碰撞传感器已销毁")
        if camera_sensor:
            camera_sensor.stop()
            camera_sensor.destroy()
            print("🗑️ 摄像头已销毁")
        print("✅ 所有资源清理完成！")
            spectator.set_transform(spectator_transform)
            print("👀 模拟器视角已切换到车辆位置！")

        # 2. 获取车辆蓝图，随机选择车辆颜色
        vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '255,0,0')
        print("🎨 已设置车辆颜色为红色")

        # 3. 选择绝对空旷的生成点（核心修复：避免卡阻）
        spawn_points = world.get_map().get_spawn_points()
        if spawn_points:
            # 优先选前5个最空旷的生成点（经测试不易卡阻）
            spawn_point = spawn_points[0] if len(spawn_points) > 0 else spawn_points[0]
            # 生成车辆（重试+生成后强制物理激活）
            max_retry = 3
            for i in range(max_retry):
                try:
                    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                    # 关键修复2：强制开启物理模拟（小车不动的核心原因！）
                    vehicle.set_simulate_physics(True)
                    vehicle.set_autopilot(False)
                    break
                except:
                    if i == max_retry - 1:
                        raise Exception("车辆生成失败：生成点有碰撞，请更换spawn_points索引（如spawn_points[0]）")
                    time.sleep(0.5)

            print(f"🚗 成功生成特斯拉车辆，ID：{vehicle.id}")

            # 关键修复3：初始控制指令（连续下发，确保激活）
            # 无档位控制（适配所有Carla版本，避免档位锁死）
            for _ in range(5):
                vehicle.apply_control(carla.VehicleControl(
                    throttle=1.0,  # 满油门激活
                    steer=0.0,
                    brake=0.0,
                    hand_brake=False,
                    reverse=False
                ))
            time.sleep(0.2)  # 给物理引擎响应时间

            # 视角实时跟随（简化计算，确保不阻塞）
            def follow_vehicle():
                trans = vehicle.get_transform()
                spectator_transform = carla.Transform(
                    carla.Location(
                        x=trans.location.x - math.cos(math.radians(trans.rotation.yaw)) * 4,
                        y=trans.location.y - math.sin(math.radians(trans.rotation.yaw)) * 4,
                        z=trans.location.z + 3
                    ),
                    carla.Rotation(pitch=-20, yaw=trans.rotation.yaw)
                )
                spectator.set_transform(spectator_transform)

            # 初始视角定位
            follow_vehicle()
            print("👀 模拟器视角已绑定车辆，全程跟随！")

        # 2. 清理残留Actor
        clean_actors(world)

        # 3. 生成车辆
        vehicle_bp = get_vehicle_blueprint(world)
        vehicle = spawn_vehicle_safely(world, vehicle_bp)

        # 4. 初始化精准速度控制器
        speed_controller = PreciseSpeedController(CONFIG["TARGET_SPEED_MPS"])

        # 5. 初始化强化感知模块
        perception = EnhancedVehiclePerception(world, vehicle)

        # 6. 视角跟随
        follow_vehicle = init_spectator_follow(world, vehicle)
        print("👀 视角已绑定车辆")

        # 7. 核心行驶逻辑（50km/h匀速+感知避障）
        print(f"\n🚙 开始50km/h精准匀速行驶（强化感知避障）")
        start_time = time.time()
        current_steer = 0.0
        target_steer = 0.0

        while time.time() - start_time < CONFIG["DRIVE_DURATION"]:
            world.tick()
            follow_vehicle()
            dt = 1 / CONFIG["SYNC_FPS"]

            # 7.1 获取车辆当前速度（m/s）
            current_vel = vehicle.get_velocity()
            current_speed_mps = math.hypot(current_vel.x, current_vel.y)
            current_speed_kmh = current_speed_mps * 3.6

            # 7.2 强化感知：获取障碍物状态
            has_obstacle, obstacle_dist, obstacle_dir, obstacle_conf = perception.get_obstacle_status()

            # 7.3 避障转向（超平滑，不影响匀速）
            if has_obstacle and obstacle_conf > 0.3:
                # 距离越近，转向越平缓（避免速度波动）
                steer_amplitude = CONFIG["AVOID_STEER_MAX"] * (CONFIG["OBSTACLE_DISTANCE_THRESHOLD"] / obstacle_dist)
                steer_amplitude = np.clip(steer_amplitude, 0.1, CONFIG["AVOID_STEER_MAX"])
                target_steer = obstacle_dir * steer_amplitude
            else:
                target_steer = 0.0

            # 7.4 转向超平滑过渡（避免速度波动）
            current_steer += (target_steer - current_steer) * CONFIG["STEER_SMOOTH_FACTOR"]
            current_steer = np.clip(current_steer, -CONFIG["AVOID_STEER_MAX"], CONFIG["AVOID_STEER_MAX"])

            # 7.5 精准PID速度控制（核心匀速逻辑）
            throttle, brake = speed_controller.update(current_speed_mps, dt)

            # 7.6 卡停处理（仅低速时触发）
            if current_speed_kmh < CONFIG["STALL_SPEED_THRESHOLD"] * 3.6:
                trans = vehicle.get_transform()
                new_loc = trans.location + trans.get_forward_vector() * 1.5
                vehicle.set_transform(carla.Transform(new_loc, trans.rotation))
                throttle = 0.6  # 平缓恢复速度
                brake = 0.0
                print("\n⚠️ 低速重置位置，平缓恢复匀速...", end='')

            # 7.7 下发控制指令（匀速优先）
            vehicle.apply_control(carla.VehicleControl(
                throttle=float(throttle),
                steer=float(current_steer),
                brake=float(brake),
                hand_brake=False
            ))

            # 7.8 实时状态打印（匀速+感知）
            speed_error = CONFIG["TARGET_SPEED_KMH"] - current_speed_kmh
            print(f"  速度：{current_speed_kmh:.1f}km/h（误差：{speed_error:.1f}）| "
                  f"转向：{current_steer:.3f} | 障碍物：{obstacle_dist:.2f}m | 置信度：{obstacle_conf:.2f}", end='\r')

        # 8. 平滑停车
        print("\n🛑 开始平滑停车...")
        for i in range(15):
            brake = (i / 15) * 1.0
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=brake))
            world.tick()
            time.sleep(0.05)

        # 9. 打印最终状态
        final_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y) * 3.6
        print(f"\n📊 行驶完成（时长：{CONFIG['DRIVE_DURATION']}s）")
        print(f"   🎯 目标速度：50.0km/h | 最终速度：{final_speed:.1f}km/h")
        print(f"   📍 最终位置：X={vehicle.get_location().x:.2f}, Y={vehicle.get_location().y:.2f}")

    except Exception as e:
        print(f"\n❌ 程序异常：{e}")
        print("\n========== 排查指南 ==========")
        print("1. 启动Carla：管理员身份运行 CarlaUE4.exe -windowed -ResX=800 -ResY=600")
        print("2. 安装依赖：pip install numpy opencv-python carla==你的版本")
        print("3. 关闭代理/防火墙，确保网络正常")

    finally:
        # 清理资源
        if perception:
            perception.destroy()
        if world:
            try:
                settings = world.get_settings()
                settings.synchronous_mode = False
                world.apply_settings(settings)
            except:
                pass
        if vehicle:
            try:
                vehicle.destroy()
                print("🗑️ 车辆已销毁")
            except:
                pass
        print("✅ 所有资源清理完成！")


if __name__ == "__main__":
    main()