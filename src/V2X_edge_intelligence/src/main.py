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
import cv2
import queue
import random

# ======================== 核心配置（车道硬约束+细障碍物检测）========================
TARGET_SPEED_KMH = 10.0  # 更低速，确保车道纠偏反应时间
TARGET_SPEED_MPS = TARGET_SPEED_KMH / 3.6
SYNC_FPS = 20

# 障碍物检测（针对电线杆等细障碍物）
LIDAR_RANGE = 15.0  # 覆盖路边障碍物
OBSTACLE_EMERGENCY_DIST = 1.0  # 1米紧急避障
OBSTACLE_WARNING_DIST = 2.5  # 提前预警
DETECT_THRESHOLD = 2  # 仅需2个点（细障碍物点少）
# 车道硬约束（核心：防止偏离撞路边障碍物）
LANE_BOUNDARY_STRICT = 0.8  # 车道边界强制纠偏力度
LANE_CENTER_BIAS = 0.1  # 轻微偏向车道中心
MAX_LANE_DEVIATION = 0.5  # 最大允许偏离车道0.5米
VISUALIZATION = True


# ======================== PID速度控制器 =========================
class SimplePID:
    def __init__(self):
        self.kp = 0.3
        self.ki = 0.008
        self.kd = 0.02
        self.error_sum = 0.0
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
    def update(self, current_speed):
        error = TARGET_SPEED_MPS - current_speed
        self.error_sum += error * (1 / SYNC_FPS)
        self.error_sum = np.clip(self.error_sum, -0.8, 0.8)
        derivative = (error - self.last_error) * SYNC_FPS
        self.last_error = error

        return throttle, brake
        throttle = self.kp * error + self.ki * self.error_sum + self.kd * derivative
        brake = 0.0 if error > -0.1 else 0.15
        return np.clip(throttle, 0.0, 1.0), brake


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
# ======================== 车道边界检测+细障碍物识别 =========================
class LaneBoundaryDetector:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()

        # 障碍物状态（细障碍物专用）
        self.has_obstacle = False
        self.obs_distance = float('inf')
        self.obs_direction = 0.0
        # 车道边界状态（核心：防止撞路边障碍物）
        self.lane_deviation = 0.0  # 偏离车道中心线距离（米）
        self.lane_steer_correction = 0.0  # 车道纠偏转向
        self.is_near_lane_edge = False  # 是否靠近车道边缘

        self.frame_queue = queue.Queue(maxsize=1) if VISUALIZATION else None

        # LiDAR（针对细障碍物优化：高密度+宽视野）
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(LIDAR_RANGE))
        lidar_bp.set_attribute('points_per_second', '25000')  # 超高密度，捕捉细障碍物
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('horizontal_fov', '90')  # 覆盖车道两侧
        lidar_bp.set_attribute('noise_stddev', '0.0')
        self.lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=1.5, z=1.2)), attach_to=vehicle)
        self.lidar.listen(self._lidar_callback)

        # 摄像头
        if VISUALIZATION:
            cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', '640')
            cam_bp.set_attribute('image_size_y', '480')
            self.cam = world.spawn_actor(cam_bp,
                                         carla.Transform(carla.Location(x=2.0, z=1.8), carla.Rotation(pitch=-8)),
                                         attach_to=vehicle)
            self.cam.listen(self._cam_callback)

    def _lidar_callback(self, data):
        """检测细障碍物（电线杆/路障/护栏等）"""
        points = np.frombuffer(data.raw_data, np.float32).reshape(-1, 4)[:, :3]
        vehicle_loc = self.vehicle.get_transform().location
        yaw = math.radians(self.vehicle.get_transform().rotation.yaw)

        # 车辆本地坐标系
        x_w = points[:, 0] - vehicle_loc.x
        y_w = points[:, 1] - vehicle_loc.y
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x_local = x_w * cos_yaw + y_w * sin_yaw
        y_local = -x_w * sin_yaw + y_w * cos_yaw

        # 过滤：覆盖车道两侧（左右4米），捕捉路边障碍物
        mask = (
                (x_local > 0.3) & (x_local < LIDAR_RANGE) &
                (abs(y_local) < 4.0) &  # 车道两侧各4米
                (points[:, 2] > 0.0) & (points[:, 2] < 4.0)  # 障碍物高度0-4米
        )
        valid_points = points[mask]

        if len(valid_points) >= DETECT_THRESHOLD:
            dists = np.sqrt((valid_points[:, 0] - vehicle_loc.x) ** 2 + (valid_points[:, 1] - vehicle_loc.y) ** 2)
            self.obs_distance = np.min(dists)
            self.has_obstacle = self.obs_distance < OBSTACLE_WARNING_DIST
            if self.has_obstacle:
                min_idx = np.argmin(dists)
                min_y_local = y_local[mask][min_idx]
                self.obs_direction = 1.0 if min_y_local > 0 else -1.0
        else:
            self.has_obstacle = False
            self.obs_distance = float('inf')

    def check_lane_boundary(self):
        """车道边界硬约束：计算偏离度，强制拉回中心"""
        vehicle_loc = self.vehicle.get_transform().location
        # 获取当前车道的中心线和边界
        current_waypoint = self.map.get_waypoint(vehicle_loc, project_to_road=True)
        lane_width = current_waypoint.lane_width  # 车道宽度（米）

        # 计算车辆到车道中心线的横向距离（偏离度）
        lane_center = current_waypoint.transform.location
        # 转换为车辆本地坐标系的横向距离（y轴）
        y_diff = (lane_center.y - vehicle_loc.y) * math.cos(math.radians(current_waypoint.transform.rotation.yaw)) - \
                 (lane_center.x - vehicle_loc.x) * math.sin(math.radians(current_waypoint.transform.rotation.yaw))
        self.lane_deviation = y_diff

        # 判断是否靠近车道边缘
        self.is_near_lane_edge = abs(self.lane_deviation) > (lane_width / 2 - MAX_LANE_DEVIATION)

        # 强制纠偏转向：偏离越多，纠偏力度越大
        if self.is_near_lane_edge:
            # 靠近边缘时，强力拉回中心
            self.lane_steer_correction = np.clip(self.lane_deviation / (lane_width / 4), -LANE_BOUNDARY_STRICT,
                                                 LANE_BOUNDARY_STRICT)
        else:
            # 轻微偏离时，柔和纠偏
            self.lane_steer_correction = np.clip(self.lane_deviation / (lane_width / 2), -0.3, 0.3) + LANE_CENTER_BIAS

    def _cam_callback(self, data):
        frame = np.frombuffer(data.raw_data, np.uint8).reshape(data.height, data.width, 4)[:, :, :3].copy()
        if not self.frame_queue.empty():
            try:


def main():
        for actor in world.get_actors():
            if actor.type_id.startswith("vehicle"):
                actor.destroy()
                self.frame_queue.get_nowait()
            except:
                continue
    time.sleep(1)
                pass
        self.frame_queue.put(frame, block=False)

    def draw_status(self):
        if not VISUALIZATION:
            return
        try:
            frame = self.frame_queue.get(timeout=0.01)
            speed = math.hypot(self.vehicle.get_velocity().x, self.vehicle.get_velocity().y) * 3.6
            # 叠加车道偏离+障碍物检测状态
            cv2.putText(frame, f"Speed: {speed:.1f}km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Lane Deviation: {self.lane_deviation:.2f}m", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if self.is_near_lane_edge else (0, 255, 0), 2)
            cv2.putText(frame, f"Obs Dist: {self.obs_distance:.2f}m", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)
            cv2.imshow("Lane & Obstacle Detection", frame)
            cv2.waitKey(1)
        except:
            pass

def get_vehicle_blueprint(world: carla.World) -> carla.ActorBlueprint:
    def destroy(self):
        self.lidar.stop()
        self.lidar.destroy()
        if VISUALIZATION:
            self.cam.stop()
            self.cam.destroy()
            cv2.destroyAllWindows()


# ======================== 修复：动态生成路边障碍物（适配所有Carla版本）========================
def spawn_roadside_obstacle(world, vehicle):
    """
    动态生成路边障碍物（适配所有Carla版本）：
    1. 优先找电线杆/路灯，找不到则用路障（static.prop.streetbarrier，所有版本都有）
    2. 生成在车道边缘，测试避障
    """
    bp_lib = world.get_blueprint_library()
    for vehicle_name in CONFIG["PREFERRED_VEHICLES"]:
    # 定义优先级列表：优先细障碍物，兜底用路障
    obstacle_blueprints = [
        'static.prop.pole',
        'static.prop.streetlight',
        'static.prop.streetbarrier',  # 兜底：所有版本都有
        'static.prop.trafficcone',
        'static.prop.barrier'
    ]

    # 查找可用的蓝图
    obstacle_bp = None
    for bp_name in obstacle_blueprints:
        try:
            bp = bp_lib.find(vehicle_name)
            bp.set_attribute('color', '255,0,0')
            return bp
        except:
            obstacle_bp = bp_lib.find(bp_name)
            print(f"✅ 找到可用障碍物蓝图：{bp_name}")
            break
        except IndexError:
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
    if obstacle_bp is None:
        # 终极兜底：随机选一个静态道具
        static_bps = [bp for bp in bp_lib if bp.id.startswith('static.prop.')]
        if static_bps:
            obstacle_bp = random.choice(static_bps)
            print(f"✅ 使用随机静态道具：{obstacle_bp.id}")
        else:
            raise RuntimeError("❌ 没有找到任何静态障碍物蓝图！")

    # 在车道右侧边缘0.5米处生成（前方10米）
    current_waypoint = world.get_map().get_waypoint(vehicle.get_transform().location)
    lane_width = current_waypoint.lane_width
    pole_waypoint = current_waypoint.next(10.0)[0]
    # 车道边缘位置（右侧0.5米）
    obstacle_loc = pole_waypoint.transform.location + pole_waypoint.transform.get_right_vector() * (
                lane_width / 2 + 0.5)
    obstacle_loc.z += 0.2  # 离地高度，避免碰撞

    # 生成障碍物（增加重试）
    for attempt in range(2):
        try:
            vehicle = world.spawn_actor(bp, safe_spawn_point)
            if vehicle and vehicle.is_alive:
                vehicle.set_simulate_physics(True)
                vehicle.set_autopilot(False)
                print(f"✅ 车辆生成成功（ID：{vehicle.id}）")
            obstacle = world.spawn_actor(obstacle_bp, carla.Transform(obstacle_loc))
            print(f"✅ 生成路边障碍物：车道右侧0.5米，前方10米处（{obstacle_loc.x:.1f}, {obstacle_loc.y:.1f}）")
            return [obstacle]
        except RuntimeError as e:
            if "collision" in str(e).lower():
                # 微调位置避免碰撞
                obstacle_loc.x += 0.5
                obstacle_loc.y += 0.5
                continue
            else:
                raise e
    raise RuntimeError("❌ 障碍物生成失败（位置碰撞）")


# ======================== 安全生成车辆（解决碰撞问题）========================
def spawn_vehicle_safely(world, bp):
    """
    安全生成车辆，避免碰撞：
    1. 筛选无碰撞的生成点
    2. 重试机制
    3. 自定义安全位置兜底
    """
    spawn_points = world.get_map().get_spawn_points()
    # 重试3次生成
    for attempt in range(3):
        if spawn_points:
            # 随机选择生成点，优先选车道中心的
            random.shuffle(spawn_points)
            for spawn_point in spawn_points:
                try:
                    # 检查生成点是否在行驶车道上
                    wp = world.get_map().get_waypoint(spawn_point.location)
                    if wp.lane_type != carla.LaneType.Driving:
                        continue
                    # 尝试生成车辆
                    vehicle = world.spawn_actor(bp, spawn_point)
                    print(
                        f"✅ 第{attempt + 1}次尝试：成功生成车辆（位置：{spawn_point.location.x:.1f}, {spawn_point.location.y:.1f}）")
                    return vehicle
                except RuntimeError as e:
                    if "collision" in str(e).lower():
                        continue
                    else:
                        raise e
        else:
            # 无默认生成点，使用自定义安全位置
            safe_loc = carla.Location(x=100.0, y=100.0, z=0.5)  # 自定义远离建筑的位置
            safe_transform = carla.Transform(safe_loc, carla.Rotation(yaw=0))
            try:
                vehicle = world.spawn_actor(bp, safe_transform)
                print(f"✅ 使用自定义安全位置生成车辆（位置：{safe_loc.x:.1f}, {safe_loc.y:.1f}）")
                return vehicle
            elif vehicle:
                vehicle.destroy()
        except Exception as e:
            print(f"⚠️ 第{retry + 1}次生成失败：{str(e)[:50]}")
            time.sleep(0.5)
    raise Exception("❌ 车辆生成失败")

            except RuntimeError as e:
                print(f"❌ 自定义位置生成失败：{e}")
                attempt += 1
    raise RuntimeError("❌ 所有生成点都有碰撞，无法生成车辆！")

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
# ======================== 核心逻辑（车道硬约束+障碍物避障）========================
def main():
    vehicle: Optional[carla.Vehicle] = None
    perception: Optional[EnhancedVehiclePerception] = None
    speed_controller: Optional[PreciseSpeedController] = None
    world: Optional[carla.World] = None
    # 1. 连接Carla
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1 / SYNC_FPS
    world.apply_settings(settings)

    # 2. 清理所有残留
    for actor in world.get_actors():
        if actor.type_id in ['vehicle.*', 'sensor.*', 'static.prop.*']:
            actor.destroy()
    time.sleep(1)

    # 3. 生成车辆（安全生成，避免碰撞）
    bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    bp.set_attribute('color', '255,0,0')
    try:
        vehicle = spawn_vehicle_safely(world, bp)
    except RuntimeError as e:
        print(e)
        return
    vehicle.set_simulate_physics(True)
    vehicle.set_autopilot(False)

    # 4. 生成路边障碍物（核心修复：适配所有版本）
    try:
        obstacles = spawn_roadside_obstacle(world, vehicle)
    except RuntimeError as e:
        print(e)
        # 销毁车辆后退出
        vehicle.destroy()
        return

    # 5. 第三人称视角（清晰看车道+障碍物）
    spectator = world.get_spectator()

    def third_person_view():
        trans = vehicle.get_transform()
        spectator_loc = trans.location - trans.get_forward_vector() * 4.5 + carla.Location(
            z=2.5) + trans.get_right_vector() * 0.5
        spectator_rot = carla.Rotation(pitch=-20, yaw=trans.rotation.yaw, roll=0)
        spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))

    # 6. 初始化检测器和控制器
    detector = LaneBoundaryDetector(world, vehicle)
    pid = SimplePID()
    current_steer = 0.0

    # 7. 核心行驶循环
    print("\n🚗 开始测试：车道硬约束 + 路边障碍物避障")
    print("核心规则：严格贴车道中心，1米内避开障碍物，零碰撞")
    print("按Ctrl+C停止\n")
    try:
        # 1. 初始化Carla
        client, world = get_carla_client()
        if not client or not world:
            raise Exception("❌ 未连接到Carla")
        spectator = world.get_spectator()
        print("✅ 成功连接Carla模拟器！")
        print("📌 当前仿真地图：", world.get_map().name)
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
        while True:
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
            third_person_view()

            # 1. 检测车道边界（优先级最高）
            detector.check_lane_boundary()

            # 2. 速度控制
            current_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
            throttle, brake = pid.update(current_speed)

            # 3. 转向逻辑：车道硬约束 > 1米紧急避障 > 预警避障
            target_steer = 0.0
            if detector.is_near_lane_edge:
                # 靠近车道边缘：强制拉回中心
                print(f"🔴 靠近车道边缘！偏离{detector.lane_deviation:.2f}米 | 强制拉回中心", end='\r')
                target_steer = detector.lane_steer_correction
                throttle *= 0.1  # 降速纠偏
            elif detector.obs_distance < OBSTACLE_EMERGENCY_DIST:
                # 1米内障碍物：紧急避障+车道约束
                print(f"⚠️ 紧急避障：距离障碍物{detector.obs_distance:.2f}米 | 贴车道绕开", end='\r')
                brake = 1.0
                throttle = 0.0
                # 避障+车道纠偏：既绕开又不越线
                target_steer = (-detector.obs_direction * 0.6) + detector.lane_steer_correction
            elif detector.has_obstacle:
                # 预警避障：贴车道绕行
                print(f"🔶 预警避障：距离障碍物{detector.obs_distance:.2f}米 | 顺车道绕开", end='\r')
                throttle *= 0.2
                target_steer = (-detector.obs_direction * 0.3) + detector.lane_steer_correction
            else:
                target_steer = 0.0

            # 7.4 转向超平滑过渡（避免速度波动）
            current_steer += (target_steer - current_steer) * CONFIG["STEER_SMOOTH_FACTOR"]
            current_steer = np.clip(current_steer, -CONFIG["AVOID_STEER_MAX"], CONFIG["AVOID_STEER_MAX"])
                # 正常行驶：严格贴车道中心
                print(f"✅ 正常行驶：车道偏离{detector.lane_deviation:.2f}米 | 速度{current_speed * 3.6:.1f}km/h",
                      end='\r')
                target_steer = detector.lane_steer_correction

            # 7.5 精准PID速度控制（核心匀速逻辑）
            throttle, brake = speed_controller.update(current_speed_mps, dt)
            # 转向平滑+硬限制
            current_steer += (target_steer - current_steer) * 0.25
            current_steer = np.clip(current_steer, -0.7, 0.7)

            # 7.6 卡停处理（仅低速时触发）
            if current_speed_kmh < CONFIG["STALL_SPEED_THRESHOLD"] * 3.6:
                trans = vehicle.get_transform()
                new_loc = trans.location + trans.get_forward_vector() * 1.5
                vehicle.set_transform(carla.Transform(new_loc, trans.rotation))
                throttle = 0.6  # 平缓恢复速度
                brake = 0.0
                print("\n⚠️ 低速重置位置，平缓恢复匀速...", end='')

            # 7.7 下发控制指令（匀速优先）
            # 下发控制
            vehicle.apply_control(carla.VehicleControl(
                throttle=float(throttle),
                steer=float(current_steer),
                brake=float(brake),
                hand_brake=False
                throttle=throttle, steer=current_steer, brake=brake, hand_brake=False
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
            # 可视化
            detector.draw_status()

    except KeyboardInterrupt:
        print("\n\n🛑 测试停止，清理资源...")
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
        # 清理所有资源
        detector.destroy()
        vehicle.destroy()
        for obs in obstacles:
            obs.destroy()
        # 恢复Carla设置
        settings.synchronous_mode = False
        world.apply_settings(settings)
        cv2.destroyAllWindows()
        print("✅ 清理完成！")


if __name__ == "__main__":