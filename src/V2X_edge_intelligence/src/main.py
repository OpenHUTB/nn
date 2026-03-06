#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuJoCo自动巡航小车 - 优化版
核心功能：
1. 恒定速度巡航（可配置）
2. 前方障碍物检测（0.5米阈值）
3. 90度精准转向避障（左/右随机）
4. R键复位
5. 状态机管理（巡航/停止/转向/恢复）
CARLA 0.9.10+ 自动驾驶控制程序
特性：
1. 完全动态加载CARLA，无任何硬编码绝对路径
2. 增强功能：多地图切换、实时数据监控、日志记录、智能避障
3. 保留核心：晚转弯、大转向角度、平稳速度控制
"""

# v2x_balance_zones.py（三区平均分配+低速精准控速）
import sys
import os
import carla
import time
import numpy as np
import json
import math
import matplotlib.pyplot as plt
import cv2
from collections import deque
import random
import sys
import os
import time
from typing import Tuple, Optional, Dict


# ===================== 核心配置 =====================
class Config:
    # 速度配置
    CRUISE_SPEED = 0.0015  # 巡航速度（原两倍速）
    TURN_SPEED = 0.00075  # 转向速度（巡航速度的50%）
    # 距离配置
    OBSTACLE_THRESHOLD = 0.5  # 障碍物检测阈值（米）
    SAFE_DISTANCE = 0.2  # 紧急停止距离（米）
    # 转向配置
    TURN_ANGLE = math.pi / 2  # 90度转向（弧度）
    STEER_LIMIT = 0.3  # 转向角度限制（匹配XML中的ctrlrange）
    # 模型配置
    MODEL_PATH = "wheeled_car.xml"
    CHASSIS_NAME = "chassis"
    # 障碍物名称列表
    OBSTACLE_NAMES = [
        'obs_box1', 'obs_box2', 'obs_box3', 'obs_box4',
        'obs_ball1', 'obs_ball2', 'obs_ball3',
        'wall1', 'wall2', 'front_dark_box'
import math
import random
import logging
from typing import Tuple, Optional, Dict
from collections import deque

import cv2
import numpy as np
import mujoco
import mujoco.viewer
import carla
from pynput import keyboard
import matplotlib.pyplot as plt

# ===================== 基础配置 =====================
# 获取脚本所在目录（核心：动态路径解析，避免绝对路径）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 相对路径配置（所有资源文件都基于脚本目录）
CONFIG = {
    # MuJoCo配置
    "MODEL_REL_PATH": "wheeled_car.xml",  # 相对于脚本的路径
    "CHASSIS_NAME": "chassis",
    # 速度配置
    "CRUISE_SPEED": 0.0015,
    "TURN_SPEED": 0.00075,
    # 距离配置
    "OBSTACLE_THRESHOLD": 0.5,
    "SAFE_DISTANCE": 0.2,
    # 转向配置
    "TURN_ANGLE": math.pi / 2,
    "STEER_LIMIT": 0.3,
    # 障碍物名称
    "OBSTACLE_NAMES": [
        'obs_box1', 'obs_box2', 'obs_box3', 'obs_box4',
        'obs_ball1', 'obs_ball2', 'obs_ball3',
        'wall1', 'wall2', 'front_dark_box'
    ],
    # 视图配置
    "CAM_DISTANCE": 2.5,
    "CAM_ELEVATION": -25,
    # CARLA配置
    "TARGET_SPEED": 20.0,  # km/h
    "PID_KP": 0.3,
    "PID_KI": 0.02,
    "PID_KD": 0.01,
    "LOOKAHEAD_DISTANCE": 8.0,
    "MAX_STEER_ANGLE": 30.0,
    "CAMERA_WIDTH": 800,
    "CAMERA_HEIGHT": 600,
    "CAMERA_FOV": 90,
    "VEHICLE_MODEL": "vehicle.tesla.model3",
    "PLOT_SIZE": (12, 10),
    "WAYPOINT_COUNT": 50
}

# ===================== 日志配置（新增） =====================
# ===================== 1. 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'carla_drive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ===================== 核心配置 =====================
class Config:
    # 速度控制
    TARGET_SPEED = 20.0  # km/h
    PID_KP = 0.3
    PID_KI = 0.02
    PID_KD = 0.01

    # 纯追踪算法参数
    LOOKAHEAD_DISTANCE = 8.0  # 前瞻距离（米）
    MAX_STEER_ANGLE = 30.0  # 最大转向角（度）

    # 摄像头配置
    CAMERA_WIDTH = 800
    CAMERA_HEIGHT = 600
    CAMERA_FOV = 90

    # 车辆配置
    VEHICLE_MODEL = "vehicle.tesla.model3"

    # 可视化配置
    PLOT_SIZE = (12, 10)
    WAYPOINT_COUNT = 50  # 预先生成的道路航点数量


# ===================== 工具类 =====================
class Tools:
    @staticmethod
    def normalize_angle(angle):
        """将角度归一化到[-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    @staticmethod
    def get_vehicle_pose(vehicle):
        """获取车辆的位置、朝向和速度"""
        transform = vehicle.get_transform()
        loc = transform.location
        yaw = math.radians(transform.rotation.yaw)
        vel = vehicle.get_velocity()
        speed = 3.6 * np.linalg.norm([vel.x, vel.y, vel.z])
        return loc, yaw, speed

    @staticmethod
    def clear_all_actors(world):
        """清理所有车辆和传感器"""
        for actor in world.get_actors():
            try:
                if actor.type_id.startswith('vehicle') or actor.type_id.startswith('sensor'):
                    actor.destroy()
            except Exception as e:
                logger.warning(f"清理Actor失败: {e}")
        time.sleep(0.5)
        logger.info("已清理所有残留Actor")

    @staticmethod
    def focus_vehicle(world, vehicle):
        """将CARLA客户端视角聚焦到车辆"""
        spectator = world.get_spectator()
        t = vehicle.get_transform()
        spectator.set_transform(carla.Transform(t.location + carla.Location(x=0, y=-8, z=5), t.rotation))
        logger.info("CARLA客户端已聚焦到车辆")

    @staticmethod
    def generate_road_waypoints(world, start_loc, count=50, step=2.0):
        """
        从起点沿道路生成连续的原生航点
        :param world: CARLA世界对象
        :param start_loc: 起点位置
        :param count: 航点数量
        :param step: 每个航点的步长（米）
        :return: 航点列表[(x, y, z), ...]
        """
        waypoints = []
        map = world.get_map()
        wp = map.get_waypoint(start_loc)
        for i in range(count):
            waypoints.append((wp.transform.location.x, wp.transform.location.y, wp.transform.location.z))
            # 沿道路下一个航点（直走，不考虑分叉）
            next_wps = wp.next(step)
            if next_wps:
                wp = next_wps[0]
            else:
# ===================== 1. 动态配置CARLA路径（无硬编码绝对路径） =====================
def setup_carla_path():
    """
    动态查找并配置CARLA路径（优先级：
    1. 环境变量 CARLA_PYTHON_API
    2. 当前目录及子目录
    3. 用户主目录
    """
    # 尝试从环境变量获取
    carla_egg_env = os.getenv('CARLA_PYTHON_API')
    if carla_egg_env and os.path.exists(carla_egg_env):
        egg_path = carla_egg_env
        logger.info(f"🔍 从环境变量获取CARLA路径：{egg_path}")
    else:
        # 动态搜索常见位置（无硬编码绝对路径）
        search_paths = [
            os.getcwd(),  # 当前目录
            os.path.expanduser("~"),  # 用户主目录
            # 相对路径搜索（CARLA通常的PythonAPI相对位置）
            os.path.join(os.getcwd(), "PythonAPI", "carla", "dist"),
            os.path.join(os.path.dirname(os.getcwd()), "PythonAPI", "carla", "dist")
        ]

        egg_path = None
        # 搜索所有.py3.7相关的egg文件（适配0.9.10）
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
            for file in os.listdir(search_path):
                if file.startswith("carla-0.9.10-py3.7") and file.endswith(".egg"):
                    egg_path = os.path.join(search_path, file)
                    logger.info(f"🔍 自动找到CARLA egg文件：{egg_path}")
                    break
            if egg_path:
                break
        logger.info(f"生成了{len(waypoints)}个原生道路航点")
        return waypoints


# --------------------------
# 1. 障碍物检测器（优化性能+无延迟检测）
# --------------------------
class ObstacleDetector:
    def __init__(self, world, vehicle, max_distance=50.0, detect_interval=1):
        self.world = world
    # 验证并添加到路径
    if egg_path and os.path.exists(egg_path):
        if egg_path not in sys.path:
            sys.path.insert(0, egg_path)
        logger.info(f"✅ CARLA egg路径已添加：{egg_path}")
        return True
    else:
        logger.error("\n❌ 未找到CARLA egg文件！")
        logger.info("📌 请通过以下方式配置：")
        logger.info("   1. 设置环境变量：CARLA_PYTHON_API=你的egg文件路径")
        logger.info("   2. 或将egg文件放到当前脚本目录")
        logger.info("   3. 确保CARLA版本为0.9.10（py3.7）")
        return False


# 配置CARLA路径
if not setup_carla_path():
    sys.exit(1)

# 导入CARLA（动态路径配置后）
try:
    import carla

    logger.info("✅ CARLA模块导入成功！")
except Exception as e:
    logger.error(f"\n❌ 导入CARLA失败：{str(e)}")
    sys.exit(1)


# ===================== 2. 核心：三区平均分配+低速精准控速 =====================
class RoadSideUnit:
    def __init__(self, carla_world, vehicle):
        self.world = carla_world
        self.vehicle = vehicle
        self.max_distance = max_distance
        self.detect_interval = detect_interval  # 检测间隔（帧）
        self.frame_count = 0
        self.last_obstacle_info = {
            'has_obstacle': False,
            'distance': float('inf'),
            'relative_angle': 0.0,
            'obstacle_type': None,
            'obstacle_speed': 0.0,
            'relative_speed': 0.0  # 自车与障碍物的相对速度
        # 1. 三区坐标（等距分配，每区长度一致）
        spawn_loc = vehicle.get_location()
        # 高速区：生成位置前5-15米（长度10米）
        self.high_zone_start = carla.Location(spawn_loc.x, spawn_loc.y + 5, spawn_loc.z)
        self.high_zone_end = carla.Location(spawn_loc.x, spawn_loc.y + 15, spawn_loc.z)
        # 中速区：生成位置前15-25米（长度10米）
        self.mid_zone_start = carla.Location(spawn_loc.x, spawn_loc.y + 15, spawn_loc.z)
        self.mid_zone_end = carla.Location(spawn_loc.x, spawn_loc.y + 25, spawn_loc.z)
        # 低速区：生成位置前25-35米（长度10米）
        self.low_zone_start = carla.Location(spawn_loc.x, spawn_loc.y + 25, spawn_loc.z)
        self.low_zone_end = carla.Location(spawn_loc.x, spawn_loc.y + 35, spawn_loc.z)

        # 2. 三区计时（确保每区停留约10秒）
        self.current_zone = "high"  # 初始区：高速
        self.zone_start_time = time.time()
        self.zone_duration = 10  # 每区停留10秒（30秒测试，三区各10秒）
        self.speed_map = {"high": 40, "mid": 25, "low": 10}

    def get_balance_speed_limit(self):
        """核心：计时强制切换+位置双重判断，确保三区平均分配"""
        current_time = time.time()
        vehicle_loc = self.vehicle.get_location()
        vehicle_y = vehicle_loc.y  # 沿行驶方向的核心坐标

        # 1. 计时判断：每区停留10秒强制切换
        if current_time - self.zone_start_time > self.zone_duration:
            if self.current_zone == "high":
                self.current_zone = "mid"
            elif self.current_zone == "mid":
                self.current_zone = "low"
            elif self.current_zone == "low":
                self.current_zone = "high"  # 循环切换（避免一直停低速）
            self.zone_start_time = current_time  # 重置计时
            logger.info(f"⏰ 计时触发区域切换：{self.current_zone}")

        # 2. 位置双重验证：确保区域与位置匹配
        spawn_y = self.vehicle.get_location().y
        if spawn_y + 5 <= vehicle_y < spawn_y + 15:
            self.current_zone = "high"
        elif spawn_y + 15 <= vehicle_y < spawn_y + 25:
            self.current_zone = "mid"
        elif spawn_y + 25 <= vehicle_y < spawn_y + 35:
            self.current_zone = "low"

        # 返回对应速度和区域名称
        speed_limit = self.speed_map[self.current_zone]
        zone_name = {
            "high": "高速区(40km/h)",
            "mid": "中速区(25km/h)",
            "low": "低速区(10km/h)"
        }[self.current_zone]
        return speed_limit, zone_name

    def send_speed_command(self, vehicle_id, speed_limit, zone_type):
        command = {
            "vehicle_id": vehicle_id,
            "speed_limit_kmh": speed_limit,
            "zone_type": zone_type,
            "timestamp": time.time()
        }
        logger.info(f"\n📡 路侧V2X指令：{json.dumps(command, indent=2, ensure_ascii=False)}")
        return command

    def get_vehicle_speed(self, vehicle):
        """获取车辆速度（km/h）"""
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        return speed * 3.6

    def get_obstacle_info(self):
        """检测前方障碍物信息（无延迟检测）"""
        self.frame_count += 1
        if self.frame_count % self.detect_interval != 0:
            return self.last_obstacle_info

        try:
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            forward_vector = vehicle_transform.get_forward_vector()
            self_speed = self.get_vehicle_speed(self.vehicle)  # 自车速度

            # 减少get_actors调用频率，只获取车辆
            all_vehicles = self.world.get_actors().filter('vehicle.*')
            min_distance = float('inf')
            closest_obstacle = None
            relative_angle = 0.0
            obstacle_speed = 0.0

            for other_vehicle in all_vehicles:
                if other_vehicle.id == self.vehicle.id:
                    continue

                other_location = other_vehicle.get_location()
                distance = vehicle_location.distance(other_location)
                if distance > self.max_distance:
                    continue

                # 计算相对角度（仅前方±70度，扩大检测范围）
                relative_vector = carla.Location(
                    other_location.x - vehicle_location.x,
                    other_location.y - vehicle_location.y,
                    0
                )
                forward_2d = carla.Vector3D(forward_vector.x, forward_vector.y, 0)
                relative_2d = carla.Vector3D(relative_vector.x, relative_vector.y, 0)

                # 向量归一化
                forward_norm = math.sqrt(forward_2d.x ** 2 + forward_2d.y ** 2)
                relative_norm = math.sqrt(relative_2d.x ** 2 + relative_2d.y ** 2)
                if forward_norm == 0 or relative_norm == 0:
                    continue

                dot_product = forward_2d.x * relative_2d.x + forward_2d.y * relative_2d.y
                cos_angle = dot_product / (forward_norm * relative_norm)
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angle_deg = math.degrees(math.acos(cos_angle))

                # 扩大检测角度到±70度，更早发现障碍物
                if angle_deg <= 70 and distance < min_distance:
                    min_distance = distance
                    closest_obstacle = other_vehicle
                    obstacle_speed = self.get_vehicle_speed(other_vehicle)
                    # 确定角度方向
                    relative_angle = angle_deg if relative_2d.y >= 0 else -angle_deg

            # 计算相对速度（自车速度 - 前车速度，正数表示自车更快）
            relative_speed = self_speed - obstacle_speed if closest_obstacle else 0.0

            # 更新障碍物信息
            if closest_obstacle is not None:
                self.last_obstacle_info = {
                    'has_obstacle': True,
                    'distance': min_distance,
                    'relative_angle': relative_angle,
                    'obstacle_type': closest_obstacle.type_id,
                    'obstacle_speed': obstacle_speed,
                    'relative_speed': relative_speed
                }
class VehicleUnit:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.vehicle.set_autopilot(False)
        self.control = carla.VehicleControl()
        self.control.steer = 0.0  # 强制直行
        self.control.hand_brake = False
        logger.info("✅ 车辆已设置为手动直行（精准控速）")

    def get_actual_speed(self):
        """获取车辆实际速度（km/h）"""
        velocity = self.vehicle.get_velocity()
        speed_kmh = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6
        return round(speed_kmh, 1)

    def precise_speed_control(self, target_speed):
        """核心修复：低速区加大油门，精准到10km/h"""
        actual_speed = self.get_actual_speed()

        # 1. 高速区：38-42km/h（精准控速）
        if target_speed == 40:
            if actual_speed > 42:
                self.control.throttle = 0.0
                self.control.brake = 0.4
            elif actual_speed < 38:
                self.control.throttle = 0.9
                self.control.brake = 0.0
            else:
                self.last_obstacle_info = {
                    'has_obstacle': False,
                    'distance': float('inf'),
                    'relative_angle': 0.0,
                    'obstacle_type': None,
                    'obstacle_speed': 0.0,
                    'relative_speed': 0.0
                }

        except Exception as e:
            logger.error(f"障碍物检测错误: {e}")

        return self.last_obstacle_info

    def visualize_obstacles(self, image, vehicle_transform):
        """在图像上可视化障碍物检测结果"""
        if not self.last_obstacle_info['has_obstacle']:
            return image

        height, width = image.shape[:2]
        distance = self.last_obstacle_info['distance']
        angle = self.last_obstacle_info['relative_angle']

        # 计算障碍物在图像中的位置
        x_pos = int(width / 2 + (angle / 70) * (width / 2))  # 适配70度检测范围
        x_pos = max(0, min(width - 1, x_pos))

        # 根据距离设置颜色和大小
        if distance < 15:
            color = (0, 0, 255)
            radius = 15
        elif distance < 30:
            color = (0, 165, 255)
            radius = 10
        else:
            color = (0, 255, 255)
            radius = 5

        # 绘制障碍物指示器
        cv2.circle(image, (x_pos, int(height * 0.8)), radius, color, -1)
        cv2.putText(image, f"{distance:.1f}m", (x_pos - 20, int(height * 0.8) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # 绘制相对速度
        rel_speed = self.last_obstacle_info['relative_speed']
        cv2.putText(image, f"RelSpeed: {rel_speed:.1f}km/h", (x_pos - 20, int(height * 0.8) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image


# --------------------------
# 2. 传统控制器（核心控制逻辑+优化避障）
# --------------------------
class TraditionalController:
    """基于路点的传统控制器，整合避障"""

    def __init__(self, world, obstacle_detector):
        self.world = world
        self.map = world.get_map()
        self.waypoint_distance = 10.0
        self.obstacle_detector = obstacle_detector
        # 调整避障阈值：扩大距离，提前减速
        self.emergency_brake_distance = 12.0  # 从6米改为12米
        self.safe_following_distance = 18.0  # 从10米改为18米
        self.early_warning_distance = 30.0  # 新增：30米提前预警

    def apply_obstacle_avoidance(self, throttle, brake, steer, vehicle, obstacle_info):
        """传统控制器的避障逻辑（优化刹车力度+相对速度）"""
        if not obstacle_info['has_obstacle']:
            return throttle, brake, steer

        distance = obstacle_info['distance']
        angle = obstacle_info['relative_angle']
        vehicle_speed = self.obstacle_detector.get_vehicle_speed(vehicle)
        relative_speed = obstacle_info['relative_speed']  # 自车与前车的相对速度

        # 1. 提前预警（30米内）：轻微减速，降低油门
        if distance < self.early_warning_distance and relative_speed > 0:
            throttle *= 0.5  # 油门减半
            if vehicle_speed > 30:
                brake = 0.2  # 轻微刹车

        # 2. 紧急刹车（12米内）：全力刹车+手刹
        if distance < self.emergency_brake_distance:
            logger.warning(f"紧急刹车！距离前车: {distance:.1f}m, 相对速度: {relative_speed:.1f}km/h")
            return 0.0, 1.0, 0.0  # brake=1.0 + 后续拉手刹

        # 3. 安全跟车（18米内）：动态调整刹车力度
        elif distance < self.safe_following_distance:
            # 根据距离和相对速度计算所需刹车力度
            required_distance = max(8.0, vehicle_speed * 0.5)  # 增加安全车距系数
            distance_ratio = (distance - required_distance) / self.safe_following_distance
            distance_ratio = max(0.0, min(1.0, distance_ratio))

            # 相对速度越大，刹车越重
            brake_strength = (1 - distance_ratio) * 0.8 + (relative_speed / 20) * 0.2
            brake_strength = max(0.3, min(0.8, brake_strength))

            if distance < required_distance:
                throttle = 0.0
                brake = brake_strength
                self.control.throttle = 0.2
                self.control.brake = 0.0

        # 2. 中速区：23-27km/h（精准控速）
        elif target_speed == 25:
            if actual_speed > 27:
                self.control.throttle = 0.0
                self.control.brake = 0.3
            elif actual_speed < 23:
                self.control.throttle = 0.6
                self.control.brake = 0.0
            else:
                throttle = 0.1
                brake = 0.0

            # 尝试变道
            if abs(angle) < 15:
                location = vehicle.get_location()
                waypoint = self.map.get_waypoint(location)
                left_lane = waypoint.get_left_lane()
                right_lane = waypoint.get_right_lane()

                if left_lane and left_lane.lane_type == carla.LaneType.Driving:
                    steer = -0.3
                elif right_lane and right_lane.lane_type == carla.LaneType.Driving:
                    steer = 0.3
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

            self.lidar_sensor.listen(lidar_callback)
            print("✅ 强化LiDAR初始化成功（64线+降噪）")
        except Exception as e:
            print(f"⚠️ LiDAR初始化失败：{e}")

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
    def _init_camera(self):
        """强化摄像头：高分辨率+实时可视化"""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            logger.info(f"✅ 成功加载MuJoCo模型: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"❌ 加载MuJoCo模型失败: {e}")

    def _init_ids(self):
        """初始化body ID缓存"""
        # 小车底盘ID
        self.chassis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.config["CHASSIS_NAME"]
        )
        if self.chassis_id == -1:
            raise RuntimeError(f"❌ 未找到底盘: {self.config['CHASSIS_NAME']}")

        # 障碍物ID
        self.obstacle_ids = {}
        for name in self.config["OBSTACLE_NAMES"]:
            obs_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if obs_id != -1:
                self.obstacle_ids[name] = obs_id
        logger.info(f"✅ 加载{len(self.obstacle_ids)}个障碍物ID")

    def reset_car(self):
        """复位小车到初始状态"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.03  # 设置初始高度
        self.car_state = CarState.CRUISING
        self.target_steer_angle = 0.0
        self.turn_progress = 0.0
        logger.info("\n🔄 MuJoCo小车已复位")

    def detect_obstacle(self) -> Tuple[int, float, Optional[str]]:
        """检测前方障碍物"""
        chassis_pos = self.data.body(self.chassis_id).xpos
        min_distance = float('inf')
        closest_obs = None

        for obs_name, obs_id in self.obstacle_ids.items():
            try:
                obs_pos = self.data.body(obs_id).xpos
                dx = obs_pos[0] - chassis_pos[0]
                dy = obs_pos[1] - chassis_pos[1]
                distance = math.hypot(dx, dy)

                if dx > 0 and abs(dy) < 0.3 and distance < self.config["OBSTACLE_THRESHOLD"]:
                    if distance < min_distance:
                        min_distance = distance
                        closest_obs = obs_name
            except Exception:
                continue

        # 判断状态
        if closest_obs is None:
            return 0, 0.0, None
        elif min_distance < self.config["SAFE_DISTANCE"]:
            return 2, min_distance, closest_obs
        else:
            return 1, min_distance, closest_obs

    def _set_steer(self, angle: float):
        """设置转向角度"""
        clamped_angle = np.clip(angle, -self.config["STEER_LIMIT"], self.config["STEER_LIMIT"])
        self.data.ctrl[0] = clamped_angle  # 前左转向
        self.data.ctrl[1] = clamped_angle  # 前右转向

    def _set_speed(self, speed: float):
        """设置驱动速度"""
        self.data.ctrl[2] = speed  # 前左驱动
        self.data.ctrl[3] = speed  # 前右驱动
        self.data.ctrl[4] = speed  # 后左驱动
        self.data.ctrl[5] = speed  # 后右驱动

    def update(self):
        """更新小车状态和控制指令"""
        # 检查复位键
        if self.key_manager.is_pressed('r'):
            self.reset_car()
            self.key_manager.reset_key('r')
            return

        # 检测障碍物
        obs_status, obs_dist, obs_name = self.detect_obstacle()

        # 状态机逻辑
        if self.car_state == CarState.CRUISING:
            if obs_status == 2:
                self.car_state = CarState.STOPPED
                self._set_speed(0.0)
                logger.warning(f"\n🛑 紧急停止！障碍物：{obs_name} (距离: {obs_dist:.2f}m)")
            elif obs_status == 1:
                self.car_state = CarState.STOPPED
                self._set_speed(0.0)
                logger.warning(f"\n⚠️  检测到障碍物：{obs_name} (距离: {obs_dist:.2f}m)")
            else:
                self._set_steer(0.0)
                self._set_speed(self.config["CRUISE_SPEED"])

        elif self.car_state == CarState.STOPPED:
            self.turn_progress += 1
            self._set_speed(0.0)

            if self.turn_progress > 10:
                self.turn_direction = 1 if random.random() > 0.5 else -1
                self.target_steer_angle = self.turn_direction * self.config["TURN_ANGLE"]
                self.car_state = CarState.TURNING
                self.turn_progress = 0.0
                dir_text = "右" if self.turn_direction > 0 else "左"
                logger.info(f"\n🔄 开始{dir_text}转90度避障")

        elif self.car_state == CarState.TURNING:
            self.turn_progress += 0.01
            current_steer = self.target_steer_angle * self.turn_progress
            self._set_steer(current_steer)
            self._set_speed(self.config["TURN_SPEED"])

            if self.turn_progress >= 1.0 or abs(current_steer) >= self.config["STEER_LIMIT"]:
                self.car_state = CarState.RESUMING
                self.turn_progress = 0.0
                logger.info("\n✅ 转向完成，开始恢复巡航")

        elif self.car_state == CarState.RESUMING:
            self.turn_progress += 0.01
            current_steer = self.target_steer_angle * (1 - self.turn_progress)
            self._set_steer(current_steer)

            # 修复：CRUISING是状态枚举，应该用CRUISE_SPEED
            current_speed = self.config["TURN_SPEED"] + (
                    self.config["CRUISE_SPEED"] - self.config["TURN_SPEED"]
            ) * self.turn_progress
            self._set_speed(current_speed)

            if self.turn_progress >= 1.0:
                if obs_status == 0:
                    self.car_state = CarState.CRUISING
                    logger.info("\n🚗 前方安全，恢复正常巡航")
                else:
                    self.car_state = CarState.STOPPED
                    logger.warning("\n⚠️  转向后仍有障碍，重新停止")
                self.turn_progress = 0.0

    def get_status_info(self) -> str:
        """获取状态信息"""
        vel = np.linalg.norm(self.data.qvel[:3])
        current_steer = math.degrees((self.data.ctrl[0] + self.data.ctrl[1]) / 2)

        info = f"状态: {self.car_state:<8} | 速度: {vel:.5f} m/s"
        if abs(current_steer) > 0.1:
            info += f" | 转向: {current_steer:.1f}°"

        if self.car_state == CarState.CRUISING:
            obs_status, obs_dist, obs_name = self.detect_obstacle()
            if obs_status > 0 and obs_name:
                info += f" | 前方障碍: {obs_name} ({obs_dist:.2f}m)"

        return info


# ===================== CARLA工具类 =====================
class CarlaTools:
    """CARLA辅助工具类"""

    @staticmethod
    def normalize_angle(angle):
        """将角度归一化到[-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    @staticmethod
    def get_vehicle_pose(vehicle):
        """获取车辆的位置、朝向和速度"""
        transform = vehicle.get_transform()
        loc = transform.location
        yaw = math.radians(transform.rotation.yaw)
        vel = vehicle.get_velocity()
        speed = 3.6 * np.linalg.norm([vel.x, vel.y, vel.z])
        return loc, yaw, speed

    @staticmethod
    def clear_all_actors(world):
        """清理所有车辆和传感器"""
        for actor in world.get_actors():
            try:
                if actor.type_id.startswith('vehicle') or actor.type_id.startswith('sensor'):
                    actor.destroy()
            except Exception as e:
                logger.warning(f"清理Actor失败: {e}")
        time.sleep(0.5)
        logger.info("已清理所有残留Actor")

    @staticmethod
    def focus_vehicle(world, vehicle):
        """将CARLA客户端视角聚焦到车辆"""
        spectator = world.get_spectator()
        t = vehicle.get_transform()
        spectator.set_transform(
            carla.Transform(t.location + carla.Location(x=0, y=-8, z=5), t.rotation)
        )
        logger.info("CARLA客户端已聚焦到车辆")

    @staticmethod
    def generate_road_waypoints(world, start_loc, count=50, step=2.0):
        """生成道路航点"""
        waypoints = []
        map = world.get_map()
        wp = map.get_waypoint(start_loc)
        for i in range(count):
            waypoints.append((wp.transform.location.x, wp.transform.location.y, wp.transform.location.z))
            next_wps = wp.next(step)
            if next_wps:
                wp = next_wps[0]
            else:
                break
        logger.info(f"生成了{len(waypoints)}个原生道路航点")
        return waypoints


# ===================== 主函数 =====================
def main():
    """主程序入口"""
    # 初始化MuJoCo控制器
    try:
        mujoco_controller = MujoCoCarController()
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(e)
        return

    # 启动MuJoCo可视化
    viewer = None
    try:
        viewer = mujoco.viewer.launch_passive(mujoco_controller.model, mujoco_controller.data)
        viewer.cam.distance = CONFIG["CAM_DISTANCE"]
        viewer.cam.elevation = CONFIG["CAM_ELEVATION"]

        # MuJoCo主循环
        logger.info("\n🚀 MuJoCo自动巡航小车启动（按R键复位）")
        logger.info("=====================================")

        while viewer.is_running():
            # 更新小车控制
            mujoco_controller.update()

            # 执行仿真步骤
            mujoco.mj_step(mujoco_controller.model, mujoco_controller.data)

# 主函数（匀速+强化感知）
def main():
    config = Config()
    tools = Tools()
                self.control.throttle = 0.1
                self.control.brake = 0.0

        # 3. 低速区：9-11km/h（加大油门，确保到10km/h）
        elif target_speed == 10:
            if actual_speed > 11:
                self.control.throttle = 0.0
                self.control.brake = 0.2
            elif actual_speed < 9:
                self.control.throttle = 0.4  # 加大油门（原0.2→0.4）
                self.control.brake = 0.0
            else:
                self.control.throttle = 0.15  # 维持油门
                self.control.brake = 0.0

    # 初始化变量
    vehicle = None
    third_camera = None
    front_camera = None
    camera = None
    visualizer = None
    third_image = None
    front_image = None
        self.vehicle.apply_control(self.control)
        return actual_speed

    def receive_speed_command(self, command):
        """接收并执行速度指令"""
        target_speed = command["speed_limit_kmh"]
        actual_speed = self.precise_speed_control(target_speed)
        logger.info(
            f"🚗 车载执行：目标{target_speed}km/h → 实际{actual_speed}km/h | 油门={round(self.control.throttle, 1)} 刹车={round(self.control.brake, 1)}")

        # 2. 清理残留Actor
        clean_actors(world)

# ===================== 3. 近距离视角 =====================
def set_near_observation_view(world, vehicle):
    """设置车辆后方近距离观察视角"""
    try:
        # 连接CARLA服务器（0.9.10兼容）
        client = carla.Client('localhost', 2000)
        client.set_timeout(30.0)
        world = client.load_world('Town01')  # 0.9.10支持Town01
        logger.info("成功连接CARLA并加载Town01地图")
        spectator = world.get_spectator()
        vehicle_transform = vehicle.get_transform()
        forward_vector = vehicle_transform.rotation.get_forward_vector()
        right_vector = vehicle_transform.rotation.get_right_vector()
        view_location = vehicle_transform.location - forward_vector * 8 + right_vector * 2 + carla.Location(z=2)
        view_rotation = carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw, roll=0)
        spectator.set_transform(carla.Transform(view_location, view_rotation))
        logger.info("✅ 初始视角已设置：车辆后方近距离")
        logger.info("📌 视角操作：鼠标拖拽=旋转 | 滚轮=缩放 | WASD=移动")
    except Exception as e:
        logger.warning(f"⚠️ 设置视角失败：{e}")

        # 设置同步模式（0.9.10关键配置）
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        logger.info("已设置同步模式，delta=0.1s")

        # 清理Actor
        tools.clear_all_actors(world)

        # 设置天气
        weather = carla.WeatherParameters(
            cloudiness=30.0,
            precipitation=0.0,
            sun_altitude_angle=70.0
        )
        world.set_weather(weather)
        logger.info("已设置天气：晴朗，30%云量")

        # 获取出生点
        map = world.get_map()
        spawn_points = map.get_spawn_points()

def get_valid_spawn_point(world):
    """获取有效生成点（容错处理）"""
    try:
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise Exception("无可用出生点")
        spawn_point = spawn_points[10]
        logger.info(f"使用出生点：{spawn_point.location}")

        # 生成主车辆
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(config.VEHICLE_MODEL)
        vehicle_bp.set_attribute('color', '255,0,0')
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        if not vehicle:
            raise Exception("无法生成主车辆")
        vehicle.set_autopilot(False)
        logger.info(f"车辆{config.VEHICLE_MODEL}生成成功，位置: {spawn_point.location}")

        # 聚焦车辆
        tools.focus_vehicle(world, vehicle)

        # 生成障碍物车辆（调整生成位置，确保在前车前方）
        obstacle_count = 3
        for i in range(obstacle_count):
            spawn_idx = (i + 12) % len(spawn_points)  # 从15改为12，更靠近主车辆
            other_vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
            other_vehicle = world.try_spawn_actor(other_vehicle_bp, spawn_points[spawn_idx])
            if other_vehicle:
                other_vehicle.set_autopilot(True)
                logger.info(f"生成障碍物车辆 {other_vehicle.type_id} 在位置 {spawn_points[spawn_idx].location}")

        # 配置传感器（0.9.10兼容）
        # 后视角相机
        third_camera_bp = blueprint_library.find('sensor.camera.rgb')
        third_camera_bp.set_attribute('image_size_x', '640')
        third_camera_bp.set_attribute('image_size_y', '480')
        third_camera_bp.set_attribute('fov', '110')
        third_camera_transform = carla.Transform(
            carla.Location(x=-5.0, y=0.0, z=3.0),
            carla.Rotation(pitch=-15.0)
        )
        third_camera = world.spawn_actor(third_camera_bp, third_camera_transform, attach_to=vehicle)

        # 前视角相机
        front_camera_bp = blueprint_library.find('sensor.camera.rgb')
        front_camera_bp.set_attribute('image_size_x', '640')
        front_camera_bp.set_attribute('image_size_y', '480')
        front_camera_bp.set_attribute('fov', '90')
        front_camera_transform = carla.Transform(
            carla.Location(x=2.0, y=0.0, z=1.5),
            carla.Rotation(pitch=0.0)
        )
        front_camera = world.spawn_actor(front_camera_bp, front_camera_transform, attach_to=vehicle)

        # 主摄像头（用于可视化）
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(config.CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(config.CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(config.CAMERA_FOV))
        camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=2.0, z=1.8)), attach_to=vehicle)

        # 摄像头数据
        camera_data = {'image': np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)}

        # 传感器回调函数
        def third_camera_callback(image):
            nonlocal third_image
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            third_image = array[:, :, :3]

        def front_camera_callback(image):
            nonlocal front_image
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            front_image = array[:, :, :3]

        # 注册回调
        third_camera.listen(third_camera_callback)
        front_camera.listen(front_camera_callback)
        camera.listen(lambda img: camera_callback(img, camera_data))
        time.sleep(2.0)  # 等待传感器初始化
        logger.info("传感器初始化完成")

        # 生成初始道路航点
        initial_waypoints = tools.generate_road_waypoints(world, spawn_point.location, config.WAYPOINT_COUNT)

        # 初始化核心组件
        obstacle_detector = ObstacleDetector(world, vehicle)
        traditional_controller = TraditionalController(world, obstacle_detector)
        speed_controller = PIDSpeedController(config)
        path_controller = PurePursuitController(config)

        # 初始化可视化
        visualizer = Visualizer(config, spawn_point.location, initial_waypoints)

        # 控制变量
        throttle = 0.3
        steer = 0.0
        brake = 0.0
        frame_count = 0
        stuck_count = 0
        last_position = vehicle.get_location()

        # 主循环
        print("\n🚀 自动巡航小车启动（按R键复位）")
        print("=====================================")
        while viewer.is_running():
            # 更新小车控制
            controller.update()
            # 执行仿真步骤
            mujoco.mj_step(controller.model, controller.data)
            # 打印状态信息
            status_info = controller.get_status_info()
            print(f"\r{status_info}", end='', flush=True)
            # 同步视图
            viewer.sync()

    print("\n\n👋 程序结束")
                    current_target_speed = min(BASE_SPEED, current_target_speed + SPEED_TRANSITION_RATE / 2)
                    steer = steer * 0.9 if abs(steer) > 0.05 else 0.0

                # 计算转向
                target_steer = calculate_steer_angle(vehicle.get_transform().rotation.yaw, target_yaw)
                current_steer = current_steer + (target_steer - current_steer) * STEER_RESPONSE_FACTOR
                steer = current_steer
                throttle = CONTROL_CONFIG["normal_throttle"]

                # 速度控制
                current_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
                speed_error = current_target_speed - current_speed

                if abs(speed_error) < SPEED_DEADZONE:
                    control.throttle = last_throttle * 0.85
                    control.brake = 0.0
                elif speed_error > 0:
                    control.throttle = min(last_throttle + ACCELERATION_FACTOR, 0.25)
                    control.brake = 0.0
                else:
                    control.brake = min(last_brake + DECELERATION_FACTOR, 0.2)
                    control.throttle = 0.0

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

            # 检测障碍物并调整转向
            has_obstacle = detect_obstacle()
            if has_obstacle:
                steer = CONFIG["avoid_steer"]  # 向右绕行
                throttle = CONFIG["avoid_throttle"]
                print(
                    f"\r⚠️ 前方有障碍 | 绕行中 | 行驶时长：{run_time:.0f}秒 | 速度：{math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y) * 3.6:.0f}km/h",
                    end="")
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
                target_steer = 0.0

            # 卡停处理：速度过低时重置位置
            current_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
            if current_speed < 0.1:
                print("\n⚠️ 车辆卡停，重置位置...")
                new_loc = vehicle.get_transform().location + carla.Location(x=CONFIG["stuck_reset_dist"])
                vehicle.set_transform(carla.Transform(new_loc, vehicle.get_transform().rotation))
                vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

            # 打印状态
            logger.debug(
                f"速度：{current_speed:.1f}km/h | 位置：({vehicle_loc.x:.1f}, {vehicle_loc.y:.1f}) | 转向：{steer:.2f}")
            time.sleep(0.01)
            raise Exception("无可用生成点")
        valid_spawn = spawn_points[10] if len(spawn_points) >= 10 else spawn_points[0]
        logger.info(f"✅ 车辆生成位置：(x={valid_spawn.location.x:.1f}, y={valid_spawn.location.y:.1f})")
        return valid_spawn
    except Exception as e:
        logger.error(f"❌ 获取生成点失败：{e}")
        raise


# ===================== 4. 辅助函数：获取CARLA启动指令（无绝对路径） =====================
def get_carla_launch_cmd():
    """获取CARLA启动指令（适配不同系统）"""
    if sys.platform == "win32":
        return "CarlaUE4.exe"  # Windows（需在CARLA根目录运行）
    elif sys.platform == "linux":
        return "./CarlaUE4.sh"  # Linux
    else:
        return "CarlaUE4"  # 其他系统


# ===================== 4. 主逻辑 =====================
def main():
    # 1. 连接CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world()
        logger.info(f"\n✅ 连接CARLA成功！服务器版本：{client.get_server_version()}")
    except Exception as e:
        logger.error(f"程序异常：{e}")
        import traceback
        traceback.print_exc()
        logger.error(f"\n❌ 连接CARLA失败：{str(e)}")
        logger.info(f"📌 请先启动CARLA服务器：{get_carla_launch_cmd()}")
        sys.exit(1)

    # 手动终止处理（Ctrl+C）
    except KeyboardInterrupt:
        logger.info("\n\n👋 用户中断程序")
    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
    finally:
        # 清理资源
        logger.info("\n正在清理资源...")
        cv2.destroyAllWindows()

        # 停止传感器
        if third_camera:
            third_camera.stop()
        if front_camera:
            front_camera.stop()
        if camera:
            camera.stop()

        # 关闭可视化
        if visualizer:
            plt.close('all')

        # 销毁所有Actor
        tools.clear_all_actors(world)

        # 关闭同步模式
    # 2. 生成车辆
    vehicle = None
    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute('color', '255,0,0')  # 红色车身
        valid_spawn = get_valid_spawn_point(world)
        vehicle = world.spawn_actor(vehicle_bp, valid_spawn)
        logger.info(f"✅ 车辆生成成功，ID：{vehicle.id}（红色车身）")
    except Exception as e:
        logger.error(f"\n❌ 生成车辆失败：{str(e)}")
        sys.exit(1)

    # 3. 初始化V2X+视角
    try:
        rsu = RoadSideUnit(world, vehicle)
        vu = VehicleUnit(vehicle)
        set_near_observation_view(world, vehicle)

        # 4. 均衡测试（30秒，三区各10秒）
        logger.info("\n✅ 开始三区均衡变速测试（30秒）...")
        logger.info("📌 高速/中速/低速区各停留10秒，低速精准到10km/h！")
        start_time = time.time()

        # 设置同步模式（提高控速精度）
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        time.sleep(1)
        logger.info("资源清理完成，仿真结束")
        try:
            while time.time() - start_time < 30:
                speed_limit, zone_type = rsu.get_balance_speed_limit()
                command = rsu.send_speed_command(vehicle.id, speed_limit, zone_type)
                vu.receive_speed_command(command)
                world.tick()  # 同步物理帧
                time.sleep(0.1)  # 提高响应速度
        except KeyboardInterrupt:
            logger.info("\n⚠️  用户中断测试")
        finally:
            # 恢复异步模式
            settings.synchronous_mode = False
            world.apply_settings(settings)

    except Exception as e:
        logger.error(f"\n❌ 测试过程出错：{e}")
    finally:
        # 紧急停车+资源清理（容错处理）
        if vehicle:
            try:
                vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0, steer=0.0))
                time.sleep(2)
                vehicle.destroy()
                logger.info("\n✅ 测试结束，车辆已销毁")
            except Exception as e:
                logger.warning(f"⚠️  清理车辆失败：{e}")


if __name__ == "__main__":
    # 打印系统信息（便于调试）
    logger.info(f"🔍 当前Python解释器路径：{sys.executable}")
    logger.info(f"🔍 当前Python版本：{sys.version.split()[0]}")
    logger.info(f"🔍 操作系统：{sys.platform}")

    main()
