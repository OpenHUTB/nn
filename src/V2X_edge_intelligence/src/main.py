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
import logging
from datetime import datetime
import carla

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
                    steer = 0.2 if angle >= 0 else -0.2

        return throttle, brake, steer

    def get_control(self, vehicle):
        """生成传统控制指令（优先避障，弱化基础速度控制）"""
        # 获取车辆状态
        transform = vehicle.get_transform()
        location = vehicle.get_location()
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6

# ====================== 2. CARLA动态加载（优化版，无绝对路径） ======================
def load_carla_dynamically():
    """动态加载CARLA，优先环境变量，其次相对路径"""
    try:
        import carla
        logger.info("✅ CARLA模块已直接加载")
        return carla
    except ImportError:
        # 动态路径列表（无任何硬编码绝对路径）
        carla_search_paths = [
            os.path.join(os.environ.get('CARLA_ROOT', ''), 'PythonAPI', 'carla', 'dist'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../PythonAPI/carla/dist'),
            os.path.expanduser('~/CARLA/PythonAPI/carla/dist'),
            os.path.expanduser('~/Documents/CARLA/PythonAPI/carla/dist'),
            os.path.expanduser('~/carla/PythonAPI/carla/dist'),
            os.path.expanduser('~/.local/share/carla/PythonAPI/carla/dist')
        ]

        # 遍历查找egg文件
        carla_egg_path = None
        for search_path in carla_search_paths:
            if not search_path or not os.path.exists(search_path):
                continue
            for file in os.listdir(search_path):
                if file.endswith('.egg') and 'carla' in file:
                    carla_egg_path = os.path.join(search_path, file)
                    break
            if carla_egg_path:
                break

        if carla_egg_path:
            sys.path.append(carla_egg_path)
            import carla
            logger.info(f"✅ 动态加载CARLA成功：{carla_egg_path}")
            return carla
        else:
            error_msg = (
                "\n❌ CARLA加载失败！请按以下方式配置：\n"
                "1. 配置CARLA_ROOT环境变量（推荐）：\n"
                "   Windows: set CARLA_ROOT=你的CARLA安装目录\n"
                "   Linux/Mac: export CARLA_ROOT=你的CARLA安装目录\n"
                "2. 确保PythonAPI路径正确：PythonAPI/carla/dist 下有carla-*.egg文件"
            )
            logger.error(error_msg)
            sys.exit(1)


# 执行动态加载
carla = load_carla_dynamically()

# ====================== 3. 配置参数（保留核心，新增多地图配置） ======================
# 速度参数
BASE_SPEED = 1.5  # 直道速度
CURVE_TARGET_SPEED = 1.0  # 弯道速度
SPEED_DEADZONE = 0.1
ACCELERATION_FACTOR = 0.04
DECELERATION_FACTOR = 0.06
SPEED_TRANSITION_RATE = 0.03

# 转向参数（晚转弯+大角度核心）
LOOKAHEAD_DISTANCE = 20.0
WAYPOINT_STEP = 1.0
CURVE_DETECTION_THRESHOLD = 2.0
TURN_TRIGGER_DISTANCE_IDX = 4  # 前方5米触发转向
STEER_ANGLE_MAX = 0.85  # 最大转向角
STEER_RESPONSE_FACTOR = 0.4  # 转向响应速度
STEER_AMPLIFY = 1.6  # 转向放大系数
MIN_STEER = 0.2  # 最小转向角

# 生成点偏移
SPAWN_OFFSET_X = -2.0
SPAWN_OFFSET_Y = 0.0
SPAWN_OFFSET_Z = 0.0

# 新增：多地图配置
SUPPORTED_MAPS = {
    "Town01": "城镇1（简单道路）",
    "Town02": "城镇2（乡村道路）",
    "Town03": "城镇3（高速公路）",
    "Town04": "城镇4（混合道路）"
}
DEFAULT_MAP = "Town01"

# 新增：控制配置
CONTROL_CONFIG = {
    "init_control_times": 12,
    "init_control_interval": 0.05,
    "init_total_delay": 0.8,
    "normal_throttle": 0.85,
    "avoid_throttle": 0.5,
    "avoid_steer": 0.6,
    "loop_interval": 0.008,
    "detect_distance": 10.0,
    "stuck_reset_dist": 2.0
}


# ====================== 4. 核心工具函数（增强版） ======================
def get_road_direction_ahead(vehicle, world):
    """获取前方道路方向，晚转弯逻辑"""
    vehicle_transform = vehicle.get_transform()
    carla_map = world.get_map()

    waypoints = []
    current_wp = carla_map.get_waypoint(vehicle_transform.location)
    next_wp = current_wp

    for _ in range(int(LOOKAHEAD_DISTANCE / WAYPOINT_STEP)):
        next_wps = next_wp.next(WAYPOINT_STEP)
        if not next_wps:
            break
        next_wp = next_wps[0]
        waypoints.append(next_wp)

    if len(waypoints) < 3:
        return vehicle_transform.rotation.yaw, False, 0.0

    target_wp_idx = min(TURN_TRIGGER_DISTANCE_IDX, len(waypoints) - 1)
    target_wp = waypoints[target_wp_idx]
    target_yaw = target_wp.transform.rotation.yaw

    current_yaw = vehicle_transform.rotation.yaw
    yaw_diff = target_yaw - current_yaw
    yaw_diff = (yaw_diff + 180) % 360 - 180
    is_curve = abs(yaw_diff) > CURVE_DETECTION_THRESHOLD

    return target_yaw, is_curve, yaw_diff


def calculate_steer_angle(current_yaw, target_yaw):
    """计算超大角度转向"""
    yaw_diff = target_yaw - current_yaw
    yaw_diff = (yaw_diff + 180) % 360 - 180

    steer = (yaw_diff / 180.0 * STEER_ANGLE_MAX) * STEER_AMPLIFY
    steer = max(-STEER_ANGLE_MAX, min(STEER_ANGLE_MAX, steer))

    if abs(steer) > 0.05 and abs(steer) < MIN_STEER:
        steer = MIN_STEER * (1 if steer > 0 else -1)

    return steer


def detect_obstacle_enhanced(vehicle, world, detect_distance=10.0):
    """增强版障碍物检测：检测车辆、行人、静态障碍物"""
    trans = vehicle.get_transform()
    vehicle_location = trans.location
    vehicle_forward = trans.get_forward_vector()

    # 1. 检测道路合法性
    for check_dist in range(2, int(detect_distance) + 1, 2):
        check_loc = vehicle_location + vehicle_forward * check_dist
        waypoint = world.get_map().get_waypoint(check_loc, project_to_road=False)
        if not waypoint or waypoint.lane_type != carla.LaneType.Driving:
            return True

    # 2. 检测其他车辆/行人（新增功能）
    actors = world.get_actors()
    for actor in actors:
        if actor.type_id.startswith(("vehicle", "walker")) and actor.id != vehicle.id:
            actor_loc = actor.get_location()
            distance = vehicle_location.distance(actor_loc)
            # 检测前方detect_distance米内的障碍物
            if distance < detect_distance:
                # 计算障碍物在车辆前方的角度
                vec = actor_loc - vehicle_location
                dot = vec.x * vehicle_forward.x + vec.y * vehicle_forward.y
                if dot > 0:  # 只检测前方障碍物
                    return True

    return False


def follow_vehicle_enhanced(vehicle, spectator, follow_mode="third_person"):
    """增强版视角跟随：支持第三人称/俯视视角"""
    trans = vehicle.get_transform()
    if follow_mode == "third_person":
        # 第三人称视角
        spectator_loc = carla.Location(
            x=trans.location.x - math.cos(math.radians(trans.rotation.yaw)) * 7,
            y=trans.location.y - math.sin(math.radians(trans.rotation.yaw)) * 7,
            z=trans.location.z + 4.5
        )
        spectator_rot = carla.Rotation(pitch=-30, yaw=trans.rotation.yaw)
    elif follow_mode == "top_down":
        # 俯视视角
        spectator_loc = carla.Location(
            x=trans.location.x,
            y=trans.location.y,
            z=40.0
        )
        spectator_rot = carla.Rotation(pitch=-85, yaw=trans.rotation.yaw)

    spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))


def print_drive_status(vehicle, run_time, has_obstacle, steer):
    """打印增强版行驶状态（新增功能）"""
    velocity = vehicle.get_velocity()
    current_speed_mps = math.hypot(velocity.x, velocity.y)
    current_speed_kmh = current_speed_mps * 3.6
    location = vehicle.get_location()

    status = "⚠️ 避障中" if has_obstacle else "✅ 正常行驶"
    status_msg = (
        f"\r{status} | 行驶时长：{run_time:.0f}s "
        f"| 速度：{current_speed_kmh:.0f}km/h "
        f"| 转向：{steer:.2f} "
        f"| 位置：({location.x:.1f}, {location.y:.1f})"
    )
    print(status_msg, end="")
    logger.info(status_msg.strip())


# ====================== 5. 主函数（增强版） ======================
def main(selected_map=DEFAULT_MAP):
    """主函数：增强版自动驾驶逻辑"""
    # 初始化资源
    client = None
    world = None
    vehicle = None
    camera_sensor = None
    collision_sensor = None
    spectator = None
    is_vehicle_alive = False
    run_time = 0

# ====================== 小车控制器（核心逻辑封装） ======================
class CruiseCarController:
    def __init__(self, config: Config):
        self.config = config
        self.key_manager = KeyManager()
    try:
        # 1. 连接CARLA服务器
        client = carla.Client("localhost", 2000)
        client.set_timeout(60.0)
        world = client.load_world(selected_map)
        logger.info(f"✅ 成功连接CARLA，加载地图：{selected_map} ({SUPPORTED_MAPS[selected_map]})")

        # 配置世界设置
        world_settings = world.get_settings()
        world_settings.synchronous_mode = False
        world_settings.fixed_delta_seconds = 0.1
        world.apply_settings(world_settings)

        # 设置天气（新增功能）
        world.set_weather(carla.WeatherParameters.ClearNoon)
        logger.info("✅ 设置天气为：晴朗正午")

        # 2. 清理旧Actor
        for actor in world.get_actors().filter('vehicle.*'):
            actor.destroy()
        logger.info("✅ 清理旧车辆完成")

        # 3. 生成车辆
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute('color', '255,0,0')  # 红色车身
        logger.info("✅ 选择特斯拉Model3，红色车身")

        # 获取生成点并偏移
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise Exception("❌ 未找到生成点")

        original_spawn_point = spawn_points[0]
        spawn_point = carla.Transform(
            carla.Location(
                x=original_spawn_point.location.x + SPAWN_OFFSET_X,
                y=original_spawn_point.location.y + SPAWN_OFFSET_Y,
                z=original_spawn_point.location.z + SPAWN_OFFSET_Z
            ),
            original_spawn_point.rotation
        )
        logger.info(f"✅ 生成点偏移：左移{abs(SPAWN_OFFSET_X)}米")

        # 加载模型
        # 重试生成车辆
        max_spawn_retry = 5
        for retry in range(max_spawn_retry):
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                if vehicle and vehicle.is_alive:
                    vehicle.set_simulate_physics(True)
                    vehicle.set_autopilot(False)
                    is_vehicle_alive = True
                    logger.info(f"✅ 车辆生成成功（重试{retry + 1}次），ID：{vehicle.id}")
                    break
            except Exception as e:
                if retry == max_spawn_retry - 1:
                    raise Exception(f"❌ 车辆生成失败：{e}")
                time.sleep(0.8)

        # 4. 初始化车辆控制
        control = carla.VehicleControl()
        control.hand_brake = False
        control.manual_gear_shift = False
        control.gear = 1

        # 激活车辆（确保能动）
        logger.info("🔋 激活车辆物理状态...")
        for _ in range(CONTROL_CONFIG["init_control_times"]):
            vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, steer=0.0, brake=0.0
            ))
            time.sleep(CONTROL_CONFIG["init_control_interval"])
        time.sleep(CONTROL_CONFIG["init_total_delay"])

        # 5. 设置视角
        spectator = world.get_spectator()
        follow_vehicle_enhanced(vehicle, spectator, "third_person")
        logger.info("✅ 视角已绑定车辆（第三人称）")

        # 6. 挂载传感器（新增功能：可选保存图片）
        # 碰撞传感器
        try:
            self.model = mujoco.MjModel.from_xml_path(config.MODEL_PATH)
            self.data = mujoco.MjData(self.model)
            print(f"✅ 成功加载模型: {config.MODEL_PATH}")
            collision_bp = bp_lib.find("sensor.other.collision")
            collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

            def collision_cb(event):
                nonlocal steer
                logger.warning("💥 检测到碰撞，自动调整方向！")
                steer = -steer if abs(steer) > 0 else -CONTROL_CONFIG["avoid_steer"]
                vehicle.apply_control(carla.VehicleControl(
                    throttle=CONTROL_CONFIG["avoid_throttle"],
                    steer=steer,
                    brake=0.0
                ))

            collision_sensor.listen(collision_cb)
            logger.info("🛡️ 碰撞传感器挂载成功")
        except Exception as e:
            raise RuntimeError(f"❌ 加载模型失败: {e}")

        dx = target_loc.x - location.x
        dy = target_loc.y - location.y

        local_x = dx * math.cos(vehicle_yaw) + dy * math.sin(vehicle_yaw)
        local_y = -dx * math.sin(vehicle_yaw) + dy * math.cos(vehicle_yaw)

        # 复位小车
        self.reset_car()
            logger.warning(f"⚠️ 碰撞传感器挂载失败：{e}")

        # 障碍物调整（优先执行）
        throttle, brake, steer = self.apply_obstacle_avoidance(throttle, brake, steer, vehicle, obstacle_info)

        # 遍历所有障碍物
        for obs_name, obs_id in self.obstacle_ids.items():
        # RGB摄像头（可选启用）
        enable_camera = False  # 可改为True启用
        if enable_camera:
            try:
                obs_pos = self.data.body(obs_id).xpos
                # 计算相对距离
                dx = obs_pos[0] - chassis_pos[0]  # 前向距离
                dy = obs_pos[1] - chassis_pos[1]  # 横向距离
                distance = math.hypot(dx, dy)

                # 只检测前方且横向偏移小于小车宽度的障碍
                if dx > 0 and abs(dy) < 0.3 and distance < self.config.OBSTACLE_THRESHOLD:
                    if distance < min_distance:
                        min_distance = distance
                        closest_obs = obs_name
            except Exception:
                continue

        # 判断状态
        if closest_obs is None:
            return 0, 0.0, None
        elif min_distance < self.config.SAFE_DISTANCE:
            return 2, min_distance, closest_obs
        else:
            return 1, min_distance, closest_obs

    def _set_steer(self, angle: float):
        """设置转向角度（限制在合法范围内）"""
        clamped_angle = np.clip(angle, -self.config.STEER_LIMIT, self.config.STEER_LIMIT)
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
            # 巡航状态：检测到障碍则停止
            if obs_status == 2:
                self.car_state = CarState.STOPPED
                self._set_speed(0.0)
                print(f"\n🛑 紧急停止！障碍物：{obs_name} (距离: {obs_dist:.2f}m)")
            elif obs_status == 1:
                self.car_state = CarState.STOPPED
                self._set_speed(0.0)
                print(f"\n⚠️  检测到障碍物：{obs_name} (距离: {obs_dist:.2f}m)，正在停止")
                # 创建保存目录（相对路径，无绝对路径）
                camera_dir = os.path.join(os.path.dirname(__file__), "camera_images")
                os.makedirs(camera_dir, exist_ok=True)

                camera_bp = bp_lib.find("sensor.camera.rgb")
                camera_bp.set_attribute('image_size_x', '800')
                camera_bp.set_attribute('image_size_y', '600')
                camera_bp.set_attribute('fov', '90')
                camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
                camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

                def camera_callback(image):
                    image.save_to_disk(os.path.join(camera_dir, f"frame_{image.frame_number}.png"))

                camera_sensor.listen(camera_callback)
                logger.info(f"📹 摄像头挂载成功，图片保存至：{camera_dir}")
            except Exception as e:
                logger.warning(f"⚠️ 摄像头挂载失败：{e}")

        # 7. 核心自动驾驶循环
        logger.info("\n🚗 开始自动驾驶（按Ctrl+C停止）")
        print("\n" + "=" * 80)
        current_steer = 0.0
        current_target_speed = BASE_SPEED
        last_throttle = 0.0
        last_brake = 0.0
        steer = 0.0

        while True:
            # 检查车辆状态
            if not vehicle or not vehicle.is_alive:
                logger.error("❌ 车辆异常消失")
                break

            # 更新视角
            follow_vehicle_enhanced(vehicle, spectator, "third_person")

            # 障碍物检测（增强版）
            has_obstacle = detect_obstacle_enhanced(vehicle, world, CONTROL_CONFIG["detect_distance"])

            if has_obstacle:
                # 避障逻辑
                steer = CONTROL_CONFIG["avoid_steer"]
                throttle = CONTROL_CONFIG["avoid_throttle"]
            else:
                # 正常巡航
                self._set_steer(0.0)
                self._set_speed(self.config.CRUISE_SPEED)

        elif self.car_state == CarState.STOPPED:
            # 停止状态：延迟后开始转向
            self.turn_progress += 1
            self._set_speed(0.0)

            if self.turn_progress > 10:  # 等待10帧确保停止
                # 随机选择转向方向
                self.turn_direction = 1 if random.random() > 0.5 else -1
                self.target_steer_angle = self.turn_direction * self.config.TURN_ANGLE
                self.car_state = CarState.TURNING
                self.turn_progress = 0.0
                dir_text = "右" if self.turn_direction > 0 else "左"
                print(f"\n🔄 开始{dir_text}转90度避障")

        elif self.car_state == CarState.TURNING:
            # 转向状态：精准控制转向角度
            self.turn_progress += 0.01
            # 计算当前转向角度（渐进式）
            current_steer = self.target_steer_angle * self.turn_progress
            self._set_steer(current_steer)
            self._set_speed(self.config.TURN_SPEED)

            # 转向完成判断（角度到位或进度完成）
            if self.turn_progress >= 1.0 or abs(current_steer) >= self.config.STEER_LIMIT:
                self.car_state = CarState.RESUMING
                self.turn_progress = 0.0
                print("\n✅ 转向完成，开始恢复巡航")

        elif self.car_state == CarState.RESUMING:
            # 恢复状态：逐渐回正转向并加速
            self.turn_progress += 0.01
            # 渐进回正转向
            current_steer = self.target_steer_angle * (1 - self.turn_progress)
            self._set_steer(current_steer)
            # 渐进恢复速度
            current_speed = self.config.TURN_SPEED + (
                        self.config.CRUISING - self.config.TURN_SPEED) * self.turn_progress
            self._set_speed(current_speed)

            # 恢复完成判断
            if self.turn_progress >= 1.0:
                # 检查前方是否安全
                if obs_status == 0:
                    self.car_state = CarState.CRUISING
                    print("\n🚗 前方安全，恢复正常巡航")
                # 正常行驶逻辑
                # 获取道路方向
                target_yaw, is_curve, yaw_diff = get_road_direction_ahead(vehicle, world)

                # 弯道速度控制
                if is_curve:
                    current_target_speed = max(CURVE_TARGET_SPEED, current_target_speed - SPEED_TRANSITION_RATE)
                else:
                    self.car_state = CarState.STOPPED
                    print("\n⚠️  转向后仍有障碍，重新停止")
                self.turn_progress = 0.0


# ===================== 摄像头回调 =====================
def camera_callback(image, data_dict):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    data_dict['image'] = array


# ===================== 控制器 =====================
class PIDSpeedController:
    def __init__(self, config):
        self.kp = config.PID_KP
        self.ki = config.PID_KI
        self.kd = config.PID_KD
        self.prev_err = 0.0
        self.integral = 0.0
        self.target_speed = config.TARGET_SPEED

    def calculate(self, current_speed):
        err = self.target_speed - current_speed
        self.integral = np.clip(self.integral + err * 0.05, -1.0, 1.0)
        deriv = (err - self.prev_err) / 0.05 if 0.05 != 0 else 0.0
        output = self.kp * err + self.ki * self.integral + self.kd * deriv
        self.prev_err = err
        return np.clip(output, 0.1, 1.0)


class PurePursuitController:
    def __init__(self, config):
        self.lookahead_dist = config.LOOKAHEAD_DISTANCE
        self.max_steer_rad = math.radians(config.MAX_STEER_ANGLE)

    def calculate_steer(self, vehicle_loc, vehicle_yaw, waypoints):
        """
        纯追踪算法计算转向角
        :param vehicle_loc: 车辆位置
        :param vehicle_yaw: 车辆朝向（弧度）
        :param waypoints: 道路航点列表
        :return: 转向角（-1~1）
        """
        # 1. 将航点转换为车辆坐标系
        wp_coords = np.array(waypoints)
        vehicle_x = vehicle_loc.x
        vehicle_y = vehicle_loc.y

        # 旋转和平移（车辆坐标系：x向前，y向左）
        cos_yaw = math.cos(vehicle_yaw)
        sin_yaw = math.sin(vehicle_yaw)
        translated_x = wp_coords[:, 0] - vehicle_x
        translated_y = wp_coords[:, 1] - vehicle_y
        rotated_x = translated_x * cos_yaw + translated_y * sin_yaw
        rotated_y = -translated_x * sin_yaw + translated_y * cos_yaw

        # 2. 找到距离车辆>=前瞻距离的第一个航点
        distances = np.hypot(rotated_x, rotated_y)
        valid_wp_indices = np.where(distances >= self.lookahead_dist)[0]
        if len(valid_wp_indices) == 0:
            return 0.0

        target_idx = valid_wp_indices[0]
        target_x = rotated_x[target_idx]
        target_y = rotated_y[target_idx]

        # 3. 计算转向角（纯追踪公式：steer = arctan(2*L*y/(x²+y²))，L为车辆轴距，这里简化为1.0）
        L = 1.0  # 车辆轴距（米）
        steer_rad = math.atan2(2 * L * target_y, self.lookahead_dist ** 2)

        # 4. 限制转向角
        steer_rad = np.clip(steer_rad, -self.max_steer_rad, self.max_steer_rad)
        steer = steer_rad / self.max_steer_rad

        return steer


# ===================== 可视化 =====================
class Visualizer:
    def __init__(self, config, spawn_loc, initial_waypoints):
        self.waypoints = np.array(initial_waypoints)
        self.trajectory = []
        self.spawn_loc = (spawn_loc.x, spawn_loc.y)

        plt.rcParams['backend'] = 'TkAgg'
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=config.PLOT_SIZE)

        # 绘制道路航点（CARLA原生）
        self.ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1], c='blue', s=50, label='Road Waypoints (CARLA)',
                        zorder=3)
        # 绘制生成点
        self.ax.scatter(self.spawn_loc[0], self.spawn_loc[1], c='orange', marker='s', s=150, label='Spawn Point',
                        zorder=5)
        # 轨迹和车辆
        self.traj_line, = self.ax.plot([], [], c='red', linewidth=4, label='Vehicle Trajectory', zorder=2)
        self.vehicle_dot, = self.ax.plot([], [], c='green', marker='o', markersize=20, label='Vehicle', zorder=6)

        self.ax.set_xlabel('X (m)', fontsize=14)
        self.ax.set_ylabel('Y (m)', fontsize=14)
        self.ax.set_title('CARLA Road Following (Native Waypoints)', fontsize=16)
        self.ax.legend(fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.axis('equal')

        plt.show(block=False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, vehicle_x, vehicle_y, new_waypoints=None):
        """更新轨迹和航点"""
        self.trajectory.append([vehicle_x, vehicle_y])
        if len(self.trajectory) > 1000:
            self.trajectory = self.trajectory[-1000:]

        # 更新轨迹
        if self.trajectory:
            traj = np.array(self.trajectory)
            self.traj_line.set_data(traj[:, 0], traj[:, 1])
        self.vehicle_dot.set_data(vehicle_x, vehicle_y)

        # 更新航点（如果有新航点）
        if new_waypoints is not None:
            self.waypoints = np.array(new_waypoints)
            self.ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1], c='blue', s=50, zorder=3)

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# ===================== 主函数 =====================
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

    # 初始化OpenCV窗口
    cv2.namedWindow('CARLA Autopilot (0.9.10)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CARLA Autopilot (0.9.10)', 640, 480)

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

                last_throttle = control.throttle
                last_brake = control.brake

            # 应用控制
            control.steer = steer
            control.throttle = throttle
            vehicle.apply_control(control)

            # 卡停处理
            current_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
            if current_speed < 0.1:
                logger.warning("⚠️ 车辆卡停，重置位置")
                new_loc = vehicle.get_transform().location + carla.Location(x=CONTROL_CONFIG["stuck_reset_dist"])
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

    except KeyboardInterrupt:
        logger.info(f"\n🛑 手动终止程序，总行驶时长：{run_time:.0f}秒")
    except Exception as e:
        logger.error(f"\n❌ 程序异常：{str(e)}", exc_info=True)
        print("\n🔧 修复建议：")
        print("1. 关闭CARLA，结束任务管理器中的CarlaUE4.exe进程")
        print("2. 重启CARLA：CarlaUE4.exe -windowed -ResX=800 -ResY=600")
        print("3. 确保CARLA_ROOT环境变量配置正确")
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

    print(f"📌 即将启动CARLA自动驾驶程序")
    print(f"🗺️ 选择地图：{selected_map} ({SUPPORTED_MAPS[selected_map]})")
    print(f"💡 支持的地图：{', '.join(SUPPORTED_MAPS.keys())}")
    print(f"💡 示例：python {sys.argv[0]} Town02\n")

if __name__ == "__main__":
    # 打印系统信息（便于调试）
    logger.info(f"🔍 当前Python解释器路径：{sys.executable}")
    logger.info(f"🔍 当前Python版本：{sys.version.split()[0]}")
    logger.info(f"🔍 操作系统：{sys.platform}")

    main()
