#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA Waypoint Following（原生道路航点版）
核心：直接使用CARLA地图的道路航点，车辆100%沿道路行驶
适配：CARLA 0.9.10，无任何硬编码绝对路径
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
import logging

# ===================== 日志配置（新增） =====================
# ===================== 1. 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
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

        # 获取障碍物信息
        obstacle_info = self.obstacle_detector.get_obstacle_info()

        # 获取路点
        waypoint = self.map.get_waypoint(location, project_to_road=True)
        next_waypoints = waypoint.next(self.waypoint_distance)
        target_waypoint = next_waypoints[0] if next_waypoints else waypoint

        # 计算转向
        vehicle_yaw = math.radians(transform.rotation.yaw)
        target_loc = target_waypoint.transform.location

        dx = target_loc.x - location.x
        dy = target_loc.y - location.y

        local_x = dx * math.cos(vehicle_yaw) + dy * math.sin(vehicle_yaw)
        local_y = -dx * math.sin(vehicle_yaw) + dy * math.cos(vehicle_yaw)

        if abs(local_x) < 0.1:
            steer = 0.0
        else:
            angle = math.atan2(local_y, local_x)
            steer = np.clip(angle / math.radians(45), -1.0, 1.0)

        # 基础速度控制（降低优先级）
        if speed < 20:
            throttle = 0.4  # 从0.6降低到0.4，减少油门
            brake = 0.0
        elif speed < 40:
            throttle = 0.2  # 从0.4降低到0.2
            brake = 0.0
        else:
            throttle = 0.1
            brake = 0.2

        # 障碍物调整（优先执行）
        throttle, brake, steer = self.apply_obstacle_avoidance(throttle, brake, steer, vehicle, obstacle_info)

        # 低速强油门（仅当无障碍物时生效）
        if speed < 5.0 and not obstacle_info['has_obstacle']:
            throttle = 0.4
            brake = 0.0

        return throttle, brake, steer


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
        logger.info("自动驾驶系统启动 - 仅使用传统控制器")
        logger.info("控制键: q-退出, r-重置车辆")

        while True:
            world.tick()
            frame_count += 1

            # 获取车辆状态
            vehicle_loc, vehicle_yaw, current_speed = tools.get_vehicle_pose(vehicle)
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_velocity = vehicle.get_velocity()
            vehicle_speed = math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2)

            # 检测障碍物
            obstacle_info = obstacle_detector.get_obstacle_info()

            # 实时更新道路航点（每10帧更新一次，减少计算量）
            new_waypoints = None
            if frame_count % 10 == 0:
                new_waypoints = tools.generate_road_waypoints(world, vehicle_location, config.WAYPOINT_COUNT)

            # 卡住检测（优化：有障碍物时不触发）
            distance_moved = vehicle_location.distance(last_position)
            is_moving = distance_moved > 0.2 or vehicle_speed > 1.0

            if obstacle_info['has_obstacle'] and obstacle_info[
                'distance'] < traditional_controller.safe_following_distance:
                stuck_count = 0
            elif not is_moving:
                stuck_count += 1
            else:
                stuck_count = 0
            last_position = vehicle_location

            # 更新可视化
            visualizer.update(vehicle_loc.x, vehicle_loc.y, new_waypoints)

            # 卡住恢复（仅当无障碍物时执行）
            if stuck_count > 20:  # 从15帧改为20帧，降低误触发
                logger.warning("检测到车辆卡住，执行恢复程序...")
                # 紧急刹车
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
                time.sleep(0.5)
                # 倒车或转向
                if obstacle_info['has_obstacle'] and obstacle_info['distance'] < 15:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, reverse=True))
                    time.sleep(1.0)
                else:
                    recovery_steer = random.choice([-0.5, 0.5])
                    vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=recovery_steer, brake=0.0))
                    time.sleep(1.0)
                stuck_count = 0

            # 生成控制指令（仅使用传统控制器）
            throttle, brake, steer = traditional_controller.get_control(vehicle)

            # 紧急刹车时拉手刹
            if brake >= 1.0:
                vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0, steer=steer, brake=1.0, hand_brake=True
                ))
            else:
                # 应用控制
                control = carla.VehicleControl(
                    throttle=throttle,
                    steer=steer,
                    brake=brake,
                    hand_brake=False,
                    reverse=False
                )
                vehicle.apply_control(control)

            # 图像显示
            if third_image is not None:
                display_image = third_image.copy()
                # 可视化障碍物
                display_image = obstacle_detector.visualize_obstacles(display_image, vehicle_transform)
                # 绘制信息
                cv2.putText(display_image, f"Speed: {vehicle_speed * 3.6:.1f} km/h", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"Mode: Traditional", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"Throttle: {throttle:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"Steer: {steer:.2f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"Brake: {brake:.2f}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # 障碍物信息
                if obstacle_info['has_obstacle']:
                    cv2.putText(display_image, f"Obstacle: {obstacle_info['distance']:.1f}m", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_image, f"RelSpeed: {obstacle_info['relative_speed']:.1f}km/h", (10, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(display_image, "Obstacle: None", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # 卡住警告
                if stuck_count > 5:
                    cv2.putText(display_image, "STUCK DETECTED!", (10, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('CARLA Autopilot (0.9.10)', display_image)

                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("用户按下q键，退出程序")
                    break
                elif key == ord('r'):
                    vehicle.set_transform(spawn_point)
                    throttle = 0.3
                    steer = 0.0
                    brake = 0.0
                    stuck_count = 0
                    logger.info("车辆已重置")

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