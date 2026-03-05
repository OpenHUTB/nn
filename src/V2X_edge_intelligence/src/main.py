#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA Waypoint Following（原生道路航点版）
核心：直接使用CARLA地图的道路航点，车辆100%沿道路行驶
适配：CARLA 0.9.10，无任何硬编码绝对路径
"""

import sys
import os
import carla
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from collections import deque
import random
import logging

# ===================== 日志配置（新增） =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
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
                break
        logger.info(f"生成了{len(waypoints)}个原生道路航点")
        return waypoints


# --------------------------
# 1. 障碍物检测器（优化性能+无延迟检测）
# --------------------------
class ObstacleDetector:
    def __init__(self, world, vehicle, max_distance=50.0, detect_interval=1):
        self.world = world
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
        }

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

    # 初始化变量
    vehicle = None
    third_camera = None
    front_camera = None
    camera = None
    visualizer = None
    third_image = None
    front_image = None

    # 初始化OpenCV窗口
    cv2.namedWindow('CARLA Autopilot (0.9.10)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CARLA Autopilot (0.9.10)', 640, 480)

    try:
        # 连接CARLA服务器（0.9.10兼容）
        client = carla.Client('localhost', 2000)
        client.set_timeout(30.0)
        world = client.load_world('Town01')  # 0.9.10支持Town01
        logger.info("成功连接CARLA并加载Town01地图")

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

    except Exception as e:
        logger.error(f"程序异常：{e}")
        import traceback
        traceback.print_exc()

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
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        time.sleep(1)
        logger.info("资源清理完成，仿真结束")


if __name__ == "__main__":
    main()