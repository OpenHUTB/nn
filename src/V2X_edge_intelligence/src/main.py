#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA自动巡航小车 - 无绝对路径优化版
适配：CARLA 0.9.10，全动态路径，无任何硬编码绝对路径
核心功能：
1. 基于CARLA原生道路航点的自动巡航
2. 前方障碍物检测与避障（动态阈值）
3. PID速度控制 + 纯追踪转向控制
4. 可视化与状态监控
5. 自动复位/防卡滞逻辑
"""
import sys
import os
import carla
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import random
import logging
from typing import Tuple, Optional, Dict
from collections import deque


# ===================== 路径自动适配（核心优化） =====================
def setup_carla_path():
    """
    自动查找并添加CARLA PythonAPI路径
    无需硬编码绝对路径，适配不同安装位置
    """
    try:
        import carla
        return True
    except ImportError:
        # 获取当前脚本目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 定义可能的CARLA路径（相对当前脚本）
        search_paths = [
            # 同级目录
            os.path.join(script_dir, "PythonAPI", "carla", "dist"),
            # 上级目录
            os.path.join(script_dir, "..", "PythonAPI", "carla", "dist"),
            # 上上级目录（适配CARLA安装目录）
            os.path.join(script_dir, "..", "..", "PythonAPI", "carla", "dist"),
            # 用户主目录下的CARLA
            os.path.join(os.path.expanduser("~"), "CARLA_0.9.10", "PythonAPI", "carla", "dist")
        ]

        # 遍历查找CARLA egg文件
        for search_path in search_paths:
            abs_search_path = os.path.abspath(search_path)
            if os.path.exists(abs_search_path):
                for file in os.listdir(abs_search_path):
                    if file.startswith("carla-0.9.10") and file.endswith(".egg"):
                        egg_path = os.path.join(abs_search_path, file)
                        sys.path.insert(0, egg_path)
                        try:
                            import carla
                            logging.info(f"✅ 自动找到CARLA: {egg_path}")
                            return True
                        except ImportError:
                            continue
        logging.error("❌ 未找到CARLA 0.9.10 PythonAPI，请检查安装路径")
        return False


# 初始化CARLA路径（优先执行）
if not setup_carla_path():
    sys.exit(1)

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ===================== 全局配置（集中管理） =====================
class Config:
    # 速度配置
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

    # 避障配置
    EMERGENCY_BRAKE_DISTANCE = 12.0  # 紧急刹车距离
    SAFE_FOLLOWING_DISTANCE = 18.0  # 安全跟车距离
    EARLY_WARNING_DISTANCE = 30.0  # 提前预警距离
    OBSTACLE_DETECT_ANGLE = 70.0  # 障碍物检测角度（±）


# ===================== 工具类（无路径依赖） =====================
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
        spectator.set_transform(carla.Transform(
            t.location + carla.Location(x=0, y=-8, z=5),
            t.rotation
        ))
        logger.info("CARLA客户端已聚焦到车辆")

    @staticmethod
    def generate_road_waypoints(world, start_loc, count=50, step=2.0):
        """从起点沿道路生成连续的原生航点（无路径依赖）"""
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


# ===================== 障碍物检测器 =====================
class ObstacleDetector:
    def __init__(self, world, vehicle, max_distance=50.0, detect_interval=1):
        self.world = world
        self.vehicle = vehicle
        self.max_distance = max_distance
        self.detect_interval = detect_interval
        self.frame_count = 0
        self.last_obstacle_info = {
            'has_obstacle': False,
            'distance': float('inf'),
            'relative_angle': 0.0,
            'obstacle_type': None,
            'obstacle_speed': 0.0,
            'relative_speed': 0.0
        }

    def get_vehicle_speed(self, vehicle):
        """获取车辆速度（km/h）"""
        velocity = vehicle.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6

    def get_obstacle_info(self):
        """检测前方障碍物信息（无延迟）"""
        self.frame_count += 1
        if self.frame_count % self.detect_interval != 0:
            return self.last_obstacle_info

        try:
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            forward_vector = vehicle_transform.get_forward_vector()
            self_speed = self.get_vehicle_speed(self.vehicle)

            # 只获取车辆类障碍物（优化性能）
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

                # 计算相对角度（仅前方±70度）
                relative_vector = carla.Location(
                    other_location.x - vehicle_location.x,
                    other_location.y - vehicle_location.y,
                    0
                )
                forward_2d = carla.Vector3D(forward_vector.x, forward_vector.y, 0)
                relative_2d = carla.Vector3D(relative_vector.x, relative_vector.y, 0)

                # 向量归一化
                forward_norm = math.hypot(forward_2d.x, forward_2d.y)
                relative_norm = math.hypot(relative_2d.x, relative_2d.y)
                if forward_norm == 0 or relative_norm == 0:
                    continue

                dot_product = forward_2d.x * relative_2d.x + forward_2d.y * relative_2d.y
                cos_angle = dot_product / (forward_norm * relative_norm)
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angle_deg = math.degrees(math.acos(cos_angle))

                # 仅检测前方±70度范围内的障碍物
                if angle_deg <= Config.OBSTACLE_DETECT_ANGLE and distance < min_distance:
                    min_distance = distance
                    closest_obstacle = other_vehicle
                    obstacle_speed = self.get_vehicle_speed(other_vehicle)
                    relative_angle = angle_deg if relative_2d.y >= 0 else -angle_deg

            # 更新障碍物信息
            relative_speed = self_speed - obstacle_speed if closest_obstacle else 0.0
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
        x_pos = int(width / 2 + (angle / Config.OBSTACLE_DETECT_ANGLE) * (width / 2))
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
        rel_speed = self.last_obstacle_info['relative_speed']
        cv2.putText(image, f"RelSpeed: {rel_speed:.1f}km/h", (x_pos - 20, int(height * 0.8) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image


# ===================== 控制器 =====================
class PIDSpeedController:
    def __init__(self):
        self.kp = Config.PID_KP
        self.ki = Config.PID_KI
        self.kd = Config.PID_KD
        self.prev_err = 0.0
        self.integral = 0.0
        self.target_speed = Config.TARGET_SPEED

    def calculate(self, current_speed):
        err = self.target_speed - current_speed
        self.integral = np.clip(self.integral + err * 0.05, -1.0, 1.0)
        deriv = (err - self.prev_err) / 0.05 if 0.05 != 0 else 0.0
        output = self.kp * err + self.ki * self.integral + self.kd * deriv
        self.prev_err = err
        return np.clip(output, 0.1, 1.0)


class PurePursuitController:
    def __init__(self):
        self.lookahead_dist = Config.LOOKAHEAD_DISTANCE
        self.max_steer_rad = math.radians(Config.MAX_STEER_ANGLE)

    def calculate_steer(self, vehicle_loc, vehicle_yaw, waypoints):
        """纯追踪算法计算转向角"""
        if not waypoints:
            return 0.0

        # 转换为车辆坐标系
        wp_coords = np.array(waypoints)
        translated_x = wp_coords[:, 0] - vehicle_loc.x
        translated_y = wp_coords[:, 1] - vehicle_loc.y

        cos_yaw = math.cos(vehicle_yaw)
        sin_yaw = math.sin(vehicle_yaw)
        rotated_x = translated_x * cos_yaw + translated_y * sin_yaw
        rotated_y = -translated_x * sin_yaw + translated_y * cos_yaw

        # 找到前瞻点
        distances = np.hypot(rotated_x, rotated_y)
        valid_indices = np.where(distances >= self.lookahead_dist)[0]
        if len(valid_indices) == 0:
            return 0.0

        target_idx = valid_indices[0]
        target_x = rotated_x[target_idx]
        target_y = rotated_y[target_idx]

        # 计算转向角
        L = 1.0  # 车辆轴距
        steer_rad = math.atan2(2 * L * target_y, self.lookahead_dist ** 2)
        steer_rad = np.clip(steer_rad, -self.max_steer_rad, self.max_steer_rad)
        steer = steer_rad / self.max_steer_rad

        return steer


class TraditionalController:
    """基于路点的传统控制器，整合避障"""

    def __init__(self, world, obstacle_detector):
        self.world = world
        self.map = world.get_map()
        self.waypoint_distance = 10.0
        self.obstacle_detector = obstacle_detector

    def apply_obstacle_avoidance(self, throttle, brake, steer, vehicle, obstacle_info):
        """避障逻辑"""
        if not obstacle_info['has_obstacle']:
            return throttle, brake, steer

        distance = obstacle_info['distance']
        angle = obstacle_info['relative_angle']
        vehicle_speed = self.obstacle_detector.get_vehicle_speed(vehicle)
        relative_speed = obstacle_info['relative_speed']

        # 提前预警
        if distance < Config.EARLY_WARNING_DISTANCE and relative_speed > 0:
            throttle *= 0.5
            if vehicle_speed > 30:
                brake = 0.2

        # 紧急刹车
        if distance < Config.EMERGENCY_BRAKE_DISTANCE:
            logger.warning(f"紧急刹车！距离: {distance:.1f}m, 相对速度: {relative_speed:.1f}km/h")
            return 0.0, 1.0, 0.0

        # 安全跟车
        elif distance < Config.SAFE_FOLLOWING_DISTANCE:
            required_distance = max(8.0, vehicle_speed * 0.5)
            distance_ratio = (distance - required_distance) / Config.SAFE_FOLLOWING_DISTANCE
            distance_ratio = max(0.0, min(1.0, distance_ratio))

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
        """生成控制指令"""
        # 基础状态
        transform = vehicle.get_transform()
        location = vehicle.get_location()
        velocity = vehicle.get_velocity()
        speed = math.hypot(velocity.x, velocity.y, velocity.z) * 3.6

        # 获取障碍物信息
        obstacle_info = self.obstacle_detector.get_obstacle_info()

        # 获取路点
        waypoint = self.map.get_waypoint(location, project_to_road=True)
        next_waypoints = waypoint.next(self.waypoint_distance)
        target_waypoint = next_waypoints[0] if next_waypoints else waypoint

        # 计算基础转向
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

        # 基础速度控制
        if speed < 20:
            throttle = 0.4
            brake = 0.0
        elif speed < 40:
            throttle = 0.2
            brake = 0.0
        else:
            throttle = 0.1
            brake = 0.2

        # 低速强油门（无障碍时）
        if speed < 5.0 and not obstacle_info['has_obstacle']:
            throttle = 0.4
            brake = 0.0

        # 应用避障逻辑
        throttle, brake, steer = self.apply_obstacle_avoidance(
            throttle, brake, steer, vehicle, obstacle_info
        )

        return throttle, brake, steer


# ===================== 可视化 =====================
class Visualizer:
    def __init__(self, spawn_loc, initial_waypoints):
        self.waypoints = np.array(initial_waypoints)
        self.trajectory = []
        self.spawn_loc = (spawn_loc.x, spawn_loc.y)

        plt.rcParams['backend'] = 'TkAgg'
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=Config.PLOT_SIZE)

        # 绘制初始元素
        self.ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1],
                        c='blue', s=50, label='Road Waypoints (CARLA)', zorder=3)
        self.ax.scatter(self.spawn_loc[0], self.spawn_loc[1],
                        c='orange', marker='s', s=150, label='Spawn Point', zorder=5)
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

        # 更新航点
        if new_waypoints is not None:
            self.waypoints = np.array(new_waypoints)
            self.ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1], c='blue', s=50, zorder=3)

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# ===================== 主函数 =====================
def main():
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
        # 连接CARLA服务器
        client = carla.Client('localhost', 2000)
        client.set_timeout(30.0)
        world = client.load_world('Town01')
        logger.info("成功连接CARLA并加载Town01地图")

        # 设置同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        logger.info("已设置同步模式，delta=0.1s")

        # 清理Actor
        Tools.clear_all_actors(world)

        # 设置天气
        weather = carla.WeatherParameters(
            cloudiness=30.0,
            precipitation=0.0,
            sun_altitude_angle=70.0
        )
        world.set_weather(weather)

        # 获取出生点
        map = world.get_map()
        spawn_points = map.get_spawn_points()
        if not spawn_points:
            raise Exception("无可用出生点")
        spawn_point = spawn_points[10]
        logger.info(f"使用出生点：{spawn_point.location}")

        # 生成主车辆
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(Config.VEHICLE_MODEL)
        vehicle_bp.set_attribute('color', '255,0,0')
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(False)
        logger.info(f"车辆{Config.VEHICLE_MODEL}生成成功")

        # 聚焦车辆
        Tools.focus_vehicle(world, vehicle)

        # 生成障碍物车辆
        obstacle_count = 3
        for i in range(obstacle_count):
            spawn_idx = (i + 12) % len(spawn_points)
            other_vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
            other_vehicle = world.try_spawn_actor(other_vehicle_bp, spawn_points[spawn_idx])
            if other_vehicle:
                other_vehicle.set_autopilot(True)
                logger.info(f"生成障碍物车辆 {other_vehicle.type_id}")

        # 配置摄像头
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(Config.CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(Config.CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(Config.CAMERA_FOV))

        # 后视角相机
        third_camera_transform = carla.Transform(
            carla.Location(x=-5.0, y=0.0, z=3.0),
            carla.Rotation(pitch=-15.0)
        )
        third_camera = world.spawn_actor(camera_bp, third_camera_transform, attach_to=vehicle)

        # 前视角相机
        front_camera_transform = carla.Transform(
            carla.Location(x=2.0, y=0.0, z=1.5),
            carla.Rotation(pitch=0.0)
        )
        front_camera = world.spawn_actor(camera_bp, front_camera_transform, attach_to=vehicle)

        # 主摄像头
        camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=2.0, z=1.8)), attach_to=vehicle)

        # 摄像头数据缓存
        camera_data = {'image': np.zeros((Config.CAMERA_HEIGHT, Config.CAMERA_WIDTH, 3), dtype=np.uint8)}

        # 摄像头回调
        def camera_callback(image, data_dict):
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                (image.height, image.width, 4))[:, :, :3]
            data_dict['image'] = array

        def third_camera_callback(image):
            nonlocal third_image
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                (image.height, image.width, 4))[:, :, :3]
            third_image = array

        def front_camera_callback(image):
            nonlocal front_image
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                (image.height, image.width, 4))[:, :, :3]
            front_image = array

        # 注册回调
        third_camera.listen(third_camera_callback)
        front_camera.listen(front_camera_callback)
        camera.listen(lambda img: camera_callback(img, camera_data))
        time.sleep(2.0)
        logger.info("传感器初始化完成")

        # 初始化核心组件
        initial_waypoints = Tools.generate_road_waypoints(world, spawn_point.location, Config.WAYPOINT_COUNT)
        obstacle_detector = ObstacleDetector(world, vehicle)
        traditional_controller = TraditionalController(world, obstacle_detector)
        speed_controller = PIDSpeedController()
        path_controller = PurePursuitController()
        visualizer = Visualizer(spawn_point.location, initial_waypoints)

        # 控制变量
        throttle = 0.3
        steer = 0.0
        brake = 0.0
        frame_count = 0
        stuck_count = 0
        last_position = vehicle.get_location()

        # 主循环
        logger.info("\n🚀 自动巡航小车启动（按q退出，r重置）")
        while True:
            world.tick()
            frame_count += 1

            # 获取车辆状态
            vehicle_loc, vehicle_yaw, current_speed = Tools.get_vehicle_pose(vehicle)
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location

            # 检测障碍物
            obstacle_info = obstacle_detector.get_obstacle_info()

            # 实时更新航点（每10帧）
            new_waypoints = None
            if frame_count % 10 == 0:
                new_waypoints = Tools.generate_road_waypoints(
                    world, vehicle_location, Config.WAYPOINT_COUNT)

            # 防卡滞检测
            distance_moved = vehicle_location.distance(last_position)
            vehicle_speed = math.hypot(
                vehicle.get_velocity().x,
                vehicle.get_velocity().y,
                vehicle.get_velocity().z
            )
            is_moving = distance_moved > 0.2 or vehicle_speed > 1.0

            if obstacle_info['has_obstacle'] and obstacle_info['distance'] < Config.SAFE_FOLLOWING_DISTANCE:
                stuck_count = 0
            elif not is_moving:
                stuck_count += 1
            else:
                stuck_count = 0
            last_position = vehicle_location

            # 卡滞恢复
            if stuck_count > 20:
                logger.warning("检测到车辆卡滞，执行恢复程序...")
                # 紧急刹车
                vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
                time.sleep(0.5)
                # 倒车/转向
                if obstacle_info['has_obstacle'] and obstacle_info['distance'] < 15:
                    vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0, steer=0.0, brake=0.0, reverse=True))
                    time.sleep(1.0)
                else:
                    recovery_steer = random.choice([-0.5, 0.5])
                    vehicle.apply_control(carla.VehicleControl(
                        throttle=0.6, steer=recovery_steer, brake=0.0))
                    time.sleep(1.0)
                stuck_count = 0

            # 生成控制指令
            throttle, brake, steer = traditional_controller.get_control(vehicle)

            # 应用控制
            if brake >= 1.0:
                vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0, steer=steer, brake=1.0, hand_brake=True))
            else:
                vehicle.apply_control(carla.VehicleControl(
                    throttle=throttle, steer=steer, brake=brake, hand_brake=False))

            # 可视化更新
            visualizer.update(vehicle_loc.x, vehicle_loc.y, new_waypoints)

            # 图像显示
            if third_image is not None:
                display_image = third_image.copy()
                display_image = obstacle_detector.visualize_obstacles(display_image, vehicle_transform)

                # 绘制状态信息
                cv2.putText(display_image, f"Speed: {current_speed:.1f} km/h", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"Throttle: {throttle:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"Steer: {steer:.2f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"Brake: {brake:.2f}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if obstacle_info['has_obstacle']:
                    cv2.putText(display_image, f"Obstacle: {obstacle_info['distance']:.1f}m", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_image, "Obstacle: None", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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

        # 销毁Actor
        Tools.clear_all_actors(world)

        # 关闭同步模式
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        time.sleep(1)
        logger.info("资源清理完成，仿真结束")


if __name__ == "__main__":
    main()