#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA 多车辆协同控制版：V2.0 增强感知（LiDAR+碰撞检测+障碍物避障）
"""

import sys
import os
import carla
import numpy as np
import math
import pygame
import traceback
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import random

# ===================== 全局配置 =======================
# CARLA连接
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT = 20.0

# 多车辆配置
VEHICLE_COUNT = 3
VEHICLE_MODELS = [
    "vehicle.tesla.model3",
    "vehicle.bmw.grandtourer",
    "vehicle.audi.a2"
]
SPAWN_INTERVAL = 1.0
SPAWN_RETRY_MAX = 8
SPAWN_RETRY_DELAY = 0.5
SPAWN_DISTANCE_LIMIT = 15.0  # 放宽距离限制到15米

# 车辆控制参数
VEHICLE_WHEELBASE = 2.9
VEHICLE_REAR_AXLE_OFFSET = 1.45
LOOKAHEAD_DIST_STRAIGHT = 7.0
LOOKAHEAD_DIST_CURVE = 4.0
STEER_GAIN_STRAIGHT = 0.7
STEER_GAIN_CURVE = 1.0
STEER_DEADZONE = 0.05
STEER_LOWPASS_ALPHA = 0.6
MAX_STEER = 1.0
DIR_CHANGE_GENTLE = 0.03
DIR_CHANGE_SHARP = 0.08
BASE_SPEEDS = [25.0, 22.0, 20.0]
PID_KP = 0.2
PID_KI = 0.01
PID_KD = 0.02

# ACC跟车配置（V1.0保留）
SAFE_TIME_GAP = 1.5  # 安全时距（秒）
MIN_SAFE_DISTANCE = 5.0  # 最小安全距离（米）
EMERGENCY_DECEL_RATE = 5.0  # 紧急制动减速度（km/h/帧）
LEAD_BRAKE_THRESHOLD = -10.0  # 前车急刹加速度阈值（km/h/s）

# LiDAR与障碍物检测配置（V2.0新增）
LIDAR_RANGE = 30.0  # LiDAR检测范围（米）
LIDAR_POINTS_PER_SECOND = 100000  # LiDAR点云密度
LIDAR_ROTATION_FREQ = 30  # LiDAR刷新率（Hz）
OBSTACLE_DETECTION_WIDTH = 2.0  # 检测宽度（左右各2米）
OBSTACLE_MIN_HEIGHT = 0.5  # 障碍物最小高度（过滤地面）
OBSTACLE_WARNING_DIST = 8.0  # 障碍物预警距离（米）
OBSTACLE_EMERGENCY_DIST = 5.0  # 障碍物紧急制动距离（米）
OBSTACLE_DECEL_RATE = 8.0  # 障碍物制动减速度（km/h/帧）

# 交通规则配置
TRAFFIC_LIGHT_STOP_DISTANCE = 4.0
TRAFFIC_LIGHT_DETECTION_RANGE = 50.0
TRAFFIC_LIGHT_ANGLE_THRESHOLD = 60.0
STOP_SPEED_THRESHOLD = 0.2
GREEN_LIGHT_ACCEL_FACTOR = 0.25
STOP_LINE_SIM_DISTANCE = 5.0
RED_LIGHT_DURATION = 3.0
GREEN_LIGHT_DURATION = 5.0
YELLOW_LIGHT_DURATION = 2.0

# 相机配置
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CAMERA_FOV = 120
CAMERA_POS = carla.Transform(carla.Location(x=-6.0, z=2.5), carla.Rotation(pitch=-5))

# 全局变量
current_view_vehicle_id = 1
vehicle_agents = []
COLLISION_FLAG = {}  # 碰撞标志（V2.0扩展）
OBSTACLE_FLAG = {}   # 障碍物标志（V2.0新增）

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - 车辆%(vehicle_id)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("multi_vehicle_simulation_v2.log"), logging.StreamHandler()]
)

# ===================== 核心工具函数 ======================
def is_actor_alive(actor):
    try:
        return actor.is_alive()
    except TypeError:
        return actor.is_alive

def get_traffic_light_stop_line(traffic_light):
    try:
        return traffic_light.get_stop_line_location()
    except AttributeError:
        tl_transform = traffic_light.get_transform()
        forward_vec = tl_transform.get_forward_vector()
        stop_line_loc = tl_transform.location - forward_vec * STOP_LINE_SIM_DISTANCE
        stop_line_loc.z = tl_transform.location.z
        return stop_line_loc

def calculate_dir_change(current_wp):
    waypoints = [current_wp]
    for i in range(5):
        next_wps = waypoints[-1].next(1.0)
        if next_wps:
            waypoints.append(next_wps[0])
        else:
            break

    if len(waypoints) < 4:
        return 0.0, 0

    dirs = []
    for i in range(1, len(waypoints)):
        wp_prev = waypoints[i-1]
        wp_curr = waypoints[i]
        dir_rad = math.atan2(
            wp_curr.transform.location.y - wp_prev.transform.location.y,
            wp_curr.transform.location.x - wp_prev.transform.location.x
        )
        dirs.append(dir_rad)

    dir_change = 0.0
    for i in range(1, len(dirs)):
        dir_change += abs(dirs[i] - dirs[i-1]) * 2

    if dir_change < DIR_CHANGE_GENTLE:
        curve_level = 0
    elif dir_change < DIR_CHANGE_SHARP:
        curve_level = 1
    else:
        curve_level = 2

    return dir_change, curve_level

def get_forward_waypoint(vehicle, map):
    vehicle_transform = vehicle.get_transform()
    current_wp = map.get_waypoint(
        vehicle_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    # 所有车辆使用同一车道的路点
    global vehicle_agents
    if len(vehicle_agents) > 0:
        try:
            lead_vehicle_wp = map.get_waypoint(vehicle_agents[0].vehicle.get_transform().location, project_to_road=True)
            current_wp = map.get_waypoint(vehicle_transform.location, project_to_road=True, lane_id=lead_vehicle_wp.lane_id)
        except:
            pass

    road_direction = current_wp.transform.get_forward_vector()
    vehicle_direction = vehicle_transform.get_forward_vector()
    dot_product = road_direction.x * vehicle_direction.x + road_direction.y * vehicle_direction.y

    if dot_product < 0.0:
        forward_wps = current_wp.next(10.0)
        if forward_wps:
            current_wp = forward_wps[0]
        else:
            current_wp = map.get_waypoint(
                vehicle_transform.location + vehicle_direction * 5.0,
                project_to_road=True
            )

    return current_wp

def get_valid_spawn_points(map, count, base_location=None, radius=100.0):
    """
    获取有效的出生点（增加容错性，避免索引越界）
    """
    # 1. 获取地图所有出生点
    all_spawn_points = map.get_spawn_points()
    if not all_spawn_points:
        raise RuntimeError("地图中无任何出生点")

    # 2. 初始化候选点列表
    candidate_points = []

    # 3. 如果有基准位置，先筛选附近的点；否则直接使用所有点
    if base_location:
        filtered_points = []
        for sp in all_spawn_points:
            dist = sp.location.distance(base_location)
            if dist <= radius:
                filtered_points.append((dist, sp))
        # 按距离排序
        filtered_points.sort(key=lambda x: x[0])
        candidate_points = [sp for _, sp in filtered_points]

    # 4. 如果候选点为空，直接使用所有出生点（容错）
    if not candidate_points:
        candidate_points = all_spawn_points
        print(f"警告：基准位置{base_location}附近无出生点，使用全局出生点")

    # 5. 筛选集中的出生点（放宽条件）
    valid_points = []
    # 确保基准点存在（核心修复：避免candidate_points[0]索引越界）
    if not candidate_points:
        candidate_points = all_spawn_points

    base_sp = candidate_points[0]
    valid_points.append(base_sp)

    # 6. 筛选其他点，放宽距离限制
    for sp in candidate_points[1:]:
        try:
            # 检查与已选点的距离（放宽到15米）
            if all(sp.location.distance(vp.location) <= SPAWN_DISTANCE_LIMIT for vp in valid_points):
                wp = map.get_waypoint(sp.location, project_to_road=True)
                if wp.lane_type == carla.LaneType.Driving and 0.0 <= sp.location.z <= 2.0:
                    valid_points.append(sp)
            if len(valid_points) >= count:
                break
        except:
            continue

    # 7. 如果数量不够，进一步放宽条件（距离限制到20米）
    if len(valid_points) < count:
        for sp in candidate_points:
            if sp not in valid_points:
                try:
                    if all(sp.location.distance(vp.location) <= SPAWN_DISTANCE_LIMIT * 1.5 for vp in valid_points):
                        wp = map.get_waypoint(sp.location, project_to_road=True)
                        if wp.lane_type == carla.LaneType.Driving and 0.0 <= sp.location.z <= 2.0:
                            valid_points.append(sp)
                    if len(valid_points) >= count:
                        break
                except:
                    continue

    # 8. 如果还是不够，直接取前N个点（最终容错）
    if len(valid_points) < count:
        print(f"警告：无法找到{count}个集中的出生点，直接取前{count}个可用点")
        for sp in candidate_points:
            if sp not in valid_points:
                wp = map.get_waypoint(sp.location, project_to_road=True)
                if wp.lane_type == carla.LaneType.Driving and 0.0 <= sp.location.z <= 2.0:
                    valid_points.append(sp)
            if len(valid_points) >= count:
                break

    # 9. 最终检查：确保数量足够
    if len(valid_points) < count:
        # 直接取所有可用点，不足的话重复使用（极端情况）
        while len(valid_points) < count:
            valid_points.append(valid_points[0])
        print(f"警告：出生点数量不足，重复使用已有点")

    # 10. 统一出生点朝向
    try:
        forward_vec = valid_points[0].transform.get_forward_vector()
        for sp in valid_points:
            sp.rotation.yaw = math.degrees(math.atan2(forward_vec.y, forward_vec.x))
    except:
        pass

    return valid_points[:count]

def check_spawn_collision(world, spawn_point, radius=3.0):
    # 检查周围车辆和行人
    vehicles = world.get_actors().filter("vehicle.*")
    for vehicle in vehicles:
        if is_actor_alive(vehicle):
            dist = vehicle.get_transform().location.distance(spawn_point.location)
            if dist < radius:
                return False

    walkers = world.get_actors().filter("walker.*")
    for walker in walkers:
        if is_actor_alive(walker):
            dist = walker.get_transform().location.distance(spawn_point.location)
            if dist < radius:
                return False

    return True

# ===================== 传感器管理类（V2.0重构：相机+LiDAR+碰撞）=====================
class VehicleSensors:
    def __init__(self, world, vehicle, vehicle_id):
        self.world = world
        self.vehicle = vehicle
        self.vehicle_id = vehicle_id
        self.logger = logging.getLogger(__name__)
        self.logger = logging.LoggerAdapter(self.logger, {"vehicle_id": vehicle_id})

        # 传感器实例
        self.camera = None
        self.lidar = None
        self.collision_sensor = None

        # 数据存储
        self.image_surface = None
        self.obstacle_distances = []  # 前方障碍物距离列表
        self.last_obstacle_dist = float('inf')  # 最近障碍物距离

        # 创建所有传感器
        self._create_camera()
        self._create_lidar()
        self._create_collision_sensor()

    def _create_camera(self):
        """创建RGB相机传感器"""
        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(640))
        camera_bp.set_attribute("image_size_y", str(360))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))

        # 生成相机（附加到车辆）
        self.camera = self.world.spawn_actor(camera_bp, CAMERA_POS, attach_to=self.vehicle)
        # 注册图像回调函数
        self.camera.listen(self._on_image)

    def _create_lidar(self):
        """创建LiDAR传感器"""
        lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        # 设置LiDAR参数
        lidar_bp.set_attribute("range", str(LIDAR_RANGE))
        lidar_bp.set_attribute("points_per_second", str(LIDAR_POINTS_PER_SECOND))
        lidar_bp.set_attribute("rotation_frequency", str(LIDAR_ROTATION_FREQ))
        lidar_bp.set_attribute("channels", "32")  # 32线LiDAR
        lidar_bp.set_attribute("upper_fov", "15")
        lidar_bp.set_attribute("lower_fov", "-25")
        lidar_bp.set_attribute("points_per_second", str(LIDAR_POINTS_PER_SECOND))

        # LiDAR挂载位置（车顶）
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        # 注册LiDAR回调函数
        self.lidar.listen(self._on_lidar_data)

    def _create_collision_sensor(self):
        """创建碰撞传感器"""
        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        # 注册碰撞回调函数
        self.collision_sensor.listen(self._on_collision)

    def _on_image(self, image):
        """相机图像回调：转换为Pygame Surface"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = np.swapaxes(array, 0, 1)
        self.image_surface = pygame.surfarray.make_surface(array)

    def _on_lidar_data(self, data):
        """LiDAR点云回调：解析并提取前方障碍物"""
        try:
            # 将点云数据转换为numpy数组 (x, y, z, intensity)
            points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
            
            # 过滤条件：
            # 1. 前方（x>0）
            # 2. 左右范围内（|y| < 检测宽度）
            # 3. 非地面（z > 最小高度）
            front_obstacle_points = points[
                (points[:, 0] > 0) & 
                (np.abs(points[:, 1]) < OBSTACLE_DETECTION_WIDTH) & 
                (points[:, 2] > OBSTACLE_MIN_HEIGHT)
            ]

            # 计算最近障碍物距离
            if len(front_obstacle_points) > 0:
                self.last_obstacle_dist = np.min(front_obstacle_points[:, 0])
                self.obstacle_distances.append(self.last_obstacle_dist)
                # 只保留最近10帧数据（平滑滤波）
                if len(self.obstacle_distances) > 10:
                    self.obstacle_distances.pop(0)
            else:
                self.last_obstacle_dist = float('inf')
                self.obstacle_distances.clear()

        except Exception as e:
            self.logger.error(f"LiDAR数据解析失败：{e}")

    def _on_collision(self, event):
        """碰撞回调：记录碰撞信息并触发紧急停车"""
        try:
            collision_actor_type = event.other_actor.type_id
            collision_location = event.transform.location
            self.logger.error(
                f"发生碰撞！碰撞对象：{collision_actor_type} | 碰撞位置：({collision_location.x:.1f}, {collision_location.y:.1f})"
            )
            # 设置碰撞标志
            global COLLISION_FLAG
            COLLISION_FLAG[self.vehicle_id] = True
        except Exception as e:
            self.logger.error(f"碰撞检测回调失败：{e}")

    def get_average_obstacle_distance(self):
        """获取平滑后的障碍物距离"""
        if len(self.obstacle_distances) == 0:
            return float('inf')
        return np.mean(self.obstacle_distances)

    def destroy(self):
        """销毁所有传感器"""
        sensors = [self.camera, self.lidar, self.collision_sensor]
        for sensor in sensors:
            if sensor:
                try:
                    sensor.stop()
                    sensor.destroy()
                except:
                    pass
        self.logger.info("所有传感器已销毁")

# ===================== 车辆控制类 ======================
class VehicleAgent:
    def __init__(self, world, map, vehicle_id, spawn_point, vehicle_model, base_speed):
        self.vehicle_id = vehicle_id
        self.world = world
        self.map = map
        self.base_speed = base_speed
        self.logger = logging.getLogger(__name__)
        self.logger = logging.LoggerAdapter(self.logger, {"vehicle_id": vehicle_id})

        # ACC跟车相关属性（V1.0保留）
        self.last_lead_speed = 0.0  # 前车上次速度
        self.stuck_count = 0  # 卡死计数
        
        # 全局状态初始化（V2.0扩展）
        global COLLISION_FLAG, OBSTACLE_FLAG
        COLLISION_FLAG[self.vehicle_id] = False
        OBSTACLE_FLAG[self.vehicle_id] = False

        # 生成车辆
        self.vehicle_bp = self.world.get_blueprint_library().find(vehicle_model)
        if self.vehicle_bp.has_attribute("color"):
            color = random.choice(self.vehicle_bp.get_attribute("color").recommended_values)
            self.vehicle_bp.set_attribute("color", color)

        self.vehicle = self._spawn_vehicle_with_retry(spawn_point)
        if not self.vehicle:
            raise RuntimeError(f"车辆{vehicle_id}生成失败")

        # 创建传感器（V2.0替换原相机类）
        self.sensors = VehicleSensors(world, self.vehicle, vehicle_id)

        # 初始化控制器
        self.pp_controller = AdaptivePurePursuit(VEHICLE_WHEELBASE)
        self.speed_controller = SpeedController(PID_KP, PID_KI, PID_KD, base_speed)
        self.traffic_light_manager = TrafficLightManager(vehicle_id)

        self.is_alive = True
        self.logger.info(f"生成成功，车型：{vehicle_model}，出生点：({spawn_point.location.x:.1f},{spawn_point.location.y:.1f})")

    def _spawn_vehicle_with_retry(self, initial_spawn_point):
        all_spawn_points = self.map.get_spawn_points()
        if not all_spawn_points:
            self.logger.error("地图中无有效出生点")
            return None

        candidate_points = [initial_spawn_point]
        candidate_points += random.sample(all_spawn_points, min(10, len(all_spawn_points)))

        for retry in range(SPAWN_RETRY_MAX):
            spawn_point = candidate_points[retry % len(candidate_points)]
            spawn_point.location.z += 0.3
            spawn_point.rotation.yaw += random.randint(-5, 5)

            if not check_spawn_collision(self.world, spawn_point):
                self.logger.warning(f"第{retry+1}次重试：出生点有碰撞风险，跳过")
                time.sleep(SPAWN_RETRY_DELAY)
                continue

            try:
                return self.world.spawn_actor(self.vehicle_bp, spawn_point)
            except Exception as e:
                self.logger.warning(f"第{retry+1}次重试失败：{e}")
                time.sleep(SPAWN_RETRY_DELAY)

        self.logger.error(f"超过{SPAWN_RETRY_MAX}次重试，生成失败")
        return None

    def update(self):
        if not self.is_alive or not is_actor_alive(self.vehicle):
            self.is_alive = False
            self.logger.error("车辆已销毁，停止更新")
            return False

        try:
            # 获取车辆状态
            vehicle_transform = self.vehicle.get_transform()
            vehicle_vel = self.vehicle.get_velocity()
            current_speed = math.hypot(vehicle_vel.x, vehicle_vel.y) * 3.6

            # 路径跟踪
            current_wp = get_forward_waypoint(self.vehicle, self.map)
            dir_change, curve_level = calculate_dir_change(current_wp)
            lookahead_dist = self.pp_controller.get_adaptive_lookahead(dir_change)
            target_wps = current_wp.next(lookahead_dist)
            target_point = target_wps[0].transform.location if target_wps else vehicle_transform.location

            # 速度控制（基础弯道速度）
            curve_speed_factors = [1.0, 0.7, 0.4]
            speed_factor = curve_speed_factors[min(curve_level, 2)]
            base_target_speed = self.base_speed * speed_factor
            base_target_speed = max(8.0, base_target_speed)

            # ========== V1.0保留：精细化ACC跟车+紧急避障逻辑 ==========
            if self.vehicle_id > 1 and len(vehicle_agents) >= self.vehicle_id:
                try:
                    lead_agent = vehicle_agents[self.vehicle_id - 2]
                    lead_vehicle = lead_agent.vehicle
                    lead_vehicle_transform = lead_vehicle.get_transform()
                    
                    # 计算前车速度和加速度
                    lead_vel = lead_vehicle.get_velocity()
                    lead_speed = math.hypot(lead_vel.x, lead_vel.y) * 3.6
                    lead_acc = (lead_speed - lead_agent.last_lead_speed) / 0.03  # 30Hz刷新率，计算加速度
                    lead_agent.last_lead_speed = lead_speed  # 更新前车上次速度
                    
                    # 计算安全跟车距离（安全时距+最小安全距）
                    safe_dist = (current_speed / 3.6) * SAFE_TIME_GAP + MIN_SAFE_DISTANCE
                    dist_to_lead = vehicle_transform.location.distance(lead_vehicle_transform.location)
                    
                    # 动态调整目标速度
                    if dist_to_lead < safe_dist - 2:
                        # 过近：减速至前车速度-2（不低于5km/h）
                        base_target_speed = max(5.0, lead_speed - 2)
                    elif dist_to_lead > safe_dist + 2:
                        # 过远：加速至前车速度+2（不超基础速度）
                        base_target_speed = min(self.base_speed * speed_factor, lead_speed + 2)
                    else:
                        # 安全距离：与前车速度同步
                        base_target_speed = lead_speed
                    
                    # 紧急避障：前车急刹（加速度<阈值）
                    if lead_acc < LEAD_BRAKE_THRESHOLD:
                        base_target_speed = max(0.0, current_speed - EMERGENCY_DECEL_RATE)
                        self.logger.warning(f"前车急刹（加速度{lead_acc:.1f}km/h/s）！紧急减速至{base_target_speed:.1f}km/h")
                        
                except Exception as e:
                    self.logger.warning(f"ACC跟车计算异常：{e}")

            # ========== V2.0新增：LiDAR障碍物检测与避障 ==========
            obstacle_dist = self.sensors.get_average_obstacle_distance()
            OBSTACLE_FLAG[self.vehicle_id] = obstacle_dist < OBSTACLE_WARNING_DIST
            
            if obstacle_dist < OBSTACLE_EMERGENCY_DIST:
                # 紧急制动：直接减速至0
                base_target_speed = max(0.0, current_speed - OBSTACLE_DECEL_RATE)
                self.logger.warning(f"前方{obstacle_dist:.1f}米检测到障碍物！紧急制动，目标速度：{base_target_speed:.1f}km/h")
            elif obstacle_dist < OBSTACLE_WARNING_DIST:
                # 预警减速：降低至基础速度的50%
                base_target_speed = max(8.0, base_target_speed * 0.5)
                self.logger.warning(f"前方{obstacle_dist:.1f}米检测到障碍物！预警减速，目标速度：{base_target_speed:.1f}km/h")

            # 交通灯处理
            target_speed, traffic_light_status = self.traffic_light_manager.handle_traffic_light_logic(
                self.vehicle, current_speed, base_target_speed
            )

            # 计算控制指令
            steer = self.pp_controller.calculate_steer(vehicle_transform, target_point, dir_change)
            throttle = self.speed_controller.calculate(target_speed, current_speed)
            brake = 1.0 - throttle if current_speed > target_speed + 1 else 0.0

            # 状态优先级：碰撞 > 红灯 > 障碍物 > 正常行驶
            if COLLISION_FLAG.get(self.vehicle_id, False):
                throttle = 0.0
                brake = 1.0
                self.logger.error("检测到碰撞，紧急停车！")
            elif "Red (Stopped)" in traffic_light_status or target_speed == 0.0:
                throttle = 0.0
                brake = 1.0
            elif obstacle_dist < OBSTACLE_EMERGENCY_DIST:
                throttle = 0.0
                brake = 1.0

            # 应用控制
            control = carla.VehicleControl()
            control.steer = steer
            control.throttle = throttle
            control.brake = brake
            self.vehicle.apply_control(control)

            # 日志输出（新增障碍物信息）
            obstacle_status = f"障碍物{obstacle_dist:.1f}m" if obstacle_dist < OBSTACLE_WARNING_DIST else "无障碍物"
            self.logger.info(
                f"速度：{current_speed:5.1f}km/h | 目标：{target_speed:5.1f} | "
                f"弯道：{['直道', '缓弯', '急弯'][curve_level]:<3} | 灯状态：{traffic_light_status} | "
                f"ACC：{'激活' if self.vehicle_id>1 else '未激活'} | {obstacle_status}"
            )

            return True

        except Exception as e:
            self.logger.error(f"更新失败：{e}", exc_info=True)
            return False

    def destroy(self):
        # 销毁传感器
        self.sensors.destroy()
        # 销毁车辆
        if self.vehicle and is_actor_alive(self.vehicle):
            self.vehicle.destroy()
        self.logger.info("车辆及传感器资源已清理")

# ===================== 控制器类 ======================
class AdaptivePurePursuit:
    def __init__(self, wheelbase):
        self.wheelbase = wheelbase
        self.last_steer = 0.0
        self.last_lookahead = LOOKAHEAD_DIST_STRAIGHT

    def calculate_steer(self, vehicle_transform, target_point, dir_change):
        forward_vec = vehicle_transform.get_forward_vector()
        rear_axle_loc = carla.Location(
            x=vehicle_transform.location.x - forward_vec.x * VEHICLE_REAR_AXLE_OFFSET,
            y=vehicle_transform.location.y - forward_vec.y * VEHICLE_REAR_AXLE_OFFSET,
            z=vehicle_transform.location.z
        )

        dx = target_point.x - rear_axle_loc.x
        dy = target_point.y - rear_axle_loc.y
        yaw = math.radians(vehicle_transform.rotation.yaw)

        dx_vehicle = dx * math.cos(yaw) + dy * math.sin(yaw)
        dy_vehicle = -dx * math.sin(yaw) + dy * math.cos(yaw)

        steer_gain = np.interp(
            dir_change,
            [0, DIR_CHANGE_SHARP],
            [STEER_GAIN_STRAIGHT, STEER_GAIN_CURVE]
        )
        steer_gain = np.clip(steer_gain, STEER_GAIN_STRAIGHT, STEER_GAIN_CURVE)

        if dx_vehicle < 0.1:
            steer = self.last_steer
        else:
            steer_rad = math.atan2(2 * self.wheelbase * dy_vehicle, dx_vehicle ** 2 + dy_vehicle ** 2)
            steer = steer_rad / math.pi
            steer *= steer_gain

        if abs(steer) < STEER_DEADZONE:
            steer = 0.0
        steer = STEER_LOWPASS_ALPHA * steer + (1 - STEER_LOWPASS_ALPHA) * self.last_steer
        steer = np.clip(steer, -MAX_STEER, MAX_STEER)

        self.last_steer = steer
        return steer

    def get_adaptive_lookahead(self, dir_change):
        lookahead_dist = np.interp(
            dir_change,
            [0, DIR_CHANGE_SHARP],
            [LOOKAHEAD_DIST_STRAIGHT, LOOKAHEAD_DIST_CURVE]
        )
        lookahead_dist = np.clip(lookahead_dist, LOOKAHEAD_DIST_CURVE, LOOKAHEAD_DIST_STRAIGHT)
        self.last_lookahead = lookahead_dist
        return lookahead_dist

class SpeedController:
    def __init__(self, kp, ki, kd, base_speed):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.base_speed = base_speed
        self.last_error = 0.0
        self.integral = 0.0

    def calculate(self, target_speed, current_speed):
        error = target_speed - current_speed
        p = self.kp * error
        self.integral += self.ki * error
        self.integral = np.clip(self.integral, -1.0, 1.0)
        d = self.kd * (error - self.last_error)
        self.last_error = error
        return np.clip(p + self.integral + d, 0.0, 1.0)  # 修复原代码i未定义问题

class TrafficLightManager:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.tracked_light = None
        self.is_stopped_at_red = False
        self.red_light_stop_time = 0
        self.logger = logging.getLogger(__name__)
        self.logger = logging.LoggerAdapter(self.logger, {"vehicle_id": vehicle_id})

    def _calculate_angle_between_vehicle_and_light(self, vehicle_transform, light_transform):
        vehicle_forward = vehicle_transform.get_forward_vector()
        vehicle_forward = np.array([vehicle_forward.x, vehicle_forward.y])
        vehicle_forward = vehicle_forward / np.linalg.norm(vehicle_forward)

        light_dir = light_transform.location - vehicle_transform.location
        light_dir = np.array([light_dir.x, light_dir.y])
        if np.linalg.norm(light_dir) < 0.1:
            return 0.0
        light_dir = light_dir / np.linalg.norm(light_dir)

        angle = math.acos(np.clip(np.dot(vehicle_forward, light_dir), -1.0, 1.0))
        angle = math.degrees(angle)
        return angle

    def get_lane_traffic_light(self, vehicle, world):
        vehicle_transform = vehicle.get_transform()
        vehicle_loc = vehicle_transform.location

        if self.tracked_light and is_actor_alive(self.tracked_light):
            dist = self.tracked_light.get_transform().location.distance(vehicle_loc)
            angle = self._calculate_angle_between_vehicle_and_light(vehicle_transform, self.tracked_light.get_transform())
            if dist < TRAFFIC_LIGHT_DETECTION_RANGE and angle < TRAFFIC_LIGHT_ANGLE_THRESHOLD:
                return self.tracked_light

        traffic_lights = world.get_actors().filter("traffic.traffic_light")
        valid_lights = []

        for light in traffic_lights:
            if not is_actor_alive(light):
                continue
            dist = light.get_transform().location.distance(vehicle_loc)
            if dist > TRAFFIC_LIGHT_DETECTION_RANGE:
                continue
            angle = self._calculate_angle_between_vehicle_and_light(vehicle_transform, light.get_transform())
            if angle < TRAFFIC_LIGHT_ANGLE_THRESHOLD:
                valid_lights.append((dist, light))

        if valid_lights:
            valid_lights.sort(key=lambda x: x[0])
            self.tracked_light = valid_lights[0][1]
            return self.tracked_light

        self.tracked_light = None
        return None

    def handle_traffic_light_logic(self, vehicle, current_speed, base_target_speed):
        world = vehicle.get_world()
        traffic_light = self.get_lane_traffic_light(vehicle, world)

        if not traffic_light:
            self.is_stopped_at_red = False
            self.red_light_stop_time = 0
            return base_target_speed, "No Light"

        stop_line_loc = get_traffic_light_stop_line(traffic_light)
        dist_to_stop_line = vehicle.get_transform().location.distance(stop_line_loc)

        if traffic_light.get_state() == carla.TrafficLightState.Green:
            if self.is_stopped_at_red:
                recovery_speed = current_speed + (base_target_speed - current_speed) * GREEN_LIGHT_ACCEL_FACTOR
                target_speed = max(STOP_SPEED_THRESHOLD, recovery_speed)
                if abs(target_speed - base_target_speed) < 0.5:
                    self.is_stopped_at_red = False
                    self.logger.info(f"绿灯恢复行驶，目标速度：{target_speed:.1f}km/h")
                return target_speed, "Green"
            return base_target_speed, "Green"

        elif traffic_light.get_state() == carla.TrafficLightState.Yellow:
            self.is_stopped_at_red = False
            yellow_speed = max(5.0, base_target_speed * 0.3)
            self.logger.warning(f"黄灯减速，目标速度：{yellow_speed:.1f}km/h")
            return yellow_speed, "Yellow"

        elif traffic_light.get_state() == carla.TrafficLightState.Red:
            if dist_to_stop_line > TRAFFIC_LIGHT_STOP_DISTANCE:
                self.is_stopped_at_red = False
                red_speed = max(2.0, current_speed * 0.1)
                self.logger.warning(f"红灯减速，距离停止线：{dist_to_stop_line:.1f}m，目标速度：{red_speed:.1f}km/h")
                return red_speed, "Red"
            else:
                if current_speed <= STOP_SPEED_THRESHOLD:
                    self.is_stopped_at_red = True
                    self.red_light_stop_time += 1
                    wait_seconds = self.red_light_stop_time // 30
                    self.logger.info(f"红灯停车等待：{wait_seconds}s")
                    return 0.0, "Red (Stopped)"
                else:
                    self.logger.warning("红灯紧急制动")
                    return 0.0, "Red (Braking)"

        return base_target_speed, "Unknown"

# ===================== 交通灯控制线程 ======================
def cycle_traffic_light_states(world, stop_event):
    logger = logging.getLogger(__name__)
    logger = logging.LoggerAdapter(logger, {"vehicle_id": "系统"})
    while not stop_event.is_set():
        traffic_lights = world.get_actors().filter("traffic.traffic_light")
        if not traffic_lights:
            time.sleep(1)
            continue

        # 红灯
        for tl in traffic_lights:
            if is_actor_alive(tl):
                try:
                    tl.set_state(carla.TrafficLightState.Red)
                except:
                    pass
        logger.info(f"所有交通灯切换为红灯，持续{RED_LIGHT_DURATION}秒")
        stop_event.wait(RED_LIGHT_DURATION)
        if stop_event.is_set():
            break

        # 绿灯
        for tl in traffic_lights:
            if is_actor_alive(tl):
                try:
                    tl.set_state(carla.TrafficLightState.Green)
                except:
                    pass
        logger.info(f"所有交通灯切换为绿灯，持续{GREEN_LIGHT_DURATION}秒")
        stop_event.wait(GREEN_LIGHT_DURATION)
        if stop_event.is_set():
            break

        # 黄灯
        for tl in traffic_lights:
            if is_actor_alive(tl):
                try:
                    tl.set_state(carla.TrafficLightState.Yellow)
                except:
                    pass
        logger.info(f"所有交通灯切换为黄灯，持续{YELLOW_LIGHT_DURATION}秒")
        stop_event.wait(YELLOW_LIGHT_DURATION)
        if stop_event.is_set():
            break

    logger.info("交通灯线程停止")

# ===================== 主函数 ======================
def main():
    global current_view_vehicle_id, vehicle_agents
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"CARLA多车辆视角（{VEHICLE_COUNT}辆车）- V2.0 LiDAR感知 - 按1/2/3切换视角，按S切换分屏，按V切换俯视视角")

    client = None
    world = None
    map = None
    tl_cycle_thread = None
    tl_stop_event = threading.Event()
    show_split_screen = True
    show_top_view = False
    top_view_camera = None

    # 清理函数
    def cleanup():
        print("\n开始清理资源...")
        tl_stop_event.set()
        if tl_cycle_thread and tl_cycle_thread.is_alive():
            tl_cycle_thread.join(timeout=2)

        if top_view_camera:
            top_view_camera.stop()
            top_view_camera.destroy()

        for agent in vehicle_agents:
            agent.destroy()

        if world:
            for actor in world.get_actors():
                if actor.type_id.startswith(("vehicle.", "walker.", "sensor.")):
                    if is_actor_alive(actor):
                        actor.destroy()

        pygame.quit()
        print("资源清理完成")

    # 注册退出回调
    import atexit
    import signal
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

    try:
        # 连接CARLA
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(CARLA_TIMEOUT)
        try:
            world = client.load_world("Town04")
            print("成功加载Town04地图")
        except Exception as e:
            world = client.get_world()
            print(f"警告：Town04地图加载失败（{e}），使用当前地图")
        map = world.get_map()

        # 清理残留演员
        print("清理残留演员...")
        for actor in world.get_actors():
            if actor.type_id.startswith(("vehicle.", "walker.", "sensor.")):
                if is_actor_alive(actor):
                    actor.destroy()
        time.sleep(3.0)
        print("清理完成")

        # 自动获取地图的第一个出生点作为基准（避免手动坐标无效）
        base_location = None
        all_spawn_points = map.get_spawn_points()
        if all_spawn_points:
            base_location = all_spawn_points[0].location
            print(f"使用地图第一个出生点作为基准：({base_location.x:.1f}, {base_location.y:.1f})")
        else:
            base_location = carla.Location(x=220.0, y=150.0, z=0.5)

        # 获取有效的出生点
        print(f"获取{VEHICLE_COUNT}个有效出生点...")
        valid_spawn_points = get_valid_spawn_points(map, VEHICLE_COUNT, base_location)
        for i, sp in enumerate(valid_spawn_points):
            print(f"  出生点{i+1}：({sp.location.x:.1f},{sp.location.y:.1f})")

        # 生成车辆
        print(f"\n分步生成车辆（间隔{SPAWN_INTERVAL}秒）...")
        for i in range(VEHICLE_COUNT):
            vehicle_model = VEHICLE_MODELS[i % len(VEHICLE_MODELS)]
            base_speed = BASE_SPEEDS[i % len(BASE_SPEEDS)]
            spawn_point = valid_spawn_points[i]

            try:
                print(f"\n生成车辆{i+1}（车型：{vehicle_model}）...")
                agent = VehicleAgent(world, map, i+1, spawn_point, vehicle_model, base_speed)
                vehicle_agents.append(agent)
                print(f"车辆{i+1}生成成功！已挂载LiDAR+碰撞传感器")
            except Exception as e:
                print(f"车辆{i+1}生成失败：{e}")

            time.sleep(SPAWN_INTERVAL)

        if len(vehicle_agents) == 0:
            raise RuntimeError("无车辆生成成功，仿真终止")

        print(f"\n共生成{len(vehicle_agents)}辆车辆！V2.0 LiDAR感知+障碍物避障功能已启用")

        # 创建全局俯视相机
        try:
            top_view_bp = world.get_blueprint_library().find("sensor.camera.rgb")
            top_view_bp.set_attribute("image_size_x", str(WINDOW_WIDTH))
            top_view_bp.set_attribute("image_size_y", str(WINDOW_HEIGHT))
            top_view_bp.set_attribute("fov", str(90))
            top_view_transform = carla.Transform(
                vehicle_agents[0].vehicle.get_transform().location + carla.Location(z=50),
                carla.Rotation(pitch=-90)
            )
            top_view_camera = world.spawn_actor(top_view_bp, top_view_transform)
            top_view_surface = None
            top_view_camera.listen(lambda image: globals().update({
                "top_view_surface": pygame.surfarray.make_surface(
                    np.swapaxes(np.array(image.raw_data).reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1], 0, 1)
                )
            }))
        except:
            print("警告：无法创建俯视相机")

        # 启动交通灯线程
        tl_cycle_thread = threading.Thread(target=cycle_traffic_light_states, args=(world, tl_stop_event), daemon=True)
        tl_cycle_thread.start()
        print("交通灯线程启动")

        # 主循环
        clock = pygame.time.Clock()
        running = True

        while running:
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1 and len(vehicle_agents) >= 1:
                        current_view_vehicle_id = 1
                        show_split_screen = False
                        show_top_view = False
                    elif event.key == pygame.K_2 and len(vehicle_agents) >= 2:
                        current_view_vehicle_id = 2
                        show_split_screen = False
                        show_top_view = False
                    elif event.key == pygame.K_3 and len(vehicle_agents) >= 3:
                        current_view_vehicle_id = 3
                        show_split_screen = False
                        show_top_view = False
                    elif event.key == pygame.K_s:
                        show_split_screen = True
                        show_top_view = False
                    elif event.key == pygame.K_v:
                        show_top_view = True
                        show_split_screen = False
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            # 清空屏幕
            screen.fill((0, 0, 0))

            if show_top_view:
                if top_view_surface:
                    screen.blit(top_view_surface, (0, 0))
            elif show_split_screen:
                if len(vehicle_agents) == 1:
                    agent = vehicle_agents[0]
                    if agent.sensors.image_surface:
                        surface = pygame.transform.scale(agent.sensors.image_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
                        screen.blit(surface, (0, 0))
                elif len(vehicle_agents) == 2:
                    agent1 = vehicle_agents[0]
                    agent2 = vehicle_agents[1]

                    if agent1.sensors.image_surface:
                        surface1 = pygame.transform.scale(agent1.sensors.image_surface, (WINDOW_WIDTH//2, WINDOW_HEIGHT))
                        screen.blit(surface1, (0, 0))

                    if agent2.sensors.image_surface:
                        surface2 = pygame.transform.scale(agent2.sensors.image_surface, (WINDOW_WIDTH//2, WINDOW_HEIGHT))
                        screen.blit(surface2, (WINDOW_WIDTH//2, 0))
                elif len(vehicle_agents) >= 3:
                    agent1 = vehicle_agents[0]
                    agent2 = vehicle_agents[1]
                    agent3 = vehicle_agents[2]

                    if agent1.sensors.image_surface:
                        surface1 = pygame.transform.scale(agent1.sensors.image_surface, (WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
                        screen.blit(surface1, (0, 0))

                    if agent2.sensors.image_surface:
                        surface2 = pygame.transform.scale(agent2.sensors.image_surface, (WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
                        screen.blit(surface2, (WINDOW_WIDTH//2, 0))

                    if agent3.sensors.image_surface:
                        surface3 = pygame.transform.scale(agent3.sensors.image_surface, (WINDOW_WIDTH, WINDOW_HEIGHT//2))
                        screen.blit(surface3, (0, WINDOW_HEIGHT//2))
            else:
                target_agent = None
                for agent in vehicle_agents:
                    if agent.vehicle_id == current_view_vehicle_id:
                        target_agent = agent
                        break

                if target_agent and target_agent.sensors.image_surface:
                    surface = pygame.transform.scale(target_agent.sensors.image_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
                    screen.blit(surface, (0, 0))

            # 更新车辆状态
            with ThreadPoolExecutor(max_workers=VEHICLE_COUNT) as executor:
                futures = [executor.submit(agent.update) for agent in vehicle_agents]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"车辆更新异常：{e}")

            # 刷新屏幕
            pygame.display.flip()
            clock.tick(30)

    except Exception as e:
        print(f"仿真异常：{e}")
        traceback.print_exc()
    finally:
        cleanup()

if __name__ == "__main__":
    main()