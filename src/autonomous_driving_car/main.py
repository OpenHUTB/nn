#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA 多车辆协同控制版：V3.0 终极稳定版
核心特性：
1. 彻底修复传感器销毁警告
2. LiDAR点云处理性能优化（降采样+缓存）
3. 车辆状态实时监控与故障自动恢复
4. 精准障碍物避障+ACC跟车+交通灯合规
5. 多视角流畅切换+性能监控
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
from collections import deque

# ===================== 全局配置（V3.0优化）=======================
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
SPAWN_DISTANCE_LIMIT = 15.0

# 车辆控制参数（V3.0优化）
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
PID_KI = 0.008  # 降低积分系数，减少饱和
PID_KD = 0.02

# ACC跟车配置
SAFE_TIME_GAP = 1.5
MIN_SAFE_DISTANCE = 5.0
EMERGENCY_DECEL_RATE = 5.0
LEAD_BRAKE_THRESHOLD = -10.0

# LiDAR与障碍物检测配置（V3.0性能优化）
LIDAR_RANGE = 30.0
LIDAR_POINTS_PER_SECOND = 50000  # 降采样，减少计算量
LIDAR_ROTATION_FREQ = 20  # 降低刷新率，提升性能
OBSTACLE_DETECTION_WIDTH = 2.0
OBSTACLE_MIN_HEIGHT = 0.5
OBSTACLE_MAX_HEIGHT = 3.0  # 新增最大高度过滤，避免误检高空物体
OBSTACLE_WARNING_DIST = 8.0
OBSTACLE_EMERGENCY_DIST = 5.0
OBSTACLE_DECEL_RATE = 8.0
OBSTACLE_CACHE_SIZE = 5  # 障碍物距离缓存大小，平滑滤波

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

# 性能监控配置
PERF_MONITOR_INTERVAL = 1.0  # 性能监控输出间隔（秒）
VEHICLE_RESTART_THRESHOLD = 5  # 车辆连续故障次数阈值，超过则重启

# 全局变量
current_view_vehicle_id = 1
vehicle_agents = []
COLLISION_FLAG = {}
OBSTACLE_FLAG = {}
last_perf_time = time.time()
perf_stats = {"frame_count": 0, "avg_fps": 0.0}

# 日志配置（V3.0增强）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - 车辆%(vehicle_id)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("multi_vehicle_simulation_v3.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ===================== 核心工具函数（V3.0重构）=====================
def is_actor_alive(actor):
    """安全检查actor是否存活（终极版）"""
    if actor is None:
        return False
    try:
        return actor.is_alive()
    except (TypeError, AttributeError):
        try:
            return actor.is_alive
        except:
            return False

def is_sensor_listening(sensor):
    """检查传感器是否在监听数据"""
    if sensor is None or not is_actor_alive(sensor):
        return False
    try:
        sensor.listen(lambda data: None)
        return True
    except:
        return False

def safe_sensor_stop(sensor, sensor_name, logger):
    """安全停止传感器监听"""
    if sensor is None or not is_actor_alive(sensor):
        return
    try:
        if is_sensor_listening(sensor):
            sensor.stop()
            logger.debug(f"{sensor_name}监听已停止")
    except Exception as e:
        logger.warning(f"停止{sensor_name}监听忽略异常：{str(e)[:50]}")

def safe_actor_destroy(actor, actor_name, logger):
    """安全销毁actor"""
    if actor is None or not is_actor_alive(actor):
        return
    try:
        actor.destroy()
        logger.debug(f"{actor_name}已销毁")
    except Exception as e:
        logger.warning(f"销毁{actor_name}忽略异常：{str(e)[:50]}")

def get_traffic_light_stop_line(traffic_light):
    """获取交通灯停止线位置（容错版）"""
    try:
        return traffic_light.get_stop_line_location()
    except AttributeError:
        tl_transform = traffic_light.get_transform()
        forward_vec = tl_transform.get_forward_vector()
        stop_line_loc = tl_transform.location - forward_vec * STOP_LINE_SIM_DISTANCE
        stop_line_loc.z = tl_transform.location.z
        return stop_line_loc

def calculate_dir_change(current_wp):
    """计算道路方向变化（优化版）"""
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

    curve_level = 0
    if dir_change >= DIR_CHANGE_GENTLE:
        curve_level = 1
    if dir_change >= DIR_CHANGE_SHARP:
        curve_level = 2

    return dir_change, curve_level

def get_forward_waypoint(vehicle, map, wp_cache=None):
    """获取前进方向路点（带缓存优化）"""
    vehicle_transform = vehicle.get_transform()
    cache_key = (round(vehicle_transform.location.x, 1), round(vehicle_transform.location.y, 1))
    
    # 缓存命中直接返回
    if wp_cache and cache_key in wp_cache:
        return wp_cache[cache_key]
    
    current_wp = map.get_waypoint(
        vehicle_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    # 所有车辆使用同一车道
    global vehicle_agents
    if len(vehicle_agents) > 0:
        try:
            lead_vehicle_wp = map.get_waypoint(
                vehicle_agents[0].vehicle.get_transform().location,
                project_to_road=True
            )
            current_wp = map.get_waypoint(
                vehicle_transform.location,
                project_to_road=True,
                lane_id=lead_vehicle_wp.lane_id
            )
        except:
            pass

    # 方向检查与修正
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
    
    # 更新缓存（有效期短，避免过时）
    if wp_cache:
        wp_cache[cache_key] = current_wp
        # 缓存清理：只保留最近100个
        if len(wp_cache) > 100:
            wp_cache.pop(next(iter(wp_cache)))
    
    return current_wp

def get_valid_spawn_points(map, count, base_location=None, radius=100.0):
    """获取有效出生点（终极容错版）"""
    all_spawn_points = map.get_spawn_points()
    if not all_spawn_points:
        raise RuntimeError("地图中无任何出生点")

    candidate_points = []
    if base_location:
        filtered_points = [(sp.location.distance(base_location), sp) for sp in all_spawn_points]
        filtered_points = [sp for dist, sp in sorted(filtered_points) if dist <= radius]
        candidate_points = filtered_points if filtered_points else all_spawn_points
    else:
        candidate_points = all_spawn_points

    # 筛选集中的出生点
    valid_points = []
    if candidate_points:
        base_sp = candidate_points[0]
        valid_points.append(base_sp)

        for sp in candidate_points[1:]:
            if len(valid_points) >= count:
                break
            try:
                if all(sp.location.distance(vp.location) <= SPAWN_DISTANCE_LIMIT for vp in valid_points):
                    wp = map.get_waypoint(sp.location, project_to_road=True)
                    if wp.lane_type == carla.LaneType.Driving and 0.0 <= sp.location.z <= 2.0:
                        valid_points.append(sp)
            except:
                continue

    # 最终容错
    while len(valid_points) < count:
        valid_points.append(valid_points[0] if valid_points else all_spawn_points[0])

    # 统一朝向
    try:
        forward_vec = valid_points[0].transform.get_forward_vector()
        for sp in valid_points:
            sp.rotation.yaw = math.degrees(math.atan2(forward_vec.y, forward_vec.x))
    except:
        pass

    return valid_points[:count]

def check_spawn_collision(world, spawn_point, radius=3.0):
    """检查出生点碰撞"""
    for vehicle in world.get_actors().filter("vehicle.*"):
        if is_actor_alive(vehicle) and vehicle.get_transform().location.distance(spawn_point.location) < radius:
            return False
    for walker in world.get_actors().filter("walker.*"):
        if is_actor_alive(walker) and walker.get_transform().location.distance(spawn_point.location) < radius:
            return False
    return True

def print_performance_stats():
    """打印性能统计信息"""
    global perf_stats, last_perf_time
    current_time = time.time()
    if current_time - last_perf_time < PERF_MONITOR_INTERVAL:
        return
    
    # 计算FPS
    elapsed = current_time - last_perf_time
    fps = perf_stats["frame_count"] / elapsed if elapsed > 0 else 0
    perf_stats["avg_fps"] = (perf_stats["avg_fps"] * 0.9) + (fps * 0.1)  # 指数平滑
    
    # 车辆状态统计
    alive_vehicles = sum(1 for agent in vehicle_agents if agent.is_alive)
    collision_count = sum(1 for v_id in COLLISION_FLAG if COLLISION_FLAG[v_id])
    obstacle_count = sum(1 for v_id in OBSTACLE_FLAG if OBSTACLE_FLAG[v_id])
    
    # 控制台输出
    print(f"\n=== 性能监控 [{time.strftime('%H:%M:%S')}] ===")
    print(f"平均FPS: {perf_stats['avg_fps']:.1f} | 活跃车辆: {alive_vehicles}/{VEHICLE_COUNT}")
    print(f"碰撞车辆数: {collision_count} | 障碍物预警数: {obstacle_count}")
    print("="*50)
    
    # 重置统计
    perf_stats["frame_count"] = 0
    last_perf_time = current_time

# ===================== 传感器管理类（V3.0终极版）=====================
class VehicleSensors:
    def __init__(self, world, vehicle, vehicle_id):
        self.world = world
        self.vehicle = vehicle
        self.vehicle_id = vehicle_id
        self.logger = logging.LoggerAdapter(logging.getLogger(__name__), {"vehicle_id": vehicle_id})
        
        # 状态标记
        self.is_destroyed = False
        self.init_success = False
        
        # 传感器实例
        self.camera = None
        self.lidar = None
        self.collision_sensor = None
        
        # 数据存储（V3.0优化）
        self.image_surface = None
        self.obstacle_dist_cache = deque(maxlen=OBSTACLE_CACHE_SIZE)  # 环形缓存
        self.last_obstacle_dist = float('inf')
        
        # 初始化传感器
        try:
            self._create_camera()
            self._create_lidar()
            self._create_collision_sensor()
            self.init_success = True
            self.logger.info("所有传感器初始化成功")
        except Exception as e:
            self.logger.error(f"传感器初始化失败：{str(e)[:100]}")
            self.destroy()

    def _create_camera(self):
        """创建RGB相机"""
        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "360")
        camera_bp.set_attribute("fov", str(CAMERA_FOV))
        camera_bp.set_attribute("sensor_tick", "0.033")  # 30Hz
        
        self.camera = self.world.spawn_actor(camera_bp, CAMERA_POS, attach_to=self.vehicle)
        self.camera.listen(self._on_image)

    def _create_lidar(self):
        """创建LiDAR传感器（V3.0性能优化）"""
        lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        # 性能优化参数
        lidar_bp.set_attribute("range", str(LIDAR_RANGE))
        lidar_bp.set_attribute("points_per_second", str(LIDAR_POINTS_PER_SECOND))
        lidar_bp.set_attribute("rotation_frequency", str(LIDAR_ROTATION_FREQ))
        lidar_bp.set_attribute("channels", "16")  # 16线代替32线，降低计算量
        lidar_bp.set_attribute("upper_fov", "10")
        lidar_bp.set_attribute("lower_fov", "-20")
        lidar_bp.set_attribute("sensor_tick", str(1.0/LIDAR_ROTATION_FREQ))
        
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.lidar.listen(self._on_lidar_data)

    def _create_collision_sensor(self):
        """创建碰撞传感器"""
        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self._on_collision)

    def _on_image(self, image):
        """相机图像回调（非阻塞）"""
        if self.is_destroyed:
            return
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            array = array[:, :, :3][:, :, ::-1]  # BGR转RGB
            array = np.swapaxes(array, 0, 1)
            self.image_surface = pygame.surfarray.make_surface(array)
        except Exception as e:
            self.logger.error(f"图像处理失败：{str(e)[:50]}")

    def _on_lidar_data(self, data):
        """LiDAR点云回调（V3.0优化）"""
        if self.is_destroyed:
            return
        try:
            # 点云降采样（每N个点取1个）
            points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)[::2]  # 降采样50%
            
            # 精准过滤障碍物
            front_obstacle_points = points[
                (points[:, 0] > 0) &                # 前方
                (np.abs(points[:, 1]) < OBSTACLE_DETECTION_WIDTH) &  # 左右范围
                (points[:, 2] > OBSTACLE_MIN_HEIGHT) &  # 最小高度
                (points[:, 2] < OBSTACLE_MAX_HEIGHT)     # 最大高度
            ]

            # 更新障碍物距离
            if len(front_obstacle_points) > 0:
                self.last_obstacle_dist = np.min(front_obstacle_points[:, 0])
                self.obstacle_dist_cache.append(self.last_obstacle_dist)
            else:
                self.last_obstacle_dist = float('inf')
                if self.obstacle_dist_cache:
                    self.obstacle_dist_cache.popleft()

        except Exception as e:
            self.logger.error(f"LiDAR处理失败：{str(e)[:50]}")

    def _on_collision(self, event):
        """碰撞回调"""
        if self.is_destroyed:
            return
        try:
            collision_actor = event.other_actor
            collision_type = collision_actor.type_id if collision_actor else "未知"
            collision_loc = event.transform.location
            self.logger.error(
                f"碰撞发生！对象：{collision_type} | 位置：({collision_loc.x:.1f}, {collision_loc.y:.1f})"
            )
            global COLLISION_FLAG
            COLLISION_FLAG[self.vehicle_id] = True
        except Exception as e:
            self.logger.error(f"碰撞检测失败：{str(e)[:50]}")

    def get_smooth_obstacle_distance(self):
        """获取平滑后的障碍物距离"""
        if not self.obstacle_dist_cache:
            return float('inf')
        return np.mean(self.obstacle_dist_cache)

    def destroy(self):
        """安全销毁传感器（终极版）"""
        if self.is_destroyed:
            return
        
        self.is_destroyed = True
        self.logger.info("开始销毁传感器")
        
        # 停止监听
        safe_sensor_stop(self.camera, "RGB相机", self.logger)
        safe_sensor_stop(self.lidar, "LiDAR", self.logger)
        safe_sensor_stop(self.collision_sensor, "碰撞传感器", self.logger)
        
        # 销毁传感器
        safe_actor_destroy(self.camera, "RGB相机", self.logger)
        safe_actor_destroy(self.lidar, "LiDAR", self.logger)
        safe_actor_destroy(self.collision_sensor, "碰撞传感器", self.logger)
        
        # 清空引用
        self.camera = None
        self.lidar = None
        self.collision_sensor = None
        self.image_surface = None
        
        self.logger.info("传感器销毁完成")

# ===================== 车辆控制类（V3.0终极版）=====================
class VehicleAgent:
    def __init__(self, world, map, vehicle_id, spawn_point, vehicle_model, base_speed):
        self.vehicle_id = vehicle_id
        self.world = world
        self.map = map
        self.base_speed = base_speed
        self.logger = logging.LoggerAdapter(logging.getLogger(__name__), {"vehicle_id": vehicle_id})
        
        # 状态管理
        self.is_alive = True
        self.fault_count = 0  # 故障计数
        self.wp_cache = {}     # 路点缓存
        self.last_update_success = True
        
        # ACC跟车属性
        self.last_lead_speed = 0.0
        self.last_lead_acc = 0.0
        
        # 全局状态初始化
        global COLLISION_FLAG, OBSTACLE_FLAG
        COLLISION_FLAG[self.vehicle_id] = False
        OBSTACLE_FLAG[self.vehicle_id] = False

        # 生成车辆
        self.vehicle = None
        self.sensors = None
        try:
            self._spawn_vehicle(spawn_point, vehicle_model)
            self._init_sensors()
            self._init_controllers()
            self.logger.info(f"车辆生成成功 | 车型：{vehicle_model} | 出生点：({spawn_point.location.x:.1f},{spawn_point.location.y:.1f})")
        except Exception as e:
            self.logger.error(f"车辆初始化失败：{str(e)[:100]}")
            self.is_alive = False

    def _spawn_vehicle(self, spawn_point, vehicle_model):
        """生成车辆（带重试）"""
        vehicle_bp = self.world.get_blueprint_library().find(vehicle_model)
        if vehicle_bp.has_attribute("color"):
            color = random.choice(vehicle_bp.get_attribute("color").recommended_values)
            vehicle_bp.set_attribute("color", color)

        candidate_points = [spawn_point] + random.sample(self.map.get_spawn_points(), min(5, len(self.map.get_spawn_points())))
        
        for retry in range(SPAWN_RETRY_MAX):
            current_sp = candidate_points[retry % len(candidate_points)]
            current_sp.location.z += 0.3
            current_sp.rotation.yaw += random.randint(-5, 5)
            
            if not check_spawn_collision(self.world, current_sp):
                self.logger.warning(f"出生点碰撞风险，重试{retry+1}/{SPAWN_RETRY_MAX}")
                time.sleep(SPAWN_RETRY_DELAY)
                continue
            
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, current_sp)
                self.logger.debug(f"车辆生成重试{retry+1}成功")
                return
            except Exception as e:
                self.logger.warning(f"车辆生成重试{retry+1}失败：{str(e)[:50]}")
                time.sleep(SPAWN_RETRY_DELAY)
        
        raise RuntimeError(f"超过{SPAWN_RETRY_MAX}次重试，车辆生成失败")

    def _init_sensors(self):
        """初始化传感器"""
        self.sensors = VehicleSensors(self.world, self.vehicle, self.vehicle_id)
        if not self.sensors.init_success:
            raise RuntimeError("传感器初始化失败")

    def _init_controllers(self):
        """初始化控制器"""
        self.pp_controller = AdaptivePurePursuit(VEHICLE_WHEELBASE)
        self.speed_controller = SpeedController(PID_KP, PID_KI, PID_KD, self.base_speed)
        self.traffic_light_manager = TrafficLightManager(self.vehicle_id)

    def _restart_vehicle(self):
        """重启故障车辆"""
        self.logger.warning(f"车辆故障次数达到阈值，尝试重启")
        
        # 销毁旧车辆
        self.destroy()
        
        # 重新生成
        try:
            spawn_points = get_valid_spawn_points(self.map, 1, self.vehicle.get_transform().location if self.vehicle else None)
            self._spawn_vehicle(spawn_points[0], VEHICLE_MODELS[self.vehicle_id % len(VEHICLE_MODELS)])
            self._init_sensors()
            self._init_controllers()
            
            # 重置状态
            self.is_alive = True
            self.fault_count = 0
            self.last_update_success = True
            COLLISION_FLAG[self.vehicle_id] = False
            OBSTACLE_FLAG[self.vehicle_id] = False
            
            self.logger.info("车辆重启成功")
        except Exception as e:
            self.logger.error(f"车辆重启失败：{str(e)[:100]}")
            self.is_alive = False

    def update(self):
        """更新车辆状态（V3.0增强）"""
        if not self.is_alive or not is_actor_alive(self.vehicle):
            self.is_alive = False
            self.logger.error("车辆已销毁，停止更新")
            return False

        try:
            # 获取车辆基础状态
            vehicle_transform = self.vehicle.get_transform()
            vehicle_vel = self.vehicle.get_velocity()
            current_speed = math.hypot(vehicle_vel.x, vehicle_vel.y) * 3.6

            # 路径跟踪
            current_wp = get_forward_waypoint(self.vehicle, self.map, self.wp_cache)
            dir_change, curve_level = calculate_dir_change(current_wp)
            lookahead_dist = self.pp_controller.get_adaptive_lookahead(dir_change)
            
            target_wps = current_wp.next(lookahead_dist)
            target_point = target_wps[0].transform.location if target_wps else vehicle_transform.location

            # 基础速度计算（弯道减速）
            curve_speed_factors = [1.0, 0.7, 0.4]
            speed_factor = curve_speed_factors[min(curve_level, 2)]
            base_target_speed = max(8.0, self.base_speed * speed_factor)

            # ========== ACC跟车逻辑（V3.0优化）==========
            if self.vehicle_id > 1 and len(vehicle_agents) >= self.vehicle_id:
                try:
                    lead_agent = vehicle_agents[self.vehicle_id - 2]
                    if lead_agent.is_alive and is_actor_alive(lead_agent.vehicle):
                        lead_vehicle = lead_agent.vehicle
                        lead_transform = lead_vehicle.get_transform()
                        lead_vel = lead_vehicle.get_velocity()
                        lead_speed = math.hypot(lead_vel.x, lead_vel.y) * 3.6
                        
                        # 加速度平滑
                        lead_acc = (lead_speed - lead_agent.last_lead_speed) / (1.0/30)
                        self.last_lead_acc = 0.8 * self.last_lead_acc + 0.2 * lead_acc
                        lead_agent.last_lead_speed = lead_speed
                        
                        # 安全距离计算
                        safe_dist = (current_speed / 3.6) * SAFE_TIME_GAP + MIN_SAFE_DISTANCE
                        dist_to_lead = vehicle_transform.location.distance(lead_transform.location)
                        
                        # 动态速度调整
                        if dist_to_lead < safe_dist - 2:
                            base_target_speed = max(5.0, lead_speed - 2)
                        elif dist_to_lead > safe_dist + 2:
                            base_target_speed = min(self.base_speed * speed_factor, lead_speed + 2)
                        else:
                            base_target_speed = lead_speed
                        
                        # 前车急刹检测
                        if self.last_lead_acc < LEAD_BRAKE_THRESHOLD:
                            base_target_speed = max(0.0, current_speed - EMERGENCY_DECEL_RATE)
                            self.logger.warning(f"前车急刹！加速度{self.last_lead_acc:.1f}km/h/s，紧急减速")
                except Exception as e:
                    self.logger.warning(f"ACC跟车异常：{str(e)[:50]}")

            # ========== 障碍物避障逻辑（V3.0优化）==========
            obstacle_dist = self.sensors.get_smooth_obstacle_distance()
            OBSTACLE_FLAG[self.vehicle_id] = obstacle_dist < OBSTACLE_WARNING_DIST
            
            if obstacle_dist < OBSTACLE_EMERGENCY_DIST:
                base_target_speed = max(0.0, current_speed - OBSTACLE_DECEL_RATE)
                self.logger.warning(f"前方{obstacle_dist:.1f}米障碍物！紧急制动")
            elif obstacle_dist < OBSTACLE_WARNING_DIST:
                base_target_speed = max(8.0, base_target_speed * 0.5)
                self.logger.warning(f"前方{obstacle_dist:.1f}米障碍物！预警减速")

            # ========== 交通灯处理 ==========
            target_speed, traffic_light_status = self.traffic_light_manager.handle_traffic_light_logic(
                self.vehicle, current_speed, base_target_speed
            )

            # ========== 控制指令计算 ==========
            steer = self.pp_controller.calculate_steer(vehicle_transform, target_point, dir_change)
            throttle = self.speed_controller.calculate(target_speed, current_speed)
            brake = 1.0 - throttle if current_speed > target_speed + 1 else 0.0

            # 状态优先级：碰撞 > 红灯 > 障碍物 > 正常行驶
            if COLLISION_FLAG.get(self.vehicle_id, False):
                throttle = 0.0
                brake = 1.0
                self.logger.error("碰撞触发紧急停车")
            elif "Red (Stopped)" in traffic_light_status or target_speed <= STOP_SPEED_THRESHOLD:
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

            # 日志输出
            obstacle_status = f"障碍物{obstacle_dist:.1f}m" if obstacle_dist < OBSTACLE_WARNING_DIST else "无障碍物"
            self.logger.info(
                f"速度：{current_speed:5.1f}km/h | 目标：{target_speed:5.1f} | "
                f"弯道：{['直道', '缓弯', '急弯'][curve_level]:<3} | 灯状态：{traffic_light_status} | "
                f"ACC：{'激活' if self.vehicle_id>1 else '未激活'} | {obstacle_status}"
            )

            self.last_update_success = True
            self.fault_count = 0  # 重置故障计数
            return True

        except Exception as e:
            self.logger.error(f"更新失败：{str(e)[:100]}", exc_info=False)
            self.last_update_success = False
            self.fault_count += 1
            
            # 故障重启逻辑
            if self.fault_count >= VEHICLE_RESTART_THRESHOLD:
                self._restart_vehicle()
            
            return False

    def destroy(self):
        """安全销毁车辆（V3.0终极版）"""
        self.logger.info("开始销毁车辆资源")
        
        # 销毁传感器
        if self.sensors:
            self.sensors.destroy()
        
        # 销毁车辆
        safe_actor_destroy(self.vehicle, f"车辆{self.vehicle_id}", self.logger)
        
        # 清空状态
        self.vehicle = None
        self.sensors = None
        self.is_alive = False
        self.wp_cache.clear()
        
        self.logger.info("车辆资源销毁完成")

# ===================== 控制器类（V3.0优化）=====================
class AdaptivePurePursuit:
    """自适应纯追踪控制器"""
    def __init__(self, wheelbase):
        self.wheelbase = wheelbase
        self.last_steer = 0.0
        self.last_lookahead = LOOKAHEAD_DIST_STRAIGHT

    def calculate_steer(self, vehicle_transform, target_point, dir_change):
        """计算转向角（优化滤波）"""
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

        # 转向增益自适应
        steer_gain = np.interp(
            dir_change,
            [0, DIR_CHANGE_SHARP],
            [STEER_GAIN_STRAIGHT, STEER_GAIN_CURVE]
        )
        steer_gain = np.clip(steer_gain, STEER_GAIN_STRAIGHT, STEER_GAIN_CURVE)

        # 计算转向角
        if dx_vehicle < 0.1:
            steer = self.last_steer
        else:
            steer_rad = math.atan2(2 * self.wheelbase * dy_vehicle, dx_vehicle ** 2 + dy_vehicle ** 2)
            steer = steer_rad / math.pi
            steer *= steer_gain

        # 死区和低通滤波
        if abs(steer) < STEER_DEADZONE:
            steer = 0.0
        steer = STEER_LOWPASS_ALPHA * steer + (1 - STEER_LOWPASS_ALPHA) * self.last_steer
        steer = np.clip(steer, -MAX_STEER, MAX_STEER)

        self.last_steer = steer
        return steer

    def get_adaptive_lookahead(self, dir_change):
        """自适应前瞻距离"""
        lookahead_dist = np.interp(
            dir_change,
            [0, DIR_CHANGE_SHARP],
            [LOOKAHEAD_DIST_STRAIGHT, LOOKAHEAD_DIST_CURVE]
        )
        lookahead_dist = np.clip(lookahead_dist, LOOKAHEAD_DIST_CURVE, LOOKAHEAD_DIST_STRAIGHT)
        self.last_lookahead = lookahead_dist
        return lookahead_dist

class SpeedController:
    """PID速度控制器（V3.0优化）"""
    def __init__(self, kp, ki, kd, base_speed):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.base_speed = base_speed
        self.last_error = 0.0
        self.integral = 0.0
        self.integral_limit = 0.5  # 积分限幅

    def calculate(self, target_speed, current_speed):
        """计算油门"""
        error = target_speed - current_speed
        
        # PID计算
        p = self.kp * error
        self.integral += self.ki * error
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)  # 积分限幅
        d = self.kd * (error - self.last_error)
        
        # 输出限幅
        output = np.clip(p + self.integral + d, 0.0, 1.0)
        
        self.last_error = error
        return output

class TrafficLightManager:
    """交通灯管理器（V3.0缓存优化）"""
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.tracked_light = None
        self.is_stopped_at_red = False
        self.red_light_stop_time = 0
        self.tl_state_cache = {}  # 交通灯状态缓存
        self.tl_cache_time = {}   # 缓存时间
        self.logger = logging.LoggerAdapter(logging.getLogger(__name__), {"vehicle_id": vehicle_id})

    def _calculate_angle_between_vehicle_and_light(self, vehicle_transform, light_transform):
        """计算车辆与交通灯的夹角"""
        vehicle_forward = vehicle_transform.get_forward_vector()
        vehicle_forward = np.array([vehicle_forward.x, vehicle_forward.y])
        vehicle_forward = vehicle_forward / np.linalg.norm(vehicle_forward)

        light_dir = light_transform.location - vehicle_transform.location
        light_dir = np.array([light_dir.x, light_dir.y])
        if np.linalg.norm(light_dir) < 0.1:
            return 0.0
        light_dir = light_dir / np.linalg.norm(light_dir)

        angle = math.acos(np.clip(np.dot(vehicle_forward, light_dir), -1.0, 1.0))
        return math.degrees(angle)

    def get_lane_traffic_light(self, vehicle, world):
        """获取车道对应的交通灯（带缓存）"""
        vehicle_transform = vehicle.get_transform()
        vehicle_loc = vehicle_transform.location
        current_time = time.time()

        # 缓存检查
        if self.tracked_light and is_actor_alive(self.tracked_light):
            tl_id = self.tracked_light.id
            if tl_id in self.tl_state_cache and current_time - self.tl_cache_time.get(tl_id, 0) < 1.0:
                dist = self.tracked_light.get_transform().location.distance(vehicle_loc)
                angle = self._calculate_angle_between_vehicle_and_light(vehicle_transform, self.tracked_light.get_transform())
                if dist < TRAFFIC_LIGHT_DETECTION_RANGE and angle < TRAFFIC_LIGHT_ANGLE_THRESHOLD:
                    return self.tracked_light

        # 重新查找交通灯
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

        # 更新追踪的交通灯
        if valid_lights:
            valid_lights.sort(key=lambda x: x[0])
            self.tracked_light = valid_lights[0][1]
            self.tl_state_cache[self.tracked_light.id] = self.tracked_light.get_state()
            self.tl_cache_time[self.tracked_light.id] = current_time
        else:
            self.tracked_light = None

        return self.tracked_light

    def handle_traffic_light_logic(self, vehicle, current_speed, base_target_speed):
        """处理交通灯逻辑"""
        world = vehicle.get_world()
        traffic_light = self.get_lane_traffic_light(vehicle, world)

        if not traffic_light:
            self.is_stopped_at_red = False
            self.red_light_stop_time = 0
            return base_target_speed, "No Light"

        # 获取停止线和距离
        stop_line_loc = get_traffic_light_stop_line(traffic_light)
        dist_to_stop_line = vehicle.get_transform().location.distance(stop_line_loc)

        # 交通灯状态处理
        tl_state = traffic_light.get_state()
        if tl_state == carla.TrafficLightState.Green:
            if self.is_stopped_at_red:
                recovery_speed = current_speed + (base_target_speed - current_speed) * GREEN_LIGHT_ACCEL_FACTOR
                target_speed = max(STOP_SPEED_THRESHOLD, recovery_speed)
                if abs(target_speed - base_target_speed) < 0.5:
                    self.is_stopped_at_red = False
                    self.logger.info(f"绿灯恢复行驶，目标速度：{target_speed:.1f}km/h")
                return target_speed, "Green"
            return base_target_speed, "Green"

        elif tl_state == carla.TrafficLightState.Yellow:
            self.is_stopped_at_red = False
            yellow_speed = max(5.0, base_target_speed * 0.3)
            self.logger.warning(f"黄灯减速，目标速度：{yellow_speed:.1f}km/h")
            return yellow_speed, "Yellow"

        elif tl_state == carla.TrafficLightState.Red:
            if dist_to_stop_line > TRAFFIC_LIGHT_STOP_DISTANCE:
                self.is_stopped_at_red = False
                red_speed = max(2.0, current_speed * 0.1)
                self.logger.warning(f"红灯减速，距离停止线：{dist_to_stop_line:.1f}m")
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
    """交通灯状态循环"""
    logger = logging.LoggerAdapter(logging.getLogger(__name__), {"vehicle_id": "系统"})
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

# ===================== 主函数（V3.0终极版）=====================
def main():
    global current_view_vehicle_id, vehicle_agents, perf_stats
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(
        f"CARLA多车辆控制 V3.0 | 车辆数：{VEHICLE_COUNT} | "
        f"按1/2/3切换视角 | S分屏 | V俯视 | ESC退出"
    )

    # 初始化核心变量
    client = None
    world = None
    map = None
    tl_cycle_thread = None
    tl_stop_event = threading.Event()
    show_split_screen = True
    show_top_view = False
    top_view_camera = None
    top_view_surface = None

    # 安全清理函数
    def cleanup():
        print("\n=== 开始安全清理资源 ===")
        tl_stop_event.set()
        
        # 等待交通灯线程结束
        if tl_cycle_thread and tl_cycle_thread.is_alive():
            tl_cycle_thread.join(timeout=2)

        # 销毁俯视相机
        if top_view_camera:
            safe_sensor_stop(top_view_camera, "俯视相机", logging.getLogger(__name__))
            safe_actor_destroy(top_view_camera, "俯视相机", logging.getLogger(__name__))

        # 销毁所有车辆
        global vehicle_agents
        for agent in vehicle_agents:
            try:
                agent.destroy()
            except Exception as e:
                print(f"销毁车辆{agent.vehicle_id}忽略异常：{str(e)[:50]}")

        # 清理残留actor
        if world:
            for actor in world.get_actors():
                if actor.type_id.startswith(("vehicle.", "walker.", "sensor.")):
                    safe_actor_destroy(actor, actor.type_id, logging.getLogger(__name__))

        # 退出pygame
        pygame.quit()
        print("=== 资源清理完成 ===")

    # 注册退出处理
    import atexit
    import signal
    atexit.register(cleanup)
    
    def signal_handler(sig, frame):
        print("\n接收到退出信号，开始清理...")
        cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # 连接CARLA服务器
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(CARLA_TIMEOUT)
        try:
            world = client.load_world("Town04")
            print("成功加载Town04地图")
        except Exception as e:
            world = client.get_world()
            print(f"警告：Town04加载失败（{e}），使用当前地图")
        map = world.get_map()

        # 清理残留演员
        print("清理残留演员...")
        for actor in world.get_actors():
            if actor.type_id.startswith(("vehicle.", "walker.", "sensor.")):
                safe_actor_destroy(actor, actor.type_id, logging.getLogger(__name__))
        time.sleep(2.0)

        # 获取出生点
        base_location = map.get_spawn_points()[0].location if map.get_spawn_points() else carla.Location(x=220.0, y=150.0)
        print(f"基准出生点：({base_location.x:.1f}, {base_location.y:.1f})")
        
        valid_spawn_points = get_valid_spawn_points(map, VEHICLE_COUNT, base_location)
        for i, sp in enumerate(valid_spawn_points):
            print(f"出生点{i+1}：({sp.location.x:.1f}, {sp.location.y:.1f})")

        # 生成车辆
        print(f"\n分步生成{VEHICLE_COUNT}辆车辆...")
        for i in range(VEHICLE_COUNT):
            vehicle_model = VEHICLE_MODELS[i % len(VEHICLE_MODELS)]
            base_speed = BASE_SPEEDS[i % len(BASE_SPEEDS)]
            spawn_point = valid_spawn_points[i]

            try:
                agent = VehicleAgent(world, map, i+1, spawn_point, vehicle_model, base_speed)
                if agent.is_alive:
                    vehicle_agents.append(agent)
                    print(f"车辆{i+1}生成成功")
                else:
                    print(f"车辆{i+1}生成失败")
            except Exception as e:
                print(f"车辆{i+1}生成异常：{str(e)[:100]}")

            time.sleep(SPAWN_INTERVAL)

        if len(vehicle_agents) == 0:
            raise RuntimeError("无车辆生成成功，仿真终止")

        # 创建俯视相机
        try:
            top_view_bp = world.get_blueprint_library().find("sensor.camera.rgb")
            top_view_bp.set_attribute("image_size_x", str(WINDOW_WIDTH))
            top_view_bp.set_attribute("image_size_y", str(WINDOW_HEIGHT))
            top_view_bp.set_attribute("fov", "90")
            
            top_view_transform = carla.Transform(
                vehicle_agents[0].vehicle.get_transform().location + carla.Location(z=50),
                carla.Rotation(pitch=-90)
            )
            top_view_camera = world.spawn_actor(top_view_bp, top_view_transform)
            
            def top_view_callback(image):
                nonlocal top_view_surface
                try:
                    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
                    array = array[:, :, :3][:, :, ::-1]
                    array = np.swapaxes(array, 0, 1)
                    top_view_surface = pygame.surfarray.make_surface(array)
                except:
                    pass
            
            top_view_camera.listen(top_view_callback)
            print("俯视相机创建成功")
        except Exception as e:
            print(f"俯视相机创建失败：{e}")
            top_view_camera = None

        # 启动交通灯线程
        tl_cycle_thread = threading.Thread(target=cycle_traffic_light_states, args=(world, tl_stop_event), daemon=True)
        tl_cycle_thread.start()
        print("交通灯控制线程启动")

        # 主循环
        clock = pygame.time.Clock()
        running = True
        executor = ThreadPoolExecutor(max_workers=VEHICLE_COUNT)  # 复用线程池，提升性能

        print("\n=== 仿真开始 ===")
        print("操作说明：")
        print("  1/2/3 - 切换单车辆视角")
        print("  S     - 切换分屏视角")
        print("  V     - 切换俯视视角")
        print("  ESC   - 退出仿真")
        print("="*50)

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

            # 视角渲染
            if show_top_view and top_view_surface:
                screen.blit(top_view_surface, (0, 0))
            elif show_split_screen:
                # 分屏渲染
                if len(vehicle_agents) >= 1 and vehicle_agents[0].sensors and vehicle_agents[0].sensors.image_surface:
                    surf1 = pygame.transform.scale(vehicle_agents[0].sensors.image_surface, (WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
                    screen.blit(surf1, (0, 0))
                
                if len(vehicle_agents) >= 2 and vehicle_agents[1].sensors and vehicle_agents[1].sensors.image_surface:
                    surf2 = pygame.transform.scale(vehicle_agents[1].sensors.image_surface, (WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
                    screen.blit(surf2, (WINDOW_WIDTH//2, 0))
                
                if len(vehicle_agents) >= 3 and vehicle_agents[2].sensors and vehicle_agents[2].sensors.image_surface:
                    surf3 = pygame.transform.scale(vehicle_agents[2].sensors.image_surface, (WINDOW_WIDTH, WINDOW_HEIGHT//2))
                    screen.blit(surf3, (0, WINDOW_HEIGHT//2))
            else:
                # 单车辆视角
                target_agent = next((a for a in vehicle_agents if a.vehicle_id == current_view_vehicle_id), None)
                if target_agent and target_agent.sensors and target_agent.sensors.image_surface:
                    surf = pygame.transform.scale(target_agent.sensors.image_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
                    screen.blit(surf, (0, 0))

            # 更新车辆状态（复用线程池）
            futures = []
            for agent in vehicle_agents:
                if agent.is_alive:
                    futures.append(executor.submit(agent.update))
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"车辆更新异常：{str(e)[:50]}")

            # 性能监控
            perf_stats["frame_count"] += 1
            print_performance_stats()

            # 刷新屏幕
            pygame.display.flip()
            clock.tick(30)

        # 关闭线程池
        executor.shutdown(wait=True)

    except Exception as e:
        print(f"\n仿真异常：{e}")
        traceback.print_exc()
    finally:
        cleanup()

if __name__ == "__main__":
    main()