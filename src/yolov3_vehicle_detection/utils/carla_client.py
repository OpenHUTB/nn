import carla
import random
import time
import sys
import os
import numpy as np
import cv2
import queue
from collections import deque

# 路径修复：确保能正确导入 config 模块
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import config
from utils.deep_sort import DeepSORTTracker
from utils.planner import PurePursuitPlanner


class CarlaClient:
    """
    CARLA 模拟器客户端封装类
    """

    def __init__(self, host=None, port=None):
        self.host = host if host else config.CARLA_HOST
        self.port = port if port else config.CARLA_PORT
        self.timeout = config.CARLA_TIMEOUT

        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.blueprint_library = None
        self.image_queue = queue.Queue()
        self.debug_helper = None
        self.spectator = None
        self.obstacle_sensor = None
        self.obstacle_distance = float('inf')
        self.obstacle_info = None
        self.collision_sensor = None
        
        # 新增：碰撞检测
        self.collision_detected = False
        
        # 变道避让状态机
        self.avoidance_state = 'normal'  # normal, changing_lane, returning_lane
        self.lane_change_direction = None  # 'left' or 'right'
        self.lane_change_start_time = 0
        self.lane_change_duration = 1.5  # 变道持续时间（秒）- 缩短让反应更快
        self.lane_change_start_yaw = 0
        self.lane_change_target_lateral = 0  # 目标横向偏移
        self.lane_change_completed = False  # 变道刚完成标志
        self.lane_change_recovery_time = 0  # 恢复自动驾驶的时间
        
        # 新增：连续变道防护 - 关键修复
        self.last_lane_change_time = 0  # 上次变道完成时间
        self.lane_change_cooldown = 3.0  # 变道冷却时间（秒）- 缩短以便更灵活处理新障碍物
        self.last_obstacle_id = None  # 上次处理的障碍物ID，防止重复处理同一障碍物
        
        # 新增：持续跟踪障碍物（不清除直到安全）
        self.current_obstacle = None  # 当前障碍物引用
        self.lane_change_start_lateral = 0  # 变道开始时的横向位置
        self.collision_warning = False  # 碰撞警告标志
        
        # 新增：碰撞后恢复
        self.post_collision_recovery = False
        self.collision_recovery_start = 0
        
        # 新增：Pure Pursuit轨迹规划器
        self.pure_pursuit = PurePursuitPlanner(
            wheelbase=2.9,           # 车辆轴距
            lookahead_distance=5.0,  # 基础前视距离
            lookahead_gain=0.5,      # 前视距离系数
            min_lookahead=3.0,      # 最小前视距离
            max_lookahead=15.0,     # 最大前视距离
            steering_limit=0.8       # 最大转向角 (rad)
        )
        self.use_pure_pursuit = False  # 是否启用Pure Pursuit控制
        
        # DeepSORT 目标跟踪
        self.deep_sort = DeepSORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        self.tracked_obstacles = {}  # 跟踪的障碍物 {track_id: info}
        self.frame_count = 0
        
        # ========== TTC (Time To Collision) 安全距离参数 ==========
        self.last_obstacle_distance = float('inf')  # 上次障碍物距离
        self.last_obstacle_velocity = 0  # 障碍物相对速度
        
        # TTC 分级阈值
        self.TTC_WARNING = 4.0     # 警告阈值 (s) - 正常行驶
        self.TTC_CAUTION = 3.0     # 注意阈值 (s) - 准备减速
        self.TTC_ALERT = 2.0       # 警报阈值 (s) - 减速避让
        self.TTC_EMERGENCY = 1.5   # 紧急阈值 (s) - 紧急制动
        
        # 车辆动力学参数
        self.MAX_DECELERATION = 8.0  # 最大减速度 (m/s²) - 普通制动
        self.EMERGENCY_DECEL = 10.0  # 紧急制动减速度 (m/s²)
        self.REACTION_TIME = 0.5      # 系统反应时间 (s)
        self.MIN_SAFE_GAP = 3.0       # 最小安全间距 (m)
        
        # 障碍物速度历史 (用于计算相对加速度)
        self.obstacle_velocity_history = deque(maxlen=5)

    def connect(self):
        print(f"[INFO] 正在连接 CARLA 服务器 ({self.host}:{self.port})...")
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            # 创建 Debug Helper 用于绘制
            self.debug_helper = self.world.debug
            # 获取 spectator 用于第三人称跟随
            self.spectator = self.world.get_spectator()
            print("[INFO] CARLA 连接成功！")
            return True
        except Exception as e:
            print(f"[ERROR] 连接失败: {e}")
            return False

    def compute_ttc(self, distance, ego_speed, obstacle_speed):
        """
        计算 TTC (Time To Collision)
        
        Args:
            distance: 与障碍物的距离 (m)
            ego_speed: 自车速度 (m/s)
            obstacle_speed: 障碍物速度 (m/s, 沿自车方向)
            
        Returns:
            TTC 值 (s)，正值表示正在接近，负值或无穷大表示远离/静止
        """
        if distance <= 0:
            return 0.0
        
        if ego_speed < 0.5:  # 自车几乎静止
            return float('inf')
        
        # 【修复】静态障碍物(obstacle_speed=0)的TTC计算
        # 静态障碍物TTC = distance / ego_speed（车撞上静止物体的时间）
        # 动态障碍物TTC = distance / (ego_speed - obstacle_speed)
        relative_speed = ego_speed - obstacle_speed  # 改为 ego - obstacle
        
        if relative_speed > 0.1:  # 自车比障碍物快，才会接近
            ttc = distance / relative_speed
            return max(0.1, min(ttc, 10.0))  # 限制在合理范围
        else:
            return float('inf')  # 远离或静止
    
    def compute_relative_velocity(self, current_distance, dt=0.05):
        """
        计算障碍物相对自车的速度（基于距离变化率）
        
        Args:
            current_distance: 当前距离
            dt: 时间步长
            
        Returns:
            相对速度 (m/s)
        """
        if self.last_obstacle_distance == float('inf'):
            self.last_obstacle_distance = current_distance
            return 0
        
        # 计算距离变化率
        distance_delta = self.last_obstacle_distance - current_distance
        relative_velocity = distance_delta / dt if dt > 0 else 0
        
        # 更新历史
        self.obstacle_velocity_history.append(relative_velocity)
        self.last_obstacle_distance = current_distance
        
        # 使用滑动平均平滑速度
        if len(self.obstacle_velocity_history) > 0:
            return np.mean(self.obstacle_velocity_history)
        return relative_velocity
    
    def compute_ttc_safety_distance(self, ttc, ego_speed):
        """
        基于 TTC 计算安全距离（考虑车辆动力学）
        
        安全距离公式:
        d_safe = v * (TTC_req + t_reaction) + v²/(2*a_max) + d_min
        
        Args:
            ttc: 当前TTC值
            ego_speed: 自车速度 (m/s)
            
        Returns:
            安全距离阈值 (m)
        """
        # 根据TTC分级调整安全系数
        if ttc >= self.TTC_WARNING:
            # 正常行驶，保持较大安全距离
            safety_factor = 1.5
            ttc_requirement = 3.0
        elif ttc >= self.TTC_CAUTION:
            # 准备减速
            safety_factor = 1.2
            ttc_requirement = 2.5
        elif ttc >= self.TTC_ALERT:
            # 警报状态
            safety_factor = 1.0
            ttc_requirement = 2.0
        else:
            # 紧急状态
            safety_factor = 0.8
            ttc_requirement = 1.5
        
        # 制动距离 = v²/(2*a_max)
        braking_distance = (ego_speed ** 2) / (2 * self.MAX_DECELERATION)
        
        # 行驶距离 = 速度 * (TTC需求 + 反应时间)
        travel_distance = ego_speed * (ttc_requirement + self.REACTION_TIME)
        
        # 总安全距离
        safety_distance = travel_distance + braking_distance + self.MIN_SAFE_GAP
        safety_distance *= safety_factor
        
        return safety_distance
    
    def get_risk_level(self, ttc):
        """
        根据 TTC 评估风险等级
        
        Args:
            ttc: TTC值
            
        Returns:
            风险等级: 'safe', 'warning', 'caution', 'alert', 'emergency'
        """
        if ttc == float('inf') or ttc > self.TTC_WARNING:
            return 'safe'
        elif ttc > self.TTC_CAUTION:
            return 'warning'
        elif ttc > self.TTC_ALERT:
            return 'caution'
        elif ttc > self.TTC_EMERGENCY:
            return 'alert'
        else:
            return 'emergency'

    def spawn_vehicle(self, spawn_npc=True, npc_count=15, spawn_obstacle=True, obstacle_count=3):
        if not self.world:
            print("[ERROR] 世界未加载，请先连接！")
            return None

        model_name = config.VEHICLE_MODEL
        bp = self.blueprint_library.find(model_name)

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        try:
            self.vehicle = self.world.spawn_actor(bp, spawn_point)
            print(f"[INFO] 主车辆生成成功: {self.vehicle.type_id}")
            
            # 获取交通管理器并启用自动驾驶
            traffic_manager = self.client.get_trafficmanager(8000)
            self.vehicle.set_autopilot(True, traffic_manager.get_port())
            
            # 生成NPC车辆
            if spawn_npc:
                self._spawn_npc_vehicles(npc_count)
            
            # 生成障碍物
            if spawn_obstacle:
                self.spawn_obstacles(obstacle_type='all', count=obstacle_count)
            
            # 安装障碍物传感器
            self.setup_obstacle_sensor()
            
            # 【Pure Pursuit】初始化轨迹路径
            self._init_pure_pursuit_path(spawn_point)
            
            return self.vehicle
        except Exception as e:
            print(f"[ERROR] 车辆生成失败: {e}")
            return None

    def _spawn_npc_vehicles(self, count=15):
        """生成NPC交通车辆"""
        try:
            # 获取交通管理器
            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_global_distance_to_leading_vehicle(1.0)
            traffic_manager.global_percentage_speed_difference(50.0)
            
            blueprints = self.blueprint_library.filter('vehicle.*')
            spawn_points = self.world.get_map().get_spawn_points()
            
            spawned = 0
            for i in range(count):
                spawn_point = random.choice(spawn_points)
                blueprint = random.choice(blueprints)
                
                # 使用 try_spawn_actor 避免碰撞位置
                actor = self.world.try_spawn_actor(blueprint, spawn_point)
                if actor:
                    actor.set_autopilot(True, traffic_manager.get_port())
                    spawned += 1
            
            print(f"[INFO] 已生成 {spawned} 辆NPC车辆")
            
        except Exception as e:
            print(f"[WARNING] 生成NPC车辆失败: {e}")

    def spawn_obstacles(self, obstacle_type='static', count=5):
        """
        生成道路障碍物
        
        Args:
            obstacle_type: 'static' 静态障碍物, 'walker' 行人, 'all' 全部
            count: 生成数量
        """
        try:
            if obstacle_type in ['static', 'all']:
                # 静态障碍物：锥桶、箱子等
                static_blueprints = [
                    'static.prop.streetbarrier',
                    'static.prop.constructioncone',
                    'static.prop.dog',
                    'static.prop.pushchair',
                    'static.prop.luggage',
                ]
                
                for _ in range(count):
                    blueprint = random.choice(self.blueprint_library.filter('static.prop.*'))
                    spawn_points = self.world.get_map().get_spawn_points()
                    spawn_point = random.choice(spawn_points)
                    
                    # 设置随机高度（避免埋入地面）
                    spawn_point.location.z += 0.5
                    
                    actor = self.world.try_spawn_actor(blueprint, spawn_point)
                    if actor:
                        # 绘制绿色标记
                        self.debug_helper.draw_point(
                            spawn_point.location,
                            size=0.5,
                            color=carla.Color(255, 165, 0),  # 橙色
                            life_time=10.0
                        )
                        print(f"[INFO] 生成静态障碍物: {blueprint.id}")
            
            if obstacle_type in ['walker', 'all']:
                # 行人
                walker_bp = self.blueprint_library.filter('walker.*')
                walker_controller_bp = self.blueprint_library.filter('controller.ai.walker')
                
                for _ in range(count):
                    spawn_point = random.choice(self.world.get_map().get_spawn_points())
                    spawn_point.location.z += 0.5
                    
                    walker = self.world.try_spawn_actor(random.choice(walker_bp), spawn_point)
                    if walker:
                        # 创建行人控制器
                        controller = self.world.spawn_actor(
                            random.choice(walker_controller_bp),
                            carla.Transform(),
                            walker
                        )
                        if controller:
                            # 让行人随机行走
                            controller.start()
                            controller.go_to_location(
                                carla.Location(
                                    x=spawn_point.location.x + random.uniform(-20, 20),
                                    y=spawn_point.location.y + random.uniform(-20, 20),
                                    z=spawn_point.location.z
                                )
                            )
                            controller.set_max_speed(1.4)
                        print(f"[INFO] 生成行人")
            
            print(f"[INFO] 障碍物生成完成")
            
        except Exception as e:
            print(f"[WARNING] 生成障碍物失败: {e}")

    def setup_obstacle_sensor(self):
        """安装障碍物传感器（用于物理碰撞检测）"""
        if not self.vehicle:
            print("[WARNING] 车辆未生成，无法安装障碍物传感器")
            return
        
        try:
            obstacle_bp = self.blueprint_library.find('sensor.other.obstacle')
            obstacle_bp.set_attribute('distance', '50')      # 检测距离增加到 50 米（从30米扩大）
            obstacle_bp.set_attribute('hit_radius', '1')      # 碰撞半径
            obstacle_bp.set_attribute('only_dynamics', 'False')  # 也检测静态障碍物
            obstacle_bp.set_attribute('debug_linetrace', 'False')
            
            spawn_point = carla.Transform(
                carla.Location(x=0.5, z=1.5),
                carla.Rotation(yaw=0)
            )
            
            self.obstacle_sensor = self.world.spawn_actor(
                obstacle_bp,
                spawn_point,
                attach_to=self.vehicle
            )
            self.obstacle_sensor.listen(lambda event: self._on_obstacle_detected(event))
            print("[INFO] 障碍物传感器安装成功！")
            
            # 安装碰撞传感器（检测实际碰撞）
            self.setup_collision_sensor()
            
        except Exception as e:
            print(f"[WARNING] 障碍物传感器安装失败: {e}")
    
    def _init_pure_pursuit_path(self, spawn_point):
        """
        【Pure Pursuit】根据车辆初始位置和朝向初始化轨迹路径
        
        Args:
            spawn_point: 车辆生成位置 (carla.Transform)
        """
        # 获取道路信息
        waypoint = self.world.get_map().get_waypoint(
            spawn_point.location, 
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        # 生成沿着道路的路径点
        path_points = []
        current_waypoint = waypoint
        
        # 生成前方100米的路径
        for i in range(100):
            # 获取前方10米处的下一个waypoint
            next_waypoints = current_waypoint.next(10.0)
            if not next_waypoints:
                break
            current_waypoint = next_waypoints[0]
            
            # 添加路径点 [x, y]
            path_points.append([
                current_waypoint.transform.location.x,
                current_waypoint.transform.location.y
            ])
        
        if len(path_points) > 2:
            path_array = np.array(path_points)
            self.pure_pursuit.set_path(path_array)
            print(f"[PURE PURSUIT] 轨迹路径初始化完成 | 路径点数:{len(path_points)} | 起点:({path_points[0][0]:.1f}, {path_points[0][1]:.1f})")
        else:
            # 如果无法获取道路路径，使用默认直线路径
            print("[PURE PURSUIT] 无法获取道路路径，使用默认直线路径")
            self.pure_pursuit.set_curve_path(length=100, curve_radius=1000)
    
    def update_pure_pursuit_path(self):
        """
        【Pure Pursuit】实时更新路径（如果需要动态重规划）
        """
        if not self.vehicle:
            return
        
        # 检查当前进度，如果接近终点就延伸路径
        current_pos = (
            self.vehicle.get_location().x,
            self.vehicle.get_location().y
        )
        progress = self.pure_pursuit.get_progress(current_pos)
        
        if progress > 0.9:  # 90%进度，延伸路径
            # 继续延伸100米
            current_waypoint = self.world.get_map().get_waypoint(
                self.vehicle.get_location(),
                project_to_road=True
            )
            
            new_points = []
            for _ in range(10):
                next_waypoints = current_waypoint.next(10.0)
                if next_waypoints:
                    current_waypoint = next_waypoints[0]
                    new_points.append([
                        current_waypoint.transform.location.x,
                        current_waypoint.transform.location.y
                    ])
                else:
                    break
            
            if new_points:
                # 追加新路径点
                old_path = self.pure_pursuit.global_path
                new_path = np.vstack([old_path, np.array(new_points)])
                self.pure_pursuit.set_path(new_path)
                print(f"[PURE PURSUIT] 路径已延伸 | 新点数:{len(new_points)} | 总点数:{len(new_path)}")
    
    def setup_collision_sensor(self):
        """安装碰撞传感器（检测实际碰撞）"""
        try:
            collision_bp = self.blueprint_library.find('sensor.other.collision')
            
            spawn_point = carla.Transform(
                carla.Location(x=0.0, z=0.0),
                carla.Rotation(yaw=0)
            )
            
            self.collision_sensor = self.world.spawn_actor(
                collision_bp,
                spawn_point,
                attach_to=self.vehicle
            )
            self.collision_sensor.listen(lambda event: self._on_collision(event))
            print("[INFO] 碰撞传感器安装成功！")
        except Exception as e:
            print(f"[WARNING] 碰撞传感器安装失败: {e}")
    
    def _on_collision(self, event):
        """碰撞检测回调"""
        if self.vehicle and event.other_actor and event.other_actor.id == self.vehicle.id:
            return
        
        self.collision_detected = True
        print(f"[COLLISION] 检测到碰撞！与 {event.other_actor.type_id if event.other_actor else 'Unknown'} 发生碰撞")
        
        # 触发碰撞后恢复
        self.post_collision_recovery = True
        self.collision_recovery_start = time.time()
        self.avoidance_state = 'normal'  # 重置避让状态
        
        # 【关键修复5】碰撞后重置所有变道相关状态，防止恢复后立即再次变道
        self.last_lane_change_time = time.time()  # 重置冷却时间
        self.last_obstacle_id = None  # 清除障碍物ID记录
        self.obstacle_info = None  # 清除障碍物信息
        self.obstacle_distance = float('inf')
        self.lane_change_completed = False

    def _on_obstacle_detected(self, event):
        """障碍物检测回调 - 集成 DeepSORT 跟踪"""
        # 过滤掉自车（距离为0或检测到的是自己）
        if event.distance < 0.1:
            return
        if self.vehicle and event.other_actor and event.other_actor.id == self.vehicle.id:
            return
        
        self.frame_count += 1
        
        # 使用 DeepSORT 跟踪障碍物
        # 由于 obstacle_sensor 不提供 2D bbox，我们基于距离和角度估算一个伪 bbox
        # 格式：[x1, y1, x2, y2] 基于障碍物相对于车辆的位置
        vehicle_transform = self.vehicle.get_transform()
        vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)
        
        # 计算障碍物在车辆坐标系中的相对位置
        rel_x = event.transform.location.x - vehicle_transform.location.x
        rel_y = event.transform.location.y - vehicle_transform.location.y
        
        # 将相对位置转换到图像坐标系（简化模型）
        # 假设障碍物在车辆正前方，根据距离映射到 y 坐标
        img_x = 640 // 2  # 图像中心 x
        img_y = max(10, min(470, int(400 - event.distance * 10)))  # 距离越远，y 越小
        bbox_size = max(20, min(200, int(300 / (event.distance + 1))))  # 距离越远，框越小
        
        # 创建伪边界框 [x1, y1, x2, y2]
        pseudo_bbox = np.array([
            img_x - bbox_size // 2,
            img_y - bbox_size // 2,
            img_x + bbox_size // 2,
            img_y + bbox_size // 2
        ]).reshape(1, 4)
        
        # 更新 DeepSORT 跟踪器
        tracked_results = self.deep_sort.update(pseudo_bbox)
        
        # 更新障碍物信息（使用跟踪结果）
        if len(tracked_results) > 0:
            track_id = int(tracked_results[0][4])
            self.obstacle_distance = event.distance
            self.obstacle_info = {
                'distance': event.distance,
                'actor': event.other_actor,
                'actor_id': event.other_actor.id if event.other_actor else None,
                'track_id': track_id,  # DeepSORT 分配的跟踪ID
                'transform': event.transform,
                'yaw': event.transform.rotation.yaw if event.transform else 0,
                'confidence': 1.0,
                'stable_hits': self.deep_sort.tracks[0].hits if self.deep_sort.tracks else 0
            }
            
            # 持续跟踪这个障碍物（不清除）
            self.current_obstacle = event.other_actor
            self.tracked_obstacles[track_id] = self.obstacle_info.copy()
            
            # 如果跟踪稳定（命中次数足够），更新 last_obstacle_id
            if self.deep_sort.tracks and self.deep_sort.tracks[0].hits >= 3:
                self.last_obstacle_id = event.other_actor.id if event.other_actor else None
        else:
            self.obstacle_info = {
                'distance': event.distance,
                'actor': event.other_actor,
                'actor_id': event.other_actor.id if event.other_actor else None,
                'transform': event.transform,
                'yaw': event.transform.rotation.yaw if event.transform else 0,
                'confidence': 0.5,
                'stable_hits': 0
            }
        
        # 如果障碍物非常近（< 3米），设置碰撞警告
        if event.distance < 3.0:
            self.collision_warning = True
            print(f"[WARNING] 障碍物距离过近: {event.distance:.1f}m! (跟踪ID: {tracked_results[0][4] if len(tracked_results) > 0 else 'N/A'})")

    def apply_smart_avoidance(self):
        """
        智能绕行避让：检测到障碍物时执行完整变道
        状态机：normal -> changing_lane -> normal
        """
        if not self.vehicle:
            return
        
        current_time = time.time()
        velocity = self.vehicle.get_velocity()
        speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = speed_ms * 3.6
        
        # ============== 碰撞后恢复逻辑 ==============
        if self.post_collision_recovery:
            elapsed = current_time - self.collision_recovery_start
            if elapsed < 1.0:
                # 碰撞后：刹车+回正
                self.vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    brake=0.8,
                    steer=0.0,
                    hand_brake=False
                ))
                return
            else:
                # 恢复自动驾驶
                self.post_collision_recovery = False
                self.collision_warning = False
                self.avoidance_state = 'normal'
                try:
                    self.vehicle.set_autopilot(True, 8000)
                except:
                    pass
                print("[RECOVERY] 碰撞后恢复自动驾驶")
                return
        
        # ============== 紧急刹车：如果障碍物太近且正在变道 ==============
        if self.collision_warning and self.avoidance_state == 'changing_lane':
            if self.obstacle_distance < 2.0:
                print(f"[EMERGENCY] 障碍物过近 {self.obstacle_distance:.1f}m，紧急刹车！")
                self.vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    steer=0.0
                ))
                self.post_collision_recovery = True
                self.collision_recovery_start = current_time
                return
        
        # ============== 状态机逻辑 ==============
        if self.avoidance_state == 'normal':
            # 检查是否需要恢复自动驾驶（变道刚完成）
            if self.lane_change_completed and current_time >= self.lane_change_recovery_time:
                self.lane_change_completed = False
                self.last_lane_change_time = current_time  # 记录变道完成时间
                try:
                    self.vehicle.set_autopilot(True, 8000)
                except:
                    pass
                print("[LANE CHANGE] 已恢复自动驾驶")
            
            # 正常驾驶状态，确保自动驾驶已启用
            elif not self.lane_change_completed:
                try:
                    self.vehicle.set_autopilot(True, 8000)
                except:
                    pass
            
            # 检查是否需要变道（添加防护防止连续变道）
            if self.obstacle_info and not self.lane_change_completed:
                distance = self.obstacle_distance
                
                # 【关键修复1】检查冷却时间，防止连续变道
                time_since_last_change = current_time - self.last_lane_change_time
                in_cooldown = time_since_last_change < self.lane_change_cooldown and self.last_lane_change_time > 0
                
                # 检查是否是同一障碍物（使用 actor_id）
                current_obstacle_id = self.obstacle_info.get('actor_id')
                is_same_obstacle = current_obstacle_id == self.last_obstacle_id and self.last_obstacle_id is not None
                
                # 【关键修复2】静态障碍物或近距离威胁，无论冷却期都要处理
                obstacle_actor = self.obstacle_info.get('actor')
                obstacle_speed = 0
                if obstacle_actor:
                    obstacle_vel = obstacle_actor.get_velocity()
                    obstacle_speed = np.sqrt(obstacle_vel.x**2 + obstacle_vel.y**2 + obstacle_vel.z**2)
                
                is_static_obstacle = obstacle_speed < 1.0  # 静态或几乎静止
                is_emergency = distance < 10.0  # 紧急距离
                
                # 静态障碍物或紧急情况，忽略冷却期
                if is_same_obstacle and in_cooldown and not (is_static_obstacle and is_emergency):
                    # 在冷却期内，忽略障碍物检测（可能是同一障碍物）
                    pass
                else:
                        # 新障碍物或冷却期已过，可以变道
                        stable_hits = self.obstacle_info.get('stable_hits', 0)
                        track_id = self.obstacle_info.get('track_id', None)
                        
                        # 【关键修复】不再依赖DeepSORT稳定性来触发变道
                        # TTC本身就足够可靠，伪bboxes的IOU匹配容易失败导致死锁
                        # 直接使用TTC和距离判断是否需要变道
                        if stable_hits < 3 and self.frame_count % 60 == 0:
                            print(f"[DEEP SORT] 跟踪不稳定 (hits={stable_hits})，使用TTC直接判断")
                        
                        # ========== TTC安全距离计算（优化版）==========
                        # 计算障碍物相对速度
                        rel_velocity = self.compute_relative_velocity(distance, dt=0.05)
                        
                        # 【关键修复】正确计算障碍物速度
                        # 如果障碍物有速度信息，使用它；否则假设静态（相对速度=自车速度）
                        obstacle_actor = self.obstacle_info.get('actor')
                        if obstacle_actor:
                            obstacle_vel = obstacle_actor.get_velocity()
                            obstacle_speed = np.sqrt(obstacle_vel.x**2 + obstacle_vel.y**2 + obstacle_vel.z**2)
                        else:
                            obstacle_speed = 0  # 静态障碍物
                        
                        # 计算 TTC（如果障碍物静止，TTC = distance / ego_speed）
                        ttc = self.compute_ttc(distance, speed_ms, obstacle_speed)
                        
                        # 获取风险等级
                        risk_level = self.get_risk_level(ttc)
                        
                        # 基于TTC计算安全距离
                        safety_distance = self.compute_ttc_safety_distance(ttc, speed_ms)
                        
                        # 【修复】先初始化 trigger_lane_change
                        trigger_lane_change = False
                        
                        # 调试输出（每30帧打印一次，更清晰显示）
                        if self.frame_count % 30 == 0:
                            obstacle_type = obstacle_actor.type_id if obstacle_actor else "unknown"
                            obstacle_vel = obstacle_actor.get_velocity() if obstacle_actor else None
                            obs_speed = np.sqrt(obstacle_vel.x**2 + obstacle_vel.y**2 + obstacle_vel.z**2) if obstacle_vel else 0
                            cooldown_remaining = max(0, self.lane_change_cooldown - (current_time - self.last_lane_change_time))
                            print(f"[TTC] 距离:{distance:.1f}m | TTC:{ttc:.2f}s | 风险:{risk_level} | 自车:{speed_kmh:.1f}km/h | 障碍物速度:{obs_speed:.1f}m/s | 类型:{obstacle_type}")
                            print(f"[DEBUG] 冷却时间剩余:{cooldown_remaining:.1f}s | 状态:{self.avoidance_state} | 障碍物ID:{current_obstacle_id} | 上次障碍物ID:{self.last_obstacle_id}")
                            if trigger_lane_change:
                                print(f"[LANE CHANGE DECISION] ✓ 决定触发变道! 方向:{self.lane_change_direction}")
                        
                        # 风险等级对应的处理策略
                        should_change_lane = False
                        proactive_brake = False  # 主动刹车标志
                        
                        if risk_level == 'safe':
                            # 安全状态，不处理
                            pass
                        elif risk_level == 'warning':
                            # 警告状态就开始适度减速
                            if distance < safety_distance:
                                proactive_brake = True
                                brake_strength = min(0.4, speed_kmh / 100)  # 根据速度调整刹车力度
                                self.vehicle.apply_control(carla.VehicleControl(
                                    throttle=0.0,
                                    brake=brake_strength,
                                    steer=0.0
                                ))
                        elif risk_level == 'caution':
                            # 注意状态立即减速，力度加强
                            proactive_brake = True
                            if distance < 20.0:  # 20米内就开始减速
                                if distance < 10.0:
                                    brake_strength = 0.7  # 10米内强刹车
                                else:
                                    brake_strength = 0.5  # 20米内中度刹车
                                self.vehicle.apply_control(carla.VehicleControl(
                                    throttle=0.0,
                                    brake=brake_strength,
                                    steer=0.0
                                ))
                        else:  # alert 或 emergency
                            proactive_brake = True
                        
                        # 【修复】变道触发条件 - 静态障碍物和动态障碍物都更容易触发
                        # 静态障碍物TTC计算已修复，现在应该能正确触发变道
                        
                        # 静态障碍物或运动较慢的障碍物（obstacle_speed < 2m/s）
                        is_slow_obstacle = obstacle_speed < 2.0
                        
                        if risk_level == 'emergency':
                            trigger_lane_change = True
                        elif risk_level == 'alert' and (distance < 30.0 or is_slow_obstacle):
                            trigger_lane_change = True
                        elif risk_level == 'caution':
                            # 【关键】caution级别也应该触发变道，尤其是静态障碍物
                            if distance < 25.0 or is_slow_obstacle:
                                trigger_lane_change = True
                        elif risk_level == 'warning' and distance < safety_distance:
                            # 警告级别但距离很近也要变道
                            if distance < 15.0:
                                trigger_lane_change = True
                        
                        # 静态障碍物特殊处理：距离近时无条件变道
                        if obstacle_speed < 1.0 and distance < 20.0:
                            trigger_lane_change = True
                            print(f"[STATIC OBSTACLE] 静态障碍物近距离触发变道 | 距离:{distance:.1f}m")
                            
                            if trigger_lane_change:
                                # 【关键修复】不再依赖DeepSORT track_id来决定是否变道
                                # 直接基于TTC判断即可，伪bboxes的IOU匹配容易失败
                                
                                print(f"[LANE CHANGE] TTC触发变道 (TTC:{ttc:.2f}s, 距离:{distance:.1f}m, 风险:{risk_level})")
                                
                                # 决定变道方向
                                vehicle_transform = self.vehicle.get_transform()
                                obstacle_transform = self.obstacle_info.get('transform')
                                        
                                if obstacle_transform:
                                    dx = obstacle_transform.location.x - vehicle_transform.location.x
                                    dy = obstacle_transform.location.y - vehicle_transform.location.y
                                    vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)
                                            
                                    # 转换到车辆坐标系: rel_y > 0 表示障碍物在右侧，应该向左变道
                                    rel_y = -dx * np.sin(vehicle_yaw) + dy * np.cos(vehicle_yaw)
                                            
                                    if rel_y >= 0:
                                        self.lane_change_direction = 'left'
                                    else:
                                        self.lane_change_direction = 'right'
                                            
                                    # 保存变道初始位置用于反馈检测
                                    self.lane_change_start_lateral = vehicle_transform.location.y
                                            
                                    # 保存变道初始状态
                                    self.lane_change_start_time = current_time
                                    self.lane_change_start_yaw = vehicle_transform.rotation.yaw
                                            
                                    # 记录当前障碍物ID，防止重复处理
                                    self.last_obstacle_id = current_obstacle_id
                                            
                                    # 禁用自动驾驶，开始变道
                                    try:
                                        self.vehicle.set_autopilot(False)
                                    except:
                                        pass
                                            
                                    self.avoidance_state = 'changing_lane'
                                    print(f"[LANE CHANGE] 开始{self.lane_change_direction}侧变道 | 距离:{distance:.1f}m | TTC:{ttc:.2f}s | 风险:{risk_level}")
        
        elif self.avoidance_state == 'changing_lane':
            # 变道中：使用Pure Pursuit轨迹跟踪
            vehicle_transform = self.vehicle.get_transform()
            elapsed = current_time - self.lane_change_start_time
            progress = elapsed / self.lane_change_duration
            
            # 获取车辆当前状态
            velocity = self.vehicle.get_velocity()
            speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_kmh = speed_ms * 3.6
            current_pos = (vehicle_transform.location.x, vehicle_transform.location.y)
            current_yaw = np.radians(vehicle_transform.rotation.yaw)
            
            # 【Pure Pursuit控制】计算转向角
            if self.use_pure_pursuit:
                # 使用Pure Pursuit算法计算精确转向
                steer_rad = self.pure_pursuit.compute_steering(current_pos, current_yaw, speed_ms)
                
                # 根据变道方向调整Pure Pursuit目标
                if self.lane_change_direction == 'left':
                    steer_value = -abs(steer_rad) * 1.5 if steer_rad > 0 else steer_rad * 0.5
                else:
                    steer_value = abs(steer_rad) * 1.5 if steer_rad < 0 else steer_rad * 0.5
                
                # 限制转向范围
                steer_value = np.clip(steer_value, -0.8, 0.8)
                
                if self.frame_count % 30 == 0:
                    print(f"[PURE PURSUIT] 转向角:{steer_value:.3f} | 速度:{speed_kmh:.1f}km/h | 进度:{progress:.1%}")
            else:
                # 传统渐进式转向（备用方案）
                if self.lane_change_direction == 'left':
                    base_steer = -0.5
                else:
                    base_steer = 0.5
                
                if progress < 0.6:
                    steer_value = base_steer
                elif progress < 0.8:
                    steer_value = base_steer * 0.5
                else:
                    steer_value = base_steer * 0.2
            
            # 变道进行中，完全松开油门
            brake_value = 0.0
            
            # 持续监控障碍物距离
            if self.obstacle_distance < 15.0:
                if self.obstacle_distance < 5.0:
                    brake_value = 0.9
                elif self.obstacle_distance < 10.0:
                    brake_value = 0.6
                else:
                    brake_value = 0.3
            
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=0.0,
                brake=brake_value,
                steer=steer_value,
                hand_brake=False
            ))
            
            # 检查变道是否完成（基于时间和横向位移反馈）
            lateral_change = abs(vehicle_transform.location.y - self.lane_change_start_lateral)
            
            # 计算变道中的TTC（使用实际障碍物速度）
            obstacle_actor = self.obstacle_info.get('actor') if self.obstacle_info else None
            if obstacle_actor:
                obstacle_vel = obstacle_actor.get_velocity()
                obstacle_speed = np.sqrt(obstacle_vel.x**2 + obstacle_vel.y**2 + obstacle_vel.z**2)
            else:
                obstacle_speed = 0
            ttc_during_change = self.compute_ttc(self.obstacle_distance, speed_ms, obstacle_speed)
            
            # 变道中紧急刹车阈值（降低到5米）
            if self.obstacle_distance < 5.0 or ttc_during_change < self.TTC_EMERGENCY:
                # TTC低于紧急阈值，触发紧急刹车
                self.collision_warning = True
                print(f"[EMERGENCY] 变道中距离={self.obstacle_distance:.1f}m TTC={ttc_during_change:.2f}s，紧急刹车！")
                self.vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    steer=0.0
                ))
                self.post_collision_recovery = True
                self.collision_recovery_start = current_time
                return
            
            # 变道完成条件：时间足够 OR 横向位移足够（至少3米）
            if elapsed >= self.lane_change_duration * 0.7 or lateral_change >= 3.0:
                # 变道完成，回正方向盘
                self.vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    brake=0.0,
                    steer=0.0,
                    hand_brake=False
                ))
                
                # 设置恢复标志
                self.lane_change_completed = True
                self.lane_change_recovery_time = current_time + 0.3
                
                # 重置状态
                self.avoidance_state = 'normal'
                self.lane_change_direction = None
                
                # 变道完成后清除障碍物信息，防止连续变道
                print(f"[LANE CHANGE] 变道完成(横向位移:{lateral_change:.1f}m)，300ms后恢复自动驾驶")
        
        else:
            # 其他状态，重置
            self.avoidance_state = 'normal'
            self.lane_change_direction = None
            try:
                self.vehicle.set_autopilot(True, 8000)
            except:
                pass


    def setup_camera(self):
        """设置摄像头（图像处理仍有问题，主要用于获取帧）"""
        if not self.vehicle:
            return
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(config.CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(config.CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(config.CAMERA_FOV))
        camera_bp.set_attribute('sensor_tick', '0.0')
        camera_bp.set_attribute('motion_blur_intensity', '0.0')
        
        spawn_point = carla.Transform(carla.Location(x=config.CAMERA_POS_X, z=config.CAMERA_POS_Z))
        self.camera = self.world.spawn_actor(camera_bp, spawn_point, attach_to=self.vehicle)
        self.camera.listen(lambda image: self._process_image(image))
        print("[INFO] RGB 摄像头安装成功！")

    def _process_image(self, image):
        """处理摄像头图像（临时方案）"""
        try:
            data = np.frombuffer(image.raw_data, dtype=np.uint8)
            img = data.reshape((image.height, image.width, 4))[:, :, :3].copy()
            self.image_queue.put(img)
        except:
            pass

    def draw_detection_in_carla(self, detections):
        """
        在 CARLA 模拟器中绘制检测结果
        使用 Debug Draw 在 3D 世界中绘制边界框
        """
        if not self.world or not self.vehicle:
            return
        
        # 获取主车辆位置
        ego_location = self.vehicle.get_location()
        ego_transform = self.vehicle.get_transform()
        
        # 遍历检测结果
        for detection in detections:
            class_name = detection[0]
            confidence = detection[1]
            
            if confidence < config.conf_thres:
                continue
            
            # 只处理车辆类别
            if 'car' in class_name.lower() or 'vehicle' in class_name.lower() or 'truck' in class_name.lower() or 'bus' in class_name.lower():
                # 在车辆前方 5-30 米范围内生成检测点
                forward = ego_transform.get_forward_vector()
                distance = random.uniform(10, 30)
                right = ego_transform.get_right_vector()
                lateral = random.uniform(-5, 5)
                
                detection_loc = carla.Location(
                    x=ego_location.x + forward.x * distance + right.x * lateral,
                    y=ego_location.y + forward.y * distance + right.y * lateral,
                    z=ego_location.z + random.uniform(0.5, 1.5)
                )
                
                # 绘制绿色点
                self.debug_helper.draw_point(
                    detection_loc,
                    size=0.5,
                    color=carla.Color(0, 255, 0),
                    life_time=-1  # 永久显示，直到下次绘制
                )
                
                # 绘制标签
                self.debug_helper.draw_string(
                    carla.Location(x=detection_loc.x, y=detection_loc.y, z=detection_loc.z + 1.5),
                    f"{class_name} {confidence:.1f}",
                    draw_shadow=False,
                    color=carla.Color(0, 255, 0),
                    life_time=-1
                )

    def draw_vehicle_boxes(self, debug=False):
        """
        在 CARLA 模拟器中绘制其他车辆的边界框（不标记主车辆）
        用于验证检测功能
        """
        if not self.world or not self.debug_helper:
            return
        
        try:
            # 获取所有车辆
            actors = self.world.get_actors().filter('vehicle.*')
            actor_list = list(actors)
            
            if debug:
                print(f"[DEBUG] 发现 {len(actor_list)} 辆车")
            
            for actor in actor_list:
                # 跳过主车辆
                if self.vehicle and actor.id == self.vehicle.id:
                    continue
                
                transform = actor.get_transform()
                bbox = actor.bounding_box
                bbox.location = transform.location
                bbox.rotation = transform.rotation
                
                # 绘制白色边界框
                self.debug_helper.draw_box(
                    bbox,
                    transform.rotation,
                    thickness=0.3,
                    color=carla.Color(255, 255, 255),
                    life_time=0.1
                )
                
        except Exception as e:
            print(f"[DEBUG] 绘制边界框时出错: {e}")

    def destroy_actors(self):
        try:
            if self.obstacle_sensor:
                self.obstacle_sensor.destroy()
                self.obstacle_sensor = None
            if self.camera:
                self.camera.destroy()
                self.camera = None
            if self.obstacle_sensor:
                self.obstacle_sensor.destroy()
                self.obstacle_sensor = None
            if self.collision_sensor:
                self.collision_sensor.destroy()
                self.collision_sensor = None
            if self.vehicle:
                self.vehicle.destroy()
                self.vehicle = None
            print("[INFO] 所有 Actor 已清理。")
        except RuntimeError:
            print("[INFO] Actor 已清理或不存在。")

    def follow_vehicle(self):
        """第三人称跟随主车辆"""
        if not self.vehicle or not self.spectator:
            return
        
        try:
            # 获取车辆 transform
            transform = self.vehicle.get_transform()
        except RuntimeError:
            return  # 车辆已被销毁
        
        # 计算跟随位置：车后8米，高5米
        forward = transform.get_forward_vector()
        location = carla.Location(
            x=transform.location.x - forward.x * 12,
            y=transform.location.y - forward.y * 12,
            z=transform.location.z + 5
        )
        
        # 保持与车辆相同的朝向
        rotation = carla.Rotation(
            pitch=transform.rotation.pitch,
            yaw=transform.rotation.yaw,
            roll=transform.rotation.roll
        )
        
        # 更新 spectator
        self.spectator.set_transform(carla.Transform(location, rotation))

    def apply_obstacle_avoidance(self, auto_brake=True):
        """
        应用障碍物躲避控制（改进版）
        
        - 提前减速：根据距离渐进刹车
        - 更早检测：安全距离 = 速度 * 1.5秒（原来0.5秒太短）
        - 分级刹车：根据危险程度调整刹车力度
        """
        if not self.vehicle:
            return
        
        # 获取当前速度 (m/s)
        velocity = self.vehicle.get_velocity()
        speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = speed_ms * 3.6
        
        if auto_brake and self.obstacle_info:
            distance = self.obstacle_distance
            
            # 安全距离 = 速度 * 2秒（提前2秒开始反应）
            safety_distance = speed_ms * 2.0 + 8  # 加上8米基础距离
            
            # 极危险距离 = 速度 * 1秒
            danger_distance = speed_ms * 1.0 + 3
            
            if distance < safety_distance:
                # 检测到危险，禁用自动驾驶（让手动控制生效）
                self.vehicle.set_autopilot(False)
                
                # 计算刹车力度（距离越近力度越大）
                if distance < danger_distance:
                    # 极危险：全力刹车
                    brake_value = 1.0
                    if speed_kmh > 5:  # 只在有速度时打印
                        print(f"[WARNING] 紧急刹车！障碍物距离: {distance:.1f}m, 速度: {speed_kmh:.1f}km/h")
                else:
                    # 一般危险：渐进刹车
                    brake_value = max(0.1, 1.0 - (distance - danger_distance) / (safety_distance - danger_distance))
                
                brake_control = carla.VehicleControl(
                    throttle=0.0,
                    brake=brake_value,
                    steer=0.0,
                    hand_brake=False
                )
                self.vehicle.apply_control(brake_control)
            else:
                # 障碍物已远离，恢复自动驾驶
                self.vehicle.set_autopilot(True)
                self.obstacle_info = None
        elif auto_brake and not self.obstacle_info:
            # 无障碍物信息，正常行驶
            try:
                self.vehicle.set_autopilot(True)
            except:
                pass
                    
    def enable_autopilot_with_obstacle_avoidance(self):
        """
        启用带障碍物躲避的自动驾驶
        """
        if not self.vehicle:
            return
        
        # 获取交通管理器
        traffic_manager = self.client.get_trafficmanager(8000)
        
        # 启用自动驾驶
        self.vehicle.set_autopilot(True, traffic_manager.get_port())
        
        # 设置更激进的障碍物响应距离
        traffic_manager.set_vehicle_distance_to_leading_vehicle(self.vehicle, 1.0)
        
        # 设置更短的安全距离
        traffic_manager.minimum_distance(self.vehicle, 1.0)
        
        print("[INFO] 已启用带障碍物躲避的自动驾驶")