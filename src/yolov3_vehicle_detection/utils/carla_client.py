import carla
import random
import time
import sys
import os
import numpy as np
import cv2
import queue

# 路径修复：确保能正确导入 config 模块
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import config
from utils.deep_sort import DeepSORTTracker


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
        self.lane_change_cooldown = 5.0  # 变道冷却时间（秒）- 防止连续变道
        self.last_obstacle_id = None  # 上次处理的障碍物ID，防止重复处理同一障碍物
        
        # 新增：持续跟踪障碍物（不清除直到安全）
        self.current_obstacle = None  # 当前障碍物引用
        self.lane_change_start_lateral = 0  # 变道开始时的横向位置
        self.collision_warning = False  # 碰撞警告标志
        
        # 新增：碰撞后恢复
        self.post_collision_recovery = False
        self.collision_recovery_start = 0
        
        # DeepSORT 目标跟踪
        self.deep_sort = DeepSORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        self.tracked_obstacles = {}  # 跟踪的障碍物 {track_id: info}
        self.frame_count = 0

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
            obstacle_bp.set_attribute('distance', '30')      # 检测距离 30 米
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
                if time_since_last_change < self.lane_change_cooldown and self.last_lane_change_time > 0:
                    # 在冷却期内，忽略障碍物检测（可能是同一障碍物）
                    pass
                else:
                    # 【关键修复2】检查是否是同一障碍物（使用 actor_id）
                    current_obstacle_id = self.obstacle_info.get('actor_id')
                    if current_obstacle_id == self.last_obstacle_id and self.last_obstacle_id is not None:
                        # 是同一个障碍物，检查它是否仍然在前方且是威胁
                        if distance > 20:  # 距离远了就清除
                            self.last_obstacle_id = None
                            self.obstacle_info = None  # 清除障碍物信息
                            self.obstacle_distance = float('inf')
                        else:
                            # 仍在范围内但刚变过道，跳过
                            pass
                    else:
                        # 新障碍物或冷却期已过，可以变道
                        # 【DeepSORT优化】检查跟踪是否稳定
                        stable_hits = self.obstacle_info.get('stable_hits', 0)
                        track_id = self.obstacle_info.get('track_id', None)
                        
                        # 只有稳定跟踪（至少3帧命中）才触发变道，避免误检
                        if stable_hits < 3:
                            if self.frame_count % 30 == 0:  # 每30帧打印一次
                                print(f"[DEEP SORT] 跟踪不稳定 (hits={stable_hits})，等待更多帧...")
                            pass
                        else:
                            # 安全距离 = 速度 * 2秒 + 8米（缩短反应时间）
                            safety_distance = speed_ms * 2.0 + 8
                            
                            if distance < safety_distance:
                                # 【DeepSORT优化】使用 track_id 避免重复触发
                                if track_id is not None:
                                    tracked_ids = [info.get('track_id') for info in self.tracked_obstacles.values() if info.get('track_id') == track_id]
                                    if len(tracked_ids) > 1:
                                        print(f"[DEEP SORT] 跳过重复跟踪 (track_id={track_id})")
                                    else:
                                        print(f"[DEEP SORT] 稳定跟踪确认 (track_id={track_id}, hits={stable_hits})")
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
                                            
                                            # 【关键修复3】记录当前障碍物ID，防止重复处理
                                            self.last_obstacle_id = current_obstacle_id
                                            
                                            # 禁用自动驾驶，开始变道
                                            try:
                                                self.vehicle.set_autopilot(False)
                                            except:
                                                pass
                                            
                                            self.avoidance_state = 'changing_lane'
                                            print(f"[LANE CHANGE] 开始{self.lane_change_direction}侧变道，障碍物距离: {distance:.1f}m，速度: {speed_kmh:.1f}km/h")
        
        elif self.avoidance_state == 'changing_lane':
            # 变道中：使用渐进式转向
            elapsed = current_time - self.lane_change_start_time
            progress = elapsed / self.lane_change_duration
            
            # 渐进式转向：开始时转向最大，后期逐渐回正
            if self.lane_change_direction == 'left':
                base_steer = -0.5  # 更激进的转向
            else:
                base_steer = 0.5
            
            # 根据进度调整转向（开始大，后期小）
            if progress < 0.6:
                steer_value = base_steer
            elif progress < 0.8:
                steer_value = base_steer * 0.5
            else:
                steer_value = base_steer * 0.2  # 接近完成时小幅度调整
            
            # 变道进行中，保持转向
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=0.2,  # 稍微减速
                brake=0.0,
                steer=steer_value,
                hand_brake=False
            ))
            
            # 检查变道是否完成（基于时间和横向位移反馈）
            vehicle_transform = self.vehicle.get_transform()
            lateral_change = abs(vehicle_transform.location.y - self.lane_change_start_lateral)
            
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
                
                # 【关键修复4】变道完成后清除障碍物信息，防止连续变道
                # 等待冷却期结束后才会再次检测障碍物
                print(f"[LANE CHANGE] 变道完成(横向位移:{lateral_change:.1f}m)，300ms后恢复自动驾驶，冷却{self.lane_change_cooldown}秒")
            else:
                # 还在变道中，持续监控障碍物
                if self.obstacle_info and self.obstacle_distance < 1.5:
                    # 障碍物太近，触发紧急刹车
                    self.collision_warning = True
        
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