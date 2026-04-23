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

    def setup_obstacle_sensor(self):
        """
        设置障碍物传感器，用于检测前方障碍物
        使用 CARLA 自带的 sensor.other.obstacle 传感器
        """
        if not self.vehicle:
            print("[WARNING] 车辆未生成，无法安装障碍物传感器")
            return
        
        try:
            # 创建障碍物传感器
            obstacle_bp = self.blueprint_library.find('sensor.other.obstacle')
            obstacle_bp.set_attribute('distance', '30')      # 检测距离 30 米
            obstacle_bp.set_attribute('hit_radius', '1')      # 碰撞半径
            obstacle_bp.set_attribute('only_dynamics', 'False')  # 也检测静态障碍物
            obstacle_bp.set_attribute('debug_linetrace', 'False')  # 关闭调试线条减少干扰
            
            # 安装在车辆前方
            spawn_point = carla.Transform(
                carla.Location(x=0.5, z=1.5),
                carla.Rotation(yaw=0)
            )
            
            self.obstacle_sensor = self.world.spawn_actor(
                obstacle_bp,
                spawn_point,
                attach_to=self.vehicle
            )
            
            # 设置回调函数
            self.obstacle_sensor.listen(lambda event: self._on_obstacle_detected(event))
            
            print("[INFO] 障碍物传感器安装成功！")
            print(f"[DEBUG] 传感器绑定到车辆 ID: {self.vehicle.id}")
            
        except Exception as e:
            print(f"[WARNING] 障碍物传感器安装失败: {e}")

    def _on_obstacle_detected(self, event):
        """
        障碍物检测回调函数
        当检测到障碍物时更新障碍物信息
        """
        # 过滤掉自车（距离为0或检测到的是自己）
        if event.distance < 0.1:
            return
        if self.vehicle and event.other_actor and event.other_actor.id == self.vehicle.id:
            return
        
        self.obstacle_distance = event.distance
        self.obstacle_info = {
            'distance': event.distance,
            'actor': event.other_actor,
            'transform': event.transform
        }
        
        # 绘制检测线（红色表示检测到障碍物）
        if self.debug_helper:
            self.debug_helper.draw_line(
                event.transform.location,
                self.vehicle.get_location(),
                thickness=0.1,
                color=carla.Color(255, 0, 0),  # 红色
                life_time=0.5
            )
            
            # 在障碍物位置绘制红色点
            self.debug_helper.draw_point(
                event.transform.location,
                size=0.3,
                color=carla.Color(255, 0, 0),
                life_time=0.5
            )

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