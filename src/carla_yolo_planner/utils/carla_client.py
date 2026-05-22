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
        self.cameras = {}  # 多摄像头字典: {'front', 'back', 'left', 'right'}
        self.current_camera = 'front'  # 当前激活的摄像头
        self.blueprint_library = None
        self.image_queue = queue.Queue()
        self.debug_helper = None
        self.spectator = None
        
        # 摄像头配置
        self.camera_configs = {
            'front': {
                'location': carla.Location(x=0.3, z=1.3),
                'rotation': carla.Rotation(pitch=0, yaw=0, roll=0),
                'name': '前视'
            },
            'back': {
                'location': carla.Location(x=-0.8, z=1.3),
                'rotation': carla.Rotation(pitch=0, yaw=180, roll=0),
                'name': '后视'
            },
            'left': {
                'location': carla.Location(x=0, z=1.3, y=0.4),
                'rotation': carla.Rotation(pitch=0, yaw=-90, roll=0),
                'name': '左视'
            },
            'right': {
                'location': carla.Location(x=0, z=1.3, y=-0.4),
                'rotation': carla.Rotation(pitch=0, yaw=90, roll=0),
                'name': '右视'
            }
        }
        
        # 碰撞检测相关
        self.collision_detected = False
        self.collision_time = 0
        self.stuck_counter = 0
        self.last_velocity = 0
        self.spawn_points = None  # 保存生成点列表用于重置
        
        # HUD显示相关
        self.hud_fps = 0
        self.hud_detection_count = 0
        self.hud_brake_status = ""
        self.hud_speed = 0  # 车辆速度（km/h）

    def connect(self):
        print(f"[INFO] 正在连接 CARLA 服务器 ({self.host}:{self.port})...")
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            
            # 设置世界为同步模式，确保自动驾驶正常工作
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)
            
            self.blueprint_library = self.world.get_blueprint_library()
            # 创建 Debug Helper 用于绘制
            self.debug_helper = self.world.debug
            # 获取 spectator 用于第三人称跟随
            self.spectator = self.world.get_spectator()
            # 保存生成点列表用于重置
            self.spawn_points = self.world.get_map().get_spawn_points()
            print("[INFO] CARLA 连接成功！")
            return True
        except Exception as e:
            print(f"[ERROR] 连接失败: {e}")
            return False

    def spawn_vehicle(self, spawn_npc=True, npc_count=15):
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
            self.traffic_manager = self.client.get_trafficmanager(8000)
            
            # 关键设置：禁用Traffic Manager同步模式（因为世界已经是同步的）
            self.traffic_manager.set_synchronous_mode(False)
            
            # 设置自动驾驶参数
            self.traffic_manager.set_global_distance_to_leading_vehicle(3.0)  # 跟车距离
            self.traffic_manager.set_global_speed_limit(50.0)  # 限速50 km/h
            
            # 为车辆设置自动驾驶
            self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
            
            # 设置车辆的自动驾驶行为
            self.traffic_manager.ignore_lights_percentage(self.vehicle, 0)  # 遵守红绿灯
            self.traffic_manager.ignore_signs_percentage(self.vehicle, 0)   # 遵守标志
            self.traffic_manager.ignore_vehicles_percentage(self.vehicle, 0) # 不忽略其他车辆
            self.traffic_manager.ignore_walkers_percentage(self.vehicle, 0)  # 不忽略行人
            
            # 设置更激进的驾驶行为
            self.traffic_manager.set_desired_speed(self.vehicle, 45.0)  # 期望速度
            self.traffic_manager.set_distance_to_leading_vehicle(self.vehicle, 3.0)  # 跟车距离
            
            print("[INFO] 自动驾驶已启用！")
            
            # 生成NPC车辆
            if spawn_npc:
                self._spawn_npc_vehicles(npc_count)
            
            return self.vehicle
        except Exception as e:
            print(f"[ERROR] 车辆生成失败: {e}")
            return None

    def _spawn_npc_vehicles(self, count=15):
        """生成NPC交通车辆"""
        try:
            # 获取交通管理器
            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_global_distance_to_leading_vehicle(2.0)
            traffic_manager.global_percentage_speed_difference(30.0)
            
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
    
    def tick(self):
        """推进世界模拟（同步模式下必须调用）"""
        if self.world:
            self.world.tick()
            return True
        return False

    def setup_camera(self):
        """设置多摄像头系统"""
        if not self.vehicle:
            return
            
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(config.CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(config.CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(config.CAMERA_FOV))
        camera_bp.set_attribute('sensor_tick', '0.0')
        camera_bp.set_attribute('motion_blur_intensity', '0.0')
        
        # 创建四个摄像头
        for cam_name, config_data in self.camera_configs.items():
            spawn_point = carla.Transform(config_data['location'], config_data['rotation'])
            camera = self.world.spawn_actor(camera_bp, spawn_point, attach_to=self.vehicle)
            
            # 只有当前摄像头才将图像放入队列
            if cam_name == self.current_camera:
                camera.listen(lambda image: self._process_image(image))
            else:
                camera.listen(lambda image: None)  # 非当前摄像头不处理
            
            self.cameras[cam_name] = camera
        
        print(f"[INFO] 多摄像头系统安装成功！当前视角: {self.camera_configs[self.current_camera]['name']}")
        print("[INFO] 按数字键 1-4 切换视角: 1-前视 | 2-后视 | 3-左视 | 4-右视")
    
    def switch_camera(self, camera_name):
        """
        切换到指定摄像头
        Args:
            camera_name: 'front', 'back', 'left', 'right'
        """
        if camera_name not in self.camera_configs:
            print(f"[WARNING] 未知摄像头: {camera_name}")
            return False
        
        if camera_name == self.current_camera:
            return True  # 已是当前摄像头
        
        # 停止当前摄像头的图像监听
        if self.current_camera in self.cameras:
            self.cameras[self.current_camera].listen(lambda image: None)
        
        # 切换到新摄像头
        self.current_camera = camera_name
        
        # 启动新摄像头的图像监听
        if self.current_camera in self.cameras:
            self.cameras[self.current_camera].listen(lambda image: self._process_image(image))
        
        cam_name = self.camera_configs[self.current_camera]['name']
        print(f"[INFO] 已切换到 {cam_name} 视角")
        return True
    
    def get_current_camera_name(self):
        """获取当前摄像头的中文名称"""
        return self.camera_configs[self.current_camera]['name']

    def _process_image(self, image):
        """处理摄像头图像（临时方案）"""
        try:
            data = np.frombuffer(image.raw_data, dtype=np.uint8)
            img = data.reshape((image.height, image.width, 4))[:, :, :3].copy()
            self.image_queue.put(img)
        except:
            pass
    
    def draw_detection_in_carla(self, detections, classes):
        """
        在 CARLA 模拟器中绘制检测结果
        使用 Debug Draw 在 3D 世界中绘制边界框和标签
        
        Args:
            detections: YOLO检测结果列表，格式为 [x, y, w, h, class_id, confidence]
            classes: 类别名称列表
        """
        if not self.world or not self.vehicle:
            return
        
        # 获取主车辆位置和变换
        ego_location = self.vehicle.get_location()
        ego_transform = self.vehicle.get_transform()
        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()
        
        # 绘制安全区域（驾驶走廊）
        self._draw_safe_corridor(ego_location, ego_transform)
        
        # 遍历检测结果
        for detection in detections:
            if len(detection) < 6:
                continue
                
            x, y, w, h, class_id, confidence = detection
            
            if confidence < config.conf_thres:
                continue
            
            # 获取类别名称
            class_name = classes[int(class_id)] if (classes and 0 <= int(class_id) < len(classes)) else f"class_{class_id}"
            
            # 根据检测框位置估算距离（检测框在图像下方表示距离更近）
            normalized_y = y / config.CAMERA_HEIGHT
            distance = 3 + (1 - normalized_y) * 27  # 距离范围 3-30 米
            
            # 根据检测框水平位置计算横向偏移
            img_center_x = config.CAMERA_WIDTH // 2
            normalized_x = (x + w/2 - img_center_x) / (config.CAMERA_WIDTH / 2)
            lateral = normalized_x * 7  # 最大偏移7米
            
            # 计算检测目标的3D位置
            detection_loc = carla.Location(
                x=ego_location.x + forward.x * distance + right.x * lateral,
                y=ego_location.y + forward.y * distance + right.y * lateral,
                z=ego_location.z + 0.3
            )
            
            # 根据距离设置颜色（近红远绿）
            if distance < 8:
                color = carla.Color(255, 0, 0)      # 红色 - 危险
                size = 0.5
            elif distance < 15:
                color = carla.Color(255, 165, 0)    # 橙色 - 警告
                size = 0.4
            else:
                color = carla.Color(0, 255, 0)      # 绿色 - 安全
                size = 0.3
            
            # 绘制检测点
            self.debug_helper.draw_point(
                detection_loc,
                size=size,
                color=color,
                life_time=0.15
            )
            
            # 绘制检测框（立方体）
            box = carla.BoundingBox(detection_loc, carla.Vector3D(1.5, 0.8, 0.5))
            self.debug_helper.draw_box(
                box,
                carla.Rotation(),
                thickness=0.2,
                color=color,
                life_time=0.15
            )
            
            # 绘制标签
            self.debug_helper.draw_string(
                carla.Location(x=detection_loc.x, y=detection_loc.y, z=detection_loc.z + 1.2),
                f"{class_name} {confidence:.2f}",
                draw_shadow=True,
                color=color,
                life_time=0.15
            )
    
    def _draw_safe_corridor(self, location, transform):
        """
        在 CARLA 中绘制驾驶安全走廊（蓝色区域）
        """
        forward = transform.get_forward_vector()
        right = transform.get_right_vector()
        
        # 安全走廊宽度（基于配置）
        corridor_width = config.SAFE_ZONE_RATIO * 8  # 转换为实际距离
        
        # 绘制安全走廊的四个角点
        points = []
        for d in [0, 25]:  # 0米和25米处
            for w in [-corridor_width/2, corridor_width/2]:
                point = carla.Location(
                    x=location.x + forward.x * d + right.x * w,
                    y=location.y + forward.y * d + right.y * w,
                    z=location.z + 0.1
                )
                points.append(point)
        
        # 绘制安全走廊区域（蓝色透明）
        # 绘制前后边界线
        self.debug_helper.draw_line(points[0], points[1], color=carla.Color(0, 128, 255), thickness=0.15, life_time=0.1)
        self.debug_helper.draw_line(points[2], points[3], color=carla.Color(0, 128, 255), thickness=0.15, life_time=0.1)
        # 绘制左右边界线
        self.debug_helper.draw_line(points[0], points[2], color=carla.Color(0, 128, 255), thickness=0.15, life_time=0.1)
        self.debug_helper.draw_line(points[1], points[3], color=carla.Color(0, 128, 255), thickness=0.15, life_time=0.1)
        
        # 绘制中心线
        center_line_start = carla.Location(
            x=location.x + forward.x * 0,
            y=location.y + forward.y * 0,
            z=location.z + 0.15
        )
        center_line_end = carla.Location(
            x=location.x + forward.x * 25,
            y=location.y + forward.y * 25,
            z=location.z + 0.15
        )
        self.debug_helper.draw_line(center_line_start, center_line_end, color=carla.Color(255, 255, 0), thickness=0.1, life_time=0.1)
        
        # 在起点绘制标签
        self.debug_helper.draw_string(
            carla.Location(x=location.x, y=location.y, z=location.z + 1.5),
            "SAFE ZONE",
            draw_shadow=True,
            color=carla.Color(0, 128, 255),
            life_time=0.1
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
            # 销毁所有摄像头
            for cam_name, camera in self.cameras.items():
                if camera:
                    camera.destroy()
            self.cameras.clear()
            print("[INFO] 所有摄像头已清理。")
            
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
            transform = self.vehicle.get_transform()
        except RuntimeError:
            return
        
        forward = transform.get_forward_vector()
        location = carla.Location(
            x=transform.location.x - forward.x * 8,
            y=transform.location.y - forward.y * 8,
            z=transform.location.z + 4
        )
        
        rotation = carla.Rotation(
            pitch=-15,
            yaw=transform.rotation.yaw,
            roll=transform.rotation.roll
        )
        
        self.spectator.set_transform(carla.Transform(location, rotation))
    
    def update_hud_info(self, fps, detection_count, speed_kmh=0, brake_status=""):
        """更新 HUD 显示信息"""
        self.hud_fps = fps
        self.hud_detection_count = detection_count
        self.hud_speed = speed_kmh
        self.hud_brake_status = brake_status
    
    def draw_hud(self):
        """在 CARLA 画面上绘制 HUD 信息"""
        if not self.world or not self.vehicle:
            return
        
        try:
            # 获取 spectator 位置（用于在画面固定位置显示）
            spectator_transform = self.spectator.get_transform()
            spectator_location = spectator_transform.location
            
            # 计算 HUD 显示位置（在 spectator 前方固定距离）
            forward = spectator_transform.get_forward_vector()
            hud_base_location = carla.Location(
                spectator_location.x + forward.x * 3,
                spectator_location.y + forward.y * 3,
                spectator_location.z - 1
            )
            
            # 显示速度（白色，最大字号）
            speed_text = f"  Speed: {self.hud_speed:.1f} km/h  "
            self.debug_helper.draw_string(
                carla.Location(hud_base_location.x, hud_base_location.y, hud_base_location.z + 1.2),
                speed_text,
                draw_shadow=True,
                color=carla.Color(255, 255, 255),
                life_time=0.2  # 每帧刷新
            )
            
            # 显示 FPS（黄色）
            fps_text = f"  FPS: {self.hud_fps:.1f}  "
            self.debug_helper.draw_string(
                carla.Location(hud_base_location.x, hud_base_location.y, hud_base_location.z + 1.0),
                fps_text,
                draw_shadow=True,
                color=carla.Color(255, 255, 0),
                life_time=0.2
            )
            
            # 显示检测数量（青色）
            detection_text = f"  Detections: {self.hud_detection_count}  "
            self.debug_helper.draw_string(
                carla.Location(hud_base_location.x, hud_base_location.y, hud_base_location.z + 0.8),
                detection_text,
                draw_shadow=True,
                color=carla.Color(0, 255, 255),
                life_time=0.2
            )
            
            # 显示当前视角（橙色）
            camera_name = self.camera_configs[self.current_camera]['name']
            camera_text = f"  View: {camera_name}  "
            self.debug_helper.draw_string(
                carla.Location(hud_base_location.x, hud_base_location.y, hud_base_location.z + 0.6),
                camera_text,
                draw_shadow=True,
                color=carla.Color(255, 128, 0),
                life_time=0.2
            )
            
            # 显示刹车状态（红色）
            if self.hud_brake_status:
                brake_text = "  !!! EMERGENCY BRAKING !!!  "
                self.debug_helper.draw_string(
                    carla.Location(hud_base_location.x, hud_base_location.y, hud_base_location.z + 0.3),
                    brake_text,
                    draw_shadow=True,
                    color=carla.Color(255, 0, 0),
                    life_time=0.2
                )
                
        except Exception as e:
            if self.hud_fps % 50 == 0:
                print(f"[WARNING] HUD绘制失败: {e}")
    
    def check_collision_and_reset(self):
        """检测车辆是否卡住或碰撞，并自动重置位置"""
        if not self.vehicle:
            return
        
        try:
            # 获取车辆速度
            velocity = self.vehicle.get_velocity()
            current_speed = velocity.length()
            
            # 判断是否卡住（速度接近0但应该在移动）
            if current_speed < 0.1:  # 速度小于0.1 m/s
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            
            # 如果连续300帧卡住，重置车辆位置
            if self.stuck_counter > 300:
                print("[WARNING] 车辆已卡住，正在重置位置...")
                
                # 获取随机生成点
                if self.spawn_points is None:
                    self.spawn_points = self.world.get_map().get_spawn_points()
                
                if self.spawn_points:
                    # 选择一个新的生成点
                    new_spawn_point = random.choice(self.spawn_points)
                    
                    # 停止自动驾驶
                    self.vehicle.set_autopilot(False)
                    
                    # 重置车辆位置
                    self.vehicle.set_transform(new_spawn_point)
                    
                    # 重新启动自动驾驶
                    traffic_manager = self.client.get_trafficmanager(8000)
                    self.vehicle.set_autopilot(True, traffic_manager.get_port())
                    
                    print(f"[INFO] 车辆已重置到新位置")
                
                self.stuck_counter = 0
                
        except Exception as e:
            print(f"[WARNING] 碰撞检测失败: {e}")