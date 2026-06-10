# --------------------------
# 简化修复版：确保车辆正确生成
# --------------------------

import carla
import time
import numpy as np
import cv2
import math
from collections import deque
import random


class SimpleController:
    """简单但可靠的控制逻辑"""

    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()
        # 速度限制相关
        self.max_speed = 50.0  # km/h，最大速度限制
        self.min_speed = 10.0  # km/h，最小速度限制
        self.target_speed = 50.0  # km/h，当前目标速度
        self.speed_step = 5.0  # km/h，速度调整步长
        self.waypoint_distance = 5.0
        self.last_waypoint = None
        # self.reverse_mode = False  # 倒车模式标志（未使用）
        self.manual_reverse = False  # 手动倒车标志

    def get_control(self):
        """基于路点的简单控制"""
        # 获取车辆状态
        location = self.vehicle.get_location()
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()

        # 计算速度（考虑倒车方向）
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6  # km/h

        # 检查是否在倒车模式
        if self.manual_reverse:
            # 倒车模式：直接返回倒车控制
            return 0.3, 0.0, 0.0, True  # throttle, brake, steer, reverse

        # 获取路点
        waypoint = self.map.get_waypoint(location, project_to_road=True)

        if not waypoint:
            # 如果没有找到路点，返回保守控制
            # return 0.3, 0.0, 0.0  # 原返回值（3个值）
            return 0.3, 0.0, 0.0, False  # 新返回值（4个值，增加reverse标志）

        # 获取下一个路点
        next_waypoints = waypoint.next(self.waypoint_distance)

        if not next_waypoints:
            # 如果没有下一个路点，使用当前路点
            target_waypoint = waypoint
        else:
            target_waypoint = next_waypoints[0]

        self.last_waypoint = target_waypoint

        # 计算转向
        vehicle_yaw = math.radians(transform.rotation.yaw)
        target_loc = target_waypoint.transform.location

        # 计算相对位置
        dx = target_loc.x - location.x
        dy = target_loc.y - location.y

        local_x = dx * math.cos(vehicle_yaw) + dy * math.sin(vehicle_yaw)
        local_y = -dx * math.sin(vehicle_yaw) + dy * math.cos(vehicle_yaw)

        if abs(local_x) < 0.1:
            steer = 0.0
        else:
            angle = math.atan2(local_y, local_x)
            steer = max(-0.5, min(0.5, angle / 1.0))

        # 速度控制
        if speed < self.target_speed * 0.8:
            throttle, brake = 0.6, 0.0
        elif speed > self.target_speed * 1.2:
            throttle, brake = 0.0, 0.3
        else:
            throttle, brake = 0.3, 0.0

        # return throttle, brake, steer  # 原返回值（3个值）
        return throttle, brake, steer, False  # 新返回值（4个值，增加reverse标志）

    def toggle_reverse(self):
        """切换倒车模式"""
        self.manual_reverse = not self.manual_reverse
        if self.manual_reverse:
            print("进入倒车模式")
        else:
            print("退出倒车模式，恢复前进")

    def increase_speed_limit(self):
        """增加速度限制"""
        if self.target_speed < self.max_speed:
            self.target_speed = min(self.target_speed + self.speed_step, self.max_speed)
            print(f"速度限制增加到: {self.target_speed:.0f} km/h")
        else:
            print(f"已达到最大速度限制: {self.max_speed:.0f} km/h")

    def decrease_speed_limit(self):
        """减少速度限制"""
        if self.target_speed > self.min_speed:
            self.target_speed = max(self.target_speed - self.speed_step, self.min_speed)
            print(f"速度限制降低到: {self.target_speed:.0f} km/h")
        else:
            print(f"已达到最小速度限制: {self.min_speed:.0f} km/h")

    def get_speed_limit(self):
        """获取当前速度限制"""
        return self.target_speed


class WeatherManager:
    """天气管理器 - 提供多种天气模式"""
    
    def __init__(self, world):
        self.world = world
        self.weather_configs = {
            'sunny': {
                'cloudiness': 10.0,
                'precipitation': 0.0,
                'precipitation_deposits': 0.0,
                'wind_intensity': 10.0,
                'sun_altitude_angle': 75.0,
                'fog_density': 0.0,
                'fog_distance': 1000.0,
                'fog_falloff': 1.0,
                'wetness': 0.0
            },
            'cloudy': {
                'cloudiness': 80.0,
                'precipitation': 0.0,
                'precipitation_deposits': 0.0,
                'wind_intensity': 20.0,
                'sun_altitude_angle': 75.0,
                'fog_density': 0.0,
                'fog_distance': 1000.0,
                'fog_falloff': 1.0,
                'wetness': 0.0
            },
            'rainy': {
                'cloudiness': 90.0,
                'precipitation': 80.0,
                'precipitation_deposits': 50.0,
                'wind_intensity': 40.0,
                'sun_altitude_angle': 45.0,
                'fog_density': 30.0,
                'fog_distance': 50.0,
                'fog_falloff': 0.1,
                'wetness': 80.0
            },
            'stormy': {
                'cloudiness': 100.0,
                'precipitation': 100.0,
                'precipitation_deposits': 80.0,
                'wind_intensity': 80.0,
                'sun_altitude_angle': 30.0,
                'fog_density': 50.0,
                'fog_distance': 30.0,
                'fog_falloff': 0.05,
                'wetness': 100.0
            },
            'snowy': {
                'cloudiness': 95.0,
                'precipitation': 0.0,
                'precipitation_deposits': 100.0,
                'wind_intensity': 30.0,
                'sun_altitude_angle': 45.0,
                'fog_density': 40.0,
                'fog_distance': 40.0,
                'fog_falloff': 0.1,
                'wetness': 0.0,
                'snow_intensity': 100.0
            },
            'foggy': {
                'cloudiness': 90.0,
                'precipitation': 10.0,
                'precipitation_deposits': 0.0,
                'wind_intensity': 5.0,
                'sun_altitude_angle': 45.0,
                'fog_density': 80.0,
                'fog_distance': 20.0,
                'fog_falloff': 0.02,
                'wetness': 20.0
            },
            'night': {
                'cloudiness': 30.0,
                'precipitation': 0.0,
                'precipitation_deposits': 0.0,
                'wind_intensity': 10.0,
                'sun_altitude_angle': -15.0,
                'fog_density': 20.0,
                'fog_distance': 80.0,
                'fog_falloff': 0.1,
                'wetness': 50.0
            }
        }
        self.current_weather = 'sunny'
        self.frame_counter = 0
    
    def _apply_weather(self, config):
        """内部方法：应用天气参数到CARLA天气对象"""
        weather = carla.WeatherParameters()
        weather.cloudiness = config['cloudiness']
        weather.precipitation = config['precipitation']
        weather.precipitation_deposits = config['precipitation_deposits']
        weather.wind_intensity = config['wind_intensity']
        weather.sun_altitude_angle = config['sun_altitude_angle']
        weather.fog_density = config['fog_density']
        weather.fog_distance = config['fog_distance']
        weather.fog_falloff = config['fog_falloff']
        weather.wetness = config['wetness']
        if 'snow_intensity' in config:
            weather.snow_intensity = config['snow_intensity']
        self.world.set_weather(weather)
    
    def set_weather(self, weather_name):
        """设置天气模式"""
        if weather_name in self.weather_configs:
            self._apply_weather(self.weather_configs[weather_name])
            self.current_weather = weather_name
            print(f"天气已切换为: {self.get_weather_name(weather_name)}")
            return True
        else:
            print(f"未知天气模式: {weather_name}")
            return False
    
    def tick(self):
        """每帧调用：定期刷新天气参数，防止CARLA自动改变天气"""
        self.frame_counter += 1
        if self.frame_counter % 50 == 0:
            self._apply_weather(self.weather_configs[self.current_weather])
    
    def cycle_weather(self):
        """循环切换天气"""
        weather_list = list(self.weather_configs.keys())
        current_index = weather_list.index(self.current_weather)
        next_index = (current_index + 1) % len(weather_list)
        return self.set_weather(weather_list[next_index])
    
    def get_weather_name(self, weather_key):
        """获取天气中文名"""
        names = {
            'sunny': '晴天',
            'cloudy': '多云',
            'rainy': '雨天',
            'stormy': '暴风雨',
            'snowy': '雪天',
            'foggy': '雾天',
            'night': '夜晚'
        }
        return names.get(weather_key, weather_key)
    
    def get_current_weather_display(self):
        """获取当前天气显示名称"""
        return self.get_weather_name(self.current_weather)


class LiDARManager:
    """LiDAR传感器管理器 - 实现障碍物检测和避障"""
    
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.lidar = None
        self.lidar_data = None
        self.min_distance = float('inf')  # 最近障碍物距离
        self.obstacle_detected = False
        self.warning_distance = 15.0  # 警告距离（米）
        self.stop_distance = 5.0  # 停止距离（米）
        
        # 初始化LiDAR传感器
        self._setup_lidar()
    
    def _setup_lidar(self):
        """设置LiDAR传感器"""
        blueprint_library = self.world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '500000')
        
        lidar_transform = carla.Transform(carla.Location(x=0.8, z=1.5))
        
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        
        self.lidar.listen(lambda data: self._process_lidar_data(data))
        
        print("LiDAR传感器已启用")
    
    def _process_lidar_data(self, data):
        """处理LiDAR点云数据"""
        self.lidar_data = data
        
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        
        front_points = []
        for point in points:
            x, y, z = point[0], point[1], point[2]
            
            angle = math.atan2(y, x) * 180 / math.pi
            if -45 < angle < 45 and z > -0.5 and z < 2.0:
                distance = math.sqrt(x**2 + y**2 + z**2)
                front_points.append(distance)
        
        if front_points:
            self.min_distance = min(front_points)
            self.obstacle_detected = self.min_distance < self.warning_distance
        else:
            self.min_distance = float('inf')
            self.obstacle_detected = False
    
    def get_min_distance(self):
        """获取最近障碍物距离"""
        return self.min_distance
    
    def is_obstacle_detected(self):
        """是否检测到障碍物"""
        return self.obstacle_detected
    
    def get_warning_level(self):
        """获取警告级别"""
        if self.min_distance < self.stop_distance:
            return 'danger'
        elif self.min_distance < self.warning_distance:
            return 'warning'
        else:
            return 'safe'
    
    def destroy(self):
        """销毁LiDAR传感器"""
        if self.lidar:
            self.lidar.destroy()
            print("LiDAR传感器已销毁")


class SimpleDrivingSystem:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.cameras = {}  # 存储多个相机
        self.controller = None
        self.camera_image = None
        self.current_view = 'third_person'  # 当前视角模式：'first_person', 'third_person', 'birdseye'
        self.weather_manager = None  # 天气管理器
        self.lidar_manager = None  # LiDAR传感器管理器

    def connect(self):
        """连接到CARLA服务器"""
        print("正在连接到CARLA服务器...")

        try:
            # 尝试多种连接方式
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)

            # 检查可用地图
            try:
                available_maps = self.client.get_available_maps()
                print(f"可用地图: {available_maps}")
            except Exception as map_e:
                print(f"获取地图列表失败: {map_e}")
                print("继续尝试加载地图...")

            # 加载地图
            self.world = self.client.load_world('Town01')
            print("地图加载成功")

            # 设置同步模式
            settings = self.world.get_settings()
            settings.synchronous_mode = False  # 先使用异步模式确保连接
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

            print("连接成功！")
            return True

        except UnicodeDecodeError as e:
            print(f"连接失败 - 编码错误: {e}")
            print("尝试重新连接...")
            # 尝试重新连接
            try:
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(10.0)
                self.world = self.client.load_world('Town01')
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                print("重新连接成功！")
                return True
            except Exception as re_e:
                print(f"重新连接也失败: {re_e}")
                print("请确保:")
                print("1. CARLA服务器正在运行")
                print("2. 服务器端口为2000")
                print("3. 地图Town01可用")
                return False
        except Exception as e:
            print(f"连接失败: {e}")
            print("请确保:")
            print("1. CARLA服务器正在运行")
            print("2. 服务器端口为2000")
            print("3. 地图Town01可用")
            return False

    def spawn_vehicle(self):
        """生成车辆 - 简化版本"""
        print("正在生成车辆...")

        try:
            # 获取蓝图库
            blueprint_library = self.world.get_blueprint_library()

            # 选择车辆蓝图
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            if not vehicle_bp:
                print("未找到特斯拉蓝图，尝试其他车辆...")
                vehicle_bp = blueprint_library.filter('vehicle.*')[0]

            vehicle_bp.set_attribute('color', '255,0,0')  # 红色

            # 获取出生点
            spawn_points = self.world.get_map().get_spawn_points()
            print(f"找到 {len(spawn_points)} 个出生点")

            if not spawn_points:
                print("没有可用的出生点！")
                return False

            # 选择第一个出生点
            spawn_point = spawn_points[0]

            # 尝试生成车辆
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

            if not self.vehicle:
                print("无法生成车辆，尝试清理现有车辆...")
                # 清理现有车辆
                for actor in self.world.get_actors().filter('vehicle.*'):
                    actor.destroy()
                time.sleep(0.5)

                # 再次尝试
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

            if self.vehicle:
                print(f"车辆生成成功！ID: {self.vehicle.id}")
                print(f"位置: {spawn_point.location}")

                # 禁用自动驾驶
                self.vehicle.set_autopilot(False)

                return True
            else:
                print("车辆生成失败")
                return False

        except Exception as e:
            print(f"生成车辆时出错: {e}")
            return False

    def setup_camera(self):
        """设置多个相机"""
        print("正在设置相机...")

        try:
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')

            # 设置相机属性
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')
            camera_bp.set_attribute('fov', '90')

            # 第一人称相机
            first_person_transform = carla.Transform(
                carla.Location(x=2.0, z=1.2),  # 驾驶座位置
                carla.Rotation(pitch=0.0)  # 平视
            )
            first_person_camera = self.world.spawn_actor(
                camera_bp, first_person_transform, attach_to=self.vehicle
            )
            first_person_camera.listen(lambda image: self.camera_callback(image, 'first_person'))
            self.cameras['first_person'] = first_person_camera

            # 第三人称相机
            third_person_transform = carla.Transform(
                carla.Location(x=-8.0, z=6.0),  # 在车辆后方上方
                carla.Rotation(pitch=-20.0)  # 向下看
            )
            third_person_camera = self.world.spawn_actor(
                camera_bp, third_person_transform, attach_to=self.vehicle
            )
            third_person_camera.listen(lambda image: self.camera_callback(image, 'third_person'))
            self.cameras['third_person'] = third_person_camera

            # 鸟瞰图相机
            birdseye_transform = carla.Transform(
                carla.Location(x=0.0, z=30.0),  # 车辆正上方30米
                carla.Rotation(pitch=-90.0)  # 垂直向下
            )
            birdseye_camera = self.world.spawn_actor(
                camera_bp, birdseye_transform, attach_to=self.vehicle
            )
            birdseye_camera.listen(lambda image: self.camera_callback(image, 'birdseye'))
            self.cameras['birdseye'] = birdseye_camera

            print("相机设置成功 - 已创建三个视角相机")
            return True

        except Exception as e:
            print(f"设置相机时出错: {e}")
            return False

    def camera_callback(self, image, view_mode=None):
        """相机数据回调"""
        try:
            # 只有当前视角的相机数据才会被使用
            if view_mode == self.current_view:
                # 转换图像数据
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                self.camera_image = array[:, :, :3]  # RGB通道
        except:
            pass

    def update_camera_view(self):
        """更新相机视角"""
        print(f"已切换到{self.get_view_name()}视角")

    def get_view_name(self):
        """获取视角名称"""
        view_names = {
            'first_person': 'First Person',
            'third_person': 'Third Person',
            'birdseye': 'Birds Eye'
        }
        return view_names.get(self.current_view, 'Unknown')

    def setup_controller(self):
        """设置控制器"""
        self.controller = SimpleController(self.world, self.vehicle)
        print("控制器设置完成")

    def run(self):
        """主运行循环"""
        print("\n" + "=" * 50)
        print("简化自动驾驶系统")
        print("=" * 50)

        # 连接服务器
        if not self.connect():
            return

        # 生成车辆
        if not self.spawn_vehicle():
            return

        # 设置相机
        if not self.setup_camera():
            # 即使相机失败也继续运行
            print("警告：相机设置失败，继续运行...")

        # 设置控制器
        self.setup_controller()

        # 等待一会儿让系统稳定
        print("系统初始化中...")
        time.sleep(2.0)

        # 设置天气
        weather = carla.WeatherParameters(
            cloudiness=30.0,
            precipitation=0.0,
            sun_altitude_angle=70.0
        )
        self.world.set_weather(weather)

        # 初始化天气管理器
        self.weather_manager = WeatherManager(self.world)
        self.weather_manager.set_weather('sunny')

        # 初始化LiDAR传感器
        self.lidar_manager = LiDARManager(self.world, self.vehicle)

        # 生成一些NPC车辆
        self.spawn_npc_vehicles(2)

        print("\n系统准备就绪！")
        print("控制指令:")
        print("  q - 退出程序")
        print("  r - 重置车辆")
        print("  s - 紧急停止")
        print("  x - 切换倒车/前进模式（速度为0时生效）")
        print("  v - 切换视角（第一人称/第三人称/鸟瞰图）")
        print("  w - 切换天气（晴天/多云/雨天/暴风雨/雪天/雾天/夜晚）")
        print("  + - 增加速度限制")
        print("  - - 减少速度限制")
        print("\n感知与避障系统已启用:")
        print("  - LiDAR检测范围: 50米")
        print("  - 警告距离: 15米")
        print("  - 自动刹车距离: 5米")
        print("\n开始自动驾驶...\n")

        frame_count = 0
        running = True

        try:
            while running:
                # 定期刷新天气，防止CARLA自动改变天气参数
                if self.weather_manager:
                    self.weather_manager.tick()

                # 获取车辆状态
                velocity = self.vehicle.get_velocity()
                speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6

                # 获取控制指令（现在返回4个值，原代码返回3个值）
                # throttle, brake, steer = self.controller.get_control()  # 原代码
                throttle, brake, steer, reverse = self.controller.get_control()  # 新代码

                # LiDAR避障控制
                if self.lidar_manager:
                    warning_level = self.lidar_manager.get_warning_level()
                    if warning_level == 'danger':
                        throttle = 0.0
                        brake = 1.0
                    elif warning_level == 'warning':
                        throttle = throttle * 0.3
                        brake = brake + 0.2

                # 应用控制
                control = carla.VehicleControl(
                    throttle=float(throttle),
                    brake=float(brake),
                    steer=float(steer),
                    hand_brake=False,
                    # reverse=False  # 原代码
                    reverse=reverse  # 新代码，支持倒车
                )
                self.vehicle.apply_control(control)

                # 更新显示
                if self.camera_image is not None:
                    display_img = self.camera_image.copy()

                    # 添加状态信息
                    cv2.putText(display_img, f"Speed: {speed:.1f} km/h",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)
                    cv2.putText(display_img, f"Throttle: {throttle:.2f}",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)
                    cv2.putText(display_img, f"Steer: {steer:.2f}",
                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)
                    cv2.putText(display_img, f"Frame: {frame_count}",
                                (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)
                    
                    # 显示倒车状态（新功能）
                    if self.controller.manual_reverse:
                        cv2.putText(display_img, "REVERSE MODE",
                                    (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 0, 255), 2)  # 红色显示
                    
                    # 显示当前视角模式
                    cv2.putText(display_img, f"View: {self.get_view_name()}",
                                (20, 240), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)  # 绿色显示
                    
                    # 显示当前天气
                    if self.weather_manager:
                        cv2.putText(display_img, f"Weather: {self.weather_manager.get_current_weather_display()}",
                                    (20, 280), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (255, 165, 0), 2)  # 橙色显示
                    
                    # 显示速度限制
                    speed_limit = self.controller.get_speed_limit()
                    cv2.putText(display_img, f"Speed Limit: {speed_limit:.0f} km/h",
                                (20, 320), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 0), 2)  # 青色显示
                    
                    # 显示LiDAR距离和警告
                    if self.lidar_manager:
                        min_dist = self.lidar_manager.get_min_distance()
                        warning_level = self.lidar_manager.get_warning_level()
                        
                        if warning_level == 'danger':
                            cv2.putText(display_img, f"⚠️ OBSTACLE! {min_dist:.1f}m",
                                        (20, 360), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, (0, 0, 255), 2)  # 红色警告
                        elif warning_level == 'warning':
                            cv2.putText(display_img, f"⚠️ Warning: {min_dist:.1f}m",
                                        (20, 360), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, (0, 255, 255), 2)  # 黄色警告
                        else:
                            cv2.putText(display_img, f"Distance: {min_dist:.1f}m",
                                        (20, 360), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, (0, 255, 0), 2)  # 绿色安全

                    cv2.imshow('Autonomous Driving - Simple Version', display_img)

                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("正在退出...")
                    running = False
                elif key == ord('r'):
                    self.reset_vehicle()
                elif key == ord('s'):
                    # 紧急停止
                    self.vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0, brake=1.0, hand_brake=True
                    ))
                    print("紧急停止")
                elif key == ord('x'):
                    # 切换倒车模式（只在速度接近0时允许切换）
                    if speed < 1.0:  # 速度小于1km/h时允许切换
                        self.controller.toggle_reverse()
                    else:
                        print("请先减速到接近停止（速度<1km/h）再切换倒车模式")
                elif key == ord('v'):
                    # 切换视角模式
                    view_modes = ['third_person', 'first_person', 'birdseye']
                    current_index = view_modes.index(self.current_view)
                    next_index = (current_index + 1) % len(view_modes)
                    self.current_view = view_modes[next_index]
                    self.update_camera_view()
                elif key == ord('w'):
                    # 切换天气模式
                    if self.weather_manager:
                        self.weather_manager.cycle_weather()
                elif key == ord('+') or key == ord('='):
                    # 增加速度限制
                    self.controller.increase_speed_limit()
                elif key == ord('-') or key == ord('_'):
                    # 减少速度限制
                    self.controller.decrease_speed_limit()

                frame_count += 1

                # 每100帧显示一次状态
                if frame_count % 100 == 0:
                    print(f"运行中... 帧数: {frame_count}, 速度: {speed:.1f} km/h")

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"运行错误: {e}")
        finally:
            self.cleanup()

    def spawn_npc_vehicles(self, count=2):
        """生成NPC车辆（简化）"""
        print(f"正在生成 {count} 辆NPC车辆...")

        try:
            blueprint_library = self.world.get_blueprint_library()
            spawn_points = self.world.get_map().get_spawn_points()

            npc_vehicles = []

            for i in range(min(count, len(spawn_points))):
                # 跳过主车辆的出生点
                if i == 0:
                    continue

                try:
                    # 随机选择车辆类型
                    vehicle_bps = list(blueprint_library.filter('vehicle.*'))
                    if vehicle_bps:
                        vehicle_bp = random.choice(vehicle_bps)

                        # 生成NPC
                        npc = self.world.try_spawn_actor(vehicle_bp, spawn_points[i])

                        if npc:
                            npc.set_autopilot(True)
                            npc_vehicles.append(npc)
                            print(f"生成NPC车辆 {len(npc_vehicles)}")
                except:
                    pass

            print(f"成功生成 {len(npc_vehicles)} 辆NPC车辆")

        except Exception as e:
            print(f"生成NPC车辆时出错: {e}")

    def reset_vehicle(self):
        """重置车辆位置"""
        print("重置车辆...")

        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_points:
            new_spawn_point = random.choice(spawn_points)
            self.vehicle.set_transform(new_spawn_point)
            print(f"车辆已重置到新位置: {new_spawn_point.location}")

            # 等待重置完成
            time.sleep(0.5)

    def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")

        # 清理LiDAR传感器
        if self.lidar_manager:
            try:
                self.lidar_manager.destroy()
            except:
                pass

        # 清理所有相机
        for view_mode, camera in self.cameras.items():
            if camera:
                try:
                    camera.stop()
                    camera.destroy()
                except:
                    pass

        if self.vehicle:
            try:
                self.vehicle.destroy()
            except:
                pass

        # 等待销毁完成
        time.sleep(1.0)

        cv2.destroyAllWindows()
        print("清理完成")


def main():
    """主函数"""
    print("自动驾驶系统 - 简化版本")
    print("确保CARLA服务器正在运行...")

    system = SimpleDrivingSystem()
    system.run()


if __name__ == "__main__":
    main()