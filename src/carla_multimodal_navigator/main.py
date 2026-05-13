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
        self.target_speed = 30.0  # km/h
        self.waypoint_distance = 5.0
        self.last_waypoint = None
        # 限速检测相关属性
        self.speed_limit = 30.0  # 默认限速 30 km/h
        self.speed_limit_detected = False  # 是否检测到限速标志
        # 车道保持辅助(LKA)相关属性
        self.lka_enabled = False  # LKA功能开关
        self.lka_active = False  # LKA是否正在工作
        self.lane_offset = 0.0  # 车道偏移量（-1到1，负数偏左，正数偏右）
        self.lka_steer = 0.0  # LKA计算的转向角度

    def detect_speed_limits(self, location, transform):
        """检测道路限速标志"""
        # 重置限速检测状态
        self.speed_limit_detected = False
        
        # 简单的限速检测逻辑，确保车辆能够根据限速调整速度
        # 每100米切换一次限速，以便测试
        distance = math.sqrt(location.x ** 2 + location.y ** 2)
        
        if distance < 100:
            self.speed_limit = 20.0  # 学校区域
            self.speed_limit_detected = True
        elif distance < 200:
            self.speed_limit = 40.0  # 普通道路
            self.speed_limit_detected = True
        else:
            self.speed_limit = 30.0  # 默认限速
            self.speed_limit_detected = False

    def get_control(self, speed, lka_enabled=False, lane_offset=0.0):
        """基于路点的简单控制，支持车道保持辅助"""
        # 更新LKA状态
        self.lka_enabled = lka_enabled
        self.lane_offset = lane_offset

        # 获取车辆状态
        location = self.vehicle.get_location()
        transform = self.vehicle.get_transform()

        # 检测限速标志
        self.detect_speed_limits(location, transform)

        # 获取路点
        waypoint = self.map.get_waypoint(location, project_to_road=True)

        if not waypoint:
            # 如果没有找到路点，返回保守控制
            return 0.3, 0.0, 0.0

        # 获取下一个路点
        next_waypoints = waypoint.next(self.waypoint_distance)

        if not next_waypoints:
            # 如果没有下一个路点，使用当前路点
            target_waypoint = waypoint
        else:
            target_waypoint = next_waypoints[0]

        self.last_waypoint = target_waypoint

        # 计算转向（基于路点）
        vehicle_yaw = math.radians(transform.rotation.yaw)
        target_loc = target_waypoint.transform.location

        # 计算相对位置
        dx = target_loc.x - location.x
        dy = target_loc.y - location.y

        local_x = dx * math.cos(vehicle_yaw) + dy * math.sin(vehicle_yaw)
        local_y = -dx * math.sin(vehicle_yaw) + dy * math.cos(vehicle_yaw)

        if abs(local_x) < 0.1:
            path_steer = 0.0
        else:
            angle = math.atan2(local_y, local_x)
            path_steer = max(-0.5, min(0.5, angle / 1.0))

        # 车道保持辅助(LKA)转向修正
        self.lka_steer = 0.0
        self.lka_active = False
        
        if self.lka_enabled and abs(lane_offset) > 0.05 and speed > 10:
            # LKA激活条件：功能开启、有明显偏移、车速大于10km/h
            self.lka_active = True
            # 根据车道偏移计算修正转向
            # lane_offset范围是-1到1，转换为转向角度
            self.lka_steer = -lane_offset * 0.3  # LKA转向系数
            self.lka_steer = max(-0.2, min(0.2, self.lka_steer))  # 限制最大修正量

        # 合并路径规划转向和LKA转向
        steer = path_steer + self.lka_steer
        steer = max(-0.5, min(0.5, steer))  # 限制在合理范围内

        # 速度控制
        # 使用检测到的限速作为目标速度
        current_target_speed = self.speed_limit if self.speed_limit_detected else self.target_speed
        
        if speed < current_target_speed * 0.8:
            throttle, brake = 0.6, 0.0
        elif speed > current_target_speed * 1.2:
            throttle, brake = 0.0, 0.3
        else:
            throttle, brake = 0.3, 0.0

        return throttle, brake, steer


class SimpleDrivingSystem:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.cameras = {}  # 多相机字典
        self.camera_images = {}  # 多相机图像
        self.speed_sensor = None
        self.controller = None
        self.camera_image = None
        self.vehicle_speed = 0.0  # km/h
        # 视角切换相关
        self.view_mode = 'single'  # 'all' = 全部视角, 'single' = 单一视角
        self.current_view_index = 5  # 当前选中的视角索引（第三人称视角）
        self.view_names = ['front', 'rear', 'left', 'right', 'birdview', 'third']
        # 车道保持辅助(LKA)相关
        self.lka_enabled = False  # LKA功能开关
        self.lane_offset = 0.0  # 车道偏移量（-1到1）
        self.lane_detected = False  # 是否检测到车道线
        self.lane_lines = []  # 检测到的车道线
        # 天气系统相关
        self.weather_system = {
            'current_weather': 'sunny',  # 当前天气：sunny, rainy, foggy
            'weather_intensity': 0.0,  # 天气强度（0.0-1.0）
            'last_weather_change': 0.0,  # 上次天气变化时间
            'visibility': 1000.0,  # 能见度（米）
            'road_conditions': 'dry'  # 道路状况：dry, wet, icy
        }
        self.auto_weather_change = True  # 自动天气变化开关
        self.is_day = True  # 白天/黑夜标志
        
        # 交通标志识别（TSR）相关
        self.tsr_enabled = True  # TSR功能开关
        self.detected_signs = []  # 当前检测到的标志列表
        self.sign_detection_history = []  # 标志检测历史
        self.last_sign_update = 0.0  # 上次更新时间

    def connect(self):
        """连接到CARLA服务器"""
        print("正在连接到CARLA服务器...")

        try:
            # 尝试多种连接方式
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)

            # 检查可用地图
            available_maps = self.client.get_available_maps()
            print(f"可用地图: {available_maps}")

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
        """设置多相机系统"""
        print("正在设置多相机系统...")

        try:
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')

            # 设置相机属性
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')
            camera_bp.set_attribute('fov', '90')

            # 相机位置配置
            camera_configs = {
                'front': carla.Transform(
                    carla.Location(x=2.0, z=1.5),  # 车辆前方
                    carla.Rotation(pitch=0.0)  # 水平向前
                ),
                'rear': carla.Transform(
                    carla.Location(x=-3.0, z=1.5),  # 车辆后方
                    carla.Rotation(pitch=0.0, yaw=180.0)  # 水平向后
                ),
                'left': carla.Transform(
                    carla.Location(x=0.0, y=1.5, z=1.5),  # 车辆左侧
                    carla.Rotation(pitch=0.0, yaw=-90.0)  # 水平向左
                ),
                'right': carla.Transform(
                    carla.Location(x=0.0, y=-1.5, z=1.5),  # 车辆右侧
                    carla.Rotation(pitch=0.0, yaw=90.0)  # 水平向右
                ),
                'birdview': carla.Transform(
                    carla.Location(x=0.0, z=15.0),  # 车辆上方
                    carla.Rotation(pitch=-90.0)  # 垂直向下
                ),
                'third': carla.Transform(
                    carla.Location(x=-8.0, z=4.0),  # 车辆后方上方
                    carla.Rotation(pitch=-20.0)  # 向下看
                )
            }

            # 生成所有相机
            for name, transform in camera_configs.items():
                try:
                    camera = self.world.spawn_actor(
                        camera_bp, transform, attach_to=self.vehicle
                    )
                    if camera:
                        self.cameras[name] = camera
                        self.camera_images[name] = None
                        # 设置回调函数
                        camera.listen(lambda image, name=name: self.camera_callback(image, name))
                        print(f"{name}相机设置成功")
                except Exception as e:
                    print(f"设置{name}相机时出错: {e}")

            # 保留原始相机引用
            if 'front' in self.cameras:
                self.camera = self.cameras['front']
                self.camera_image = self.camera_images['front']

            print(f"多相机系统设置成功，共{len(self.cameras)}个相机")
            return len(self.cameras) > 0

        except Exception as e:
            print(f"设置多相机系统时出错: {e}")
            return False

    def setup_speed_sensor(self):
        """设置速度传感器"""
        print("正在设置速度传感器...")

        try:
            blueprint_library = self.world.get_blueprint_library()
            # 尝试查找速度传感器蓝图
            speed_sensors = blueprint_library.filter('sensor.other.speed')
            
            if speed_sensors:
                # 使用专用速度传感器
                speed_bp = speed_sensors[0]
                print("使用专用速度传感器")
            else:
                # 使用IMU传感器作为速度传感器
                print("未找到专用速度传感器，使用IMU传感器")
                imu_sensors = blueprint_library.filter('sensor.other.imu')
                if not imu_sensors:
                    print("未找到速度或IMU传感器，将使用备用方法")
                    return False
                speed_bp = imu_sensors[0]

            # 设置传感器属性
            try:
                speed_bp.set_attribute('sensor_tick', '0.01')  # 100Hz
            except Exception as e:
                print(f"设置传感器属性时出错: {e}")

            # 传感器位置（车辆中心）
            sensor_transform = carla.Transform(
                carla.Location(x=0.0, y=0.0, z=1.0),  # 车辆中心上方
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            )

            # 生成传感器
            try:
                self.speed_sensor = self.world.spawn_actor(
                    speed_bp, sensor_transform, attach_to=self.vehicle
                )
            except Exception as e:
                print(f"生成速度传感器时出错: {e}")
                return False

            # 设置回调函数
            try:
                self.speed_sensor.listen(lambda data: self.speed_sensor_callback(data))
            except Exception as e:
                print(f"设置速度传感器回调时出错: {e}")
                # 继续执行，不设置回调

            print("速度传感器设置成功")
            return True

        except Exception as e:
            print(f"设置速度传感器时出错: {e}")
            return False

    def camera_callback(self, image, name='front'):
        """多相机数据回调"""
        try:
            # 转换图像数据
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            self.camera_images[name] = array[:, :, :3]  # RGB通道
            # 同时更新主相机图像
            if name == 'front':
                self.camera_image = self.camera_images[name]
        except:
            pass

    def speed_sensor_callback(self, data):
        """速度传感器数据回调"""
        try:
            # 从传感器数据中获取速度
            # 检查数据类型并相应处理
            if hasattr(data, 'velocity'):
                # 专用速度传感器
                velocity = data.velocity
                self.vehicle_speed = 0.8 * self.vehicle_speed + 0.2 * (math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6)
            elif hasattr(data, 'accelerometer'):
                # IMU传感器，使用车辆速度作为参考
                if self.vehicle:
                    velocity = self.vehicle.get_velocity()
                    self.vehicle_speed = 0.8 * self.vehicle_speed + 0.2 * (math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6)
            else:
                # 其他类型传感器
                if self.vehicle:
                    velocity = self.vehicle.get_velocity()
                    self.vehicle_speed = 0.8 * self.vehicle_speed + 0.2 * (math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6)
        except Exception as e:
            print(f"速度传感器回调错误: {e}")
            # 回退到直接获取车辆速度
            if self.vehicle:
                try:
                    velocity = self.vehicle.get_velocity()
                    self.vehicle_speed = 0.8 * self.vehicle_speed + 0.2 * (math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6)
                except:
                    pass

    def detect_lane_lines(self):
        """使用OpenCV检测车道线并计算车道偏移量"""
        if 'front' not in self.camera_images or self.camera_images['front'] is None:
            self.lane_detected = False
            self.lane_offset = 0.0
            self.lane_lines = []
            return

        image = self.camera_images['front'].copy()
        height, width = image.shape[:2]

        # 感兴趣区域（ROI）- 只处理图像下半部分
        roi_height = int(height * 0.4)
        roi_y_start = height - roi_height
        roi = image[roi_y_start:height, :]

        # 转换为灰度图像
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # 高斯模糊
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny边缘检测
        edges = cv2.Canny(blur, 50, 150)

        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)

        self.lane_lines = []
        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 计算斜率
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # 根据斜率判断是左车道线还是右车道线
                    if abs(slope) > 0.3:  # 过滤掉接近水平的线
                        if slope < 0:
                            # 左车道线（从右上到左下）
                            left_lines.append((x1, y1 + roi_y_start, x2, y2 + roi_y_start))
                        else:
                            # 右车道线（从左上到右下）
                            right_lines.append((x1, y1 + roi_y_start, x2, y2 + roi_y_start))

        # 如果检测到车道线
        if left_lines or right_lines:
            self.lane_detected = True
            
            # 计算车道线的平均位置
            left_x_avg = 0
            right_x_avg = width
            
            if left_lines:
                left_x_avg = sum((x1 + x2) / 2 for x1, y1, x2, y2 in left_lines) / len(left_lines)
            
            if right_lines:
                right_x_avg = sum((x1 + x2) / 2 for x1, y1, x2, y2 in right_lines) / len(right_lines)
            
            # 计算车道中心和车辆位置
            lane_center = (left_x_avg + right_x_avg) / 2
            vehicle_center = width / 2
            
            # 计算车道偏移量（-1到1）
            lane_width = right_x_avg - left_x_avg if (right_x_avg - left_x_avg) > 50 else width
            self.lane_offset = (vehicle_center - lane_center) / (lane_width / 2)
            self.lane_offset = max(-1.0, min(1.0, self.lane_offset))
            
            # 保存检测到的车道线
            self.lane_lines = left_lines + right_lines
        else:
            self.lane_detected = False
            self.lane_offset = 0.0

    def draw_lane_lines(self, image):
        """在图像上绘制车道线"""
        if self.lane_lines:
            for x1, y1, x2, y2 in self.lane_lines:
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        
        # 如果检测到车道，绘制车道中心和偏移指示
        if self.lane_detected:
            height, width = image.shape[:2]
            # 绘制车道中心虚线
            lane_center_x = int(width / 2 + self.lane_offset * 50)
            cv2.line(image, (lane_center_x, height-200), (lane_center_x, height-50), (0, 255, 255), 2, cv2.LINE_AA)
            
            # 绘制车辆中心线
            cv2.line(image, (int(width/2), height-100), (int(width/2), height-50), (255, 0, 0), 2, cv2.LINE_AA)

    def setup_controller(self):
        """设置控制器"""
        self.controller = SimpleController(self.world, self.vehicle)
        print("控制器设置完成")

    def set_weather(self, weather_type, intensity=0.5):
        """手动设置天气"""
        import time
        
        current_time = time.time()
        
        # 更新天气信息
        self.weather_system['current_weather'] = weather_type
        self.weather_system['weather_intensity'] = intensity
        self.weather_system['last_weather_change'] = current_time
        
        # 更新能见度
        if weather_type == 'sunny':
            self.weather_system['visibility'] = 1000.0
            self.weather_system['road_conditions'] = 'dry'
        elif weather_type == 'rainy':
            self.weather_system['visibility'] = 500.0 * (1 - intensity * 0.5)
            self.weather_system['road_conditions'] = 'wet'
        elif weather_type == 'foggy':
            self.weather_system['visibility'] = 50.0
            self.weather_system['road_conditions'] = 'dry'
        
        print(f"天气设置为: {weather_type}，强度: {intensity:.2f}，能见度: {self.weather_system['visibility']:.1f}米，路面: {self.weather_system['road_conditions']}")
        
        # 在CARLA中设置天气
        if self.world:
            try:
                weather = carla.WeatherParameters()
                if weather_type == 'sunny':
                    weather = carla.WeatherParameters.ClearNoon if self.is_day else carla.WeatherParameters.ClearNight
                elif weather_type == 'rainy':
                    weather = carla.WeatherParameters.HardRainNoon if self.is_day else carla.WeatherParameters.HardRainNight
                elif weather_type == 'foggy':
                    weather.cloudiness = 70.0
                    weather.precipitation = 0.0
                    weather.precipitation_deposits = 0.0
                    weather.fog_distance = 50.0
                    weather.fog_density = intensity * 0.95
                    weather.wetness = 0.0
                    weather.sun_altitude_angle = 20.0 if self.is_day else -10.0
                    weather.sun_azimuth_angle = 180.0
                self.world.set_weather(weather)
            except Exception as e:
                print(f"设置天气失败: {e}")

    def toggle_day_night(self):
        """切换白天/黑夜"""
        self.is_day = not self.is_day
        time_of_day = "白天" if self.is_day else "黑夜"
        print(f"切换到{time_of_day}模式")
        
        # 在CARLA中设置光照
        if self.world:
            try:
                weather = carla.WeatherParameters()
                weather_type = self.weather_system['current_weather']
                
                if weather_type == 'sunny':
                    weather = carla.WeatherParameters.ClearNoon if self.is_day else carla.WeatherParameters.ClearNight
                elif weather_type == 'rainy':
                    weather = carla.WeatherParameters.HardRainNoon if self.is_day else carla.WeatherParameters.HardRainNight
                elif weather_type == 'foggy':
                    weather.cloudiness = 70.0
                    weather.precipitation = 0.0
                    weather.precipitation_deposits = 0.0
                    weather.fog_distance = 50.0
                    weather.fog_density = self.weather_system['weather_intensity'] * 0.95
                    weather.wetness = 0.0
                    weather.sun_altitude_angle = 20.0 if self.is_day else -10.0
                    weather.sun_azimuth_angle = 180.0
                self.world.set_weather(weather)
            except Exception as e:
                print(f"切换白天黑夜失败: {e}")

    def update_weather(self):
        """自动更新天气"""
        import time
        current_time = time.time()
        
        # 检查是否需要自动切换天气
        if self.auto_weather_change and current_time - self.weather_system['last_weather_change'] > random.uniform(30, 60):
            # 随机选择天气
            weather_types = ['sunny', 'rainy', 'foggy']
            new_weather = random.choice(weather_types)
            self.set_weather(new_weather, random.uniform(0.3, 0.8))

    def detect_traffic_signs(self):
        """模拟交通标志检测（基于CARLA传感器或模拟）"""
        import time
        current_time = time.time()
        
        if not self.tsr_enabled:
            return
        
        # 每隔一定时间随机检测标志（模拟真实检测）
        if current_time - self.last_sign_update > random.uniform(5, 15):
            self.last_sign_update = current_time
            
            # 交通标志类型定义
            sign_types = [
                {'type': 'speed_limit', 'value': 30, 'color': (0, 255, 0)},
                {'type': 'speed_limit', 'value': 50, 'color': (0, 255, 255)},
                {'type': 'speed_limit', 'value': 70, 'color': (0, 128, 255)},
                {'type': 'no_overtaking', 'value': None, 'color': (255, 0, 0)},
                {'type': 'overtaking_allowed', 'value': None, 'color': (0, 255, 0)},
                {'type': 'stop', 'value': None, 'color': (255, 0, 0)},
                {'type': 'yield', 'value': None, 'color': (255, 255, 0)},
                {'type': 'no_entry', 'value': None, 'color': (255, 0, 0)},
                {'type': 'one_way', 'value': 'right', 'color': (255, 255, 0)},
                {'type': 'school_zone', 'value': None, 'color': (255, 0, 255)}
            ]
            
            # 随机选择一个标志（模拟检测到标志）
            if random.random() > 0.3:  # 70%概率检测到标志
                detected_sign = random.choice(sign_types)
                detected_sign['timestamp'] = current_time
                
                # 添加到检测历史
                self.sign_detection_history.append(detected_sign)
                
                # 保留最近5个检测记录
                if len(self.sign_detection_history) > 5:
                    self.sign_detection_history.pop(0)
                
                # 更新当前检测到的标志（只保留最新的限速标志和其他重要标志）
                self.detected_signs = []
                
                # 获取最新的限速标志
                speed_limit_sign = None
                for sign in reversed(self.sign_detection_history):
                    if sign['type'] == 'speed_limit':
                        speed_limit_sign = sign
                        break
                
                if speed_limit_sign:
                    self.detected_signs.append(speed_limit_sign)
                
                # 获取最新的其他重要标志
                other_sign = None
                for sign in reversed(self.sign_detection_history):
                    if sign['type'] != 'speed_limit':
                        other_sign = sign
                        break
                
                if other_sign and other_sign['timestamp'] > current_time - 10:
                    self.detected_signs.append(other_sign)
                
                # 打印检测信息
                sign_name = self.get_sign_name(detected_sign)
                print(f"TSR: 检测到交通标志 - {sign_name}")
    
    def get_sign_name(self, sign):
        """获取标志名称（英文缩写，避免中文显示问题）"""
        type_names = {
            'speed_limit': f"{sign['value']}km/h",
            'no_overtaking': 'NO PASS',
            'overtaking_allowed': 'PASS OK',
            'stop': 'STOP',
            'yield': 'YIELD',
            'no_entry': 'NO ENTRY',
            'one_way': 'ONE WAY',
            'school_zone': 'SCHOOL'
        }
        return type_names.get(sign['type'], sign['type'])
    
    def draw_traffic_signs(self, image):
        """在图像上绘制交通标志"""
        if not self.tsr_enabled or not self.detected_signs:
            return
        
        height, width = image.shape[:2]
        margin = 15
        sign_size = 60
        spacing = 10
        
        # 在右上角绘制检测到的标志
        start_x = width - margin - sign_size
        start_y = margin
        
        for i, sign in enumerate(self.detected_signs):
            # 计算位置
            x = start_x
            y = start_y + i * (sign_size + spacing)
            
            # 绘制标志背景
            cv2.rectangle(image, (x, y), (x + sign_size, y + sign_size), (0, 0, 0), -1)
            cv2.rectangle(image, (x, y), (x + sign_size, y + sign_size), sign['color'], 2)
            
            # 绘制标志内容
            center_x = x + sign_size // 2
            center_y = y + sign_size // 2
            
            if sign['type'] == 'speed_limit':
                # 限速标志：圆形背景 + 数字
                cv2.circle(image, (center_x, center_y), sign_size // 3, sign['color'], -1)
                cv2.putText(image, str(sign['value']), (center_x - 12, center_y + 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            elif sign['type'] == 'no_overtaking':
                # 禁止超车标志：红圈 + 斜杠
                cv2.circle(image, (center_x, center_y), sign_size // 3, sign['color'], 2)
                cv2.line(image, (x + 10, y + 10), (x + sign_size - 10, y + sign_size - 10), sign['color'], 3)
            elif sign['type'] == 'stop':
                # 停车标志：八边形
                pts = []
                for j in range(8):
                    angle = j * 45 + 22.5
                    px = center_x + (sign_size // 3) * math.cos(math.radians(angle))
                    py = center_y + (sign_size // 3) * math.sin(math.radians(angle))
                    pts.append((int(px), int(py)))
                cv2.fillPoly(image, [np.array(pts)], sign['color'])
                cv2.putText(image, 'S', (center_x - 6, center_y + 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            elif sign['type'] == 'yield':
                # 让行标志：三角形
                pts = [
                    (center_x, y + 5),
                    (x + sign_size - 5, y + sign_size - 5),
                    (x + 5, y + sign_size - 5)
                ]
                cv2.fillPoly(image, [np.array(pts)], sign['color'])
                cv2.putText(image, 'Y', (center_x - 4, center_y + 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                # 其他标志：显示类型名称
                cv2.putText(image, self.get_sign_name(sign)[:2], (center_x - 10, center_y + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, sign['color'], 2)

    def create_multi_view_display(self, speed, throttle, steer):
        """创建多视角显示"""
        if self.view_mode == 'single':
            # 单一视角模式
            view_name = self.view_names[self.current_view_index]
            if view_name in self.camera_images and self.camera_images[view_name] is not None:
                display_img = self.camera_images[view_name].copy()
                
                # 在前视图上绘制车道线和交通标志
                if view_name == 'front':
                    self.draw_lane_lines(display_img)
                    self.draw_traffic_signs(display_img)
                
                # 添加状态信息
                cv2.putText(display_img, f"View: {view_name.upper()}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                # 添加速度和限速信息
                speed_color = (0, 255, 0) if speed <= self.controller.speed_limit else (0, 0, 255)
                cv2.putText(display_img, f"Speed: {speed:.1f} km/h",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, speed_color, 2)
                cv2.putText(display_img, f"Limit: {self.controller.speed_limit:.0f} km/h",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)
                cv2.putText(display_img, f"Throttle: {throttle:.2f}",
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                cv2.putText(display_img, f"Steer: {steer:.2f}",
                            (20, 180), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                # 添加速度传感器状态
                speed_sensor_status = "Active" if self.speed_sensor else "Inactive"
                cv2.putText(display_img, f"Speed Sensor: {speed_sensor_status}",
                            (20, 220), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)
                # 添加车道保持辅助状态
                lka_status = "ON" if self.lka_enabled else "OFF"
                lka_color = (0, 255, 0) if self.lka_enabled else (0, 165, 255)
                cv2.putText(display_img, f"LKA: {lka_status}",
                            (20, 260), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, lka_color, 2)
                if self.lka_enabled:
                    lane_status = "Detected" if self.lane_detected else "No Lane"
                    lane_color = (0, 255, 0) if self.lane_detected else (0, 0, 255)
                    cv2.putText(display_img, f"Lane: {lane_status}",
                                (20, 285), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, lane_color, 2)
                    cv2.putText(display_img, f"Offset: {self.lane_offset:.2f}",
                                (20, 310), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 0), 2)
                
                return display_img
        else:
            # 全部视角模式 - 2x3网格
            grid_width = 1280
            grid_height = 960
            display_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            display_img[:] = (50, 50, 50)  # 深灰色背景
            
            # 定义视角布局
            view_layouts = [
                (0, 0, 'front'),
                (0, 1, 'rear'),
                (1, 0, 'left'),
                (1, 1, 'right'),
                (0, 2, 'birdview'),
                (1, 2, 'third')
            ]
            
            cell_width = int(grid_width / 3)
            cell_height = int(grid_height / 2)
            
            # 绘制所有视角
            for row, col, view_name in view_layouts:
                x_start = col * cell_width
                y_start = row * cell_height
                x_end = x_start + cell_width
                y_end = y_start + cell_height
                
                # 创建视角图像
                view_img = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
                view_img[:] = (40, 40, 40)
                
                # 添加视角标签
                cv2.putText(view_img, view_name.upper(),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                
                # 绘制视角内容
                if view_name in self.camera_images and self.camera_images[view_name] is not None:
                    camera_img = self.camera_images[view_name]
                    # 缩放图像以适应单元格
                    resized_img = cv2.resize(camera_img, (cell_width - 20, cell_height - 40))
                    view_img[40:40+resized_img.shape[0], 10:10+resized_img.shape[1]] = resized_img
                
                # 放置到网格中
                display_img[y_start:y_end, x_start:x_end] = view_img
            
            # 添加模式指示
            cv2.putText(display_img, "SPACE: toggle mode | 1-5: select view | T: next view",
                        (300, grid_height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
            
            # 添加速度状态指示
            speed_color = (0, 255, 0) if speed <= self.controller.speed_limit else (0, 0, 255)
            cv2.putText(display_img, f"Speed: {speed:.1f} km/h | Limit: {self.controller.speed_limit:.0f} km/h",
                        (400, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, speed_color, 2)
            # 添加速度传感器状态
            speed_sensor_status = "Active" if self.speed_sensor else "Inactive"
            sensor_color = (0, 255, 255) if self.speed_sensor else (0, 165, 255)
            cv2.putText(display_img, f"Speed Sensor: {speed_sensor_status}",
                        (400, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, sensor_color, 2)
            
            return display_img
        
        # 如果没有图像，返回空图像
        return np.zeros((480, 640, 3), dtype=np.uint8)

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

        # 设置速度传感器
        if not self.setup_speed_sensor():
            # 即使传感器失败也继续运行，但会使用备用方法
            print("警告：速度传感器设置失败，将使用备用速度获取方法...")

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

        # 生成一些NPC车辆
        self.spawn_npc_vehicles(2)

        print("\n系统准备就绪！")
        print("控制指令:")
        print("  q - 退出程序")
        print("  r - 重置车辆")
        print("  s - 紧急停止")
        print("  l - 切换车道保持辅助(LKA)")
        print("  k - 切换交通标志识别(TSR)")
        print("  w - 切换自动天气变化")
        print("  7 - 设置晴天")
        print("  8 - 设置雨天")
        print("  9 - 设置雾天")
        print("  0 - 切换白天/黑夜")
        print("  空格键 - 切换全部/单一视角模式")
        print("  1-6 - 选择视角 (仅在单一视角模式下)")
        print("  t - 切换到下一个视角 (仅在单一视角模式下)")
        print("\n视角: 1-前视 2-后视 3-左视 4-右视 5-鸟瞰 6-第三人称")
        print("\n天气控制: W-切换自动天气 7-晴天 8-雨天 9-雾天 0-白天/黑夜")
        print("\n开始自动驾驶...\n")

        frame_count = 0
        running = True

        try:
            while running:
                # 获取速度数据（优先使用传感器数据）
                if self.vehicle_speed > 0:
                    speed = self.vehicle_speed
                else:
                    # 备用方法：直接从车辆获取速度
                    velocity = self.vehicle.get_velocity()
                    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6

                # 检测车道线（使用前视摄像头）
                self.detect_lane_lines()

                # 自动更新天气
                self.update_weather()

                # 检测交通标志
                self.detect_traffic_signs()

                # 获取控制指令（包含LKA辅助）
                throttle, brake, steer = self.controller.get_control(
                    speed, 
                    lka_enabled=self.lka_enabled, 
                    lane_offset=self.lane_offset
                )

                # 应用控制
                control = carla.VehicleControl(
                    throttle=float(throttle),
                    brake=float(brake),
                    steer=float(steer),
                    hand_brake=False,
                    reverse=False
                )
                self.vehicle.apply_control(control)

                # 创建多视角显示
                display_img = self.create_multi_view_display(speed, throttle, steer)
                cv2.imshow('Autonomous Driving - Multi View (SPACE: toggle mode | 1-5: select view)', display_img)

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
                elif key == ord('l'):
                    # 切换车道保持辅助(LKA)
                    self.lka_enabled = not self.lka_enabled
                    status = "开启" if self.lka_enabled else "关闭"
                    print(f"车道保持辅助(LKA)已{status}")
                elif key == ord('k'):
                    # 切换交通标志识别(TSR)
                    self.tsr_enabled = not self.tsr_enabled
                    status = "开启" if self.tsr_enabled else "关闭"
                    print(f"交通标志识别(TSR)已{status}")
                elif key == ord('w'):
                    # 切换自动天气变化
                    self.auto_weather_change = not self.auto_weather_change
                    status = "开启" if self.auto_weather_change else "关闭"
                    print(f"自动天气变化已{status}")
                elif key == ord('7'):
                    # 设置晴天
                    self.set_weather('sunny', 0.5)
                elif key == ord('8'):
                    # 设置雨天
                    self.set_weather('rainy', 0.5)
                elif key == ord('9'):
                    # 设置雾天
                    self.set_weather('foggy', 0.5)
                elif key == ord('0'):
                    # 切换白天/黑夜
                    self.toggle_day_night()
                elif key == 32:  # 空格键
                    # 切换视角模式
                    if self.view_mode == 'all':
                        self.view_mode = 'single'
                        print(f"切换到单一视角模式: {self.view_names[self.current_view_index].upper()}")
                    else:
                        self.view_mode = 'all'
                        print("切换到全部视角模式")
                elif key == ord('t'):
                    # 切换到下一个视角
                    if self.view_mode == 'single':
                        self.current_view_index = (self.current_view_index + 1) % len(self.view_names)
                        print(f"切换到视角: {self.view_names[self.current_view_index].upper()}")
                elif key >= ord('1') and key <= ord('6'):
                    # 数字键选择视角
                    if self.view_mode == 'single':
                        self.current_view_index = key - ord('1')
                        print(f"选择视角: {self.view_names[self.current_view_index].upper()}")

                frame_count += 1

                # 每100帧显示一次状态
                if frame_count % 100 == 0:
                    print(f"运行中... 帧数: {frame_count}, 速度: {speed:.1f} km/h, 视角: {self.view_mode}")

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

        # 清理所有相机
        for name, camera in self.cameras.items():
            if camera:
                try:
                    camera.stop()
                    camera.destroy()
                    print(f"{name}相机已清理")
                except:
                    pass

        # 清理原始相机
        if self.camera and self.camera not in self.cameras.values():
            try:
                self.camera.stop()
                self.camera.destroy()
            except:
                pass

        if self.speed_sensor:
            try:
                self.speed_sensor.stop()
                self.speed_sensor.destroy()
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