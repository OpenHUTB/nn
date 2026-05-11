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
        # self.target_speed = 30.0  # km/h，原速度限制
        self.target_speed = 50.0  # km/h，增加最高速度限制
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


class SimpleDrivingSystem:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.cameras = {}  # 存储多个相机
        self.controller = None
        self.camera_image = None
        self.current_view = 'third_person'  # 当前视角模式：'first_person', 'third_person', 'birdseye'
        self.current_map = 'Town01'  # 当前地图
        self.available_maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07']  # 可用地图列表
        self.current_weather = 'clear'  # 当前天气
        # 简化天气预设，使用肯定存在的天气类型
        self.weather_presets = {
            'clear': carla.WeatherParameters.ClearNoon,
            'rain': carla.WeatherParameters.HardRainNoon,
            'cloudy': carla.WeatherParameters.CloudyNoon,
            'wet': carla.WeatherParameters.WetNoon
        }  # 天气预设
        self.car_colors = [
            (255, 0, 0),      # 红色
            (0, 0, 255),      # 蓝色
            (0, 255, 0),      # 绿色
            (255, 255, 0),    # 黄色
            (255, 0, 255),    # 品红色
            (0, 255, 255),    # 青色
            (128, 0, 128),    # 紫色
            (255, 165, 0),    # 橙色
            (128, 128, 128),  # 灰色
            (255, 255, 255)   # 白色
        ]  # 车辆颜色列表
        self.current_color_index = 0  # 当前颜色索引
        self.screenshot_dir = 'screenshots'  # 截图保存目录
        self.is_night_mode = False  # 夜晚模式标志
        self.vehicle_models = [  # 车辆款式列表
            'vehicle.tesla.model3',      # 特斯拉 Model3
            'vehicle.chevrolet.impala',  # 雪佛兰 Impala
            'vehicle.ford.mustang',      # 福特 Mustang
            'vehicle.mercedes.coupe',    # 奔驰 Coupe
            'vehicle.nissan.patrol',     # 日产 Patrol
        ]  # 车辆款式列表
        self.current_model_index = 0  # 当前车辆款式索引
        self.show_trajectory = False  # 轨迹显示标志
        self.trajectory_points = deque(maxlen=500)  # 轨迹点集合
        self.map_img = None  # 存储地图图像
        self.map_width = 800  # 地图图像宽度
        self.map_height = 800  # 地图图像高度

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

    def generate_topology_map(self):
        """生成道路拓扑地图图像"""
        try:
            # 创建黑色背景
            self.map_img = np.zeros((self.map_width, self.map_height, 3), dtype=np.uint8)

            # 获取地图拓扑
            map = self.world.get_map()
            topology = map.get_topology()

            # 计算道路节点的范围
            all_x = []
            all_y = []
            for waypoint in map.generate_waypoints(2.0):
                all_x.append(waypoint.transform.location.x)
                all_y.append(waypoint.transform.location.y)

            if not all_x or not all_y:
                print("无法获取道路拓扑信息")
                return

            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            # 坐标缩放比例
            scale_x = (self.map_width - 100) / (max_x - min_x) if max_x != min_x else 1
            scale_y = (self.map_height - 100) / (max_y - min_y) if max_y != min_y else 1
            self.map_scale = min(scale_x, scale_y) * 0.8
            self.map_offset_x = min_x
            self.map_offset_y = min_y

            # 绘制道路
            for waypoint in map.generate_waypoints(2.0):
                wp_x = int((waypoint.transform.location.x - self.map_offset_x) * self.map_scale + 50)
                wp_y = int((waypoint.transform.location.y - self.map_offset_y) * self.map_scale + 50)

                # 绘制道路点（灰色）
                cv2.circle(self.map_img, (wp_x, wp_y), 1, (80, 80, 80), -1)

            print("道路拓扑地图生成成功")
        except Exception as e:
            print(f"生成拓扑地图时出错: {e}")

    def world_to_map(self, location):
        """将世界坐标转换为地图图像坐标"""
        try:
            x = int((location.x - self.map_offset_x) * self.map_scale + 50)
            y = int((location.y - self.map_offset_y) * self.map_scale + 50)
            return (x, y)
        except:
            return (400, 400)

    def switch_map(self):
        """切换到下一个地图"""
        try:
            # 停止所有相机
            for view_mode, camera in self.cameras.items():
                if camera:
                    try:
                        camera.stop()
                        camera.destroy()
                    except:
                        pass
            self.cameras.clear()

            # 销毁车辆
            if self.vehicle:
                try:
                    self.vehicle.destroy()
                except:
                    pass
                self.vehicle = None
            
            # 等待清理完成
            time.sleep(1.0)

            # 切换到下一个地图
            current_index = self.available_maps.index(self.current_map)
            next_index = (current_index + 1) % len(self.available_maps)
            new_map = self.available_maps[next_index]

            print(f"正在加载地图: {new_map}...")
            
            # 完全重新连接CARLA客户端
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            
            # 加载新地图
            self.world = self.client.load_world(new_map)
            self.current_map = new_map
            
            # 等待地图完全加载
            time.sleep(3.0)
            
            # 重新生成车辆
            if not self.spawn_vehicle():
                raise Exception("车辆生成失败")
            
            # 重新设置相机
            if not self.setup_camera():
                raise Exception("相机设置失败")
            
            # 重新设置控制器
            self.setup_controller()
            
            # 重新生成NPC车辆
            self.spawn_npc_vehicles(2)
            
            # 应用当前天气
            self.set_weather(self.current_weather)

            # 重新生成道路拓扑地图
            self.generate_topology_map()

            print(f"地图切换成功: {self.current_map}")
            
        except Exception as e:
            print(f"切换地图时出错: {e}")
            # 尝试重新加载Town01作为备份
            try:
                print("正在恢复到Town01...")
                self.current_map = 'Town01'
                time.sleep(1.0)
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(10.0)
                self.world = self.client.load_world(self.current_map)
                time.sleep(3.0)
                self.spawn_vehicle()
                self.setup_camera()
                self.setup_controller()
                self.set_weather(self.current_weather)
                print("已恢复到Town01")
            except Exception as e2:
                print(f"恢复失败: {e2}")

    def set_weather(self, weather_type):
        """设置天气"""
        try:
            if weather_type in self.weather_presets:
                weather = self.weather_presets[weather_type]
                self.world.set_weather(weather)
                self.current_weather = weather_type
                print(f"天气设置成功: {weather_type}")
                return True
            else:
                print(f"无效的天气类型: {weather_type}")
                return False
        except Exception as e:
            print(f"设置天气时出错: {e}")
            return False

    def switch_weather(self):
        """切换到下一个天气"""
        try:
            weather_types = list(self.weather_presets.keys())
            current_index = weather_types.index(self.current_weather)
            next_index = (current_index + 1) % len(weather_types)
            next_weather = weather_types[next_index]
            self.set_weather(next_weather)
        except Exception as e:
            print(f"切换天气时出错: {e}")

    def toggle_night_mode(self):
        """切换夜晚模式并自动打开/关闭近光灯"""
        try:
            self.is_night_mode = not self.is_night_mode
            
            if self.is_night_mode:
                # 切换到夜晚模式
                night_weather = carla.WeatherParameters(
                    cloudiness=80.0,
                    precipitation=0.0,
                    sun_altitude_angle=-30.0,  # 负角度表示夜晚
                    fog_density=30.0,
                    fog_distance=50.0
                )
                self.world.set_weather(night_weather)
                
                # 打开近光灯
                if self.vehicle:
                    self.vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.LowBeam))
                
                print("已切换到夜晚模式，近光灯已打开")
            else:
                # 切换回白天模式（使用当前天气）
                self.set_weather(self.current_weather)
                
                # 关闭近光灯
                if self.vehicle:
                    self.vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.NONE))
                
                print("已切换到白天模式，近光灯已关闭")
                
        except Exception as e:
            print(f"切换夜晚模式时出错: {e}")

    def switch_color(self):
        """切换车辆颜色"""
        try:
            if self.vehicle:
                # 获取当前车辆位置和方向
                transform = self.vehicle.get_transform()
                
                # 切换到下一个颜色
                self.current_color_index = (self.current_color_index + 1) % len(self.car_colors)
                color = self.car_colors[self.current_color_index]
                
                # 获取颜色名称
                color_names = ['Red', 'Blue', 'Green', 'Yellow', 'Magenta', 'Cyan', 'Purple', 'Orange', 'Gray', 'White']
                color_name = color_names[self.current_color_index]
                
                # 停止相机
                for view_mode, camera in self.cameras.items():
                    if camera:
                        try:
                            camera.stop()
                            camera.destroy()
                        except:
                            pass
                self.cameras.clear()
                
                # 销毁当前车辆
                self.vehicle.destroy()
                self.vehicle = None
                
                # 创建新车辆蓝图
                blueprint_library = self.world.get_blueprint_library()
                vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
                if not vehicle_bp:
                    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
                
                # 设置新颜色
                vehicle_bp.set_attribute('color', f'{color[0]},{color[1]},{color[2]}')
                
                # 首先尝试在相同位置生成新车辆
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
                
                # 如果失败，尝试使用出生点
                if not self.vehicle:
                    spawn_points = self.world.get_map().get_spawn_points()
                    for spawn_point in spawn_points[:5]:  # 尝试前5个出生点
                        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                        if self.vehicle:
                            print("车辆已移动到新位置")
                            break
                
                if self.vehicle:
                    # 禁用自动驾驶
                    self.vehicle.set_autopilot(False)
                    
                    # 重新设置相机
                    self.setup_camera()
                    
                    # 重新设置控制器
                    self.setup_controller()
                    
                    print(f"车辆颜色已切换: {color_name}")
                else:
                    print("无法生成新车辆，颜色切换失败")
                    # 重置颜色索引
                    self.current_color_index = (self.current_color_index - 1) % len(self.car_colors)
                    # 尝试恢复车辆
                    self.spawn_vehicle()
            else:
                print("车辆不存在，无法切换颜色")
        except Exception as e:
            print(f"切换车辆颜色时出错: {e}")
            # 重置颜色索引
            self.current_color_index = (self.current_color_index - 1) % len(self.car_colors)
            # 尝试恢复车辆
            if not self.vehicle:
                self.spawn_vehicle()

    def switch_vehicle_model(self):
        """切换车辆款式"""
        try:
            if self.vehicle:
                transform = self.vehicle.get_transform()
                
                # 停止所有相机
                for view_mode, camera in self.cameras.items():
                    if camera:
                        try:
                            camera.stop()
                            camera.destroy()
                        except:
                            pass
                self.cameras.clear()
                
                # 销毁当前车辆
                self.vehicle.destroy()
                self.vehicle = None
                
                # 切换到下一个车辆款式
                self.current_model_index = (self.current_model_index + 1) % len(self.vehicle_models)
                next_model = self.vehicle_models[self.current_model_index]
                
                # 获取蓝图库
                blueprint_library = self.world.get_blueprint_library()
                vehicle_bp = blueprint_library.find(next_model)
                
                # 如果找不到指定车型，随机选择一个
                if not vehicle_bp:
                    all_vehicles = blueprint_library.filter('vehicle.*')
                    if all_vehicles:
                        vehicle_bp = random.choice(list(all_vehicles))
                        print(f"未找到 {next_model}，使用随机车辆: {vehicle_bp.id}")
                
                if vehicle_bp:
                    # 设置当前颜色
                    color = self.car_colors[self.current_color_index]
                    vehicle_bp.set_attribute('color', f'{color[0]},{color[1]},{color[2]}')
                    
                    # 尝试在相同位置生成新车辆
                    self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
                    
                    # 如果失败，尝试使用出生点
                    if not self.vehicle:
                        spawn_points = self.world.get_map().get_spawn_points()
                        for spawn_point in spawn_points[:5]:
                            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                            if self.vehicle:
                                print("车辆已移动到新位置")
                                break
                    
                    if self.vehicle:
                        # 禁用自动驾驶
                        self.vehicle.set_autopilot(False)
                        
                        # 重新设置相机
                        self.setup_camera()
                        
                        # 重新设置控制器
                        self.setup_controller()
                        
                        print(f"车辆款式已切换: {next_model.split('.')[-1]}")
                    else:
                        print("无法生成新车辆，款式切换失败")
                        self.current_model_index = (self.current_model_index - 1) % len(self.vehicle_models)
                        self.spawn_vehicle()
                else:
                    print("无法获取车辆蓝图")
                    self.spawn_vehicle()
            else:
                print("车辆不存在，无法切换款式")
        except Exception as e:
            print(f"切换车辆款式时出错: {e}")
            self.current_model_index = (self.current_model_index - 1) % len(self.vehicle_models)
            if not self.vehicle:
                self.spawn_vehicle()

    def take_screenshot(self, image):
        """保存当前画面截图"""
        try:
            import os
            import time
            
            # 创建截图目录
            os.makedirs(self.screenshot_dir, exist_ok=True)
            
            # 获取当前时间戳
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 获取当前地图名称
            map_name = self.current_map
            
            # 获取当前天气
            weather_name = self.current_weather
            
            # 获取当前颜色名称
            color_names = ['Red', 'Blue', 'Green', 'Yellow', 'Magenta', 'Cyan', 'Purple', 'Orange', 'Gray', 'White']
            color_name = color_names[self.current_color_index]
            
            # 生成文件名
            filename = f"screenshot_{timestamp}_{map_name}_{weather_name}_{color_name}.png"
            filepath = os.path.join(self.screenshot_dir, filename)
            
            # 保存截图
            cv2.imwrite(filepath, image)
            
            print(f"截图已保存: {filepath}")
            
        except Exception as e:
            print(f"保存截图时出错: {e}")

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

        # 生成一些NPC车辆
        self.spawn_npc_vehicles(2)

        # 生成道路拓扑地图
        self.generate_topology_map()

        print("\n系统准备就绪！")
        print("控制指令:")
        print("  q - 退出程序")
        print("  r - 重置车辆")
        print("  s - 紧急停止")
        print("  x - 切换倒车/前进模式（速度为0时生效）")
        print("  v - 切换视角（第一人称/第三人称/鸟瞰图）")
        print("  m - 切换地图（Town01/Town02/Town03等）")
        print("  w - 切换天气（晴天/雨天/多云/湿滑）")
        print("  c - 切换车辆颜色")
        print("  p - 保存当前画面截图")
        print("  l - 切换夜晚模式（自动打开/关闭近光灯）")
        print("  u - 切换车辆款式")
        print("  d - 切换导航轨迹显示")
        print("\n开始自动驾驶...\n")

        frame_count = 0
        running = True

        try:
            while running:
                # 获取车辆状态
                velocity = self.vehicle.get_velocity()
                speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6

                # 获取控制指令（现在返回4个值，原代码返回3个值）
                # throttle, brake, steer = self.controller.get_control()  # 原代码
                throttle, brake, steer, reverse = self.controller.get_control()  # 新代码

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

                # 记录轨迹点
                if self.show_trajectory:
                    vehicle_location = self.vehicle.get_location()
                    self.trajectory_points.append(vehicle_location)

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
                    
                    # 显示当前地图
                    cv2.putText(display_img, f"Map: {self.current_map}",
                                (20, 280), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 0), 2)  # 黄色显示
                    
                    # 显示当前天气
                    cv2.putText(display_img, f"Weather: {self.current_weather}",
                                (20, 320), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 0, 255), 2)  # 品红色显示
                    
                    # 显示当前车辆颜色
                    color_names = ['Red', 'Blue', 'Green', 'Yellow', 'Magenta', 'Cyan', 'Purple', 'Orange', 'Gray', 'White']
                    current_color_name = color_names[self.current_color_index]
                    cv2.putText(display_img, f"Color: {current_color_name}",
                                (20, 360), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 128, 255), 2)  # 橙色显示
                    
                    # 显示当前车辆款式
                    current_model = self.vehicle_models[self.current_model_index]
                    model_name = current_model.split('.')[-1]
                    cv2.putText(display_img, f"Model: {model_name}",
                                (20, 400), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (128, 255, 0), 2)  # 亮绿色显示

                    # 绘制道路轨迹
                    if self.show_trajectory and self.map_img is not None:
                        # 创建地图副本
                        map_display = self.map_img.copy()

                        # 绘制轨迹
                        if len(self.trajectory_points) > 1:
                            for i in range(1, len(self.trajectory_points)):
                                pt1 = self.world_to_map(self.trajectory_points[i-1])
                                pt2 = self.world_to_map(self.trajectory_points[i])
                                cv2.line(map_display, pt1, pt2, (0, 255, 255), 2)

                        # 绘制车辆位置（红色点）
                        vehicle_loc = self.vehicle.get_location()
                        vehicle_pt = self.world_to_map(vehicle_loc)
                        cv2.circle(map_display, vehicle_pt, 5, (0, 0, 255), -1)

                        # 在主显示图像右上角显示地图
                        map_resized = cv2.resize(map_display, (200, 200))
                        display_img[10:210, display_img.shape[1]-210:display_img.shape[1]-10] = map_resized

                        # 显示轨迹状态
                        cv2.putText(display_img, f"Trajectory: ON ({len(self.trajectory_points)} pts)",
                                    (20, 440), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 255), 2)
                    else:
                        cv2.putText(display_img, "Trajectory: OFF",
                                    (20, 440), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (128, 128, 128), 2)

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
                elif key == ord('m'):
                    # 切换地图
                    self.switch_map()
                elif key == ord('w'):
                    # 切换天气
                    self.switch_weather()
                elif key == ord('c'):
                    # 切换车辆颜色
                    self.switch_color()
                elif key == ord('p'):
                    # 保存截图
                    if self.camera_image is not None:
                        self.take_screenshot(self.camera_image)
                    else:
                        print("当前没有图像可保存")
                elif key == ord('l') or key == ord('L'):
                    # 切换夜晚模式（支持大小写）
                    self.toggle_night_mode()
                elif key == ord('u') or key == ord('U'):
                    # 切换车辆款式（支持大小写）
                    self.switch_vehicle_model()
                elif key == ord('d') or key == ord('D'):
                    # 切换轨迹显示（支持大小写）
                    self.show_trajectory = not self.show_trajectory
                    if self.show_trajectory:
                        print("轨迹显示已开启")
                    else:
                        print("轨迹显示已关闭")

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