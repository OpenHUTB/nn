import sys
import os
import time
import random
import argparse
import traceback
import math
import threading
import json
from datetime import datetime

from carla_utils import setup_carla_path, import_carla_module
from config_manager import ConfigManager
from annotation_generator import AnnotationGenerator
from data_validator import DataValidator
from scene_manager import SceneManager
from data_analyzer import DataAnalyzer
from lidar_processor import LidarProcessor, MultiSensorFusion

from multi_vehicle_manager import MultiVehicleManager
from v2x_communication import V2XCommunication



carla_egg_path, remaining_argv = setup_carla_path()
carla = import_carla_module()


class Log:
    @staticmethod
    def info(msg):
        print(f"[INFO] {msg}")

    @staticmethod
    def warning(msg):
        print(f"[WARNING] {msg}")

    @staticmethod
    def error(msg):
        print(f"[ERROR] {msg}")

    @staticmethod
    def debug(msg):
        print(f"[DEBUG] {msg}")


class WeatherSystem:
    WEATHER_PRESETS = {
        'clear': {'cloudiness': 10, 'precipitation': 0, 'wind': 5},
        'rainy': {'cloudiness': 90, 'precipitation': 80, 'wind': 15},
        'cloudy': {'cloudiness': 70, 'precipitation': 10, 'wind': 10},
        'foggy': {'cloudiness': 50, 'precipitation': 0, 'fog_density': 40}
    }

    @staticmethod
    def create_weather(weather_type, time_of_day):
        weather = carla.WeatherParameters()

        if weather_type in WeatherSystem.WEATHER_PRESETS:
            preset = WeatherSystem.WEATHER_PRESETS[weather_type]
            weather.cloudiness = preset.get('cloudiness', 30)
            weather.precipitation = preset.get('precipitation', 0)
            weather.wind_intensity = preset.get('wind', 5)
            if 'fog_density' in preset:
                weather.fog_density = preset['fog_density']

        if time_of_day == 'noon':
            weather.sun_altitude_angle = 75
        elif time_of_day == 'sunset':
            weather.sun_altitude_angle = 15
        elif time_of_day == 'night':
            weather.sun_altitude_angle = -20

        return weather


class ImageProcessor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.stitched_dir = os.path.join(output_dir, "stitched")
        os.makedirs(self.stitched_dir, exist_ok=True)

    def stitch(self, image_paths, frame_num, view_type="vehicle"):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            Log.warning("PIL未安装，跳过图像拼接")
            return False

        positions = [(10, 10), (660, 10), (10, 390), (660, 390)]

        canvas = Image.new('RGB', (640 * 2 + 20, 360 * 2 + 20), (40, 40, 40))
        draw = ImageDraw.Draw(canvas)

        for idx, (cam_name, img_path) in enumerate(list(image_paths.items())[:4]):
            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).resize((640, 360))
                except:
                    img = Image.new('RGB', (640, 360), (80, 80, 80))
            else:
                img = Image.new('RGB', (640, 360), (80, 80, 80))

            canvas.paste(img, positions[idx])
            draw.text((positions[idx][0] + 5, positions[idx][1] + 5),
                      cam_name, fill=(255, 255, 200))

        output_path = os.path.join(self.stitched_dir, f"{view_type}_{frame_num:06d}.jpg")
        canvas.save(output_path, "JPEG", quality=90)
        return True


class TrafficManager:
    def __init__(self, world, config):
        self.world = world
        self.config = config
        self.vehicles = []
        self.pedestrians = []

        seed = config['scenario'].get('seed', random.randint(1, 1000))
        random.seed(seed)
        Log.info(f"随机种子: {seed}")

    def spawn_ego_vehicle(self):
        blueprint_lib = self.world.get_blueprint_library()

        common_vehicles = [
            'vehicle.tesla.model3',
            'vehicle.audi.tt',
            'vehicle.mini.cooperst',
            'vehicle.nissan.micra'
        ]

        for vtype in common_vehicles:
            if blueprint_lib.filter(vtype):
                vehicle_bp = random.choice(blueprint_lib.filter(vtype))
                break
        else:
            vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            return None

        spawn_point = random.choice(spawn_points)
        try:
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            vehicle.set_autopilot(True)
            vehicle.apply_control(carla.VehicleControl(throttle=0.2))
            Log.info(f"主车: {vehicle.type_id}")
            return vehicle
        except Exception as e:
            Log.warning(f"主车生成失败: {e}")
            return None

    def spawn_traffic(self, center_location):
        vehicles = self._spawn_vehicles()
        pedestrians = self._spawn_pedestrians(center_location)

        Log.info(f"交通生成: {vehicles}辆车, {pedestrians}个行人")
        return vehicles + pedestrians

    def _spawn_vehicles(self):
        blueprint_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            return 0

        num_vehicles = min(self.config['traffic']['background_vehicles'], 10)
        spawned = 0

        for _ in range(num_vehicles):
            try:
                vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))
                spawn_point = random.choice(spawn_points)
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                vehicle.set_autopilot(True)
                self.vehicles.append(vehicle)
                spawned += 1
            except:
                pass

        return spawned

    def _spawn_pedestrians(self, center_location):
        blueprint_lib = self.world.get_blueprint_library()

        num_peds = min(self.config['traffic']['pedestrians'], 8)
        spawned = 0

        for _ in range(num_peds):
            try:
                ped_bps = list(blueprint_lib.filter('walker.pedestrian.*'))
                if not ped_bps:
                    continue

                ped_bp = random.choice(ped_bps)

                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(5.0, 12.0)

                location = carla.Location(
                    x=center_location.x + distance * math.cos(angle),
                    y=center_location.y + distance * math.sin(angle),
                    z=center_location.z + 0.5
                )

                pedestrian = self.world.spawn_actor(ped_bp, carla.Transform(location))
                self.pedestrians.append(pedestrian)
                spawned += 1
            except Exception as e:
                Log.debug(f"行人生成失败: {e}")

        return spawned

    def cleanup(self):
        Log.info("清理交通...")

        for vehicle in self.vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except:
                pass

        for pedestrian in self.pedestrians:
            try:
                if pedestrian.is_alive:
                    pedestrian.destroy()
            except:
                pass

        self.vehicles.clear()
        self.pedestrians.clear()


class SensorManager:
    def __init__(self, world, config, data_dir):
        self.world = world
        self.config = config
        self.data_dir = data_dir
        self.sensors = []

        self.frame_counter = 0
        self.last_capture_time = 0

        self.vehicle_buffer = {}
        self.infra_buffer = {}
        self.buffer_lock = threading.Lock()

        self.image_processor = ImageProcessor(data_dir)
        self.lidar_processor = None
        self.fusion_manager = None

        if config['sensors'].get('lidar_sensors', 0) > 0:
            self.lidar_processor = LidarProcessor(data_dir)

        if config['output'].get('save_fusion', False):
            self.fusion_manager = MultiSensorFusion(data_dir)


    def setup_cameras(self, vehicle, center_location, vehicle_id=0):
        vehicle_cams = self._setup_vehicle_cameras(vehicle, vehicle_id)

    def setup_cameras(self, vehicle, center_location):
        vehicle_cams = self._setup_vehicle_cameras(vehicle)

        infra_cams = self._setup_infrastructure_cameras(center_location)

        Log.info(f"摄像头: {vehicle_cams}车辆 + {infra_cams}基础设施")
        return vehicle_cams + infra_cams


    def _setup_vehicle_cameras(self, vehicle, vehicle_id):

    def _setup_vehicle_cameras(self, vehicle):

        if not vehicle:
            return 0

        camera_configs = {
            'front_wide': {'loc': (2.0, 0, 1.8), 'rot': (0, -3, 0), 'fov': 100},
            'front_narrow': {'loc': (2.0, 0, 1.6), 'rot': (0, 0, 0), 'fov': 60},
            'right_side': {'loc': (0.5, 1.0, 1.5), 'rot': (0, -2, 45), 'fov': 90},
            'left_side': {'loc': (0.5, -1.0, 1.5), 'rot': (0, -2, -45), 'fov': 90}
        }

        installed = 0
        for cam_name, config_data in camera_configs.items():

            if self._create_camera(cam_name, config_data, vehicle, 'vehicle', vehicle_id):

            if self._create_camera(cam_name, config_data, vehicle, 'vehicle'):

                installed += 1

        return installed

    def _setup_infrastructure_cameras(self, center_location):
        camera_configs = [
            {'name': 'north', 'offset': (0, -20, 12), 'rotation': (0, -25, 180)},
            {'name': 'south', 'offset': (0, 20, 12), 'rotation': (0, -25, 0)},
            {'name': 'east', 'offset': (20, 0, 12), 'rotation': (0, -25, -90)},
            {'name': 'west', 'offset': (-20, 0, 12), 'rotation': (0, -25, 90)}
        ]

        installed = 0
        for cam_config in camera_configs:
            sensor_config = {
                'loc': (
                    center_location.x + cam_config['offset'][0],
                    center_location.y + cam_config['offset'][1],
                    center_location.z + cam_config['offset'][2]
                ),
                'rot': cam_config['rotation'],
                'fov': 90
            }

            if self._create_camera(cam_config['name'], sensor_config, None, 'infrastructure'):
                installed += 1

        return installed


    def setup_lidar(self, vehicle, vehicle_id=0):

    def setup_lidar(self, vehicle):

        if not vehicle or not self.config['sensors'].get('lidar_sensors', 0) > 0:
            return 0

        try:
            blueprint_lib = self.world.get_blueprint_library()
            lidar_bp = blueprint_lib.find('sensor.lidar.ray_cast')

            lidar_config = self.config['sensors'].get('lidar_config', {})

            lidar_bp.set_attribute('channels', str(lidar_config.get('channels', 32)))
            lidar_bp.set_attribute('range', str(lidar_config.get('range', 100)))
            lidar_bp.set_attribute('points_per_second', str(lidar_config.get('points_per_second', 56000)))
            lidar_bp.set_attribute('rotation_frequency', str(lidar_config.get('rotation_frequency', 10)))



            # 添加更多LiDAR参数

            lidar_bp.set_attribute('upper_fov', '10')
            lidar_bp.set_attribute('lower_fov', '-20')
            lidar_bp.set_attribute('horizontal_fov', '360')

            lidar_location = carla.Location(x=0, y=0, z=2.5)
            lidar_rotation = carla.Rotation(0, 0, 0)
            lidar_transform = carla.Transform(lidar_location, lidar_rotation)

            lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

            def lidar_callback(lidar_data):
                current_time = time.time()
                if current_time - self.last_capture_time >= self.config['sensors']['capture_interval']:
                    if self.lidar_processor:
                        try:
                            metadata = self.lidar_processor.process_lidar_data(lidar_data, self.frame_counter)
                            if metadata and self.fusion_manager:

                                vehicle_image_path = None
                                with self.buffer_lock:
                                    if self.vehicle_buffer:

                                # 尝试获取最新的车辆图像
                                vehicle_image_path = None
                                with self.buffer_lock:
                                    if self.vehicle_buffer:
                                        # 取第一个摄像头的图像

                                        for cam_name, img_path in self.vehicle_buffer.items():
                                            if os.path.exists(img_path):
                                                vehicle_image_path = img_path
                                                break

                                sensor_data = {
                                    'lidar': os.path.join(self.data_dir, "lidar", f"lidar_{self.frame_counter:06d}.bin")
                                }
                                if vehicle_image_path:
                                    sensor_data['camera'] = vehicle_image_path

                                self.fusion_manager.create_synchronization_file(self.frame_counter, sensor_data)
                        except Exception as e:
                            print(f"LiDAR处理失败: {e}")

            lidar_sensor.listen(lidar_callback)
            self.sensors.append(lidar_sensor)

            print("LiDAR传感器已安装")
            return 1

        except Exception as e:
            print(f"LiDAR安装失败: {e}")
            return 0


    def _create_camera(self, name, config, parent, sensor_type, vehicle_id=0):

    def _create_camera(self, name, config, parent, sensor_type):

        try:
            blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')

            img_size = self.config['sensors'].get('image_size', [1280, 720])
            blueprint.set_attribute('image_size_x', str(img_size[0]))
            blueprint.set_attribute('image_size_y', str(img_size[1]))
            blueprint.set_attribute('fov', str(config.get('fov', 90)))

            location = carla.Location(config['loc'][0], config['loc'][1], config['loc'][2])
            rotation = carla.Rotation(config['rot'][0], config['rot'][1], config['rot'][2])
            transform = carla.Transform(location, rotation)

            if parent:
                camera = self.world.spawn_actor(blueprint, transform, attach_to=parent)
            else:
                camera = self.world.spawn_actor(blueprint, transform)


            # 为不同车辆创建不同目录
            if sensor_type == 'vehicle' and vehicle_id > 0:
                save_dir = os.path.join(self.data_dir, "raw", f"vehicle_{vehicle_id}", name)
            else:
                save_dir = os.path.join(self.data_dir, "raw", sensor_type, name)

            os.makedirs(save_dir, exist_ok=True)

            callback = self._create_callback(save_dir, name, sensor_type, vehicle_id)
            save_dir = os.path.join(self.data_dir, "raw", sensor_type, name)
            os.makedirs(save_dir, exist_ok=True)

            callback = self._create_callback(save_dir, name, sensor_type)

            camera.listen(callback)

            self.sensors.append(camera)
            return True

        except Exception as e:
            Log.warning(f"创建摄像头 {name} 失败: {e}")
            return False


    def _create_callback(self, save_dir, name, sensor_type, vehicle_id=0):

    def _create_callback(self, save_dir, name, sensor_type):

        capture_interval = self.config['sensors']['capture_interval']

        def callback(image):
            current_time = time.time()

            if current_time - self.last_capture_time >= capture_interval:
                self.frame_counter += 1
                self.last_capture_time = current_time

                filename = os.path.join(save_dir, f"{name}_{self.frame_counter:06d}.png")
                image.save_to_disk(filename, carla.ColorConverter.Raw)

                with self.buffer_lock:
                    if sensor_type == 'vehicle':
                        self.vehicle_buffer[name] = filename
                        if len(self.vehicle_buffer) >= 4:

                            self.image_processor.stitch(self.vehicle_buffer, self.frame_counter,
                                                        f'vehicle_{vehicle_id}')

                            self.image_processor.stitch(self.vehicle_buffer, self.frame_counter, 'vehicle')

                            self.vehicle_buffer.clear()
                    else:
                        self.infra_buffer[name] = filename
                        if len(self.infra_buffer) >= 4:
                            self.image_processor.stitch(self.infra_buffer, self.frame_counter, 'infrastructure')
                            self.infra_buffer.clear()

        return callback

    def get_frame_count(self):
        return self.frame_counter

    def generate_sensor_summary(self):
        summary = {
            'total_sensors': len(self.sensors),
            'frame_count': self.frame_counter,
            'lidar_data': None,
            'fusion_data': None
        }

        if self.lidar_processor:
            summary['lidar_data'] = self.lidar_processor.generate_lidar_summary()

        if self.fusion_manager:
            summary['fusion_data'] = self.fusion_manager.generate_fusion_report()

        return summary

    def cleanup(self):
        Log.info(f"清理 {len(self.sensors)} 个传感器...")
        for sensor in self.sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except:
                pass
        self.sensors.clear()


class DataCollector:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.world = None

        self.ego_vehicles = []  # 多个主车

        self.ego_vehicle = None

        self.scene_center = None

        self.setup_directories()

        self.traffic_manager = None

        self.sensor_managers = {}  # 车辆ID -> SensorManager
        self.multi_vehicle_manager = None
        self.v2x_communication = None

        self.sensor_manager = None


        self.start_time = None
        self.is_running = False
        self.collected_frames = 0

    def setup_directories(self):
        scenario = self.config['scenario']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir = os.path.join(
            self.config['output']['data_dir'],
            f"{scenario['name']}_{scenario['town']}_{timestamp}"
        )

        directories = [

            "raw/vehicle_1",  # 主车1
            "raw/vehicle_2",  # 主车2（如果有）

            "raw/vehicle",

            "raw/infrastructure",
            "stitched",
            "lidar",
            "fusion",
            "calibration",

            "cooperative",
            "v2x_messages",

            "metadata"
        ]

        for subdir in directories:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

        Log.info(f"数据目录: {self.output_dir}")

    def connect(self):
        for attempt in range(1, 6):
            try:
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(10.0)

                town = self.config['scenario']['town']
                self.world = self.client.load_world(town)

                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)

                Log.info(f"连接成功: {town}")
                return True

            except Exception as e:
                Log.warning(f"连接尝试 {attempt}/5 失败: {str(e)[:50]}")
                time.sleep(2)

        return False

    def setup_scene(self):
        weather_cfg = self.config['scenario']
        weather = WeatherSystem.create_weather(weather_cfg['weather'], weather_cfg['time_of_day'])
        self.world.set_weather(weather)
        Log.info(f"天气: {weather_cfg['weather']}, 时间: {weather_cfg['time_of_day']}")

        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_points:
            self.scene_center = spawn_points[len(spawn_points) // 2].location
        else:
            self.scene_center = carla.Location(0, 0, 0)

        self.traffic_manager = TrafficManager(self.world, self.config)


        # 生成多个主车
        num_ego_vehicles = min(self.config['cooperative'].get('num_coop_vehicles', 2) + 1, 3)
        for i in range(num_ego_vehicles):
            ego_vehicle = self.traffic_manager.spawn_ego_vehicle()
            if ego_vehicle:
                self.ego_vehicles.append(ego_vehicle)
                Log.info(f"主车 {i + 1} 生成: {ego_vehicle.type_id}")

        if not self.ego_vehicles:

        self.ego_vehicle = self.traffic_manager.spawn_ego_vehicle()
        if not self.ego_vehicle:

            Log.error("主车生成失败")
            return False

        self.traffic_manager.spawn_traffic(self.scene_center)


        # 初始化V2X通信
        if self.config['v2x']['enabled']:
            self.v2x_communication = V2XCommunication(self.config['v2x'])

            # 注册车辆到V2X网络
            for i, vehicle in enumerate(self.ego_vehicles):
                location = vehicle.get_location()
                self.v2x_communication.register_node(
                    f'vehicle_{vehicle.id}',
                    (location.x, location.y, location.z),
                    {'type': 'vehicle', 'capabilities': ['bsm', 'rsm']}
                )

        # 初始化多车辆管理器
        self.multi_vehicle_manager = MultiVehicleManager(
            self.world,
            self.config,
            self.output_dir
        )

        # 设置主车
        self.multi_vehicle_manager.ego_vehicles = self.ego_vehicles

        # 生成协同车辆
        num_coop_vehicles = self.config['cooperative'].get('num_coop_vehicles', 2)
        coop_vehicles = self.multi_vehicle_manager.spawn_cooperative_vehicles(num_coop_vehicles)

        # 注册协同车辆到V2X网络
        if self.v2x_communication:
            for vehicle in coop_vehicles:
                location = vehicle.get_location()
                self.v2x_communication.register_node(
                    f'vehicle_{vehicle.id}',
                    (location.x, location.y, location.z),
                    {'type': 'vehicle', 'capabilities': ['bsm', 'rsm']}
                )



        time.sleep(3.0)
        return True

    def setup_sensors(self):

        # 为每个主车设置传感器
        for i, vehicle in enumerate(self.ego_vehicles):
            sensor_manager = SensorManager(self.world, self.config, self.output_dir)

            cameras = sensor_manager.setup_cameras(vehicle, self.scene_center, i + 1)
            if cameras == 0:
                Log.error(f"车辆 {i + 1} 没有摄像头安装成功")
                return False

            lidars = sensor_manager.setup_lidar(vehicle, i + 1)
            Log.info(f"车辆 {i + 1} 传感器: {cameras}摄像头 + {lidars}LiDAR")

            self.sensor_managers[vehicle.id] = sensor_manager

        self.sensor_manager = SensorManager(self.world, self.config, self.output_dir)

        cameras = self.sensor_manager.setup_cameras(self.ego_vehicle, self.scene_center)
        if cameras == 0:
            Log.error("没有摄像头安装成功")
            return False

        lidars = self.sensor_manager.setup_lidar(self.ego_vehicle)
        Log.info(f"传感器: {cameras}摄像头 + {lidars}LiDAR")


        return True

    def collect_data(self):
        duration = self.config['scenario']['duration']
        Log.info(f"开始数据收集，时长: {duration}秒")

        self.start_time = time.time()
        self.is_running = True

        last_update = time.time()

        last_v2x_update = time.time()
        last_perception_share = time.time()

        try:
            while time.time() - self.start_time < duration and self.is_running:
                current_time = time.time()
                elapsed = current_time - self.start_time


                # 更新车辆状态
                if self.multi_vehicle_manager:
                    self.multi_vehicle_manager.update_vehicle_states()

                # V2X通信更新
                if self.v2x_communication and current_time - last_v2x_update >= 0.1:
                    self._update_v2x_communication()
                    last_v2x_update = current_time

                # 共享感知数据
                if (self.config['cooperative'].get('enable_shared_perception', True) and
                        current_time - last_perception_share >= 2.0):
                    self._share_perception_data()
                    last_perception_share = current_time

                # 定期保存共享感知
                frame_count = sum(mgr.get_frame_count() for mgr in self.sensor_managers.values())
                if frame_count % 10 == 0 and self.multi_vehicle_manager:
                    self.multi_vehicle_manager.save_shared_perception(frame_count)

                if current_time - last_update >= 5.0:
                    total_frames = sum(mgr.get_frame_count() for mgr in self.sensor_managers.values())
                    progress = (elapsed / duration) * 100

                    Log.info(f"进度: {elapsed:.0f}/{duration}秒 ({progress:.1f}%) | 总帧数: {total_frames}")

                if current_time - last_update >= 5.0:
                    frames = self.sensor_manager.get_frame_count()
                    progress = (elapsed / duration) * 100

                    Log.info(f"进度: {elapsed:.0f}/{duration}秒 ({progress:.1f}%) | 帧数: {frames}")

                    last_update = current_time

                time.sleep(0.05)

        except KeyboardInterrupt:
            Log.info("数据收集被用户中断")
        finally:
            self.is_running = False
            elapsed = time.time() - self.start_time


            self.collected_frames = sum(mgr.get_frame_count() for mgr in self.sensor_managers.values())

            self.collected_frames = self.sensor_manager.get_frame_count()

            Log.info(f"收集完成: {self.collected_frames}帧, 用时: {elapsed:.1f}秒")

            self._save_metadata()
            self._print_summary()


    def _update_v2x_communication(self):
        """更新V2X通信"""
        if not self.v2x_communication:
            return

        # 为每辆车发送基本安全消息
        for vehicle in self.ego_vehicles + self.multi_vehicle_manager.cooperative_vehicles:
            if not vehicle.is_alive:
                continue

            try:
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

                vehicle_data = {
                    'position': (location.x, location.y, location.z),
                    'speed': speed,
                    'heading': vehicle.get_transform().rotation.yaw,
                    'acceleration': (0, 0, 0)  # 简化处理
                }

                self.v2x_communication.broadcast_basic_safety_message(
                    f'vehicle_{vehicle.id}',
                    vehicle_data
                )
            except:
                pass

        # 处理接收到的消息
        for vehicle in self.ego_vehicles:
            messages = self.v2x_communication.get_messages_for_node(f'vehicle_{vehicle.id}')
            if messages:
                Log.debug(f"车辆 {vehicle.id} 收到 {len(messages)} 条V2X消息")

    def _share_perception_data(self):
        """共享感知数据"""
        if not self.multi_vehicle_manager or not self.config['cooperative']['enable_shared_perception']:
            return

        # 模拟车辆感知数据（简化处理，实际应从传感器获取）
        for vehicle in self.ego_vehicles + self.multi_vehicle_manager.cooperative_vehicles:
            if not vehicle.is_alive:
                continue

            # 模拟检测到的物体
            detected_objects = self._simulate_object_detection(vehicle)

            if detected_objects:
                self.multi_vehicle_manager.share_perception_data(vehicle.id, detected_objects)

    def _simulate_object_detection(self, vehicle):
        """模拟对象检测（简化）"""
        detected_objects = []

        # 获取车辆周围的其他车辆
        for other_vehicle in self.ego_vehicles + self.multi_vehicle_manager.cooperative_vehicles:
            if other_vehicle.id == vehicle.id or not other_vehicle.is_alive:
                continue

            try:
                location = other_vehicle.get_location()
                distance = vehicle.get_location().distance(location)

                # 模拟检测范围（50米）
                if distance < 50.0:
                    obj_data = {
                        'class': 'vehicle',
                        'position': {'x': location.x, 'y': location.y, 'z': location.z},
                        'velocity': {'x': 0, 'y': 0, 'z': 0},
                        'confidence': max(0.7, 1.0 - distance / 50.0),
                        'size': {'width': 2.0, 'length': 4.5, 'height': 1.5},
                        'id': other_vehicle.id
                    }
                    detected_objects.append(obj_data)
            except:
                pass

        return detected_objects


    def _save_metadata(self):
        metadata = {
            'scenario': self.config['scenario'],
            'traffic': self.config['traffic'],
            'sensors': self.config['sensors'],

            'v2x': self.config['v2x'],
            'cooperative': self.config['cooperative'],


            'output': self.config['output'],
            'collection': {
                'duration': round(time.time() - self.start_time, 2),
                'total_frames': self.collected_frames,
                'frame_rate': round(self.collected_frames / max(time.time() - self.start_time, 0.1), 2)
            }
        }


        # 传感器摘要
        sensor_summaries = {}
        for vehicle_id, sensor_manager in self.sensor_managers.items():
            sensor_summaries[vehicle_id] = sensor_manager.generate_sensor_summary()
        metadata['sensor_summaries'] = sensor_summaries

        # V2X通信状态
        if self.v2x_communication:
            metadata['v2x_status'] = self.v2x_communication.get_network_status()

        # 协同摘要
        if self.multi_vehicle_manager:
            metadata['cooperative_summary'] = self.multi_vehicle_manager.generate_summary()

        if self.sensor_manager:
            sensor_summary = self.sensor_manager.generate_sensor_summary()
            metadata['sensor_summary'] = sensor_summary


        meta_path = os.path.join(self.output_dir, "metadata", "collection_info.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        Log.info(f"元数据保存: {meta_path}")

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("数据收集摘要")
        print("=" * 60)

        # 统计图像
        stitched_dir = os.path.join(self.output_dir, "stitched")
        if os.path.exists(stitched_dir):
            stitched_files = [f for f in os.listdir(stitched_dir) if f.endswith('.jpg')]
            print(f"拼接图像: {len(stitched_files)} 张")

        # 统计LiDAR
        lidar_dir = os.path.join(self.output_dir, "lidar")
        if os.path.exists(lidar_dir):
            bin_files = [f for f in os.listdir(lidar_dir) if f.endswith('.bin')]
            npy_files = [f for f in os.listdir(lidar_dir) if f.endswith('.npy')]
            print(f"LiDAR数据: {len(bin_files)} .bin文件, {len(npy_files)} .npy文件")

            if bin_files:
                total_points = 0
                for bin_file in bin_files[:3]:
                    bin_path = os.path.join(lidar_dir, bin_file)
                    if os.path.exists(bin_path):
                        file_size = os.path.getsize(bin_path)
                        points_in_file = file_size // (4 * 4)
                        total_points += points_in_file
                print(f"  估计总点数: {total_points:,}")


        # 统计协同数据
        coop_dir = os.path.join(self.output_dir, "cooperative")
        if os.path.exists(coop_dir):
            v2x_files = len([f for f in os.listdir(os.path.join(coop_dir, "v2x_messages")) if f.endswith('.json')])
            perception_files = len(
                [f for f in os.listdir(os.path.join(coop_dir, "shared_perception")) if f.endswith('.json')])
            print(f"协同数据: {v2x_files} V2X消息, {perception_files} 共享感知文件")

        # V2X统计
        if self.v2x_communication:
            v2x_status = self.v2x_communication.get_network_status()
            print(f"V2X通信: {v2x_status['stats']['messages_sent']} 发送, "
                  f"{v2x_status['stats']['messages_received']} 接收, "
                  f"{v2x_status['stats']['messages_dropped']} 丢包")

        # 车辆统计
        print(f"车辆总数: {len(self.ego_vehicles)} 主车 + "
              f"{len(self.multi_vehicle_manager.cooperative_vehicles)} 协同车")

        # 统计融合数据
        fusion_dir = os.path.join(self.output_dir, "fusion")
        if os.path.exists(fusion_dir):
            sync_files = [f for f in os.listdir(fusion_dir) if f.endswith('.json')]
            print(f"融合数据: {len(sync_files)} 个同步文件")


        print(f"\n输出目录: {self.output_dir}")
        print("=" * 60)

    def run_validation(self):
        if self.config['output'].get('validate_data', True):
            Log.info("运行数据验证...")
            DataValidator.validate_dataset(self.output_dir)

    def run_analysis(self):
        if self.config['output'].get('run_analysis', False):
            Log.info("运行数据分析...")
            DataAnalyzer.analyze_dataset(self.output_dir)

    def cleanup(self):
        Log.info("清理场景...")


        # 清理传感器
        for sensor_manager in self.sensor_managers.values():
            sensor_manager.cleanup()

        # 清理交通
        if self.traffic_manager:
            self.traffic_manager.cleanup()

        # 清理协同管理
        if self.multi_vehicle_manager:
            self.multi_vehicle_manager.cleanup()

        # 清理V2X通信
        if self.v2x_communication:
            self.v2x_communication.stop()

        # 清理车辆
        for vehicle in self.ego_vehicles:
            if vehicle and vehicle.is_alive:
                try:
                    vehicle.destroy()
                except:
                    pass


        if self.sensor_manager:
            self.sensor_manager.cleanup()

        if self.traffic_manager:
            self.traffic_manager.cleanup()

        if self.ego_vehicle and self.ego_vehicle.is_alive:
            try:
                self.ego_vehicle.destroy()
            except:
                pass


        Log.info("清理完成")


def main():

    parser = argparse.ArgumentParser(description='CVIPS 多车辆协同数据收集系统 v10.0')

    # 基础参数
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--scenario', type=str, default='multi_vehicle_cooperative', help='场景名称')

    parser = argparse.ArgumentParser(description='CVIPS 多传感器数据收集系统')

    # 基础参数
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--scenario', type=str, default='multi_sensor_scene', help='场景名称')

    parser.add_argument('--town', type=str, default='Town10HD',
                        choices=['Town03', 'Town04', 'Town05', 'Town10HD'], help='地图')
    parser.add_argument('--weather', type=str, default='clear',
                        choices=['clear', 'rainy', 'cloudy', 'foggy'], help='天气')
    parser.add_argument('--time-of-day', type=str, default='noon',
                        choices=['noon', 'sunset', 'night'], help='时间')

    # 交通参数
    parser.add_argument('--num-vehicles', type=int, default=8, help='背景车辆数')
    parser.add_argument('--num-pedestrians', type=int, default=6, help='行人数')

    parser.add_argument('--num-coop-vehicles', type=int, default=2, help='协同车辆数')



    # 收集参数
    parser.add_argument('--duration', type=int, default=60, help='收集时长(秒)')
    parser.add_argument('--capture-interval', type=float, default=2.0, help='捕捉间隔(秒)')
    parser.add_argument('--seed', type=int, help='随机种子')

    # 传感器参数
    parser.add_argument('--enable-lidar', action='store_true', help='启用LiDAR传感器')
    parser.add_argument('--enable-fusion', action='store_true', help='启用多传感器融合')

    parser.add_argument('--enable-v2x', action='store_true', help='启用V2X通信')
    parser.add_argument('--enable-cooperative', action='store_true', help='启用协同感知')


    parser.add_argument('--enable-annotations', action='store_true', help='启用自动标注')

    # 功能参数
    parser.add_argument('--run-analysis', action='store_true', help='运行数据集分析')
    parser.add_argument('--skip-validation', action='store_true', help='跳过数据验证')

    args = parser.parse_args(remaining_argv)

    # 加载配置
    config = ConfigManager.load_config(args.config)
    config = ConfigManager.merge_args(config, args)

    # 显示配置
    print("\n" + "=" * 60)

    print("CVIPS 多车辆协同数据收集系统 v10.0")

    print("CVIPS 多传感器数据收集系统 v9.0")

    print("=" * 60)

    print(f"场景: {config['scenario']['name']}")
    print(f"地图: {config['scenario']['town']}")
    print(f"天气/时间: {config['scenario']['weather']}/{config['scenario']['time_of_day']}")
    print(f"时长: {config['scenario']['duration']}秒")

    print(f"交通: {config['traffic']['background_vehicles']}背景车辆 + {config['traffic']['pedestrians']}行人")
    print(f"协同: {config['cooperative']['num_coop_vehicles']} 协同车辆")

    print(f"交通: {config['traffic']['background_vehicles']}车辆 + {config['traffic']['pedestrians']}行人")


    print(f"传感器:")
    print(
        f"  摄像头: {config['sensors']['vehicle_cameras']}车辆 + {config['sensors']['infrastructure_cameras']}基础设施")
    print(f"  LiDAR: {'启用' if config['sensors']['lidar_sensors'] > 0 else '禁用'}")
    print(f"  融合: {'启用' if config['output']['save_fusion'] else '禁用'}")

    print(f"  V2X: {'启用' if config['v2x']['enabled'] else '禁用'}")
    print(f"  协同: {'启用' if config['output']['save_cooperative'] else '禁用'}")

    print(f"  标注: {'启用' if config['output']['save_annotations'] else '禁用'}")


    collector = DataCollector(config)

    try:
        if not collector.connect():
            print("连接CARLA服务器失败")
            return

        if not collector.setup_scene():
            print("场景设置失败")
            collector.cleanup()
            return

        if not collector.setup_sensors():
            print("传感器设置失败")
            collector.cleanup()
            return

        collector.collect_data()

        collector.run_analysis()
        collector.run_validation()

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n运行错误: {e}")
        traceback.print_exc()
    finally:
        collector.cleanup()
        print(f"\n数据集已保存到: {collector.output_dir}")


if __name__ == "__main__":
    main()