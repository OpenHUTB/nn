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
import gc
import psutil

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


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        self.frame_times = []

    def sample_memory(self):
        """采样内存使用"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_samples.append(memory_mb)
        return memory_mb

    def sample_cpu(self):
        """采样CPU使用"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_samples.append(cpu_percent)
        return cpu_percent

    def record_frame_time(self, frame_time):
        """记录帧处理时间"""
        self.frame_times.append(frame_time)

    def get_performance_summary(self):
        """获取性能摘要"""
        return {
            'total_runtime': time.time() - self.start_time,
            'average_memory_mb': sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
            'max_memory_mb': max(self.memory_samples) if self.memory_samples else 0,
            'average_cpu_percent': sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
            'average_frame_time': sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0,
            'frames_per_second': 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
        }


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
    def __init__(self, output_dir, config=None):
        self.output_dir = output_dir
        self.stitched_dir = os.path.join(output_dir, "stitched")
        os.makedirs(self.stitched_dir, exist_ok=True)

        # 性能配置
        self.compress_images = config.get('compress_images', True) if config else True
        self.compression_quality = config.get('compression_quality', 85) if config else 85
        self.enable_memory_cache = config.get('enable_memory_cache', True) if config else True
        self.image_cache = {}
        self.max_cache_size = config.get('max_cache_size', 50) if config else 50

    def stitch(self, image_paths, frame_num, view_type="vehicle"):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            Log.warning("PIL未安装，跳过图像拼接")
            return False

        # 检查缓存
        cache_key = f"{view_type}_{frame_num}"
        if self.enable_memory_cache and cache_key in self.image_cache:
            # 从缓存加载
            cached_image = self.image_cache[cache_key]
            output_path = os.path.join(self.stitched_dir, f"{view_type}_{frame_num:06d}.jpg")
            cached_image.save(output_path, "JPEG", quality=self.compression_quality)
            return True

        positions = [(10, 10), (660, 10), (10, 390), (660, 390)]

        canvas = Image.new('RGB', (640 * 2 + 20, 360 * 2 + 20), (40, 40, 40))
        draw = ImageDraw.Draw(canvas)

        images_loaded = 0
        for idx, (cam_name, img_path) in enumerate(list(image_paths.items())[:4]):
            if img_path and os.path.exists(img_path):
                try:
                    # 检查图像是否已经加载到内存
                    if img_path in self.image_cache:
                        img = self.image_cache[img_path]
                    else:
                        img = Image.open(img_path).resize((640, 360))
                        # 缓存图像
                        if self.enable_memory_cache:
                            self.image_cache[img_path] = img
                            # 清理缓存如果太大
                            if len(self.image_cache) > self.max_cache_size:
                                oldest_key = next(iter(self.image_cache))
                                del self.image_cache[oldest_key]
                    images_loaded += 1
                except:
                    img = Image.new('RGB', (640, 360), (80, 80, 80))
            else:
                img = Image.new('RGB', (640, 360), (80, 80, 80))

            canvas.paste(img, positions[idx])
            draw.text((positions[idx][0] + 5, positions[idx][1] + 5),
                      cam_name, fill=(255, 255, 200))

        if images_loaded == 0:
            return False

        output_path = os.path.join(self.stitched_dir, f"{view_type}_{frame_num:06d}.jpg")

        # 压缩保存
        if self.compress_images:
            canvas.save(output_path, "JPEG", quality=self.compression_quality)
        else:
            canvas.save(output_path, "PNG")

        # 缓存结果
        if self.enable_memory_cache:
            self.image_cache[cache_key] = canvas.copy()

        # 清理内存
        del canvas
        gc.collect()

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

        # 性能优化：批量生成设置
        self.batch_spawn = config.get('batch_spawn', True)
        self.max_spawn_attempts = config.get('max_spawn_attempts', 5)

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

        # 尝试多次生成
        for attempt in range(self.max_spawn_attempts):
            try:
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                vehicle.set_autopilot(True)
                vehicle.apply_control(carla.VehicleControl(throttle=0.2))
                Log.info(f"主车: {vehicle.type_id}")
                return vehicle
            except Exception as e:
                if attempt == self.max_spawn_attempts - 1:
                    Log.warning(f"主车生成失败: {e}")
                else:
                    # 尝试不同的生成点
                    spawn_point = random.choice(spawn_points)
                    time.sleep(0.1)

        return None

    def spawn_traffic(self, center_location):
        if self.batch_spawn:
            vehicles = self._spawn_vehicles_batch()
        else:
            vehicles = self._spawn_vehicles()

        pedestrians = self._spawn_pedestrians(center_location)

        Log.info(f"交通生成: {vehicles}辆车, {pedestrians}个行人")
        return vehicles + pedestrians

    def _spawn_vehicles_batch(self):
        """批量生成车辆（提高性能）"""
        blueprint_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            return 0

        num_vehicles = min(self.config['traffic']['background_vehicles'], 10)
        spawned = 0

        # 准备批处理
        batch_commands = []
        available_points = spawn_points.copy()
        random.shuffle(available_points)

        for i in range(num_vehicles):
            if i >= len(available_points):
                break

            try:
                vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))
                spawn_point = available_points[i]

                # 创建生成命令
                batch_commands.append((vehicle_bp, spawn_point))

            except:
                pass

        # 批量生成
        for vehicle_bp, spawn_point in batch_commands:
            try:
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                vehicle.set_autopilot(True)
                self.vehicles.append(vehicle)
                spawned += 1
            except:
                pass

            # 避免过快的生成速度
            if spawned % 3 == 0:
                time.sleep(0.05)

        return spawned

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

        # 批量销毁
        actors_to_destroy = []

        for vehicle in self.vehicles:
            if vehicle.is_alive:
                actors_to_destroy.append(vehicle)

        for pedestrian in self.pedestrians:
            if pedestrian.is_alive:
                actors_to_destroy.append(pedestrian)

        # 批量销毁
        batch_size = 10
        for i in range(0, len(actors_to_destroy), batch_size):
            batch = actors_to_destroy[i:i + batch_size]
            for actor in batch:
                try:
                    actor.destroy()
                except:
                    pass

            # 避免过快的销毁速度
            if i > 0 and i % 30 == 0:
                time.sleep(0.1)

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
        self.last_performance_sample = 0

        self.vehicle_buffer = {}
        self.infra_buffer = {}
        self.buffer_lock = threading.Lock()

        # 性能监控
        self.performance_monitor = PerformanceMonitor()

        # 性能优化配置
        self.image_processor = ImageProcessor(data_dir, config.get('image_processing', {}))
        self.lidar_processor = None
        self.fusion_manager = None

        # 批处理设置
        self.batch_size = config.get('batch_size', 5)
        self.enable_async_processing = config.get('enable_async_processing', True)

        if config['sensors'].get('lidar_sensors', 0) > 0:
            self.lidar_processor = LidarProcessor(data_dir, config.get('lidar_processing', {}))

        if config['output'].get('save_fusion', False):
            self.fusion_manager = MultiSensorFusion(data_dir, config.get('fusion', {}))

    def setup_cameras(self, vehicle, center_location, vehicle_id=0):
        vehicle_cams = self._setup_vehicle_cameras(vehicle, vehicle_id)
        infra_cams = self._setup_infrastructure_cameras(center_location)

        Log.info(f"摄像头: {vehicle_cams}车辆 + {infra_cams}基础设施")
        return vehicle_cams + infra_cams

    def _setup_vehicle_cameras(self, vehicle, vehicle_id):
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

            lidar_bp.set_attribute('upper_fov', '10')
            lidar_bp.set_attribute('lower_fov', '-20')
            lidar_bp.set_attribute('horizontal_fov', '360')

            lidar_location = carla.Location(x=0, y=0, z=2.5)
            lidar_rotation = carla.Rotation(0, 0, 0)
            lidar_transform = carla.Transform(lidar_location, lidar_rotation)

            lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

            def lidar_callback(lidar_data):
                current_time = time.time()
                frame_start_time = time.time()

                if current_time - self.last_capture_time >= self.config['sensors']['capture_interval']:
                    if self.lidar_processor:
                        try:
                            metadata = self.lidar_processor.process_lidar_data(lidar_data, self.frame_counter)
                            if metadata and self.fusion_manager:
                                vehicle_image_path = None
                                with self.buffer_lock:
                                    if self.vehicle_buffer:
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

                # 记录性能
                frame_time = time.time() - frame_start_time
                self.performance_monitor.record_frame_time(frame_time)

            lidar_sensor.listen(lidar_callback)
            self.sensors.append(lidar_sensor)

            print("LiDAR传感器已安装")
            return 1

        except Exception as e:
            print(f"LiDAR安装失败: {e}")
            return 0

    def _create_camera(self, name, config, parent, sensor_type, vehicle_id=0):
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
            camera.listen(callback)

            self.sensors.append(camera)
            return True

        except Exception as e:
            Log.warning(f"创建摄像头 {name} 失败: {e}")
            return False

    def _create_callback(self, save_dir, name, sensor_type, vehicle_id=0):
        capture_interval = self.config['sensors']['capture_interval']

        def callback(image):
            current_time = time.time()
            frame_start_time = time.time()

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
                            self.vehicle_buffer.clear()
                    else:
                        self.infra_buffer[name] = filename
                        if len(self.infra_buffer) >= 4:
                            self.image_processor.stitch(self.infra_buffer, self.frame_counter, 'infrastructure')
                            self.infra_buffer.clear()

                # 定期采样性能
                if current_time - self.last_performance_sample >= 5.0:
                    memory_mb = self.performance_monitor.sample_memory()
                    cpu_percent = self.performance_monitor.sample_cpu()
                    if self.frame_counter % 10 == 0:
                        Log.debug(f"性能采样 - 内存: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")
                    self.last_performance_sample = current_time

                # 记录帧处理时间
                frame_time = time.time() - frame_start_time
                self.performance_monitor.record_frame_time(frame_time)

                # 定期垃圾回收
                if self.frame_counter % 50 == 0:
                    gc.collect()

        return callback

    def get_frame_count(self):
        return self.frame_counter

    def generate_sensor_summary(self):
        summary = {
            'total_sensors': len(self.sensors),
            'frame_count': self.frame_counter,
            'lidar_data': None,
            'fusion_data': None,
            'performance': self.performance_monitor.get_performance_summary()
        }

        if self.lidar_processor:
            summary['lidar_data'] = self.lidar_processor.generate_lidar_summary()

        if self.fusion_manager:
            summary['fusion_data'] = self.fusion_manager.generate_fusion_report()

        return summary

    def cleanup(self):
        Log.info(f"清理 {len(self.sensors)} 个传感器...")

        # 刷新批处理数据
        if self.lidar_processor:
            self.lidar_processor.flush_batch()

        # 批量销毁传感器
        batch_size = 5
        for i in range(0, len(self.sensors), batch_size):
            batch = self.sensors[i:i + batch_size]
            for sensor in batch:
                try:
                    sensor.stop()
                    sensor.destroy()
                except:
                    pass

            if i > 0 and i % 20 == 0:
                time.sleep(0.05)

        self.sensors.clear()

        # 清理缓存
        if hasattr(self.image_processor, 'image_cache'):
            self.image_processor.image_cache.clear()

        gc.collect()


class DataCollector:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.world = None
        self.ego_vehicles = []
        self.scene_center = None

        self.setup_directories()

        self.traffic_manager = None
        self.sensor_managers = {}
        self.multi_vehicle_manager = None
        self.v2x_communication = None

        self.start_time = None
        self.is_running = False
        self.collected_frames = 0

        # 性能监控
        self.performance_monitor = PerformanceMonitor()

        # 数据格式配置
        self.output_format = config.get('output_format', 'standard')  # standard, v2xformer, kitti

    def setup_directories(self):
        scenario = self.config['scenario']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir = os.path.join(
            self.config['output']['data_dir'],
            f"{scenario['name']}_{scenario['town']}_{timestamp}"
        )

        directories = [
            "raw/vehicle_1",
            "raw/vehicle_2",
            "raw/infrastructure",
            "stitched",
            "lidar",
            "fusion",
            "calibration",
            "cooperative",
            "v2x_messages",
            "v2xformer_format",  # V2XFormer格式数据
            "kitti_format",  # KITTI格式数据
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

        return True

    def collect_data(self):
        duration = self.config['scenario']['duration']
        Log.info(f"开始数据收集，时长: {duration}秒")

        self.start_time = time.time()
        self.is_running = True

        last_update = time.time()
        last_v2x_update = time.time()
        last_perception_share = time.time()
        last_performance_sample = time.time()

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

                # 性能采样
                if current_time - last_performance_sample >= 10.0:
                    memory_mb = self.performance_monitor.sample_memory()
                    cpu_percent = self.performance_monitor.sample_cpu()
                    Log.debug(f"系统性能 - 内存: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")
                    last_performance_sample = current_time

                    # 定期垃圾回收
                    gc.collect()

                if current_time - last_update >= 5.0:
                    total_frames = sum(mgr.get_frame_count() for mgr in self.sensor_managers.values())
                    progress = (elapsed / duration) * 100

                    # 计算预估剩余时间
                    if total_frames > 0:
                        frames_per_second = total_frames / elapsed
                        remaining_frames = (duration - elapsed) * frames_per_second
                        eta_seconds = (duration - elapsed)
                    else:
                        eta_seconds = duration - elapsed

                    Log.info(f"进度: {elapsed:.0f}/{duration}秒 ({progress:.1f}%) | "
                             f"总帧数: {total_frames} | "
                             f"ETA: {eta_seconds:.0f}秒")
                    last_update = current_time

                time.sleep(0.01)  # 减少CPU占用

        except KeyboardInterrupt:
            Log.info("数据收集被用户中断")
        except Exception as e:
            Log.error(f"数据收集错误: {e}")
            traceback.print_exc()
        finally:
            self.is_running = False
            elapsed = time.time() - self.start_time

            self.collected_frames = sum(mgr.get_frame_count() for mgr in self.sensor_managers.values())

            # 获取性能摘要
            performance_summary = self.performance_monitor.get_performance_summary()

            Log.info(f"收集完成: {self.collected_frames}帧, 用时: {elapsed:.1f}秒")
            Log.info(f"平均帧率: {performance_summary['frames_per_second']:.2f} FPS")
            Log.info(f"最大内存使用: {performance_summary['max_memory_mb']:.1f} MB")

            self._save_metadata()
            self._print_summary()

            # 生成标准格式数据
            if self.output_format != 'standard':
                self._convert_to_target_format()

    def _convert_to_target_format(self):
        """转换为目标数据格式"""
        Log.info(f"转换为 {self.output_format} 格式...")

        if self.output_format == 'v2xformer':
            self._convert_to_v2xformer_format()
        elif self.output_format == 'kitti':
            self._convert_to_kitti_format()

    def _convert_to_v2xformer_format(self):
        """转换为V2XFormer格式"""
        try:
            # 创建必要的目录结构
            v2x_dir = os.path.join(self.output_dir, "v2xformer_format")

            # 创建数据集结构
            splits = ['train', 'val', 'test']
            for split in splits:
                split_dir = os.path.join(v2x_dir, split)
                os.makedirs(split_dir, exist_ok=True)

                # 创建子目录
                for subdir in ['image', 'point_cloud', 'calib', 'label']:
                    os.makedirs(os.path.join(split_dir, subdir), exist_ok=True)

            # 生成数据集划分
            total_frames = self.collected_frames
            train_ratio = 0.7
            val_ratio = 0.2
            test_ratio = 0.1

            train_frames = int(total_frames * train_ratio)
            val_frames = int(total_frames * val_ratio)
            test_frames = total_frames - train_frames - val_frames

            # 生成划分文件
            splits_info = {
                'train': list(range(0, train_frames)),
                'val': list(range(train_frames, train_frames + val_frames)),
                'test': list(range(train_frames + val_frames, total_frames))
            }

            splits_file = os.path.join(v2x_dir, "splits.json")
            with open(splits_file, 'w') as f:
                json.dump(splits_info, f, indent=2)

            Log.info(f"V2XFormer格式转换完成: {v2x_dir}")

        except Exception as e:
            Log.error(f"V2XFormer格式转换失败: {e}")

    def _convert_to_kitti_format(self):
        """转换为KITTI格式"""
        try:
            kitti_dir = os.path.join(self.output_dir, "kitti_format")

            # 创建KITTI标准目录结构
            for subdir in ['training', 'testing']:
                full_dir = os.path.join(kitti_dir, subdir)
                os.makedirs(full_dir, exist_ok=True)

                for subsubdir in ['image_2', 'velodyne', 'calib', 'label_2']:
                    os.makedirs(os.path.join(full_dir, subsubdir), exist_ok=True)

            # 生成KITTI格式的校准文件
            self._generate_kitti_calibration(kitti_dir)

            Log.info(f"KITTI格式转换完成: {kitti_dir}")

        except Exception as e:
            Log.error(f"KITTI格式转换失败: {e}")

    def _generate_kitti_calibration(self, kitti_dir):
        """生成KITTI格式的校准文件"""
        calib_template = """P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.341729e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
Tr_velo_to_cam: 4.276802e-04 -9.999672e-01 -8.084491e-03 -1.198459e-02 -7.210626e-03 8.081198e-03 -9.999413e-01 -5.403984e-02 9.999738e-01 4.859485e-04 -7.206933e-03 -2.921968e-02
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01"""

        # 为每一帧生成校准文件
        for i in range(self.collected_frames):
            calib_file = os.path.join(kitti_dir, "training", "calib", f"{i:06d}.txt")
            with open(calib_file, 'w') as f:
                f.write(calib_template)

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
            'output_format': self.output_format,
            'output': self.config['output'],
            'collection': {
                'duration': round(time.time() - self.start_time, 2),
                'total_frames': self.collected_frames,
                'frame_rate': round(self.collected_frames / max(time.time() - self.start_time, 0.1), 2)
            },
            'performance': self.performance_monitor.get_performance_summary()
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

        meta_path = os.path.join(self.output_dir, "metadata", "collection_info.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        Log.info(f"元数据保存: {meta_path}")

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("数据收集摘要")
        print("=" * 60)

        # 统计原始图像
        raw_dirs = [d for d in os.listdir(self.output_dir) if d.startswith('raw')]
        total_raw_images = 0
        for raw_dir in raw_dirs:
            raw_path = os.path.join(self.output_dir, raw_dir)
            if os.path.exists(raw_path):
                # 快速统计
                for root, dirs, files in os.walk(raw_path):
                    total_raw_images += len([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])

        print(f"原始图像: {total_raw_images} 张")

        # 统计LiDAR
        lidar_dir = os.path.join(self.output_dir, "lidar")
        if os.path.exists(lidar_dir):
            import glob
            bin_files = glob.glob(os.path.join(lidar_dir, "*.bin"))
            npy_files = glob.glob(os.path.join(lidar_dir, "*.npy"))
            batch_files = glob.glob(os.path.join(lidar_dir, "*batch*.json"))
            print(f"LiDAR数据: {len(bin_files)} .bin文件, {len(npy_files)} .npy文件")
            print(f"批处理文件: {len(batch_files)} 个")

        # 统计协同数据
        coop_dir = os.path.join(self.output_dir, "cooperative")
        if os.path.exists(coop_dir):
            v2x_files = len(
                [f for f in os.listdir(os.path.join(coop_dir, "v2x_messages")) if f.endswith(('.json', '.gz'))])
            perception_files = len(
                [f for f in os.listdir(os.path.join(coop_dir, "shared_perception")) if f.endswith('.json')])
            print(f"协同数据: {v2x_files} V2X消息, {perception_files} 共享感知文件")

        # 格式转换
        if self.output_format == 'v2xformer':
            v2x_dir = os.path.join(self.output_dir, "v2xformer_format")
            if os.path.exists(v2x_dir):
                print(f"V2XFormer格式: 已生成")

        if self.output_format == 'kitti':
            kitti_dir = os.path.join(self.output_dir, "kitti_format")
            if os.path.exists(kitti_dir):
                print(f"KITTI格式: 已生成")

        # 性能统计
        performance = self.performance_monitor.get_performance_summary()
        print(f"\n性能统计:")
        print(f"  平均帧率: {performance['frames_per_second']:.2f} FPS")
        print(f"  平均内存: {performance['average_memory_mb']:.1f} MB")
        print(f"  最大内存: {performance['max_memory_mb']:.1f} MB")
        print(f"  平均CPU: {performance['average_cpu_percent']:.1f}%")

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

        # 强制垃圾回收
        gc.collect()

        Log.info("清理完成")


def main():
    parser = argparse.ArgumentParser(description='CVIPS 性能优化数据收集系统 v12.0')

    # 基础参数
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--scenario', type=str, default='performance_optimized', help='场景名称')
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

    # 性能参数
    parser.add_argument('--batch-size', type=int, default=5, help='批处理大小')
    parser.add_argument('--enable-compression', action='store_true', help='启用数据压缩')
    parser.add_argument('--enable-downsampling', action='store_true', help='启用LiDAR下采样')

    # 输出格式
    parser.add_argument('--output-format', type=str, default='standard',
                        choices=['standard', 'v2xformer', 'kitti'], help='输出数据格式')

    # 传感器参数
    parser.add_argument('--enable-lidar', action='store_true', help='启用LiDAR传感器')
    parser.add_argument('--enable-fusion', action='store_true', help='启用多传感器融合')
    parser.add_argument('--enable-v2x', action='store_true', help='启用V2X通信')
    parser.add_argument('--enable-cooperative', action='store_true', help='启用协同感知')
    parser.add_argument('--enable-enhancement', action='store_true', help='启用数据增强')  # 添加这行
    parser.add_argument('--enable-annotations', action='store_true', help='启用自动标注')

    # 功能参数
    parser.add_argument('--run-analysis', action='store_true', help='运行数据集分析')
    parser.add_argument('--skip-validation', action='store_true', help='跳过数据验证')
    parser.add_argument('--skip-quality-check', action='store_true', help='跳过质量检查')

    args = parser.parse_args(remaining_argv)

    # 加载配置
    config = ConfigManager.load_config(args.config)
    config = ConfigManager.merge_args(config, args)

    # 添加性能配置
    config['performance']['batch_size'] = args.batch_size
    config['performance']['enable_compression'] = args.enable_compression
    config['performance']['enable_downsampling'] = args.enable_downsampling
    config['output']['output_format'] = args.output_format

    # 显示配置
    print("\n" + "=" * 60)
    print("CVIPS 性能优化数据收集系统 v12.0")
    print("=" * 60)

    print(f"场景: {config['scenario']['name']}")
    print(f"地图: {config['scenario']['town']}")
    print(f"天气/时间: {config['scenario']['weather']}/{config['scenario']['time_of_day']}")
    print(f"时长: {config['scenario']['duration']}秒")
    print(f"交通: {config['traffic']['background_vehicles']}背景车辆 + {config['traffic']['pedestrians']}行人")
    print(f"协同: {config['cooperative']['num_coop_vehicles']} 协同车辆")
    print(f"输出格式: {config['output']['output_format']}")

    print(f"传感器:")
    print(
        f"  摄像头: {config['sensors']['vehicle_cameras']}车辆 + {config['sensors']['infrastructure_cameras']}基础设施")
    print(f"  LiDAR: {'启用' if config['sensors']['lidar_sensors'] > 0 else '禁用'}")
    print(f"  融合: {'启用' if config['output']['save_fusion'] else '禁用'}")
    print(f"  V2X: {'启用' if config['v2x']['enabled'] else '禁用'}")
    print(f"  协同: {'启用' if config['output']['save_cooperative'] else '禁用'}")
    print(f"  增强: {'启用' if config['enhancement']['enabled'] else '禁用'}")

    print(f"性能:")
    print(f"  批处理大小: {config['performance']['batch_size']}")
    print(f"  压缩: {'启用' if config['performance']['enable_compression'] else '禁用'}")
    print(f"  下采样: {'启用' if config['performance']['enable_downsampling'] else '禁用'}")

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