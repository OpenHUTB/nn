# !/usr/bin/env python3
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
import numpy as np

from carla_utils import setup_carla_path, import_carla_module
from config_manager import ConfigManager
from annotation_generator import AnnotationGenerator
from data_validator import DataValidator
from lidar_processor import LidarProcessor, MultiSensorFusion
from multi_vehicle_manager import MultiVehicleManager
from pedestrian_safety_monitor import PedestrianSafetyMonitor

carla_egg_path, remaining_argv = setup_carla_path()
carla = import_carla_module()


class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        self.frame_times = []
        self.first_frame_time = None
        self.last_frame_time = None

    def sample_memory(self):
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)
            return {
                'process_mb': memory_mb,
                'system_total_mb': psutil.virtual_memory().total / 1024 / 1024,
                'system_used_percent': psutil.virtual_memory().percent,
                'system_available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        except:
            memory_mb = 0
            self.memory_samples.append(memory_mb)
            return {
                'process_mb': memory_mb,
                'system_total_mb': 0,
                'system_used_percent': 0,
                'system_available_mb': 0
            }

    def sample_cpu(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_samples.append(cpu_percent)
            return {
                'total_percent': cpu_percent,
                'per_core': psutil.cpu_percent(interval=0.1, percpu=True),
                'count': psutil.cpu_count()
            }
        except:
            cpu_percent = 0
            self.cpu_samples.append(cpu_percent)
            return {
                'total_percent': cpu_percent,
                'per_core': [0],
                'count': 1
            }

    def record_frame_time(self, frame_time):
        self.frame_times.append(frame_time)
        if self.first_frame_time is None:
            self.first_frame_time = time.time()
        self.last_frame_time = time.time()

    def get_performance_summary(self):
        if not self.frame_times or len(self.frame_times) < 2:
            avg_frame_time = 0
            fps = 0
        else:
            avg_frame_time = np.mean(self.frame_times)
            fps = len(self.frame_times) / max(0.1, (self.last_frame_time - self.first_frame_time))

        summary = {
            'total_runtime': time.time() - self.start_time,
            'average_memory_mb': np.mean(self.memory_samples) if self.memory_samples else 0,
            'max_memory_mb': max(self.memory_samples) if self.memory_samples else 0,
            'min_memory_mb': min(self.memory_samples) if self.memory_samples else 0,
            'average_cpu_percent': np.mean(self.cpu_samples) if self.cpu_samples else 0,
            'max_cpu_percent': max(self.cpu_samples) if self.cpu_samples else 0,
            'average_frame_time': avg_frame_time,
            'frames_per_second': fps,
            'total_frames': len(self.frame_times)
        }

        if self.frame_times and len(self.frame_times) >= 2:
            summary['frame_time_stats'] = {
                'p50': np.percentile(self.frame_times, 50),
                'p95': np.percentile(self.frame_times, 95),
                'p99': np.percentile(self.frame_times, 99) if len(self.frame_times) > 1 else 0,
                'std': np.std(self.frame_times)
            }
        else:
            summary['frame_time_stats'] = {
                'p50': 0,
                'p95': 0,
                'p99': 0,
                'std': 0
            }

        return summary


class Log:
    @staticmethod
    def info(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO][{timestamp}] {msg}")

    @staticmethod
    def warning(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[WARNING][{timestamp}] {msg}")

    @staticmethod
    def error(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[ERROR][{timestamp}] {msg}")

    @staticmethod
    def debug(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[DEBUG][{timestamp}] {msg}")

    @staticmethod
    def performance(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[PERF][{timestamp}] {msg}")

    @staticmethod
    def safety(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[SAFETY][{timestamp}] {msg}")


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

        self.compress_images = config.get('compress_images', True) if config else True
        self.compression_quality = config.get('compression_quality', 85) if config else 85
        self.enable_memory_cache = config.get('enable_memory_cache', True) if config else True
        self.image_cache = {}
        self.max_cache_size = config.get('max_cache_size', 50) if config else 50

        self.stats = {
            'images_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0
        }

    def stitch(self, image_paths, frame_num, view_type="vehicle"):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            Log.warning("PIL未安装，跳过图像拼接")
            return False

        start_time = time.time()

        cache_key = f"{view_type}_{frame_num}"
        if self.enable_memory_cache and cache_key in self.image_cache:
            cached_image = self.image_cache[cache_key]
            output_path = os.path.join(self.stitched_dir, f"{view_type}_{frame_num:06d}.jpg")

            try:
                if self.compress_images:
                    cached_image.save(output_path, "JPEG",
                                      quality=self.compression_quality,
                                      optimize=True,
                                      progressive=True)
                else:
                    cached_image.save(output_path, "PNG", optimize=True)
            except Exception as e:
                Log.error(f"保存缓存图像失败: {e}")
                return False

            self.stats['cache_hits'] += 1
            self.stats['images_processed'] += 1
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            return True

        self.stats['cache_misses'] += 1

        positions = [(10, 10), (660, 10), (10, 390), (660, 390)]

        canvas = Image.new('RGB', (640 * 2 + 20, 360 * 2 + 20), (40, 40, 40))
        draw = ImageDraw.Draw(canvas)

        images_loaded = 0
        for idx, (cam_name, img_path) in enumerate(list(image_paths.items())[:4]):
            if img_path and os.path.exists(img_path):
                try:
                    if img_path in self.image_cache:
                        img = self.image_cache[img_path]
                    else:
                        img = Image.open(img_path)
                        img.load()
                        img = img.resize((640, 360), Image.Resampling.LANCZOS)

                        if self.enable_memory_cache:
                            self.image_cache[img_path] = img
                            if len(self.image_cache) > self.max_cache_size:
                                self._cleanup_cache()
                    images_loaded += 1
                except Exception as e:
                    Log.warning(f"加载图像失败: {e}")
                    img = Image.new('RGB', (640, 360), (80, 80, 80))
            else:
                img = Image.new('RGB', (640, 360), (80, 80, 80))

            canvas.paste(img, positions[idx])
            draw.text((positions[idx][0] + 5, positions[idx][1] + 5),
                      cam_name, fill=(255, 255, 200))

        if images_loaded == 0:
            return False

        output_path = os.path.join(self.stitched_dir, f"{view_type}_{frame_num:06d}.jpg")

        try:
            if self.compress_images:
                canvas.save(output_path, "JPEG",
                            quality=self.compression_quality,
                            optimize=True,
                            progressive=True,
                            subsampling='4:2:0')
            else:
                canvas.save(output_path, "PNG", optimize=True)
        except Exception as e:
            Log.error(f"保存图像失败: {e}")
            return False

        if self.enable_memory_cache:
            self.image_cache[cache_key] = canvas.copy()

        del canvas
        gc.collect()

        self.stats['images_processed'] += 1
        processing_time = time.time() - start_time
        self.stats['total_processing_time'] += processing_time

        if self.stats['images_processed'] % 50 == 0:
            avg_time = self.stats['total_processing_time'] / self.stats['images_processed']
            cache_hit_rate = self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
            Log.performance(f"图像处理统计: 处理{self.stats['images_processed']}张, "
                            f"平均{avg_time:.3f}秒/张, 缓存命中率{cache_hit_rate:.1%}")

        return True

    def _cleanup_cache(self):
        if len(self.image_cache) > self.max_cache_size:
            keys_to_remove = list(self.image_cache.keys())[:len(self.image_cache) - self.max_cache_size]
            for key in keys_to_remove:
                del self.image_cache[key]
            Log.debug(f"清理缓存: 移除了{len(keys_to_remove)}个缓存项")

    def get_stats(self):
        total_operations = self.stats['cache_hits'] + self.stats['cache_misses']
        if total_operations > 0:
            cache_hit_rate = self.stats['cache_hits'] / total_operations
        else:
            cache_hit_rate = 0

        avg_time = 0
        if self.stats['images_processed'] > 0:
            avg_time = self.stats['total_processing_time'] / self.stats['images_processed']

        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'average_processing_time': avg_time,
            'current_cache_size': len(self.image_cache)
        }


class TrafficManager:
    def __init__(self, world, config):
        self.world = world
        self.config = config
        self.vehicles = []
        self.pedestrians = []

        seed = config['scenario'].get('seed', random.randint(1, 1000))
        random.seed(seed)
        Log.info(f"随机种子: {seed}")

        self.batch_spawn = config.get('batch_spawn', True)
        self.max_spawn_attempts = config.get('max_spawn_attempts', 5)

        self.spawn_stats = {
            'total_attempts': 0,
            'successful_spawns': 0,
            'failed_spawns': 0,
            'total_spawn_time': 0
        }

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

        for attempt in range(self.max_spawn_attempts):
            self.spawn_stats['total_attempts'] += 1
            try:
                start_time = time.time()
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                vehicle.set_autopilot(True)
                vehicle.apply_control(carla.VehicleControl(throttle=0.2))

                spawn_time = time.time() - start_time
                self.spawn_stats['total_spawn_time'] += spawn_time
                self.spawn_stats['successful_spawns'] += 1

                Log.info(f"主车: {vehicle.type_id}, 生成时间: {spawn_time:.3f}秒")
                return vehicle
            except Exception as e:
                self.spawn_stats['failed_spawns'] += 1
                if attempt == self.max_spawn_attempts - 1:
                    Log.warning(f"主车生成失败: {e}")
                else:
                    spawn_point = random.choice(spawn_points)
                    time.sleep(0.1)

        return None

    def spawn_traffic(self, center_location):
        start_time = time.time()

        if self.batch_spawn:
            vehicles = self._spawn_vehicles_batch()
        else:
            vehicles = self._spawn_vehicles()

        pedestrians = self._spawn_pedestrians(center_location)

        total_time = time.time() - start_time
        Log.info(f"交通生成: {vehicles}辆车, {pedestrians}个行人, 用时: {total_time:.2f}秒")

        success_rate = self.spawn_stats['successful_spawns'] / max(1, self.spawn_stats['total_attempts'])
        avg_spawn_time = self.spawn_stats['total_spawn_time'] / max(1, self.spawn_stats['successful_spawns'])
        Log.performance(f"生成统计: 成功率{success_rate:.1%}, 平均生成时间{avg_spawn_time:.3f}秒")

        return vehicles + pedestrians

    def _spawn_vehicles_batch(self):
        blueprint_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            return 0

        num_vehicles = min(self.config['traffic']['background_vehicles'], 10)
        spawned = 0

        batch_commands = []
        available_points = spawn_points.copy()
        random.shuffle(available_points)

        for i in range(num_vehicles):
            if i >= len(available_points):
                break

            try:
                vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))
                spawn_point = available_points[i]
                batch_commands.append((vehicle_bp, spawn_point))
            except:
                pass

        for vehicle_bp, spawn_point in batch_commands:
            try:
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                vehicle.set_autopilot(True)
                self.vehicles.append(vehicle)
                spawned += 1

                self.spawn_stats['successful_spawns'] += 1
                self.spawn_stats['total_attempts'] += 1
            except:
                self.spawn_stats['failed_spawns'] += 1
                self.spawn_stats['total_attempts'] += 1
                pass

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

                self.spawn_stats['successful_spawns'] += 1
                self.spawn_stats['total_attempts'] += 1
            except:
                self.spawn_stats['failed_spawns'] += 1
                self.spawn_stats['total_attempts'] += 1
                pass

        return spawned

    def _spawn_pedestrians(self, center_location):
        blueprint_lib = self.world.get_blueprint_library()

        num_peds = min(self.config['traffic']['pedestrians'], 12)  # 增加行人数量
        spawned = 0

        for _ in range(num_peds):
            try:
                ped_bps = list(blueprint_lib.filter('walker.pedestrian.*'))
                if not ped_bps:
                    continue

                ped_bp = random.choice(ped_bps)

                # 在学校区域或人行横道场景中，行人更集中
                if self.config.get('scenario', {}).get('name', '').lower() in ['school_zone', 'pedestrian_crossing']:
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(3.0, 8.0)  # 更靠近中心
                else:
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(5.0, 15.0)

                location = carla.Location(
                    x=center_location.x + distance * math.cos(angle),
                    y=center_location.y + distance * math.sin(angle),
                    z=center_location.z + 0.5
                )

                pedestrian = self.world.spawn_actor(ped_bp, carla.Transform(location))
                self.pedestrians.append(pedestrian)
                spawned += 1

                self.spawn_stats['successful_spawns'] += 1
                self.spawn_stats['total_attempts'] += 1
            except Exception as e:
                self.spawn_stats['failed_spawns'] += 1
                self.spawn_stats['total_attempts'] += 1
                Log.debug(f"行人生成失败: {e}")

        return spawned

    def cleanup(self):
        Log.info("开始清理交通管理器...")

        try:
            for pedestrian in self.pedestrians:
                try:
                    if pedestrian and pedestrian.is_alive:
                        pedestrian.destroy()
                except:
                    pass

            self.pedestrians.clear()

            for vehicle in self.vehicles:
                try:
                    if vehicle and vehicle.is_alive:
                        vehicle.destroy()
                except:
                    pass

            self.vehicles.clear()

            Log.info("交通管理器清理完成")

        except Exception as e:
            Log.error(f"清理交通管理器失败: {e}")


class SensorManager:
    def __init__(self, world, config, data_dir):
        self.world = world
        self.config = config
        self.data_dir = data_dir
        self.sensors = []

        self.frame_counter = 0
        self.last_capture_time = 0
        self.last_performance_sample = 0

        self.target_fps = config['performance'].get('frame_rate_limit', 5.0)
        self.min_frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.1
        self.last_frame_time = 0
        self.frame_skip_count = 0
        self.max_frame_skip = 2

        self.vehicle_buffer = {}
        self.infra_buffer = {}
        self.buffer_lock = threading.Lock()

        self.is_running = True

        self.performance_monitor = PerformanceMonitor()

        self.image_processor = ImageProcessor(data_dir, config.get('image_processing', {}))
        self.lidar_processor = None
        self.fusion_manager = None

        self.batch_size = config['performance'].get('batch_size', 5)
        self.enable_async_processing = config.get('enable_async_processing', True)

        if config['sensors'].get('lidar_sensors', 0) > 0:
            self.lidar_processor = LidarProcessor(data_dir, config['performance'].get('lidar_processing', {}))

        if config['output'].get('save_fusion', False):
            self.fusion_manager = MultiSensorFusion(data_dir, config['performance'].get('fusion', {}))

        self.sensor_stats = {
            'total_images': 0,
            'total_lidar_frames': 0,
            'image_capture_times': [],
            'lidar_processing_times': []
        }

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
                if not self.is_running:
                    return

                try:
                    current_time = time.time()
                    frame_start_time = time.time()

                    if current_time - self.last_capture_time >= self.config['sensors']['capture_interval']:
                        if self.lidar_processor:
                            try:
                                start_process = time.time()
                                metadata = self.lidar_processor.process_lidar_data(lidar_data, self.frame_counter)
                                process_time = time.time() - start_process
                                self.sensor_stats['lidar_processing_times'].append(process_time)
                                self.sensor_stats['total_lidar_frames'] += 1

                                if metadata and self.fusion_manager:
                                    vehicle_image_path = None
                                    with self.buffer_lock:
                                        if self.vehicle_buffer:
                                            for cam_name, img_path in self.vehicle_buffer.items():
                                                if os.path.exists(img_path):
                                                    vehicle_image_path = img_path
                                                    break

                                    sensor_data = {
                                        'lidar': os.path.join(self.data_dir, "lidar",
                                                              f"lidar_{self.frame_counter:06d}.bin")
                                    }
                                    if vehicle_image_path:
                                        sensor_data['camera'] = vehicle_image_path

                                    self.fusion_manager.create_synchronization_file(self.frame_counter, sensor_data)
                            except Exception as e:
                                if self.is_running:
                                    Log.error(f"LiDAR处理失败: {e}")

                    frame_time = time.time() - frame_start_time
                    self.performance_monitor.record_frame_time(frame_time)
                except Exception as e:
                    if self.is_running:
                        Log.error(f"LiDAR回调错误: {e}")

            lidar_sensor.listen(lidar_callback)
            self.sensors.append(lidar_sensor)

            Log.info("LiDAR传感器已安装")
            return 1

        except Exception as e:
            Log.error(f"LiDAR安装失败: {e}")
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
            if not self.is_running:
                return

            try:
                current_time = time.time()
                frame_start_time = time.time()

                time_since_last_frame = current_time - self.last_frame_time
                if time_since_last_frame < self.min_frame_interval:
                    self.frame_skip_count += 1
                    if self.frame_skip_count > self.max_frame_skip:
                        self.frame_skip_count = 0
                    else:
                        return
                else:
                    self.frame_skip_count = 0

                if current_time - self.last_capture_time >= capture_interval:
                    self.frame_counter += 1
                    self.last_capture_time = current_time
                    self.last_frame_time = current_time

                    capture_start = time.time()
                    filename = os.path.join(save_dir, f"{name}_{self.frame_counter:06d}.png")
                    image.save_to_disk(filename, carla.ColorConverter.Raw)
                    capture_time = time.time() - capture_start

                    self.sensor_stats['image_capture_times'].append(capture_time)
                    self.sensor_stats['total_images'] += 1

                    with self.buffer_lock:
                        if sensor_type == 'vehicle':
                            self.vehicle_buffer[name] = filename
                            if len(self.vehicle_buffer) >= 4:
                                if self.enable_async_processing:
                                    import concurrent.futures
                                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                                        future = executor.submit(
                                            self.image_processor.stitch,
                                            self.vehicle_buffer.copy(),
                                            self.frame_counter,
                                            f'vehicle_{vehicle_id}'
                                        )
                                else:
                                    self.image_processor.stitch(self.vehicle_buffer, self.frame_counter,
                                                                f'vehicle_{vehicle_id}')
                                self.vehicle_buffer.clear()
                        else:
                            self.infra_buffer[name] = filename
                            if len(self.infra_buffer) >= 4:
                                if self.enable_async_processing:
                                    import concurrent.futures
                                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                                        future = executor.submit(
                                            self.image_processor.stitch,
                                            self.infra_buffer.copy(),
                                            self.frame_counter,
                                            'infrastructure'
                                        )
                                else:
                                    self.image_processor.stitch(self.infra_buffer, self.frame_counter, 'infrastructure')
                                self.infra_buffer.clear()

                    if current_time - self.last_performance_sample >= 5.0:
                        memory_info = self.performance_monitor.sample_memory()
                        cpu_info = self.performance_monitor.sample_cpu()

                        if self.frame_counter % 10 == 0:
                            Log.performance(f"系统监控 - 内存: {memory_info['process_mb']:.1f}MB, "
                                            f"CPU: {cpu_info['total_percent']:.1f}%, "
                                            f"线程数: {threading.active_count()}")

                        self.last_performance_sample = current_time

                    frame_time = time.time() - frame_start_time
                    self.performance_monitor.record_frame_time(frame_time)

                    if self.frame_counter % 50 == 0:
                        gc.collect()

                        if self.sensor_stats['total_images'] > 0:
                            avg_capture_time = np.mean(self.sensor_stats['image_capture_times'][-100:]) if len(
                                self.sensor_stats['image_capture_times']) > 0 else 0
                            Log.performance(f"传感器统计: 图像{self.sensor_stats['total_images']}张, "
                                            f"平均捕获时间{avg_capture_time:.3f}秒")
            except Exception as e:
                if self.is_running:
                    Log.error(f"传感器回调错误: {e}")

        return callback

    def get_frame_count(self):
        return self.frame_counter

    def generate_sensor_summary(self):
        summary = {
            'total_sensors': len(self.sensors),
            'frame_count': self.frame_counter,
            'lidar_data': None,
            'fusion_data': None,
            'performance': self.performance_monitor.get_performance_summary(),
            'sensor_stats': self.sensor_stats
        }

        if self.sensor_stats['image_capture_times']:
            summary['sensor_stats']['avg_image_capture_time'] = np.mean(self.sensor_stats['image_capture_times'])
            summary['sensor_stats']['max_image_capture_time'] = np.max(self.sensor_stats['image_capture_times'])

        if self.sensor_stats['lidar_processing_times']:
            summary['sensor_stats']['avg_lidar_process_time'] = np.mean(self.sensor_stats['lidar_processing_times'])
            summary['sensor_stats']['max_lidar_process_time'] = np.max(self.sensor_stats['lidar_processing_times'])

        if self.lidar_processor:
            summary['lidar_data'] = self.lidar_processor.generate_lidar_summary()

        if self.fusion_manager:
            summary['fusion_data'] = self.fusion_manager.generate_fusion_report()

        summary['image_processor_stats'] = self.image_processor.get_stats()

        return summary

    def cleanup(self):
        Log.info(f"安全清理 {len(self.sensors)} 个传感器...")

        self.is_running = False

        for sensor in self.sensors:
            try:
                if hasattr(sensor, 'stop'):
                    sensor.stop()
                    time.sleep(0.001)
            except:
                pass

        time.sleep(0.2)

        if self.lidar_processor:
            try:
                self.lidar_processor.flush_batch()
            except:
                pass

        for i, sensor in enumerate(self.sensors):
            try:
                if hasattr(sensor, 'destroy'):
                    sensor.destroy()
            except:
                pass
            if i % 5 == 0:
                time.sleep(0.01)

        self.sensors.clear()

        if hasattr(self.image_processor, 'image_cache'):
            try:
                self.image_processor.image_cache.clear()
            except:
                pass

        self.vehicle_buffer.clear()
        self.infra_buffer.clear()

        gc.collect()

        Log.info("传感器清理完成")


class V2XCommunication:
    def __init__(self, config):
        self.config = config
        self.nodes = {}
        self.messages = []

    def register_node(self, node_id, position, capabilities):
        self.nodes[node_id] = {
            'position': position,
            'capabilities': capabilities,
            'last_update': time.time()
        }

    def broadcast_basic_safety_message(self, sender_id, vehicle_data):
        pass

    def get_messages_for_node(self, node_id):
        return []

    def get_network_status(self):
        return {'nodes': len(self.nodes), 'messages': len(self.messages)}

    def stop(self):
        pass


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
        self.safety_monitor = None

        self.start_time = None
        self.is_running = False
        self.collected_frames = 0

        self.performance_monitor = PerformanceMonitor()

        self.output_format = config.get('output_format', 'standard')

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
            "v2xformer_format",
            "kitti_format",
            "metadata",
            "safety_reports"
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

        if self.config['v2x']['enabled']:
            self.v2x_communication = V2XCommunication(self.config['v2x'])

            for i, vehicle in enumerate(self.ego_vehicles):
                location = vehicle.get_location()
                self.v2x_communication.register_node(
                    f'vehicle_{vehicle.id}',
                    (location.x, location.y, location.z),
                    {'type': 'vehicle', 'capabilities': ['bsm', 'rsm']}
                )

        self.multi_vehicle_manager = MultiVehicleManager(
            self.world,
            self.config,
            self.output_dir
        )

        self.multi_vehicle_manager.ego_vehicles = self.ego_vehicles

        num_coop_vehicles = self.config['cooperative'].get('num_coop_vehicles', 2)
        coop_vehicles = self.multi_vehicle_manager.spawn_cooperative_vehicles(num_coop_vehicles)

        if self.v2x_communication:
            for vehicle in coop_vehicles:
                location = vehicle.get_location()
                self.v2x_communication.register_node(
                    f'vehicle_{vehicle.id}',
                    (location.x, location.y, location.z),
                    {'type': 'vehicle', 'capabilities': ['bsm', 'rsm']}
                )

        self.safety_monitor = PedestrianSafetyMonitor(self.world, self.output_dir)

        time.sleep(3.0)
        return True

    def setup_sensors(self):
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
        last_detailed_log = time.time()
        last_memory_check = time.time()
        last_safety_check = time.time()

        memory_warning_issued = False
        early_stop_triggered = False

        memory_warning_threshold = 350
        memory_critical_threshold = 400
        early_stop_threshold = 450

        try:
            while time.time() - self.start_time < duration and self.is_running:
                current_time = time.time()
                elapsed = current_time - self.start_time

                if current_time - last_memory_check >= 2.0:
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / (1024 * 1024)

                        if memory_mb > early_stop_threshold:
                            Log.error(
                                f"内存使用超过临界值({early_stop_threshold}MB): {memory_mb:.1f}MB，提前结束数据收集")
                            early_stop_triggered = True
                            break
                        elif memory_mb > memory_critical_threshold:
                            Log.warning(f"内存使用严重过高: {memory_mb:.1f}MB，进行强制清理")
                            self._force_memory_cleanup()
                            memory_warning_issued = True
                        elif memory_mb > memory_warning_threshold and not memory_warning_issued:
                            Log.warning(f"内存使用较高: {memory_mb:.1f}MB，减少数据处理")
                            memory_warning_issued = True
                        elif memory_mb < memory_warning_threshold:
                            memory_warning_issued = False

                    except Exception as e:
                        Log.debug(f"内存检查失败: {e}")

                    last_memory_check = current_time

                if self.multi_vehicle_manager:
                    self.multi_vehicle_manager.update_vehicle_states()

                v2x_interval = self.config['v2x'].get('update_interval', 2.0)
                if self.v2x_communication and current_time - last_v2x_update >= v2x_interval:
                    self._update_v2x_communication()
                    last_v2x_update = current_time

                if (not memory_warning_issued and
                        self.config['cooperative'].get('enable_shared_perception', True) and
                        current_time - last_perception_share >= 2.0):
                    self._share_perception_data()
                    last_perception_share = current_time

                # 行人安全检查
                if current_time - last_safety_check >= 1.0 and self.safety_monitor:
                    safety_report = self.safety_monitor.check_pedestrian_safety()
                    if safety_report['risk_distribution']['high'] > 0:
                        Log.safety(f"行人安全警告: {safety_report['risk_distribution']['high']}个高风险情况")
                        # 广播行人警告
                        self._broadcast_pedestrian_warnings(safety_report)
                    last_safety_check = current_time

                if current_time - last_performance_sample >= 10.0:
                    memory_info = self.performance_monitor.sample_memory()
                    cpu_info = self.performance_monitor.sample_cpu()

                    if current_time - last_detailed_log >= 60.0:
                        Log.performance(f"详细性能监控:")
                        Log.performance(f"  进程内存: {memory_info['process_mb']:.1f}MB")
                        Log.performance(f"  系统内存: {memory_info['system_used_percent']:.1f}%使用率")
                        Log.performance(f"  CPU: {cpu_info['total_percent']:.1f}% ({cpu_info['count']}核心)")
                        last_detailed_log = current_time
                    else:
                        Log.performance(f"系统监控 - 内存: {memory_info['process_mb']:.1f}MB, "
                                        f"CPU: {cpu_info['total_percent']:.1f}%, "
                                        f"活跃线程: {threading.active_count()}")

                    last_performance_sample = current_time

                    gc.collect()

                if current_time - last_update >= 5.0:
                    total_frames = sum(mgr.get_frame_count() for mgr in self.sensor_managers.values())
                    progress = (elapsed / duration) * 100

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

                time.sleep(0.01)

        except KeyboardInterrupt:
            Log.info("数据收集被用户中断")
        except Exception as e:
            Log.error(f"数据收集错误: {e}")
            traceback.print_exc()
        finally:
            self.is_running = False
            elapsed = time.time() - self.start_time

            self.collected_frames = sum(mgr.get_frame_count() for mgr in self.sensor_managers.values())

            if self.collected_frames > 0:
                total_frames_from_sensors = self.collected_frames
                performance_summary = self.performance_monitor.get_performance_summary()

                if early_stop_triggered:
                    Log.warning("数据收集因内存过高而提前终止")
                else:
                    Log.info(f"收集完成: {self.collected_frames}帧, 用时: {elapsed:.1f}秒")

                fps = total_frames_from_sensors / max(elapsed, 0.1)
                Log.info(f"平均帧率: {fps:.2f} FPS")
                Log.info(f"最大内存使用: {performance_summary['max_memory_mb']:.1f} MB")
                Log.info(f"平均CPU使用: {performance_summary['average_cpu_percent']:.1f}%")

                # 生成行人安全报告
                if self.safety_monitor:
                    final_report = self.safety_monitor.generate_final_report()
                    Log.safety(
                        f"行人安全报告: {final_report['risk_distribution']['high']}高风险, {final_report['risk_distribution']['medium']}中风险")
                    Log.safety(f"行人安全评分: {final_report['safety_score']:.1f}/100")
            else:
                Log.warning("未收集到任何数据帧")

            self._save_metadata()
            self._print_summary()

            if self.output_format != 'standard':
                self._convert_to_target_format()

    def _broadcast_pedestrian_warnings(self, safety_report):
        """广播行人警告"""
        if not self.multi_vehicle_manager:
            return

        # 检查高风险交互
        if safety_report.get('risk_distribution', {}).get('high', 0) > 0:
            for vehicle in self.ego_vehicles + self.multi_vehicle_manager.cooperative_vehicles:
                if not hasattr(vehicle, 'is_alive') or not vehicle.is_alive:
                    continue

                try:
                    location = vehicle.get_location()
                    velocity = vehicle.get_velocity()
                    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

                    # 模拟行人位置（实际应用中应从感知系统获取）
                    pedestrian_location = (
                        location.x + random.uniform(-5, 5),
                        location.y + random.uniform(-5, 5),
                        location.z
                    )

                    distance = math.sqrt(
                        (location.x - pedestrian_location[0]) ** 2 +
                        (location.y - pedestrian_location[1]) ** 2
                    )

                    if distance < 20.0:  # 只广播近距离行人
                        self.multi_vehicle_manager.share_pedestrian_warning(
                            vehicle.id,
                            pedestrian_location,
                            distance,
                            speed
                        )
                except:
                    pass

    def _force_memory_cleanup(self):
        Log.info("执行强制内存清理...")

        for sensor_manager in self.sensor_managers.values():
            if hasattr(sensor_manager, 'image_processor'):
                if hasattr(sensor_manager.image_processor, 'image_cache'):
                    try:
                        old_size = len(sensor_manager.image_processor.image_cache)
                        sensor_manager.image_processor.image_cache.clear()
                        Log.debug(f"清理图像缓存: 释放了{old_size}个缓存项")
                    except:
                        pass

        for sensor_manager in self.sensor_managers.values():
            if hasattr(sensor_manager, 'lidar_processor'):
                try:
                    sensor_manager.lidar_processor.flush_batch()
                    Log.debug("刷新LiDAR批处理数据")
                except:
                    pass

        gc.collect()

        Log.info("强制内存清理完成")

    def cleanup(self):
        Log.info("开始安全清理场景...")

        cleanup_start = time.time()

        try:
            self.is_running = False

            time.sleep(0.2)

            if self.v2x_communication:
                try:
                    Log.info("停止V2X通信...")
                    self.v2x_communication.stop()
                except:
                    pass

            if self.safety_monitor:
                try:
                    Log.info("保存行人安全数据...")
                    self.safety_monitor.save_data()
                except:
                    pass

            Log.info(f"清理 {len(self.sensor_managers)} 个传感器管理器...")
            for vehicle_id, sensor_manager in self.sensor_managers.items():
                try:
                    sensor_manager.cleanup()
                except:
                    pass

            self.sensor_managers.clear()
            time.sleep(0.1)

            if self.multi_vehicle_manager:
                try:
                    Log.info("清理多车辆管理器...")
                    self.multi_vehicle_manager.cleanup()
                except:
                    pass

            if self.traffic_manager:
                try:
                    Log.info("清理交通管理器...")
                    self.traffic_manager.cleanup()
                except:
                    pass

            Log.info(f"清理 {len(self.ego_vehicles)} 个主车...")
            for vehicle in self.ego_vehicles:
                try:
                    if hasattr(vehicle, 'destroy'):
                        vehicle.destroy()
                except:
                    pass
                time.sleep(0.01)

            self.ego_vehicles.clear()

            try:
                if self.world:
                    default_weather = carla.WeatherParameters()
                    self.world.set_weather(default_weather)
            except:
                pass

            gc.collect()

            time.sleep(0.3)

            self.traffic_manager = None
            self.multi_vehicle_manager = None
            self.v2x_communication = None
            self.safety_monitor = None
            self.scene_center = None

            gc.collect()

            Log.info(f"清理完成，总用时: {time.time() - cleanup_start:.2f}秒")

        except Exception as e:
            Log.error(f"清理过程中发生错误: {e}")
            try:
                self.sensor_managers.clear()
                self.ego_vehicles.clear()
                self.traffic_manager = None
                self.multi_vehicle_manager = None
                self.v2x_communication = None
                self.safety_monitor = None
                gc.collect()
            except:
                pass

    def _convert_to_target_format(self):
        Log.info(f"转换为 {self.output_format} 格式...")

        if self.output_format == 'v2xformer':
            self._convert_to_v2xformer_format()
        elif self.output_format == 'kitti':
            self._convert_to_kitti_format()

    def _convert_to_v2xformer_format(self):
        try:
            v2x_dir = os.path.join(self.output_dir, "v2xformer_format")

            splits = ['train', 'val', 'test']
            for split in splits:
                split_dir = os.path.join(v2x_dir, split)
                os.makedirs(split_dir, exist_ok=True)

                for subdir in ['image', 'point_cloud', 'calib', 'label']:
                    os.makedirs(os.path.join(split_dir, subdir), exist_ok=True)

            total_frames = self.collected_frames
            train_ratio = 0.7
            val_ratio = 0.2
            test_ratio = 0.1

            train_frames = int(total_frames * train_ratio)
            val_frames = int(total_frames * val_ratio)
            test_frames = total_frames - train_frames - val_frames

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
        try:
            kitti_dir = os.path.join(self.output_dir, "kitti_format")

            for subdir in ['training', 'testing']:
                full_dir = os.path.join(kitti_dir, subdir)
                os.makedirs(full_dir, exist_ok=True)

                for subsubdir in ['image_2', 'velodyne', 'calib', 'label_2']:
                    os.makedirs(os.path.join(full_dir, subsubdir), exist_ok=True)

            self._generate_kitti_calibration(kitti_dir)

            Log.info(f"KITTI格式转换完成: {kitti_dir}")

        except Exception as e:
            Log.error(f"KITTI格式转换失败: {e}")

    def _generate_kitti_calibration(self, kitti_dir):
        calib_template = """P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.341729e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
Tr_velo_to_cam: 4.276802e-04 -9.999672e-01 -8.084491e-03 -1.198459e-02 -7.210626e-03 8.081198e-03 -9.999413e-01 -5.403984e-02 9.999738e-01 4.859485e-04 -7.206933e-03 -2.921968e-02
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01"""

        for i in range(self.collected_frames):
            calib_file = os.path.join(kitti_dir, "training", "calib", f"{i:06d}.txt")
            with open(calib_file, 'w') as f:
                f.write(calib_template)

    def _update_v2x_communication(self):
        if not self.v2x_communication:
            return

        for vehicle in self.ego_vehicles + self.multi_vehicle_manager.cooperative_vehicles:
            if not hasattr(vehicle, 'is_alive') or not vehicle.is_alive:
                continue

            try:
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

                vehicle_data = {
                    'position': (location.x, location.y, location.z),
                    'speed': speed,
                    'heading': vehicle.get_transform().rotation.yaw,
                    'acceleration': (0, 0, 0)
                }

                self.v2x_communication.broadcast_basic_safety_message(
                    f'vehicle_{vehicle.id}',
                    vehicle_data
                )
            except:
                pass

        for vehicle in self.ego_vehicles:
            messages = self.v2x_communication.get_messages_for_node(f'vehicle_{vehicle.id}')
            if messages:
                Log.debug(f"车辆 {vehicle.id} 收到 {len(messages)} 条V2X消息")

    def _share_perception_data(self):
        if not self.multi_vehicle_manager or not self.config['cooperative']['enable_shared_perception']:
            return

        for vehicle in self.ego_vehicles + self.multi_vehicle_manager.cooperative_vehicles:
            if not hasattr(vehicle, 'is_alive') or not vehicle.is_alive:
                continue

            detected_objects = self._simulate_object_detection(vehicle)

            if detected_objects:
                self.multi_vehicle_manager.share_perception_data(vehicle.id, detected_objects)

    def _simulate_object_detection(self, vehicle):
        detected_objects = []

        for other_vehicle in self.ego_vehicles + self.multi_vehicle_manager.cooperative_vehicles:
            if other_vehicle.id == vehicle.id or not hasattr(other_vehicle, 'is_alive') or not other_vehicle.is_alive:
                continue

            try:
                location = other_vehicle.get_location()
                distance = vehicle.get_location().distance(location)

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
        total_frames = self.collected_frames

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
                'total_frames': total_frames,
                'frame_rate': round(total_frames / max(time.time() - self.start_time, 0.1),
                                    2) if total_frames > 0 else 0
            }
        }

        if total_frames > 0:
            performance_summary = self.performance_monitor.get_performance_summary()
            metadata['performance'] = performance_summary

            sensor_summaries = {}
            for vehicle_id, sensor_manager in self.sensor_managers.items():
                sensor_summaries[vehicle_id] = sensor_manager.generate_sensor_summary()
            metadata['sensor_summaries'] = sensor_summaries

            if self.v2x_communication:
                metadata['v2x_status'] = self.v2x_communication.get_network_status()

            if self.multi_vehicle_manager:
                metadata['cooperative_summary'] = self.multi_vehicle_manager.generate_summary()

            if self.safety_monitor:
                metadata['safety_report'] = self.safety_monitor.generate_final_report()

        meta_path = os.path.join(self.output_dir, "metadata", "collection_info.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        Log.info(f"元数据保存: {meta_path}")

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("数据收集摘要")
        print("=" * 60)

        raw_dirs = [d for d in os.listdir(self.output_dir) if d.startswith('raw')]
        total_raw_images = 0
        for raw_dir in raw_dirs:
            raw_path = os.path.join(self.output_dir, raw_dir)
            if os.path.exists(raw_path):
                for root, dirs, files in os.walk(raw_path):
                    total_raw_images += len([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])

        print(f"原始图像: {total_raw_images} 张")

        lidar_dir = os.path.join(self.output_dir, "lidar")
        if os.path.exists(lidar_dir):
            import glob
            bin_files = glob.glob(os.path.join(lidar_dir, "*.bin"))
            npy_files = glob.glob(os.path.join(lidar_dir, "*.npy"))
            batch_files = glob.glob(os.path.join(lidar_dir, "*batch*.json"))
            print(f"LiDAR数据: {len(bin_files)} .bin文件, {len(npy_files)} .npy文件")
            print(f"批处理文件: {len(batch_files)} 个")

        coop_dir = os.path.join(self.output_dir, "cooperative")
        if os.path.exists(coop_dir):
            v2x_files = len(
                [f for f in os.listdir(os.path.join(coop_dir, "v2x_messages")) if f.endswith(('.json', '.gz'))])
            perception_files = len(
                [f for f in os.listdir(os.path.join(coop_dir, "shared_perception")) if f.endswith('.json')])
            print(f"协同数据: {v2x_files} V2X消息, {perception_files} 共享感知文件")

        safety_dir = os.path.join(self.output_dir, "safety_reports")
        if os.path.exists(safety_dir):
            safety_files = len([f for f in os.listdir(safety_dir) if f.endswith('.json')])
            print(f"安全报告: {safety_files} 个")

        if self.output_format == 'v2xformer':
            v2x_dir = os.path.join(self.output_dir, "v2xformer_format")
            if os.path.exists(v2x_dir):
                print(f"V2XFormer格式: 已生成")

        if self.output_format == 'kitti':
            kitti_dir = os.path.join(self.output_dir, "kitti_format")
            if os.path.exists(kitti_dir):
                print(f"KITTI格式: 已生成")

        total_frames = self.collected_frames
        elapsed = time.time() - self.start_time

        print(f"\n性能统计:")
        if total_frames > 0:
            fps = total_frames / max(elapsed, 0.1)
            print(f"  平均帧率: {fps:.2f} FPS")

            performance = self.performance_monitor.get_performance_summary()
            print(f"  帧时统计:")
            print(f"    平均: {performance['average_frame_time']:.3f}秒")
            print(f"    P50: {performance['frame_time_stats']['p50']:.3f}秒")
            print(f"    P95: {performance['frame_time_stats']['p95']:.3f}秒")
            print(f"    P99: {performance['frame_time_stats']['p99']:.3f}秒")
            print(f"  平均内存: {performance['average_memory_mb']:.1f} MB")
            print(f"  最大内存: {performance['max_memory_mb']:.1f} MB")
            print(f"  平均CPU: {performance['average_cpu_percent']:.1f}%")
            print(f"  总帧数: {total_frames}")
        else:
            print("  未收集到有效帧数据")

        print(f"\n输出目录: {self.output_dir}")
        print("=" * 60)

    def run_validation(self):
        if self.config['output'].get('validate_data', True) and self.collected_frames > 0:
            Log.info("运行数据验证...")
            DataValidator.validate_dataset(self.output_dir)


def main():
    parser = argparse.ArgumentParser(description='CVIPS 行人安全增强数据收集系统')

    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--scenario', type=str, default='pedestrian_safety', help='场景名称')
    parser.add_argument('--town', type=str, default='Town10HD',
                        choices=['Town03', 'Town04', 'Town05', 'Town10HD'], help='地图')
    parser.add_argument('--weather', type=str, default='clear',
                        choices=['clear', 'rainy', 'cloudy', 'foggy'], help='天气')
    parser.add_argument('--time-of-day', type=str, default='noon',
                        choices=['noon', 'sunset', 'night'], help='时间')

    parser.add_argument('--num-vehicles', type=int, default=8, help='背景车辆数')
    parser.add_argument('--num-pedestrians', type=int, default=12, help='行人数')
    parser.add_argument('--num-coop-vehicles', type=int, default=2, help='协同车辆数')

    parser.add_argument('--duration', type=int, default=60, help='收集时长(秒)')
    parser.add_argument('--capture-interval', type=float, default=2.0, help='捕捉间隔(秒)')
    parser.add_argument('--seed', type=int, help='随机种子')

    parser.add_argument('--batch-size', type=int, default=5, help='批处理大小')
    parser.add_argument('--enable-compression', action='store_true', help='启用数据压缩')
    parser.add_argument('--enable-downsampling', action='store_true', help='启用LiDAR下采样')

    parser.add_argument('--output-format', type=str, default='standard',
                        choices=['standard', 'v2xformer', 'kitti'], help='输出数据格式')

    parser.add_argument('--enable-lidar', action='store_true', help='启用LiDAR传感器')
    parser.add_argument('--enable-fusion', action='store_true', help='启用多传感器融合')
    parser.add_argument('--enable-v2x', action='store_true', help='启用V2X通信')
    parser.add_argument('--enable-cooperative', action='store_true', help='启用协同感知')
    parser.add_argument('--enable-enhancement', action='store_true', help='启用数据增强')
    parser.add_argument('--enable-annotations', action='store_true', help='启用自动标注')
    parser.add_argument('--enable-safety-monitor', action='store_true', default=True, help='启用行人安全监控')

    parser.add_argument('--run-analysis', action='store_true', help='运行数据集分析')
    parser.add_argument('--skip-validation', action='store_true', help='跳过数据验证')
    parser.add_argument('--skip-quality-check', action='store_true', help='跳过质量检查')

    args = parser.parse_args(remaining_argv)

    config = ConfigManager.load_config(args.config)
    config = ConfigManager.merge_args(config, args)

    config['performance']['batch_size'] = args.batch_size
    config['performance']['enable_compression'] = args.enable_compression
    config['performance']['enable_downsampling'] = args.enable_downsampling
    config['output']['output_format'] = args.output_format

    print("\n" + "=" * 60)
    print("CVIPS 行人安全增强数据收集系统")
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
    print(f"  安全监控: {'启用' if args.enable_safety_monitor else '禁用'}")

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