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
from collections import defaultdict

from carla_utils import setup_carla_path, import_carla_module
from config_manager import ConfigManager
from annotation_generator import AnnotationGenerator
from data_validator import DataValidator
from scene_manager import SceneManager
from data_analyzer import DataAnalyzer

carla_egg_path, remaining_argv = setup_carla_path()
carla = import_carla_module()


class Log:
    """简单日志类"""

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
    """天气管理系统"""

    WEATHER_PRESETS = {
        'clear': {'cloudiness': 10, 'precipitation': 0, 'wind': 5},
        'rainy': {'cloudiness': 90, 'precipitation': 80, 'wind': 15},
        'cloudy': {'cloudiness': 70, 'precipitation': 10, 'wind': 10},
        'foggy': {'cloudiness': 50, 'precipitation': 0, 'fog_density': 40},
        'stormy': {'cloudiness': 95, 'precipitation': 90, 'wind': 25, 'fog_density': 20}
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

        # 时间设置
        if time_of_day == 'noon':
            weather.sun_altitude_angle = 75
        elif time_of_day == 'sunset':
            weather.sun_altitude_angle = 15
        elif time_of_day == 'night':
            weather.sun_altitude_angle = -20
        elif time_of_day == 'morning':
            weather.sun_altitude_angle = 30
        elif time_of_day == 'dawn':
            weather.sun_altitude_angle = 5

        return weather


class ImageProcessor:
    """图像处理器 - 支持多种输出格式"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.stitched_dir = os.path.join(output_dir, "stitched")
        os.makedirs(self.stitched_dir, exist_ok=True)

        self.has_pil = self._check_pil()

    def _check_pil(self):
        try:
            from PIL import Image
            return True
        except ImportError:
            Log.warning("PIL未安装，图像处理功能受限")
            return False

    def stitch_4_view(self, image_paths, frame_num, view_type="vehicle"):
        """四视角拼接"""
        if not self.has_pil:
            return False

        try:
            from PIL import Image, ImageDraw

            positions = [(10, 10), (660, 10), (10, 390), (660, 390)]

            canvas = Image.new('RGB', (640 * 2 + 20, 360 * 2 + 20), (40, 40, 40))
            draw = ImageDraw.Draw(canvas)

            for idx, (cam_name, img_path) in enumerate(list(image_paths.items())[:4]):
                img = self._load_image(img_path)
                canvas.paste(img, positions[idx])

                label = cam_name.replace('_', ' ').title()
                draw.text((positions[idx][0] + 5, positions[idx][1] + 5),
                          label, fill=(255, 255, 200))

            # 添加帧信息和时间戳
            timestamp = datetime.now().strftime("%H:%M:%S")
            draw.text((canvas.width // 2 - 100, 5), f"Frame {frame_num:06d}",
                      fill=(255, 255, 255))
            draw.text((canvas.width - 150, canvas.height - 25), timestamp,
                      fill=(200, 200, 200))

            output_path = os.path.join(self.stitched_dir, f"{view_type}_{frame_num:06d}.jpg")
            canvas.save(output_path, "JPEG", quality=95)

            return True

        except Exception as e:
            Log.error(f"图像拼接失败: {e}")
            return False

    def _load_image(self, img_path):
        """加载图像，处理异常"""
        if not self.has_pil:
            return None

        try:
            from PIL import Image

            if img_path and os.path.exists(img_path):
                img = Image.open(img_path)
                return img.resize((640, 360))

        except Exception as e:
            Log.debug(f"加载图像失败 {img_path}: {e}")

        # 返回灰色占位图像
        from PIL import Image
        return Image.new('RGB', (640, 360), (80, 80, 80))


class TrafficManager:
    """交通管理器 - 增强版"""

    def __init__(self, world, config):
        self.world = world
        self.config = config
        self.vehicles = []
        self.pedestrians = []
        self.static_obstacles = []

        # 设置随机种子
        seed = config['scenario'].get('seed', random.randint(1, 1000))
        random.seed(seed)
        Log.info(f"随机种子: {seed}")

    def spawn_ego_vehicle(self, vehicle_type=None):
        """生成主车"""
        blueprint_lib = self.world.get_blueprint_library()

        if vehicle_type:
            if blueprint_lib.filter(vehicle_type):
                vehicle_bp = random.choice(blueprint_lib.filter(vehicle_type))
            else:
                vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))
        else:
            # 常用车辆类型
            common_vehicles = [
                'vehicle.tesla.model3',
                'vehicle.audi.tt',
                'vehicle.mini.cooperst',
                'vehicle.nissan.micra',
                'vehicle.mercedes.coupe',
                'vehicle.bmw.grandtourer'
            ]

            for vtype in common_vehicles:
                if blueprint_lib.filter(vtype):
                    vehicle_bp = random.choice(blueprint_lib.filter(vtype))
                    break
            else:
                vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            Log.error("没有可用的生成点")
            return None

        # 尝试多个生成点
        for _ in range(3):
            spawn_point = random.choice(spawn_points)
            try:
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                vehicle.set_autopilot(True)

                # 初始控制
                control = carla.VehicleControl()
                control.throttle = 0.3
                control.steer = random.uniform(-0.1, 0.1)
                vehicle.apply_control(control)

                Log.info(f"主车: {vehicle.type_id}")
                return vehicle

            except Exception as e:
                Log.debug(f"车辆生成失败: {e}")

        return None

    def spawn_traffic(self, center_location, num_vehicles=8, num_pedestrians=6):
        """生成交通"""
        vehicles_spawned = self._spawn_vehicles(center_location, num_vehicles)
        pedestrians_spawned = self._spawn_pedestrians(center_location, num_pedestrians)

        Log.info(f"交通生成: {vehicles_spawned}辆车, {pedestrians_spawned}个行人")
        return vehicles_spawned + pedestrians_spawned

    def _spawn_vehicles(self, center_location, count):
        """生成车辆"""
        blueprint_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            return 0

        spawned = 0
        for i in range(min(count, 15)):
            try:
                vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))
                spawn_point = random.choice(spawn_points)

                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                vehicle.set_autopilot(True)

                # 设置不同速度
                speed_factor = random.uniform(0.6, 1.2)
                control = carla.VehicleControl()
                control.throttle = 0.25 * speed_factor
                vehicle.apply_control(control)

                self.vehicles.append(vehicle)
                spawned += 1

            except Exception as e:
                Log.debug(f"背景车辆生成失败: {e}")

        return spawned

    def _spawn_pedestrians(self, center_location, count):
        """生成行人"""
        blueprint_lib = self.world.get_blueprint_library()

        spawned = 0
        for i in range(min(count, 12)):
            try:
                ped_bps = list(blueprint_lib.filter('walker.pedestrian.*'))
                if not ped_bps:
                    continue

                ped_bp = random.choice(ped_bps)

                # 在中心位置周围生成
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(3.0, 15.0)

                location = carla.Location(
                    x=center_location.x + distance * math.cos(angle),
                    y=center_location.y + distance * math.sin(angle),
                    z=center_location.z + 0.5
                )

                pedestrian = self.world.spawn_actor(ped_bp, carla.Transform(location))

                # 尝试添加AI控制器
                try:
                    controller_bp = blueprint_lib.find('controller.ai.walker')
                    if controller_bp:
                        controller = self.world.spawn_actor(
                            controller_bp,
                            carla.Transform(),
                            attach_to=pedestrian
                        )
                        controller.start()

                        # 设置随机目标
                        target = self.world.get_random_location_from_navigation()
                        if target:
                            controller.go_to_location(target)

                        self.pedestrians.append((pedestrian, controller))
                    else:
                        self.pedestrians.append((pedestrian, None))

                except:
                    self.pedestrians.append((pedestrian, None))

                spawned += 1

            except Exception as e:
                Log.debug(f"行人生成失败: {e}")

        return spawned

    def add_static_obstacles(self, center_location):
        """添加静态障碍物"""
        try:
            cones = SceneManager.spawn_traffic_cones(self.world, center_location, 6)
            barriers = SceneManager.spawn_construction_barriers(self.world, center_location, 3)

            self.static_obstacles.extend(cones)
            self.static_obstacles.extend(barriers)

            Log.info(f"添加障碍物: {len(cones)}个锥桶, {len(barriers)}个路障")

        except Exception as e:
            Log.warning(f"添加障碍物失败: {e}")

    def setup_accident_scene(self, center_location):
        """设置事故场景"""
        try:
            accidents, emergency = SceneManager.setup_traffic_accident(
                self.world, center_location, 'minor'
            )

            self.vehicles.extend(accidents)
            self.vehicles.extend(emergency)

            Log.info(f"事故场景: {len(accidents)}辆事故车, {len(emergency)}辆应急车")

        except Exception as e:
            Log.warning(f"设置事故场景失败: {e}")

    def update_traffic_flow(self):
        """更新交通流"""
        # 随机改变一些车辆的速度
        for vehicle in self.vehicles[:10]:  # 只处理前10辆车
            try:
                if random.random() < 0.01:  # 1%概率
                    control = vehicle.get_control()
                    control.throttle *= random.uniform(0.8, 1.2)
                    vehicle.apply_control(control)
            except:
                pass

    def cleanup(self):
        """清理交通"""
        Log.info("清理交通系统...")

        # 清理车辆
        for vehicle in self.vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except:
                pass

        # 清理行人
        for pedestrian, controller in self.pedestrians:
            try:
                if controller and controller.is_alive:
                    controller.stop()
                    controller.destroy()
                if pedestrian.is_alive:
                    pedestrian.destroy()
            except:
                pass

        # 清理障碍物
        for obstacle in self.static_obstacles:
            try:
                if obstacle.is_alive:
                    obstacle.destroy()
            except:
                pass

        self.vehicles.clear()
        self.pedestrians.clear()
        self.static_obstacles.clear()


class SensorSystem:
    """传感器系统 - 支持多类型传感器"""

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
        self.annotation_generator = None

        # 摄像头配置
        self.camera_configs = {
            'vehicle': {
                'front_wide': {'loc': (2.0, 0, 1.8), 'rot': (0, -3, 0), 'fov': 100},
                'front_narrow': {'loc': (2.0, 0, 1.6), 'rot': (0, 0, 0), 'fov': 60},
                'right_side': {'loc': (0.5, 1.0, 1.5), 'rot': (0, -2, 45), 'fov': 90},
                'left_side': {'loc': (0.5, -1.0, 1.5), 'rot': (0, -2, -45), 'fov': 90},
                'rear': {'loc': (-1.5, 0, 1.5), 'rot': (0, -2, 180), 'fov': 90}
            },
            'infrastructure': [
                {'name': 'north', 'loc': (0, -25, 15), 'rot': (0, -25, 180)},
                {'name': 'south', 'loc': (0, 25, 15), 'rot': (0, -25, 0)},
                {'name': 'east', 'loc': (25, 0, 15), 'rot': (0, -25, -90)},
                {'name': 'west', 'loc': (-25, 0, 15), 'rot': (0, -25, 90)},
                {'name': 'top', 'loc': (0, 0, 25), 'rot': (-90, 0, 0)}
            ]
        }

    def enable_annotations(self):
        """启用自动标注"""
        self.annotation_generator = AnnotationGenerator(self.data_dir)

    def setup_cameras(self, vehicle, center_location, enable_vehicle=True, enable_infra=True):
        """设置摄像头系统"""
        installed = 0

        if enable_vehicle and vehicle:
            installed += self._setup_vehicle_cameras(vehicle)

        if enable_infra:
            installed += self._setup_infrastructure_cameras(center_location)

        Log.info(f"摄像头安装: {installed}个")
        return installed

    def _setup_vehicle_cameras(self, vehicle):
        """设置车辆摄像头"""
        installed = 0

        for cam_name, config_data in self.camera_configs['vehicle'].items():
            if self._create_camera(cam_name, config_data, vehicle, 'vehicle'):
                installed += 1

        return installed

    def _setup_infrastructure_cameras(self, center_location):
        """设置基础设施摄像头"""
        installed = 0

        for cam_config in self.camera_configs['infrastructure']:
            sensor_config = {
                'loc': (
                    center_location.x + cam_config['loc'][0],
                    center_location.y + cam_config['loc'][1],
                    center_location.z + cam_config['loc'][2]
                ),
                'rot': cam_config['rot'],
                'fov': 90
            }

            if self._create_camera(cam_config['name'], sensor_config, None, 'infrastructure'):
                installed += 1

        return installed

    def _create_camera(self, name, config, parent, sensor_type):
        """创建摄像头"""
        try:
            blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')

            # 设置摄像头参数
            img_size = self.config['sensors'].get('image_size', [1280, 720])
            blueprint.set_attribute('image_size_x', str(img_size[0]))
            blueprint.set_attribute('image_size_y', str(img_size[1]))
            blueprint.set_attribute('fov', str(config.get('fov', 90)))

            # 质量设置
            blueprint.set_attribute('motion_blur_intensity', '0')
            blueprint.set_attribute('enable_postprocess_effects', 'False')

            location = carla.Location(*config['loc'])
            rotation = carla.Rotation(*config['rot'])
            transform = carla.Transform(location, rotation)

            # 生成摄像头
            if parent:
                camera = self.world.spawn_actor(blueprint, transform, attach_to=parent)
            else:
                camera = self.world.spawn_actor(blueprint, transform)

            # 创建保存目录
            save_dir = os.path.join(self.data_dir, "raw", sensor_type, name)
            os.makedirs(save_dir, exist_ok=True)

            # 设置回调
            callback = self._create_image_callback(save_dir, name, sensor_type)
            camera.listen(callback)

            self.sensors.append(camera)
            return True

        except Exception as e:
            Log.warning(f"创建摄像头 {name} 失败: {e}")
            return False

    def _create_image_callback(self, save_dir, name, sensor_type):
        """创建图像回调函数"""
        capture_interval = self.config['sensors'].get('capture_interval', 2.0)

        def callback(image):
            current_time = time.time()

            if current_time - self.last_capture_time >= capture_interval:
                self.frame_counter += 1
                self.last_capture_time = current_time

                # 保存原始图像
                filename = os.path.join(save_dir, f"{name}_{self.frame_counter:06d}.png")
                image.save_to_disk(filename, carla.ColorConverter.Raw)

                # 更新缓冲区
                with self.buffer_lock:
                    if sensor_type == 'vehicle':
                        self.vehicle_buffer[name] = filename
                        if len(self.vehicle_buffer) >= 4:
                            self.image_processor.stitch_4_view(
                                self.vehicle_buffer, self.frame_counter, 'vehicle'
                            )
                            self.vehicle_buffer.clear()
                    else:
                        self.infra_buffer[name] = filename
                        if len(self.infra_buffer) >= 4:
                            self.image_processor.stitch_4_view(
                                self.infra_buffer, self.frame_counter, 'infrastructure'
                            )
                            self.infra_buffer.clear()

                # 生成标注
                if self.annotation_generator:
                    timestamp = datetime.now().isoformat()
                    self.annotation_generator.detect_objects(
                        self.world, self.frame_counter, timestamp
                    )

        return callback

    def get_frame_count(self):
        """获取帧数"""
        return self.frame_counter

    def cleanup(self):
        """清理传感器"""
        Log.info(f"清理 {len(self.sensors)} 个传感器...")
        for sensor in self.sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except:
                pass
        self.sensors.clear()


class DataCollector:
    """数据收集器 - 主控制器"""

    def __init__(self, config):
        self.config = config
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.scene_center = None

        self.setup_directories()

        self.traffic_manager = None
        self.sensor_system = None

        self.start_time = None
        self.is_running = False
        self.collected_frames = 0

        # 性能监控
        self.performance_stats = {
            'fps_history': [],
            'memory_usage': [],
            'start_time': 0
        }

    def setup_directories(self):
        """设置目录结构"""
        scenario = self.config['scenario']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir = os.path.join(
            self.config['output']['data_dir'],
            f"{scenario['name']}_{scenario['town']}_{timestamp}"
        )

        # 创建目录结构
        directories = [
            "raw/vehicle",
            "raw/infrastructure",
            "stitched",
            "annotations",
            "metadata",
            "calibration",
            "logs"
        ]

        for subdir in directories:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

        Log.info(f"数据目录: {self.output_dir}")

    def connect(self):
        """连接到CARLA服务器"""
        for attempt in range(1, 6):
            try:
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(15.0)

                town = self.config['scenario']['town']
                self.world = self.client.load_world(town)

                # 设置世界参数
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(settings)

                Log.info(f"连接成功: {town}")
                return True

            except Exception as e:
                Log.warning(f"连接尝试 {attempt}/5 失败: {str(e)[:50]}")
                if attempt < 5:
                    time.sleep(3)

        return False

    def setup_scene(self, scene_type=None, add_obstacles=False, add_accident=False):
        """设置场景"""
        # 设置天气
        weather_cfg = self.config['scenario']
        weather = WeatherSystem.create_weather(
            weather_cfg['weather'],
            weather_cfg['time_of_day']
        )
        self.world.set_weather(weather)

        Log.info(f"天气: {weather_cfg['weather']}, 时间: {weather_cfg['time_of_day']}")

        # 获取场景中心
        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_points:
            self.scene_center = spawn_points[len(spawn_points) // 2].location
        else:
            self.scene_center = carla.Location(0, 0, 0)

        # 应用场景配置
        if scene_type:
            self.config = SceneManager.setup_scene(
                self.world, self.config, scene_type
            )
            SceneManager.save_scene_description(
                self.output_dir, scene_type, self.config
            )

        # 初始化交通管理器
        self.traffic_manager = TrafficManager(self.world, self.config)

        # 生成主车
        self.ego_vehicle = self.traffic_manager.spawn_ego_vehicle()
        if not self.ego_vehicle:
            Log.error("主车生成失败")
            return False

        # 生成交通
        vehicles = self.config['traffic'].get('background_vehicles', 8)
        pedestrians = self.config['traffic'].get('pedestrians', 6)
        self.traffic_manager.spawn_traffic(
            self.scene_center, vehicles, pedestrians
        )

        # 添加障碍物
        if add_obstacles:
            self.traffic_manager.add_static_obstacles(self.scene_center)

        # 添加事故场景
        if add_accident:
            self.traffic_manager.setup_accident_scene(self.scene_center)

        # 等待场景稳定
        Log.info("等待场景稳定...")
        time.sleep(5.0)

        return True

    def setup_sensors(self, enable_annotations=True):
        """设置传感器"""
        self.sensor_system = SensorSystem(
            self.world,
            self.config,
            self.output_dir
        )

        if enable_annotations:
            self.sensor_system.enable_annotations()

        # 安装摄像头
        vehicle_cams = self.config['sensors'].get('vehicle_cameras', 4)
        infra_cams = self.config['sensors'].get('infrastructure_cameras', 4)

        installed = self.sensor_system.setup_cameras(
            self.ego_vehicle,
            self.scene_center,
            vehicle_cams > 0,
            infra_cams > 0
        )

        if installed == 0:
            Log.error("没有摄像头安装成功")
            return False

        Log.info(f"传感器系统就绪: {installed}个摄像头")
        return True

    def collect_data(self):
        """收集数据"""
        duration = self.config['scenario']['duration']
        Log.info(f"开始数据收集，时长: {duration}秒")

        self.start_time = time.time()
        self.is_running = True
        self.performance_stats['start_time'] = self.start_time

        last_update = time.time()
        last_fps_check = time.time()
        frame_counter = 0

        try:
            while time.time() - self.start_time < duration and self.is_running:
                current_time = time.time()
                elapsed = current_time - self.start_time

                # 更新交通
                if self.traffic_manager:
                    self.traffic_manager.update_traffic_flow()

                # 性能监控
                if current_time - last_fps_check >= 1.0:
                    current_frames = self.sensor_system.get_frame_count()
                    fps = current_frames - frame_counter
                    self.performance_stats['fps_history'].append(fps)
                    frame_counter = current_frames
                    last_fps_check = current_time

                # 进度更新
                if current_time - last_update >= 5.0:
                    frames = self.sensor_system.get_frame_count()
                    progress = (elapsed / duration) * 100
                    remaining = duration - elapsed

                    avg_fps = sum(self.performance_stats['fps_history'][-5:]) / \
                              min(len(self.performance_stats['fps_history'][-5:]), 5)

                    Log.info(
                        f"进度: {elapsed:.0f}/{duration}秒 ({progress:.1f}%) | "
                        f"帧数: {frames} | FPS: {avg_fps:.1f} | "
                        f"剩余: {remaining:.0f}秒"
                    )
                    last_update = current_time

                time.sleep(0.01)

        except KeyboardInterrupt:
            Log.info("数据收集被用户中断")
        finally:
            self.is_running = False
            elapsed = time.time() - self.start_time

            self.collected_frames = self.sensor_system.get_frame_count() if self.sensor_system else 0
            Log.info(f"收集完成: {self.collected_frames}帧, 用时: {elapsed:.1f}秒")

            self._save_collection_metadata()
            self._generate_dataset_report()

    def _save_collection_metadata(self):
        """保存收集元数据"""
        metadata = {
            'scenario': self.config['scenario'],
            'traffic': self.config['traffic'],
            'sensors': self.config['sensors'],
            'output': self.config['output'],
            'collection': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'duration_seconds': round(time.time() - self.start_time, 2),
                'total_frames': self.collected_frames,
                'frame_rate': round(self.collected_frames / (time.time() - self.start_time), 2)
            },
            'performance': {
                'average_fps': round(
                    sum(self.performance_stats['fps_history']) /
                    max(len(self.performance_stats['fps_history']), 1),
                    2
                ),
                'fps_history': self.performance_stats['fps_history'][:20]  # 只保留前20个
            }
        }

        meta_path = os.path.join(self.output_dir, "metadata", "collection_info.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        Log.info(f"元数据保存: {meta_path}")

    def _generate_dataset_report(self):
        """生成数据集报告"""
        print("\n" + "=" * 60)
        print("数据集生成报告")
        print("=" * 60)

        # 基本统计
        total_size = self._get_directory_size(self.output_dir)
        print(f"总数据大小: {total_size} MB")
        print(f"总帧数: {self.collected_frames}")

        # 文件统计
        stitched_dir = os.path.join(self.output_dir, "stitched")
        if os.path.exists(stitched_dir):
            stitched_files = [f for f in os.listdir(stitched_dir) if f.endswith('.jpg')]
            print(f"拼接图像: {len(stitched_files)} 张")

        # 原始图像统计
        raw_stats = self._count_raw_images()
        print(f"原始图像: {raw_stats['total']} 张")
        print(f"  车辆视角: {raw_stats['vehicle']} 张")
        print(f"  基础设施: {raw_stats['infrastructure']} 张")

        # 标注统计
        annotations_dir = os.path.join(self.output_dir, "annotations")
        if os.path.exists(annotations_dir):
            json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
            print(f"标注文件: {len(json_files)} 个")

        print(f"\n输出目录: {self.output_dir}")
        print("=" * 60)

    def _count_raw_images(self):
        """统计原始图像"""
        stats = {'total': 0, 'vehicle': 0, 'infrastructure': 0}

        for view_type in ['vehicle', 'infrastructure']:
            path = os.path.join(self.output_dir, "raw", view_type)
            if os.path.exists(path):
                count = 0
                for root, dirs, files in os.walk(path):
                    count += len([f for f in files if f.endswith('.png')])

                stats[view_type] = count
                stats['total'] += count

        return stats

    def _get_directory_size(self, path):
        """计算目录大小"""
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return round(total / (1024 * 1024), 2)

    def run_analysis(self):
        """运行数据集分析"""
        Log.info("运行数据集分析...")
        DataAnalyzer.analyze_dataset(self.output_dir)

    def run_validation(self):
        """运行数据验证"""
        Log.info("运行数据验证...")
        DataValidator.validate_dataset(self.output_dir)

    def cleanup(self):
        """清理场景"""
        Log.info("清理场景...")

        if self.sensor_system:
            self.sensor_system.cleanup()

        if self.traffic_manager:
            self.traffic_manager.cleanup()

        if self.ego_vehicle and self.ego_vehicle.is_alive:
            try:
                self.ego_vehicle.destroy()
            except:
                pass

        Log.info("清理完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CVIPS 高级交通场景数据集生成器')

    # 基础参数
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--scenario', type=str, default='urban_scene', help='场景名称')
    parser.add_argument('--town', type=str, default='Town10HD',
                        choices=['Town03', 'Town04', 'Town05', 'Town10HD'], help='地图')
    parser.add_argument('--weather', type=str, default='clear',
                        choices=['clear', 'rainy', 'cloudy', 'foggy', 'stormy'], help='天气')
    parser.add_argument('--time-of-day', type=str, default='noon',
                        choices=['noon', 'sunset', 'night', 'morning', 'dawn'], help='时间')

    # 交通参数
    parser.add_argument('--num-vehicles', type=int, default=8, help='背景车辆数')
    parser.add_argument('--num-pedestrians', type=int, default=6, help='行人数')
    parser.add_argument('--ego-vehicle', type=str, help='主车类型')

    # 收集参数
    parser.add_argument('--duration', type=int, default=60, help='收集时长(秒)')
    parser.add_argument('--capture-interval', type=float, default=2.0, help='捕捉间隔(秒)')
    parser.add_argument('--seed', type=int, help='随机种子')

    # 场景参数
    parser.add_argument('--scene-type', type=str,
                        choices=['intersection_4way', 'highway', 'urban_street',
                                 'night_scene', 'rainy_intersection'], help='场景类型')
    parser.add_argument('--add-obstacles', action='store_true', help='添加障碍物')
    parser.add_argument('--add-accident', action='store_true', help='添加事故场景')

    # 功能参数
    parser.add_argument('--enable-annotations', action='store_true', help='启用自动标注')
    parser.add_argument('--run-analysis', action='store_true', help='运行数据集分析')
    parser.add_argument('--run-validation', action='store_true', help='运行数据验证')
    parser.add_argument('--skip-validation', action='store_true', help='跳过数据验证')

    args = parser.parse_args(remaining_argv)

    # 加载配置
    config = ConfigManager.load_config(args.config)
    config = ConfigManager.merge_args(config, args)

    # 显示配置
    print("\n" + "=" * 60)
    print("CVIPS 交通场景数据集生成器 v8.0")
    print("=" * 60)

    print(f"场景: {config['scenario']['name']}")
    print(f"地图: {config['scenario']['town']}")
    print(f"天气/时间: {config['scenario']['weather']}/{config['scenario']['time_of_day']}")
    print(f"时长: {config['scenario']['duration']}秒")
    print(f"交通: {config['traffic']['background_vehicles']}车辆 + {config['traffic']['pedestrians']}行人")

    if args.scene_type:
        print(f"场景类型: {args.scene_type}")
    if args.add_obstacles:
        print(f"障碍物: 启用")
    if args.add_accident:
        print(f"事故场景: 启用")

    print(f"标注: {'启用' if args.enable_annotations else '禁用'}")
    print(f"分析: {'启用' if args.run_analysis else '禁用'}")
    print(f"验证: {'禁用' if args.skip_validation else '启用'}")

    # 创建收集器
    collector = DataCollector(config)

    try:
        # 连接服务器
        if not collector.connect():
            print("连接CARLA服务器失败")
            return

        # 设置场景
        if not collector.setup_scene(
                args.scene_type,
                args.add_obstacles,
                args.add_accident
        ):
            print("场景设置失败")
            collector.cleanup()
            return

        # 设置传感器
        if not collector.setup_sensors(args.enable_annotations):
            print("传感器设置失败")
            collector.cleanup()
            return

        # 收集数据
        collector.collect_data()

        # 运行分析
        if args.run_analysis:
            collector.run_analysis()

        # 运行验证
        if not args.skip_validation:
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