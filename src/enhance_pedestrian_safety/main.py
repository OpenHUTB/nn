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
from collections import deque

from carla_utils import setup_carla_path, import_carla_module
from config_manager import ConfigManager
from annotation_generator import AnnotationGenerator
from data_validator import DataValidator

carla_egg_path, remaining_argv = setup_carla_path()
carla = import_carla_module()


class Logger:
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


class Config:
    VEHICLE_TYPES = [
        'vehicle.tesla.model3',
        'vehicle.audi.tt',
        'vehicle.mini.cooperst',
        'vehicle.nissan.micra'
    ]


class WeatherManager:
    @staticmethod
    def create_weather(weather_type, time_of_day):
        weather = carla.WeatherParameters()

        if weather_type == 'clear':
            weather.cloudiness = 10.0
        elif weather_type == 'rainy':
            weather.cloudiness = 90.0
            weather.precipitation = 80.0
        elif weather_type == 'cloudy':
            weather.cloudiness = 70.0

        if time_of_day == 'noon':
            weather.sun_altitude_angle = 75
        elif time_of_day == 'sunset':
            weather.sun_altitude_angle = 15
        elif time_of_day == 'night':
            weather.sun_altitude_angle = -20

        return weather


class ImageStitcher:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.stitched_dir = os.path.join(output_dir, "stitched")
        os.makedirs(self.stitched_dir, exist_ok=True)

    def stitch(self, image_paths, frame_num, view_type="vehicle"):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            Logger.warning("PIL未安装，跳过图像拼接")
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


class TrafficSystem:
    def __init__(self, world, config):
        self.world = world
        self.config = config
        self.vehicles = []
        self.pedestrians = []

        seed = config['scenario'].get('seed', 42)
        random.seed(seed)

    def spawn_ego_vehicle(self):
        blueprint_lib = self.world.get_blueprint_library()

        for vtype in Config.VEHICLE_TYPES:
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
            Logger.info(f"主车: {vehicle.type_id}")
            return vehicle
        except Exception as e:
            Logger.warning(f"主车生成失败: {e}")
            return None

    def spawn_background_vehicles(self):
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

        Logger.info(f"背景车辆: {spawned} 辆")
        return spawned

    def spawn_pedestrians(self, center_location):
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
                Logger.debug(f"行人生成失败: {e}")

        Logger.info(f"行人: {spawned} 个")
        return spawned

    def cleanup(self):
        Logger.info("清理交通...")

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


class SensorSystem:
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

        self.image_stitcher = ImageStitcher(data_dir)

        self.camera_configs = {
            'vehicle': {
                'front_wide': {'loc': (2.0, 0, 1.8), 'rot': (0, -3, 0), 'fov': 100},
                'front_narrow': {'loc': (2.0, 0, 1.6), 'rot': (0, 0, 0), 'fov': 60},
                'right_side': {'loc': (0.5, 1.0, 1.5), 'rot': (0, -2, 45), 'fov': 90},
                'left_side': {'loc': (0.5, -1.0, 1.5), 'rot': (0, -2, -45), 'fov': 90}
            },
            'infrastructure': [
                {'name': 'north', 'offset': (0, -20, 12), 'rotation': (0, -25, 180)},
                {'name': 'south', 'offset': (0, 20, 12), 'rotation': (0, -25, 0)},
                {'name': 'east', 'offset': (20, 0, 12), 'rotation': (0, -25, -90)},
                {'name': 'west', 'offset': (-20, 0, 12), 'rotation': (0, -25, 90)}
            ]
        }

    def setup_vehicle_cameras(self, vehicle):
        if not vehicle:
            return 0

        installed = 0
        for cam_name, config_data in self.camera_configs['vehicle'].items():
            if self._create_camera(cam_name, config_data, vehicle, 'vehicle'):
                installed += 1

        Logger.info(f"车辆摄像头: {installed}")
        return installed

    def setup_infrastructure_cameras(self, center_location):
        installed = 0

        for cam_config in self.camera_configs['infrastructure']:
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

        Logger.info(f"基础设施摄像头: {installed}")
        return installed

    def _create_camera(self, name, config, parent, sensor_type):
        try:
            blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            blueprint.set_attribute('image_size_x', '1280')
            blueprint.set_attribute('image_size_y', '720')
            blueprint.set_attribute('fov', str(config.get('fov', 90)))

            location = carla.Location(config['loc'][0], config['loc'][1], config['loc'][2])
            rotation = carla.Rotation(config['rot'][0], config['rot'][1], config['rot'][2])
            transform = carla.Transform(location, rotation)

            if parent:
                camera = self.world.spawn_actor(blueprint, transform, attach_to=parent)
            else:
                camera = self.world.spawn_actor(blueprint, transform)

            save_dir = os.path.join(self.data_dir, "raw", sensor_type, name)
            os.makedirs(save_dir, exist_ok=True)

            callback = self._create_callback(save_dir, name, sensor_type)
            camera.listen(callback)

            self.sensors.append(camera)
            return True

        except Exception as e:
            Logger.warning(f"创建摄像头 {name} 失败: {e}")
            return False

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
                            self.image_stitcher.stitch(self.vehicle_buffer, self.frame_counter, 'vehicle')
                            self.vehicle_buffer.clear()
                    else:
                        self.infra_buffer[name] = filename
                        if len(self.infra_buffer) >= 4:
                            self.image_stitcher.stitch(self.infra_buffer, self.frame_counter, 'infrastructure')
                            self.infra_buffer.clear()

        return callback

    def get_frame_count(self):
        return self.frame_counter

    def cleanup(self):
        Logger.info(f"清理 {len(self.sensors)} 个传感器...")
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
        self.ego_vehicle = None
        self.scene_center = None

        self.setup_directories()

        self.traffic_system = None
        self.sensor_system = None

        self.start_time = None
        self.is_running = False

    def setup_directories(self):
        scenario = self.config['scenario']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir = os.path.join(
            self.config['output']['data_dir'],
            f"{scenario['name']}_{scenario['town']}_{timestamp}"
        )

        for subdir in ["raw/vehicle", "raw/infrastructure", "stitched", "metadata"]:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

        Logger.info(f"数据目录: {self.output_dir}")

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

                Logger.info(f"连接成功: {town}")
                return True

            except Exception as e:
                Logger.warning(f"连接尝试 {attempt}/5 失败: {str(e)[:50]}")
                time.sleep(2)

        return False

    def setup_scene(self):
        weather_cfg = self.config['scenario']
        weather = WeatherManager.create_weather(weather_cfg['weather'], weather_cfg['time_of_day'])
        self.world.set_weather(weather)
        Logger.info(f"天气: {weather_cfg['weather']}, 时间: {weather_cfg['time_of_day']}")

        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_points:
            self.scene_center = spawn_points[len(spawn_points) // 2].location
        else:
            self.scene_center = carla.Location(0, 0, 0)

        self.traffic_system = TrafficSystem(self.world, self.config)

        self.ego_vehicle = self.traffic_system.spawn_ego_vehicle()
        if not self.ego_vehicle:
            Logger.error("主车生成失败")
            return False

        self.traffic_system.spawn_background_vehicles()
        self.traffic_system.spawn_pedestrians(self.scene_center)

        time.sleep(2.0)
        return True

    def setup_sensors(self):
        self.sensor_system = SensorSystem(self.world, self.config, self.output_dir)

        vehicle_cams = self.sensor_system.setup_vehicle_cameras(self.ego_vehicle)
        infra_cams = self.sensor_system.setup_infrastructure_cameras(self.scene_center)

        total_cams = vehicle_cams + infra_cams
        if total_cams == 0:
            Logger.error("没有摄像头安装成功")
            return False

        Logger.info(f"摄像头总数: {total_cams}")
        return True

    def run_collection(self):
        duration = self.config['scenario']['duration']
        Logger.info(f"开始数据收集，时长: {duration}秒")

        self.start_time = time.time()
        self.is_running = True

        last_update = time.time()

        try:
            while time.time() - self.start_time < duration and self.is_running:
                current_time = time.time()
                elapsed = current_time - self.start_time

                if current_time - last_update >= 5.0:
                    frames = self.sensor_system.get_frame_count()
                    progress = (elapsed / duration) * 100

                    Logger.info(f"进度: {elapsed:.0f}/{duration}秒 ({progress:.1f}%) | 帧数: {frames}")
                    last_update = current_time

                time.sleep(0.05)

        except KeyboardInterrupt:
            Logger.info("数据收集被用户中断")
        finally:
            self.is_running = False
            elapsed = time.time() - self.start_time

            frames = self.sensor_system.get_frame_count() if self.sensor_system else 0
            Logger.info(f"收集完成: {frames}帧, 用时: {elapsed:.1f}秒")

            self._save_metadata(frames, elapsed)
            self._print_summary()

    def _save_metadata(self, total_frames, elapsed_time):
        metadata = {
            'scenario': self.config['scenario'],
            'traffic': self.config['traffic'],
            'sensors': self.config['sensors'],
            'collection': {
                'duration': elapsed_time,
                'total_frames': total_frames,
                'frame_rate': total_frames / elapsed_time if elapsed_time > 0 else 0
            }
        }

        meta_path = os.path.join(self.output_dir, "metadata", "collection_info.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        Logger.info(f"元数据保存: {meta_path}")

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("数据收集摘要")
        print("=" * 60)

        stitched_dir = os.path.join(self.output_dir, "stitched")
        if os.path.exists(stitched_dir):
            files = [f for f in os.listdir(stitched_dir) if f.endswith('.jpg')]
            print(f"拼接图像: {len(files)} 张")

        raw_dirs = ["vehicle", "infrastructure"]
        for raw_dir in raw_dirs:
            path = os.path.join(self.output_dir, "raw", raw_dir)
            if os.path.exists(path):
                total = 0
                for root, dirs, files in os.walk(path):
                    total += len([f for f in files if f.endswith('.png')])
                print(f"原始图像 ({raw_dir}): {total} 张")

        print(f"\n输出目录: {self.output_dir}")
        print("=" * 60)

    def cleanup(self):
        Logger.info("清理场景...")

        if self.sensor_system:
            self.sensor_system.cleanup()

        if self.traffic_system:
            self.traffic_system.cleanup()

        if self.ego_vehicle and self.ego_vehicle.is_alive:
            try:
                self.ego_vehicle.destroy()
            except:
                pass

        Logger.info("清理完成")

        # 运行数据验证
        if self.config['output'].get('validate_data', True):
            DataValidator.validate_dataset(self.output_dir)


def main():
    parser = argparse.ArgumentParser(description='CVIPS 数据收集系统')

    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--scenario', type=str, default='urban_scene', help='场景名称')
    parser.add_argument('--town', type=str, default='Town10HD',
                        choices=['Town03', 'Town04', 'Town05', 'Town10HD'], help='地图')
    parser.add_argument('--weather', type=str, default='clear',
                        choices=['clear', 'rainy', 'cloudy'], help='天气')
    parser.add_argument('--time-of-day', type=str, default='noon',
                        choices=['noon', 'sunset', 'night'], help='时间')
    parser.add_argument('--num-vehicles', type=int, default=8, help='背景车辆数')
    parser.add_argument('--num-pedestrians', type=int, default=10, help='行人数')
    parser.add_argument('--duration', type=int, default=60, help='收集时长(秒)')
    parser.add_argument('--capture-interval', type=float, default=2.0, help='捕捉间隔(秒)')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--validate-data', action='store_true', help='启用数据验证')

    args = parser.parse_args(remaining_argv)

    config = ConfigManager.load_config(args.config)
    config = ConfigManager.merge_args(config, args)

    if args.validate_data:
        config['output']['validate_data'] = True

    print("\n" + "=" * 60)
    print("CVIPS 数据收集系统")
    print("=" * 60)

    print(f"场景: {config['scenario']['name']}")
    print(f"地图: {config['scenario']['town']}")
    print(f"天气/时间: {config['scenario']['weather']}/{config['scenario']['time_of_day']}")
    print(f"时长: {config['scenario']['duration']}秒")
    print(f"交通: {config['traffic']['background_vehicles']}车辆 + {config['traffic']['pedestrians']}行人")
    print(f"验证: {'启用' if config['output'].get('validate_data', True) else '禁用'}")

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

        collector.run_collection()

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