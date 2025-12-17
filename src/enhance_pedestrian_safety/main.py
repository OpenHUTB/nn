import sys
import os
import time
import random
import argparse
import traceback
import math
import threading
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from carla_utils import setup_carla_path, import_carla_module

carla_egg_path, remaining_argv = setup_carla_path()
carla = import_carla_module()


class ImageStitcher:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.stitched_dir = os.path.join(output_dir, "stitched_images")
        os.makedirs(self.stitched_dir, exist_ok=True)

        self.font = self._load_font()

    def _load_font(self):
        font_paths = [
            "arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc"
        ]

        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, 20)
            except:
                continue
        return ImageFont.load_default()

    def stitch_ego_vehicle_images(self, image_paths, frame_num):
        positions = {
            'front_wide': (10, 10),
            'front_narrow': (660, 10),
            'right_side': (10, 390),
            'left_side': (660, 390)
        }

        images = []
        for cam_name in positions.keys():
            img_path = image_paths.get(cam_name)
            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).resize((640, 360))
                    images.append((cam_name, img))
                except:
                    images.append((cam_name, Image.new('RGB', (640, 360), (100, 100, 100))))
            else:
                images.append((cam_name, Image.new('RGB', (640, 360), (100, 100, 100))))

        if len(images) < 4:
            return False

        canvas = Image.new('RGB', (640 * 2 + 20, 360 * 2 + 20), (50, 50, 50))

        for cam_name, img in images:
            x, y = positions[cam_name]
            canvas.paste(img, (x, y))
            draw = ImageDraw.Draw(canvas)
            label = cam_name.replace('_', ' ').title()
            draw.text((x + 10, y + 10), label, fill=(255, 255, 255), font=self.font)

        draw = ImageDraw.Draw(canvas)
        title = f"CVIPS - Frame {frame_num:04d}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((canvas.width // 2 - 150, 5), title, fill=(255, 255, 255), font=self.font)
        draw.text((10, canvas.height - 30), timestamp, fill=(200, 200, 200), font=self.font)

        output_path = os.path.join(self.stitched_dir, f"ego_stitched_{frame_num:04d}.jpg")
        canvas.save(output_path, "JPEG", quality=95)

        return True


class PedestrianController:
    STATE_WAITING = "waiting"
    STATE_CROSSING = "crossing"
    STATE_WALKING = "walking"
    STATE_STOPPED = "stopped"

    def __init__(self, world):
        self.world = world
        self.pedestrians = []
        self.running = True

    def spawn_pedestrian(self, location, behavior_type="crossing"):
        blueprint_lib = self.world.get_blueprint_library()
        ped_bps = list(blueprint_lib.filter('walker.pedestrian.*'))

        if not ped_bps:
            return None

        ped_bp = random.choice(ped_bps)
        location.z += 1.0

        try:
            pedestrian = self.world.spawn_actor(ped_bp, carla.Transform(location))
            controller_bp = blueprint_lib.find('controller.ai.walker')
            controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=pedestrian)
            controller.start()

            behavior_state = {
                'state': self.STATE_WAITING if behavior_type in ["crossing", "hesitant"] else self.STATE_WALKING,
                'wait_start': time.time(),
                'wait_duration': random.uniform(2.0, 5.0),
                'target': None,
                'original': location,
                'behavior': behavior_type
            }

            if behavior_type == "walking":
                behavior_state['target'] = self._get_random_location()

            self.pedestrians.append((pedestrian, controller, behavior_state))
            return pedestrian

        except Exception as e:
            print(f"生成行人失败: {e}")
            return None

    def _get_random_location(self):
        try:
            return self.world.get_random_location_from_navigation()
        except:
            return None

    def update_behaviors(self):
        current_time = time.time()

        for pedestrian, controller, state in self.pedestrians:
            if not pedestrian.is_alive or not controller.is_alive:
                continue

            if state['behavior'] == "crossing":
                self._update_crossing(pedestrian, controller, state, current_time)
            elif state['behavior'] == "hesitant":
                self._update_hesitant(pedestrian, controller, state, current_time)
            else:
                self._update_walking(pedestrian, controller, state, current_time)

    def _update_crossing(self, pedestrian, controller, state, current_time):
        if state['state'] == self.STATE_WAITING:
            if current_time - state['wait_start'] >= state['wait_duration']:
                target = carla.Location(
                    x=state['original'].x + random.uniform(15.0, 25.0),
                    y=state['original'].y + random.uniform(-5.0, 5.0),
                    z=state['original'].z
                )
                state['target'] = target
                state['state'] = self.STATE_CROSSING
                state['cross_start'] = current_time
                controller.go_to_location(target)

        elif state['state'] == self.STATE_CROSSING:
            distance = pedestrian.get_location().distance(state['target'])
            if distance < 2.0 or current_time - state['cross_start'] > 15.0:
                state['state'] = self.STATE_STOPPED

    def _update_hesitant(self, pedestrian, controller, state, current_time):
        if state['state'] == self.STATE_WAITING:
            if current_time - state['wait_start'] >= state['wait_duration']:
                target = self._get_random_location()
                if target:
                    state['target'] = target
                    state['state'] = self.STATE_WALKING
                    state['walk_start'] = current_time
                    state['walk_duration'] = random.uniform(3.0, 8.0)
                    controller.go_to_location(target)

        elif state['state'] == self.STATE_WALKING:
            if current_time - state['walk_start'] >= state['walk_duration']:
                state['state'] = self.STATE_WAITING
                state['wait_start'] = current_time
                state['wait_duration'] = random.uniform(1.0, 4.0)

    def _update_walking(self, pedestrian, controller, state, current_time):
        if state['state'] == self.STATE_WALKING and state['target']:
            distance = pedestrian.get_location().distance(state['target'])
            if distance < 2.0:
                new_target = self._get_random_location()
                if new_target:
                    state['target'] = new_target
                    controller.go_to_location(new_target)

    def start_updates(self):
        def update_loop():
            while self.running:
                try:
                    self.update_behaviors()
                    time.sleep(0.5)
                except Exception as e:
                    print(f"行为更新错误: {e}")
                    time.sleep(1.0)

        threading.Thread(target=update_loop, daemon=True).start()

    def cleanup(self):
        self.running = False
        for pedestrian, controller, _ in self.pedestrians:
            try:
                if controller.is_alive:
                    controller.stop()
                    controller.destroy()
                if pedestrian.is_alive:
                    pedestrian.destroy()
            except:
                pass
        self.pedestrians.clear()


class DataGenerator:
    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.actors = []
        self.sensors = []
        self.frame_count = 0
        self.last_capture_time = 0

        self.setup_output_directory()
        self.stitcher = ImageStitcher(self.output_dir)
        self.ped_controller = None

        self.image_buffer = {}
        self.buffer_lock = threading.Lock()

    def setup_output_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("cvips_data", f"{self.args.scenario}_{timestamp}")

        self.raw_dirs = {}
        for view in ['front_wide', 'front_narrow', 'right_side', 'left_side']:
            dir_path = os.path.join(self.output_dir, "raw", "ego_vehicle", view)
            os.makedirs(dir_path, exist_ok=True)
            self.raw_dirs[view] = dir_path

        print(f"数据输出目录: {self.output_dir}")

    def connect_to_server(self):
        for attempt in range(1, 6):
            try:
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(15.0)

                if self.args.town:
                    self.world = self.client.load_world(self.args.town)
                else:
                    self.world = self.client.get_world()

                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)

                print(f"连接成功! 地图: {self.world.get_map().name}")
                return True

            except Exception as e:
                print(f"尝试 {attempt}/5 失败: {str(e)[:80]}")
                if attempt < 5:
                    time.sleep(3)

        return False

    def setup_scene(self):
        self.ped_controller = PedestrianController(self.world)

        self.set_weather()
        time.sleep(2.0)

        ego_vehicle = self.spawn_ego_vehicle()
        if not ego_vehicle:
            return None

        self.spawn_pedestrians()
        self.spawn_background_vehicles()
        time.sleep(5.0)

        self.ped_controller.start_updates()
        return ego_vehicle

    def set_weather(self):
        weather = carla.WeatherParameters()

        if self.args.weather == 'clear':
            weather.sun_altitude_angle = 75
            weather.cloudiness = 5.0
        elif self.args.weather == 'rainy':
            weather.sun_altitude_angle = 40
            weather.cloudiness = 90.0
            weather.precipitation = 60.0
        else:
            weather.sun_altitude_angle = 60
            weather.cloudiness = 70.0

        if self.args.time_of_day == 'night':
            weather.sun_altitude_angle = -10
        elif self.args.time_of_day == 'sunset':
            weather.sun_altitude_angle = 5

        self.world.set_weather(weather)

    def spawn_ego_vehicle(self):
        blueprint_lib = self.world.get_blueprint_library()
        vehicle_types = ['vehicle.tesla.model3', 'vehicle.audi.tt', 'vehicle.mini.cooperst']

        vehicle_bp = None
        for vtype in vehicle_types:
            if blueprint_lib.filter(vtype):
                vehicle_bp = random.choice(blueprint_lib.filter(vtype))
                break

        if not vehicle_bp:
            vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            return None

        spawn_point = random.choice(spawn_points)

        try:
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.actors.append(vehicle)
            vehicle.set_autopilot(True)
            vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0))

            print(f"主车辆: {vehicle.type_id}")
            return vehicle
        except Exception as e:
            print(f"生成主车辆失败: {e}")
            return None

    def spawn_pedestrians(self):
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            return

        print(f"生成 {self.args.num_smart_pedestrians} 个行人...")

        behaviors = ['crossing', 'hesitant', 'walking']

        for _ in range(self.args.num_smart_pedestrians):
            behavior = random.choice(behaviors)
            spawn_point = random.choice(spawn_points)

            if self.ped_controller.spawn_pedestrian(spawn_point.location, behavior):
                self.actors.append(self.ped_controller.pedestrians[-1][0])

    def spawn_background_vehicles(self):
        blueprint_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            return

        spawned = 0
        for _ in range(min(5, self.args.num_background_vehicles)):
            try:
                vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))
                spawn_point = random.choice(spawn_points)

                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                self.actors.append(vehicle)
                vehicle.set_autopilot(True)
                spawned += 1
            except:
                pass

        print(f"背景车辆: {spawned} 辆")

    def setup_cameras(self, vehicle):
        if not vehicle:
            return False

        blueprint_lib = self.world.get_blueprint_library()

        camera_configs = [
            ('front_wide', carla.Location(x=2.0, z=1.8), carla.Rotation(pitch=-3.0), 100),
            ('front_narrow', carla.Location(x=2.0, z=1.6), carla.Rotation(pitch=0), 60),
            ('right_side', carla.Location(x=0.5, y=1.0, z=1.5), carla.Rotation(pitch=-2.0, yaw=45), 90),
            ('left_side', carla.Location(x=0.5, y=-1.0, z=1.5), carla.Rotation(pitch=-2.0, yaw=-45), 90),
        ]

        installed = 0

        for name, location, rotation, fov in camera_configs:
            try:
                camera_bp = blueprint_lib.find('sensor.camera.rgb')
                camera_bp.set_attribute('image_size_x', '1280')
                camera_bp.set_attribute('image_size_y', '720')
                camera_bp.set_attribute('fov', str(fov))

                camera = self.world.spawn_actor(
                    camera_bp,
                    carla.Transform(location, rotation),
                    attach_to=vehicle
                )

                def make_callback(save_dir, cam_name):
                    def callback(image):
                        current_time = time.time()

                        if current_time - self.last_capture_time >= self.args.capture_interval:
                            self.frame_count += 1
                            self.last_capture_time = current_time

                            raw_filename = f"{save_dir}/{cam_name}_{self.frame_count:04d}.png"
                            image.save_to_disk(raw_filename, carla.ColorConverter.Raw)

                            with self.buffer_lock:
                                self.image_buffer[cam_name] = raw_filename

                                if len(self.image_buffer) == 4:
                                    self.stitcher.stitch_ego_vehicle_images(self.image_buffer, self.frame_count)
                                    self.image_buffer.clear()

                    return callback

                camera.listen(make_callback(self.raw_dirs[name], name))
                self.actors.append(camera)
                self.sensors.append(camera)
                installed += 1

            except Exception as e:
                print(f"安装 {name} 摄像头失败: {e}")

        print(f"摄像头安装: {installed}/4")
        return installed == 4

    def collect_data(self):
        print(f"\n开始数据收集...")
        print(f"时长: {self.args.total_dura.tion}秒, 间隔: {self.args.capture_interval}秒")

        start_time = time.time()
        self.frame_count = 0
        self.last_capture_time = start_time

        try:
            while time.time() - start_time < self.args.total_duration:
                elapsed = time.time() - start_time
                remaining = self.args.total_duration - elapsed

                if int(elapsed) % 10 == 0:
                    progress = (elapsed / self.args.total_duration) * 100
                    print(f"进度: {elapsed:.0f}/{self.args.total_duration}秒 ({progress:.1f}%) | "
                          f"批次: {self.frame_count} | 剩余: {remaining:.0f}秒")

                time.sleep(0.1)

            print(f"\n数据收集完成! 总批次: {self.frame_count}")

        except KeyboardInterrupt:
            print(f"\n数据收集中断, 已收集 {self.frame_count} 批次")

        self.display_summary()

    def display_summary(self):
        print("\n" + "=" * 60)
        print("数据收集摘要:")
        print("=" * 60)

        stitched_dir = os.path.join(self.output_dir, "stitched_images")
        if os.path.exists(stitched_dir):
            stitched_files = [f for f in os.listdir(stitched_dir) if f.endswith('.jpg')]
            print(f"拼接图像: {len(stitched_files)} 张")

        raw_dir = os.path.join(self.output_dir, "raw")
        if os.path.exists(raw_dir):
            total_raw = 0
            for root, dirs, files in os.walk(raw_dir):
                total_raw += len([f for f in files if f.endswith('.png')])
            print(f"原始图像: {total_raw} 张")

        print(f"\n数据目录: {self.output_dir}")

    def cleanup(self):
        if self.ped_controller:
            self.ped_controller.cleanup()

        for sensor in self.sensors:
            try:
                sensor.stop()
            except:
                pass

        destroyed = 0
        for actor in self.actors:
            try:
                if actor.is_alive:
                    actor.destroy()
                    destroyed += 1
            except:
                pass

        print(f"清理 {destroyed} 个actor")
        self.actors.clear()
        self.sensors.clear()


def main():
    parser = argparse.ArgumentParser(description='CVIPS 数据生成器')

    parser.add_argument('--scenario', type=str, default='pedestrian_scene', help='场景名称')
    parser.add_argument('--town', type=str, default='Town10HD',
                        choices=['Town03', 'Town04', 'Town05', 'Town10HD'], help='CARLA地图')
    parser.add_argument('--weather', type=str, default='clear',
                        choices=['clear', 'rainy', 'cloudy'], help='天气条件')
    parser.add_argument('--time-of-day', type=str, default='noon',
                        choices=['noon', 'sunset', 'night'], help='时间')
    parser.add_argument('--num-smart-pedestrians', type=int, default=6, help='行人数')
    parser.add_argument('--num-background-vehicles', type=int, default=4, help='背景车辆数')
    parser.add_argument('--total-duration', type=int, default=90, help='总时长(秒)')
    parser.add_argument('--capture-interval', type=float, default=2.5, help='捕捉间隔(秒)')

    args = parser.parse_args(remaining_argv)

    if args.capture_interval < 2.0:
        args.capture_interval = 2.0

    print(f"\n场景配置:")
    print(f"  场景: {args.scenario}")
    print(f"  地图: {args.town}")
    print(f"  天气/时间: {args.weather}/{args.time_of_day}")
    print(f"  行人: {args.num_smart_pedestrians}个")
    print(f"  车辆: {args.num_background_vehicles}辆")
    print(f"  时长: {args.total_duration}秒")
    print(f"  间隔: {args.capture_interval}秒")

    generator = DataGenerator(args)

    try:
        if not generator.connect_to_server():
            return

        ego_vehicle = generator.setup_scene()
        if not ego_vehicle:
            generator.cleanup()
            return

        if not generator.setup_cameras(ego_vehicle):
            generator.cleanup()
            return

        generator.collect_data()

    except KeyboardInterrupt:
        print("\n程序中断")
    except Exception as e:
        print(f"\n运行错误: {e}")
        traceback.print_exc()
    finally:
        generator.cleanup()
        print(f"\n数据保存到: {generator.output_dir}")


if __name__ == "__main__":
    main()