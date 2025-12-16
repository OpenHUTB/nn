import sys
import os
import time
import random
import argparse
import traceback
from datetime import datetime
import glob

print("=" * 80)
print("CVIPS v3.0 - 行人安全数据生成器")
print("=" * 80)

def find_carla_egg():
    """自动查找CARLA的egg文件"""
    common_paths = [
        os.path.expanduser("~/carla/*"),
        os.path.expanduser("~/Desktop/carla/*"),
        "/opt/carla/*",
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "carla"),
    ]

    for path in common_paths:
        egg_pattern = os.path.join(path, "PythonAPI", "carla", "dist", "carla-*.egg")
        egg_files = glob.glob(egg_pattern, recursive=True)
        if egg_files:
            return egg_files[0]
        egg_pattern = os.path.join(path, "dist", "carla-*.egg")
        egg_files = glob.glob(egg_pattern, recursive=True)
        if egg_files:
            return egg_files[0]
    return None

print("\n[1/5] 初始化CARLA环境...")
path_parser = argparse.ArgumentParser(add_help=False)
path_parser.add_argument('--carla-path', type=str, help='CARLA的egg文件路径或dist目录路径')
args_path, remaining_argv = path_parser.parse_known_args()

carla_egg_path = None
if args_path.carla_path:
    if os.path.isfile(args_path.carla_path) and args_path.carla_path.endswith('.egg'):
        carla_egg_path = args_path.carla_path
    elif os.path.isdir(args_path.carla_path):
        egg_files = glob.glob(os.path.join(args_path.carla_path, "carla-*.egg"))
        if egg_files:
            carla_egg_path = egg_files[0]
    if carla_egg_path:
        sys.path.append(os.path.dirname(carla_egg_path))
        print(f"✓ 通过命令行参数加载CARLA egg文件: {carla_egg_path}")
    else:
        print(f"✗ 命令行参数指定的路径中未找到CARLA egg文件: {args_path.carla_path}")
        sys.exit(1)
elif os.getenv("CARLA_PYTHON_PATH"):
    env_carla_path = os.getenv("CARLA_PYTHON_PATH")
    if os.path.isfile(env_carla_path) and env_carla_path.endswith('.egg'):
        carla_egg_path = env_carla_path
    elif os.path.isdir(env_carla_path):
        egg_files = glob.glob(os.path.join(env_carla_path, "carla-*.egg"))
        if egg_files:
            carla_egg_path = egg_files[0]
    if carla_egg_path:
        sys.path.append(os.path.dirname(carla_egg_path))
        print(f"✓ 通过环境变量加载CARLA egg文件: {carla_egg_path}")
    else:
        print(f"✗ 环境变量CARLA_PYTHON_PATH中未找到CARLA egg文件: {env_carla_path}")
        sys.exit(1)
else:
    carla_egg_path = find_carla_egg()
    if carla_egg_path:
        sys.path.append(os.path.dirname(carla_egg_path))
        print(f"✓ 自动找到CARLA egg文件: {carla_egg_path}")
    else:
        print("✗ 未找到CARLA egg文件！")
        print("提示：请通过以下方式之一配置CARLA路径：")
        print("  1. 命令行参数：--carla-path <CARLA的egg文件/ dist目录>")
        print("  2. 环境变量：设置CARLA_PYTHON_PATH=<CARLA的egg文件/ dist目录>")
        print("  3. 将CARLA放在用户目录carla/、桌面carla/或/opt/carla/（自动查找）")
        sys.exit(1)

print("\n[2/5] 导入CARLA模块...")
try:
    import carla
    print("✓ CARLA模块导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

class PedestrianSafetyGenerator:
    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.actors = []
        self.sensors = []
        self.frame_count = 0
        self.camera_last_save = {}
        self.setup_output_directory()

    def setup_output_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_name = f"{self.args.scenario}_ped_safety_{timestamp}"
        self.output_dir = os.path.join("pedestrian_safety_data", scenario_name)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"输出目录: {self.output_dir}")

    def connect_to_server(self):
        print("\n[3/5] 连接到CARLA服务器...")

        for attempt in range(1, 6):
            try:
                print(f"  尝试 {attempt}/5...")
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(15.0)

                if self.args.town:
                    self.world = self.client.load_world(self.args.town)
                else:
                    self.world = self.client.get_world()

                print(f"✓ 连接成功! 地图: {self.world.get_map().name}")
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)

                return True

            except Exception as e:
                error_msg = str(e)
                print(f"  尝试 {attempt} 失败: {error_msg[:80]}...")
                if attempt < 5:
                    print("  等待3秒后重试...")
                    time.sleep(3)

        print("✗ 连接失败")
        return False

    def setup_pedestrian_safety_scene(self):
        print("\n[4/5] 设置行人安全场景...")

        try:
            self.set_high_quality_settings()
            self.set_pedestrian_friendly_environment()
            time.sleep(2.0)

            ego_vehicle = self.spawn_ego_vehicle_near_crosswalk()
            if not ego_vehicle:
                print("⚠ 无法生成主车辆")
                return None

            crossing_pedestrians = self.spawn_crossing_pedestrians()
            self.spawn_background_traffic()

            print("等待场景稳定...")
            time.sleep(5.0)

            return ego_vehicle

        except Exception as e:
            print(f"设置场景失败: {e}")
            traceback.print_exc()
            return None

    def set_high_quality_settings(self):
        try:
            print("✓ 应用行人检测优化的渲染设置（部分参数需在CARLA UE4端配置）")
            settings = self.world.get_settings()
            settings.no_rendering_mode = False
            self.world.apply_settings(settings)

        except Exception as e:
            print(f"设置渲染质量失败: {e}")
            traceback.print_exc()

    def set_pedestrian_friendly_environment(self):
        weather = carla.WeatherParameters()

        if self.args.weather == 'clear':
            weather.sun_altitude_angle = 75
            weather.sun_azimuth_angle = 0
            weather.cloudiness = 5.0
            weather.precipitation = 0.0
            weather.precipitation_deposits = 0.0
            weather.wind_intensity = 5.0
            weather.fog_density = 0.0
            weather.wetness = 0.0
            weather.scattering_intensity = 1.0
            weather.mie_scattering_scale = 0.8
            weather.rayleigh_scattering_scale = 1.0

        elif self.args.weather == 'rainy':
            weather.sun_altitude_angle = 40
            weather.cloudiness = 90.0
            weather.precipitation = 60.0
            weather.precipitation_deposits = 50.0
            weather.wind_intensity = 30.0
            weather.fog_density = 15.0
            weather.wetness = 70.0
            weather.scattering_intensity = 1.5

        elif self.args.weather == 'cloudy':
            weather.sun_altitude_angle = 60
            weather.cloudiness = 70.0
            weather.precipitation = 0.0
            weather.wind_intensity = 10.0
            weather.fog_density = 5.0
            weather.wetness = 10.0
            weather.scattering_intensity = 1.2

        if self.args.time_of_day == 'night':
            weather.sun_altitude_angle = -10
            weather.fog_density = 5.0
            weather.wetness = 10.0
            weather.scattering_intensity = 1.8
        elif self.args.time_of_day == 'sunset':
            weather.sun_altitude_angle = 5
            weather.cloudiness = 50.0
            weather.fog_density = 10.0

        self.world.set_weather(weather)
        print(f"✓ 行人友好环境设置: {self.args.weather}, {self.args.time_of_day}")

    def find_crosswalk_locations(self):
        waypoints = self.world.get_map().generate_waypoints(10.0)
        crosswalk_points = []

        for wp in waypoints:
            if wp.is_junction or wp.lane_type == carla.LaneType.Sidewalk:
                crosswalk_points.append(wp.transform)
                if len(crosswalk_points) >= 3:
                    break

        if not crosswalk_points:
            spawn_points = self.world.get_map().get_spawn_points()
            crosswalk_points = spawn_points[:3]

        return crosswalk_points

    def spawn_ego_vehicle_near_crosswalk(self):
        blueprint_lib = self.world.get_blueprint_library()

        vehicle_types = [
            'vehicle.tesla.model3',
            'vehicle.audi.tt',
            'vehicle.mini.cooperst',
            'vehicle.nissan.patrol'
        ]

        vehicle_bp = None
        for vtype in vehicle_types:
            if blueprint_lib.filter(vtype):
                vehicle_bp = random.choice(blueprint_lib.filter(vtype))
                break

        if not vehicle_bp:
            vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))

        crosswalk_points = self.find_crosswalk_locations()
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            print("⚠ 没有生成点")
            return None

        if crosswalk_points:
            crosswalk_point = crosswalk_points[0]
            nearest_spawn_point = None
            min_distance = float('inf')

            for spawn_point in spawn_points:
                distance = spawn_point.location.distance(crosswalk_point.location)
                if distance < min_distance and distance > 10.0:
                    min_distance = distance
                    nearest_spawn_point = spawn_point

            spawn_point = nearest_spawn_point if nearest_spawn_point else random.choice(spawn_points)
        else:
            spawn_point = random.choice(spawn_points)

        try:
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.actors.append(vehicle)

            vehicle.set_autopilot(True)
            tm = self.client.get_trafficmanager(8000)
            tm.set_desired_speed(vehicle, 10.0)

            print(f"✓ 生成主车辆: {vehicle.type_id}")
            print(f"  位置: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")

            return vehicle

        except Exception as e:
            print(f"生成主车辆失败: {e}")
            traceback.print_exc()
            return None

    def spawn_crossing_pedestrians(self):
        blueprint_lib = self.world.get_blueprint_library()
        crosswalk_points = self.find_crosswalk_locations()
        if not crosswalk_points:
            print("⚠ 未找到合适的过马路区域")
            return []

        crossing_pedestrians = []
        num_crossing = min(3, self.args.num_crossing_pedestrians)

        print(f"生成 {num_crossing} 个过马路行人...")

        for i in range(num_crossing):
            try:
                ped_types = list(blueprint_lib.filter('walker.pedestrian.*'))
                if not ped_types:
                    continue

                ped_bp = random.choice(ped_types)
                crosswalk_point = crosswalk_points[i % len(crosswalk_points)]

                offset_x = random.uniform(-1.0, 1.0)
                offset_y = random.uniform(-1.0, 1.0)

                location = carla.Location(
                    x=crosswalk_point.location.x + offset_x,
                    y=crosswalk_point.location.y + offset_y,
                    z=crosswalk_point.location.z + 0.1
                )

                spawn_point = carla.Transform(location)
                pedestrian = self.world.try_spawn_actor(ped_bp, spawn_point)
                if not pedestrian:
                    print(f"  行人 {i+1} 生成位置无效，跳过")
                    continue

                self.actors.append(pedestrian)

                controller_bp = blueprint_lib.find('controller.ai.walker')
                if controller_bp:
                    controller = self.world.spawn_actor(
                        controller_bp,
                        carla.Transform(),
                        attach_to=pedestrian
                    )
                    self.actors.append(controller)

                    controller.start()
                    controller.set_max_speed(1.0)

                    wp = self.world.get_map().get_waypoint(pedestrian.get_location())
                    if wp:
                        target_location = wp.transform.location + carla.Location(
                            x=10 * (-wp.transform.rotation.yaw / 90),
                            y=10 * (wp.transform.rotation.yaw / 90),
                            z=wp.transform.location.z
                        )
                    else:
                        target_location = carla.Location(
                            x=location.x + random.uniform(10.0, 20.0) * (1 if random.random() > 0.5 else -1),
                            y=location.y + random.uniform(10.0, 20.0) * (1 if random.random() > 0.5 else -1),
                            z=location.z
                        )

                    controller.go_to_location(target_location)
                    crossing_pedestrians.append(pedestrian)
                    print(f"  过马路行人 {i + 1} 已生成")

            except Exception as e:
                print(f"  生成过马路行人失败: {e}")
                traceback.print_exc()
                continue

        print(f"✓ 生成 {len(crossing_pedestrians)} 个过马路行人")
        return crossing_pedestrians

    def spawn_background_traffic(self):
        blueprint_lib = self.world.get_blueprint_library()

        vehicles_spawned = 0
        spawn_points = self.world.get_map().get_spawn_points()
        for i in range(min(5, self.args.num_background_vehicles)):
            try:
                vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))
                if spawn_points:
                    spawn_point = random.choice(spawn_points)
                    vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                    if vehicle:
                        self.actors.append(vehicle)
                        vehicle.set_autopilot(True)
                        tm = self.client.get_trafficmanager(8000)
                        tm.set_desired_speed(vehicle, 15.0)
                        vehicles_spawned += 1
            except Exception as e:
                print(f"  生成背景车辆失败: {e}")
                continue

        pedestrians_spawned = 0
        for i in range(min(8, self.args.num_background_pedestrians)):
            try:
                ped_bp = random.choice(blueprint_lib.filter('walker.pedestrian.*'))
                location = self.world.get_random_location_from_navigation()

                if location:
                    location.z += 0.1
                    pedestrian = self.world.try_spawn_actor(ped_bp, carla.Transform(location))
                    if pedestrian:
                        self.actors.append(pedestrian)

                        controller_bp = blueprint_lib.find('controller.ai.walker')
                        controller = self.world.spawn_actor(
                            controller_bp,
                            carla.Transform(),
                            attach_to=pedestrian
                        )
                        controller.start()
                        controller.set_max_speed(0.8)

                        target_location = self.world.get_random_location_from_navigation()
                        if target_location:
                            controller.go_to_location(target_location)
                        else:
                            controller.go_to_location(location + carla.Location(x=5, y=5))

                        pedestrians_spawned += 1
                        self.actors.append(controller)
            except Exception as e:
                print(f"  生成背景行人失败: {e}")
                continue

        print(f"✓ 生成 {vehicles_spawned} 辆背景车辆和 {pedestrians_spawned} 个背景行人")

    def setup_pedestrian_safety_cameras(self, vehicle):
        if not vehicle:
            return

        blueprint_lib = self.world.get_blueprint_library()
        print("\n安装行人安全摄像头系统...")

        camera_configs = [
            ('front_wide',
             carla.Location(x=2.0, z=1.8),
             carla.Rotation(pitch=-3.0),
             100,
             "前视广角摄像头 - 检测前方行人"),

            ('front_narrow',
             carla.Location(x=2.0, z=1.6),
             carla.Rotation(pitch=0),
             60,
             "前视窄角摄像头 - 细节识别"),

            ('right_side',
             carla.Location(x=0.5, y=1.0, z=1.5),
             carla.Rotation(pitch=-2.0, yaw=45),
             90,
             "右侧摄像头 - 检测右侧行人"),

            ('left_side',
             carla.Location(x=0.5, y=-1.0, z=1.5),
             carla.Rotation(pitch=-2.0, yaw=-45),
             90,
             "左侧摄像头 - 检测左侧行人")
        ]

        installed_cameras = 0

        for name, location, rotation, fov, description in camera_configs:
            try:
                camera_bp = blueprint_lib.find('sensor.camera.rgb')

                camera_bp.set_attribute('image_size_x', '1280')
                camera_bp.set_attribute('image_size_y', '720')
                camera_bp.set_attribute('fov', str(fov))
                camera_bp.set_attribute('motion_blur_intensity', '0.0')
                camera_bp.set_attribute('motion_blur_max_distortion', '0.0')
                camera_bp.set_attribute('enable_postprocess_effects', 'False')
                camera_bp.set_attribute('gamma', '2.2')
                camera_bp.set_attribute('shutter_speed', '100')
                camera_bp.set_attribute('iso', '200')
                camera_bp.set_attribute('fstop', '2.0')
                camera_bp.set_attribute('lens_k', '0.0')
                camera_bp.set_attribute('lens_kcube', '0.0')
                camera_bp.set_attribute('lens_x_size', '0.08')
                camera_bp.set_attribute('lens_y_size', '0.08')

                transform = carla.Transform(location, rotation)
                camera = self.world.spawn_actor(camera_bp, transform, attach_to=vehicle)

                camera_dir = os.path.join(self.output_dir, name)
                os.makedirs(camera_dir, exist_ok=True)

                self.camera_last_save[name] = time.time()

                def save_image(image, save_dir=camera_dir, cam_name=name):
                    current_time = time.time()

                    if current_time - self.camera_last_save[cam_name] >= self.args.capture_interval:
                        self.frame_count += 1
                        self.camera_last_save[cam_name] = current_time

                        filename = f"{save_dir}/ped_frame_{self.frame_count:04d}.png"
                        image.save_to_disk(filename, carla.ColorConverter.Raw)

                        if self.frame_count % 10 == 0:
                            print(f"  [{cam_name}] 保存第 {self.frame_count} 帧")

                camera.listen(save_image)
                self.actors.append(camera)
                self.sensors.append(camera)

                installed_cameras += 1
                print(f"✓ {description}")

            except Exception as e:
                print(f"  安装{name}摄像头失败: {e}")
                traceback.print_exc()
                continue

        print(f"✓ 总共安装 {installed_cameras} 个行人安全摄像头")

    def collect_pedestrian_safety_data(self):
        print("\n[5/5] 开始收集行人安全数据...")
        print(f"数据收集模式: 间隔{self.args.capture_interval}秒捕捉")
        print(f"预计总时长: {self.args.total_duration}秒")
        print(f"预计帧数: {self.args.total_duration // self.args.capture_interval}")
        print("\n提示: 正在模拟行人过马路场景...")
        print("按 Ctrl+C 提前结束\n")

        start_time = time.time()
        self.frame_count = 0
        for cam_name in self.camera_last_save:
            self.camera_last_save[cam_name] = start_time

        try:
            update_interval = 5.0

            while time.time() - start_time < self.args.total_duration:
                elapsed = time.time() - start_time
                remaining = max(0, self.args.total_duration - elapsed)

                if int(elapsed) % update_interval == 0 and elapsed % update_interval < 0.1:
                    progress_percent = (elapsed / self.args.total_duration) * 100

                    print(f"  进度: {elapsed:.0f}/{self.args.total_duration}秒 "
                          f"({progress_percent:.1f}%) | "
                          f"已保存帧数: {self.frame_count} | "
                          f"剩余: {remaining:.0f}秒")

                time.sleep(0.1)

            elapsed = time.time() - start_time

            print(f"\n✓ 行人安全数据收集完成!")
            print(f"  总时长: {elapsed:.1f}秒")
            print(f"  保存帧数: {self.frame_count}")
            print(f"  实际帧率: {self.frame_count / elapsed if elapsed > 0 else 0:.2f} FPS")

            self.display_data_summary()

        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            print(f"\n数据收集中断，已收集 {self.frame_count} 帧")
            print(f"总时长: {elapsed:.1f}秒")

    def display_data_summary(self):
        print("\n" + "-" * 60)
        print("数据收集摘要:")
        print("-" * 60)

        camera_dirs = ['front_wide', 'front_narrow', 'right_side', 'left_side']

        for cam_dir in camera_dirs:
            cam_path = os.path.join(self.output_dir, cam_dir)
            if os.path.exists(cam_path):
                image_files = [f for f in os.listdir(cam_path) if f.endswith('.png')]
                print(f"  {cam_dir}: {len(image_files)} 张图像")

        print(f"\n数据目录: {self.output_dir}")
        print("建议: 检查图像质量，确保行人清晰可见")
        print("-" * 60)

    def cleanup(self):
        print("\n清理场景...")

        destroyed = 0
        for actor in self.actors:
            try:
                if actor and actor.is_alive:
                    actor.destroy()
                    destroyed += 1
            except Exception as e:
                print(f"  销毁actor失败: {e}")
                continue

        print(f"销毁 {destroyed} 个actor")
        self.actors.clear()
        self.sensors.clear()

def main():
    parser = argparse.ArgumentParser(
        description='CVIPS v3.0 - 行人安全协同感知数据生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[path_parser],
        epilog="""
示例:
  # 基本行人安全场景（晴天中午）
  python cvips_pedestrian_safety.py

  # 雨天夜晚的行人安全场景
  python cvips_pedestrian_safety.py --weather rainy --time-of-day night --total-duration 120

  # 自定义场景
  python cvips_pedestrian_safety.py --scenario crosswalk_test --capture-interval 3 --num-crossing-pedestrians 4
        """
    )

    parser.add_argument('--scenario', type=str, default='pedestrian_crossing',
                        help='场景名称')
    parser.add_argument('--town', type=str, default='Town10HD',
                        choices=['Town03', 'Town04', 'Town05', 'Town10HD'],
                        help='CARLA地图（推荐Town10HD行人多）')

    parser.add_argument('--weather', type=str, default='clear',
                        choices=['clear', 'rainy', 'cloudy'],
                        help='天气条件（推荐clear）')
    parser.add_argument('--time-of-day', type=str, default='noon',
                        choices=['noon', 'sunset', 'night'],
                        help='时间（推荐noon）')

    parser.add_argument('--num-crossing-pedestrians', type=int, default=3,
                        help='过马路行人数（重点行人）')
    parser.add_argument('--num-background-pedestrians', type=int, default=6,
                        help='背景行人数')
    parser.add_argument('--num-background-vehicles', type=int, default=4,
                        help='背景车辆数')

    parser.add_argument('--total-duration', type=int, default=60,
                        help='总收集时间(秒)')
    parser.add_argument('--capture-interval', type=float, default=2.0,
                        help='图像捕捉间隔(秒) - 建议2.0-5.0秒')

    args = parser.parse_args()

    if args.capture_interval < 1.0:
        print("⚠ 警告: 捕捉间隔太短可能导致图像变化不明显，建议使用2.0秒或更长")
        args.capture_interval = 2.0

    print(f"\n配置参数:")
    print(f"  场景: {args.scenario}")
    print(f"  地图: {args.town}")
    print(f"  天气: {args.weather}, 时间: {args.time_of_day}")
    print(f"  过马路行人: {args.num_crossing_pedestrians}")
    print(f"  总时长: {args.total_duration}秒")
    print(f"  捕捉间隔: {args.capture_interval}秒")
    print(f"  预计帧数: {args.total_duration // args.capture_interval}")

    generator = PedestrianSafetyGenerator(args)

    try:
        if not generator.connect_to_server():
            print("\n连接失败，退出")
            return

        ego_vehicle = generator.setup_pedestrian_safety_scene()

        if not ego_vehicle:
            print("\n场景设置失败")
            generator.cleanup()
            return

        generator.setup_pedestrian_safety_cameras(ego_vehicle)

        if not generator.sensors:
            print("\n摄像头安装失败")
            generator.cleanup()
            return

        generator.collect_pedestrian_safety_data()

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n运行出错: {e}")
        traceback.print_exc()
    finally:
        generator.cleanup()

        print("\n" + "=" * 80)
        print("行人安全数据收集完成!")
        print(f"数据保存到: {generator.output_dir}")
        print("=" * 80)

        print("\n使用提示:")
        print("1. 检查输出目录中的图像质量")
        print("2. 确保行人清晰可见")
        print("3. 可调整capture-interval参数控制图像间隔")
        print("4. 下次运行可尝试不同天气和时间条件")

if __name__ == "__main__":
    main()