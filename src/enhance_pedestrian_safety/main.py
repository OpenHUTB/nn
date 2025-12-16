# cvips_infrastructure_collaborative.py
import sys
import os
import time
import random
import argparse
import traceback
import math
from datetime import datetime

print("=" * 80)
print("CVIPS v4.0 - 基础设施摄像头与多车协同数据生成器")
print("=" * 80)

from carla_utils import setup_carla_path, import_carla_module

carla_egg_path, remaining_argv = setup_carla_path()
carla = import_carla_module()


class InfrastructureCollaborativeGenerator:
    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.actors = []
        self.sensors = []
        self.infrastructure_cameras = []
        self.cooperative_vehicles = []
        self.frame_count = 0
        self.last_save_time = time.time()
        self.intersection_location = None
        self.setup_output_directory()

    def setup_output_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_name = f"{self.args.scenario}_v4_{timestamp}"
        self.output_dir = os.path.join("cvips_v4_data", scenario_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.camera_dirs = {}

        for view in ['front_wide', 'front_narrow', 'right_side', 'left_side']:
            dir_path = os.path.join(self.output_dir, "ego_vehicle", view)
            os.makedirs(dir_path, exist_ok=True)
            self.camera_dirs[f"ego_{view}"] = dir_path

        for i in range(self.args.num_infra_cameras):
            for angle in ['overview', 'left_view', 'right_view', 'closeup']:
                dir_path = os.path.join(self.output_dir, "infrastructure", f"camera_{i + 1}", angle)
                os.makedirs(dir_path, exist_ok=True)
                self.camera_dirs[f"infra_{i + 1}_{angle}"] = dir_path

        for v in range(self.args.num_coop_vehicles):
            for view in ['front', 'rear', 'left', 'right']:
                dir_path = os.path.join(self.output_dir, "cooperative", f"vehicle_{v + 1}", view)
                os.makedirs(dir_path, exist_ok=True)
                self.camera_dirs[f"coop{v + 1}_{view}"] = dir_path

        print(f"输出目录: {self.output_dir}")
        print(f"目录结构: 主车({self.args.num_coop_vehicles + 1}辆) + 基础设施({self.args.num_infra_cameras}个)")

    def connect_to_server(self):
        print("\n[3/6] 连接到CARLA服务器...")

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

    def setup_collaborative_scene(self):
        print("\n[4/6] 设置协同感知场景...")

        try:
            self.set_high_quality_settings()
            self.set_environment()
            time.sleep(2.0)

            self.intersection_location = self.find_main_intersection()
            if self.intersection_location:
                print(f"✓ 找到主路口位置: ({self.intersection_location.x:.1f}, {self.intersection_location.y:.1f})")

            ego_vehicle = self.spawn_ego_vehicle_near_intersection()
            if not ego_vehicle:
                print("⚠ 无法生成主车辆")
                return None

            self.cooperative_vehicles = self.spawn_cooperative_vehicles()
            self.setup_infrastructure_cameras()
            self.spawn_traffic_and_pedestrians()

            print("等待协同场景稳定...")
            time.sleep(8.0)
            return ego_vehicle

        except Exception as e:
            print(f"设置场景失败: {e}")
            traceback.print_exc()
            return None

    def set_high_quality_settings(self):
        try:
            quality_settings = {
                'r.MotionBlurQuality': 0,
                'r.DepthOfFieldQuality': 0,
                'r.BloomQuality': 0,
                'r.LensFlareQuality': 0,
                'r.TonemapperQuality': 0,
                'r.AmbientOcclusionLevels': 0,
                'r.ShadowQuality': 3,
                'r.TextureStreaming': True,
                'r.PostProcessAAQuality': 6,
            }

            for key, value in quality_settings.items():
                self.world.get_settings().set(str(key), str(value))

            print("✓ 高质量渲染设置已应用")
        except Exception as e:
            print(f"设置渲染质量失败: {e}")

    def set_environment(self):
        weather = carla.WeatherParameters()

        if self.args.weather == 'clear':
            weather.sun_altitude_angle = 75
            weather.cloudiness = 5.0
            weather.precipitation = 0.0
            weather.fog_density = 0.0
        elif self.args.weather == 'rainy':
            weather.sun_altitude_angle = 40
            weather.cloudiness = 90.0
            weather.precipitation = 60.0
            weather.fog_density = 15.0
        elif self.args.weather == 'cloudy':
            weather.sun_altitude_angle = 60
            weather.cloudiness = 70.0
            weather.precipitation = 0.0
            weather.fog_density = 5.0

        if self.args.time_of_day == 'night':
            weather.sun_altitude_angle = -10
            weather.fog_density = 5.0
        elif self.args.time_of_day == 'sunset':
            weather.sun_altitude_angle = 5
            weather.cloudiness = 50.0

        self.world.set_weather(weather)
        print(f"✓ 环境设置: {self.args.weather}, {self.args.time_of_day}")

    def find_main_intersection(self):
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            return None

        if len(spawn_points) > 10:
            center_point = None
            min_distance = float('inf')

            for point in spawn_points:
                distance = math.sqrt(point.location.x ** 2 + point.location.y ** 2)
                if distance < min_distance:
                    min_distance = distance
                    center_point = point.location
            return center_point

        return spawn_points[0].location

    def spawn_ego_vehicle_near_intersection(self):
        blueprint_lib = self.world.get_blueprint_library()
        vehicle_types = [
            'vehicle.tesla.model3',
            'vehicle.audi.tt',
            'vehicle.bmw.grandtourer',
            'vehicle.mercedes.coupe'
        ]

        vehicle_bp = None
        for vtype in vehicle_types:
            if blueprint_lib.filter(vtype):
                vehicle_bp = random.choice(blueprint_lib.filter(vtype))
                break

        if not vehicle_bp:
            vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            print("⚠ 没有生成点")
            return None

        if self.intersection_location:
            nearest_point = None
            min_distance = float('inf')

            for point in spawn_points:
                distance = point.location.distance(self.intersection_location)
                if distance < min_distance and distance > 5.0:
                    min_distance = distance
                    nearest_point = point

            spawn_point = nearest_point if nearest_point else random.choice(spawn_points)
        else:
            spawn_point = random.choice(spawn_points)

        try:
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.actors.append(vehicle)
            vehicle.set_autopilot(True)
            vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0))

            print(f"✓ 生成主车辆: {vehicle.type_id}")
            print(f"  位置: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
            return vehicle
        except Exception as e:
            print(f"生成主车辆失败: {e}")
            return None

    def spawn_cooperative_vehicles(self):
        blueprint_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            print("⚠ 没有生成点，无法生成协同车辆")
            return []

        cooperative_vehicles = []
        num_to_spawn = min(self.args.num_coop_vehicles, len(spawn_points) - 1)
        print(f"生成 {num_to_spawn} 辆协同车辆...")

        for i in range(num_to_spawn):
            try:
                vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))
                spawn_index = (i + 5) % len(spawn_points)
                spawn_point = spawn_points[spawn_index]

                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                self.actors.append(vehicle)
                vehicle.set_autopilot(True)
                speed = 0.2 + (i * 0.05)
                vehicle.apply_control(carla.VehicleControl(throttle=speed, steer=0.0))

                cooperative_vehicles.append(vehicle)
                print(f"  协同车辆 {i + 1}: {vehicle.type_id}")
            except Exception as e:
                print(f"  生成协同车辆 {i + 1} 失败: {e}")
                continue

        print(f"✓ 生成 {len(cooperative_vehicles)} 辆协同车辆")
        return cooperative_vehicles

    def setup_infrastructure_cameras(self):
        if not self.intersection_location:
            print("⚠ 无路口位置，跳过基础设施摄像头")
            return

        blueprint_lib = self.world.get_blueprint_library()
        print(f"安装 {self.args.num_infra_cameras} 个基础设施摄像头...")

        for cam_idx in range(self.args.num_infra_cameras):
            try:
                angle_offset = (cam_idx * 360) / self.args.num_infra_cameras
                radius = 15.0 + (cam_idx * 5.0)
                angle_rad = math.radians(angle_offset)
                cam_x = self.intersection_location.x + radius * math.cos(angle_rad)
                cam_y = self.intersection_location.y + radius * math.sin(angle_rad)
                cam_z = 8.0 + (cam_idx * 2.0)

                camera_location = carla.Location(x=cam_x, y=cam_y, z=cam_z)
                look_at_location = self.intersection_location
                look_at_location.z = 0
                direction = look_at_location - camera_location
                rotation = carla.Rotation(
                    pitch=-20 - (cam_idx * 5),
                    yaw=math.degrees(math.atan2(direction.y, direction.x))
                )
                transform = carla.Transform(camera_location, rotation)

                camera_angles = ['overview', 'left_view', 'right_view', 'closeup']

                for angle_name in camera_angles:
                    try:
                        camera_bp = blueprint_lib.find('sensor.camera.rgb')

                        if angle_name == 'overview':
                            fov = 90
                        elif angle_name == 'closeup':
                            fov = 45
                        else:
                            fov = 70

                        camera_bp.set_attribute('image_size_x', '1280')
                        camera_bp.set_attribute('image_size_y', '720')
                        camera_bp.set_attribute('fov', str(fov))
                        camera_bp.set_attribute('motion_blur_intensity', '0.0')
                        camera_bp.set_attribute('enable_postprocess_effects', 'False')

                        camera = self.world.spawn_actor(camera_bp, transform)

                        def make_save_callback(save_dir, cam_id, angle):
                            def save_image(image):
                                current_time = time.time()
                                if current_time - self.last_save_time >= self.args.capture_interval:
                                    self.frame_count += 1
                                    self.last_save_time = current_time
                                    filename = f"{save_dir}/infra_{cam_id}_{angle}_frame_{self.frame_count:04d}.png"
                                    image.save_to_disk(filename, carla.ColorConverter.Raw)
                            return save_image

                        save_dir = self.camera_dirs[f"infra_{cam_idx + 1}_{angle_name}"]
                        camera.listen(make_save_callback(save_dir, cam_idx + 1, angle_name))
                        self.actors.append(camera)
                        self.infrastructure_cameras.append(camera)
                    except Exception as e:
                        print(f"    安装{angle_name}视角失败: {e}")

                print(f"  基础设施摄像头 {cam_idx + 1} 安装在: ({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f})")
            except Exception as e:
                print(f"  安装基础设施摄像头 {cam_idx + 1} 失败: {e}")
                continue

        print(f"✓ 安装 {len(self.infrastructure_cameras)} 个基础设施摄像头视角")

    def setup_ego_cameras(self, vehicle):
        if not vehicle:
            return

        blueprint_lib = self.world.get_blueprint_library()
        print("安装主车辆摄像头...")

        camera_configs = [
            ('front_wide', carla.Location(x=2.0, z=1.8), carla.Rotation(pitch=-3.0), 100),
            ('front_narrow', carla.Location(x=2.0, z=1.6), carla.Rotation(pitch=0), 60),
            ('right_side', carla.Location(x=0.5, y=1.0, z=1.5), carla.Rotation(pitch=-2.0, yaw=45), 90),
            ('left_side', carla.Location(x=0.5, y=-1.0, z=1.5), carla.Rotation(pitch=-2.0, yaw=-45), 90),
        ]

        for name, location, rotation, fov in camera_configs:
            try:
                camera_bp = blueprint_lib.find('sensor.camera.rgb')
                camera_bp.set_attribute('image_size_x', '1280')
                camera_bp.set_attribute('image_size_y', '720')
                camera_bp.set_attribute('fov', str(fov))
                camera_bp.set_attribute('motion_blur_intensity', '0.0')
                camera_bp.set_attribute('enable_postprocess_effects', 'False')

                transform = carla.Transform(location, rotation)
                camera = self.world.spawn_actor(camera_bp, transform, attach_to=vehicle)

                def make_save_callback(save_dir, cam_name):
                    def save_image(image):
                        current_time = time.time()
                        if current_time - self.last_save_time >= self.args.capture_interval:
                            self.frame_count += 1
                            self.last_save_time = current_time
                            filename = f"{save_dir}/ego_{cam_name}_frame_{self.frame_count:04d}.png"
                            image.save_to_disk(filename, carla.ColorConverter.Raw)
                    return save_image

                save_dir = self.camera_dirs[f"ego_{name}"]
                camera.listen(make_save_callback(save_dir, name))
                self.actors.append(camera)
                self.sensors.append(camera)
                print(f"  主车{name}摄像头已安装")
            except Exception as e:
                print(f"  安装主车{name}摄像头失败: {e}")

        print("✓ 主车辆摄像头安装完成")

    def setup_cooperative_cameras(self):
        if not self.cooperative_vehicles:
            return

        blueprint_lib = self.world.get_blueprint_library()
        print("安装协同车辆摄像头...")

        for v_idx, vehicle in enumerate(self.cooperative_vehicles):
            camera_configs = [
                ('front', carla.Location(x=1.5, z=1.4), carla.Rotation(pitch=0), 90),
                ('rear', carla.Location(x=-1.5, z=1.4), carla.Rotation(pitch=0, yaw=180), 90),
                ('left', carla.Location(x=0.0, y=-0.8, z=1.4), carla.Rotation(pitch=0, yaw=-90), 90),
                ('right', carla.Location(x=0.0, y=0.8, z=1.4), carla.Rotation(pitch=0, yaw=90), 90),
            ]

            for name, location, rotation, fov in camera_configs:
                try:
                    camera_bp = blueprint_lib.find('sensor.camera.rgb')
                    camera_bp.set_attribute('image_size_x', '1280')
                    camera_bp.set_attribute('image_size_y', '720')
                    camera_bp.set_attribute('fov', str(fov))
                    camera_bp.set_attribute('motion_blur_intensity', '0.0')

                    transform = carla.Transform(location, rotation)
                    camera = self.world.spawn_actor(camera_bp, transform, attach_to=vehicle)

                    def make_save_callback(save_dir, v_id, cam_name):
                        def save_image(image):
                            current_time = time.time()
                            if current_time - self.last_save_time >= self.args.capture_interval:
                                self.frame_count += 1
                                self.last_save_time = current_time
                                filename = f"{save_dir}/coop{v_id}_{cam_name}_frame_{self.frame_count:04d}.png"
                                image.save_to_disk(filename, carla.ColorConverter.Raw)
                        return save_image

                    save_dir = self.camera_dirs[f"coop{v_idx + 1}_{name}"]
                    camera.listen(make_save_callback(save_dir, v_idx + 1, name))
                    self.actors.append(camera)
                    self.sensors.append(camera)
                except Exception as e:
                    print(f"  协同车辆{v_idx + 1} {name}摄像头安装失败: {e}")

            print(f"  协同车辆{v_idx + 1}摄像头安装完成")

        print("✓ 所有协同车辆摄像头安装完成")

    def spawn_traffic_and_pedestrians(self):
        blueprint_lib = self.world.get_blueprint_library()

        vehicles_spawned = 0
        for i in range(min(8, self.args.num_background_vehicles)):
            try:
                vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))
                spawn_points = self.world.get_map().get_spawn_points()
                if spawn_points and len(spawn_points) > i + 15:
                    spawn_point = spawn_points[i + 15]
                    vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                    self.actors.append(vehicle)
                    vehicle.set_autopilot(True)
                    vehicles_spawned += 1
            except:
                pass

        pedestrians_spawned = 0
        if self.intersection_location:
            for i in range(min(10, self.args.num_pedestrians)):
                try:
                    ped_bp = random.choice(blueprint_lib.filter('walker.pedestrian.*'))
                    angle = random.uniform(0, 2 * math.pi)
                    radius = random.uniform(5.0, 20.0)
                    location = carla.Location(
                        x=self.intersection_location.x + radius * math.cos(angle),
                        y=self.intersection_location.y + radius * math.sin(angle),
                        z=self.intersection_location.z + 1.0
                    )
                    pedestrian = self.world.spawn_actor(ped_bp, carla.Transform(location))
                    self.actors.append(pedestrian)

                    controller_bp = blueprint_lib.find('controller.ai.walker')
                    controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=pedestrian)
                    controller.start()

                    target_angle = angle + random.uniform(-math.pi / 2, math.pi / 2)
                    target_radius = random.uniform(5.0, 15.0)
                    target_location = carla.Location(
                        x=self.intersection_location.x + target_radius * math.cos(target_angle),
                        y=self.intersection_location.y + target_radius * math.sin(target_angle),
                        z=location.z
                    )
                    controller.go_to_location(target_location)
                    pedestrians_spawned += 1
                    self.actors.append(controller)
                except Exception as e:
                    continue

        print(f"✓ 生成 {vehicles_spawned} 辆背景车辆和 {pedestrians_spawned} 个行人")

    def collect_collaborative_data(self):
        print("\n[5/6] 开始收集协同感知数据...")
        print(f"数据收集模式: 间隔{self.args.capture_interval}秒捕捉")
        print(f"预计总时长: {self.args.total_duration}秒")
        print(f"预计总帧数: {self.args.total_duration // self.args.capture_interval}")
        print("\n协同感知场景运行中...")
        print("按 Ctrl+C 提前结束\n")

        start_time = time.time()
        self.frame_count = 0
        self.last_save_time = start_time

        total_cameras = 4 + self.args.num_coop_vehicles * 4 + self.args.num_infra_cameras * 4
        print(f"摄像头总数: {total_cameras}个视角")
        print(f"每次捕捉保存: {total_cameras}张图像")

        try:
            update_interval = 5.0

            while time.time() - start_time < self.args.total_duration:
                elapsed = time.time() - start_time
                remaining = max(0, self.args.total_duration - elapsed)

                if int(elapsed) % update_interval == 0 and elapsed % update_interval < 0.1:
                    progress_percent = (elapsed / self.args.total_duration) * 100
                    estimated_total_frames = total_cameras * self.frame_count

                    print(f"  进度: {elapsed:.0f}/{self.args.total_duration}秒 "
                          f"({progress_percent:.1f}%) | "
                          f"已保存批次: {self.frame_count} | "
                          f"总图像数: {estimated_total_frames} | "
                          f"剩余: {remaining:.0f}秒")

                time.sleep(0.1)

            elapsed = time.time() - start_time
            estimated_total_frames = total_cameras * self.frame_count

            print(f"\n✓ 协同感知数据收集完成!")
            print(f"  总时长: {elapsed:.1f}秒")
            print(f"  保存批次: {self.frame_count}")
            print(f"  估计总图像数: {estimated_total_frames}")
            print(f"  批次间隔: {self.args.capture_interval}秒")

            self.display_collaborative_summary(total_cameras)

        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            estimated_total_frames = total_cameras * self.frame_count
            print(f"\n数据收集中断，已收集 {self.frame_count} 批次")
            print(f"估计总图像数: {estimated_total_frames}")

    def display_collaborative_summary(self, total_cameras):
        print("\n" + "=" * 60)
        print("协同感知数据收集摘要:")
        print("=" * 60)

        print(f"\n摄像头配置:")
        print(f"  主车辆: 4个视角")
        print(f"  协同车辆: {self.args.num_coop_vehicles}辆 × 4个视角 = {self.args.num_coop_vehicles * 4}个视角")
        print(f"  基础设施: {self.args.num_infra_cameras}个 × 4个视角 = {self.args.num_infra_cameras * 4}个视角")
        print(f"  总计: {total_cameras}个视角")

        print(f"\n数据详情:")

        ego_dir = os.path.join(self.output_dir, "ego_vehicle")
        if os.path.exists(ego_dir):
            for view in ['front_wide', 'front_narrow', 'right_side', 'left_side']:
                view_dir = os.path.join(ego_dir, view)
                if os.path.exists(view_dir):
                    count = len([f for f in os.listdir(view_dir) if f.endswith('.png')])
                    print(f"  主车-{view}: {count}张")

        infra_dir = os.path.join(self.output_dir, "infrastructure")
        if os.path.exists(infra_dir):
            for cam_dir in os.listdir(infra_dir):
                cam_path = os.path.join(infra_dir, cam_dir)
                if os.path.isdir(cam_path):
                    for angle in ['overview', 'left_view', 'right_view', 'closeup']:
                        angle_dir = os.path.join(cam_path, angle)
                        if os.path.exists(angle_dir):
                            count = len([f for f in os.listdir(angle_dir) if f.endswith('.png')])
                            print(f"  基础设施{cam_dir}-{angle}: {count}张")

        print(f"\n数据目录: {self.output_dir}")
        print("=" * 60)

    def cleanup(self):
        print("\n清理协同感知场景...")

        for sensor in self.sensors + self.infrastructure_cameras:
            try:
                sensor.stop()
            except:
                pass

        destroyed = 0
        for actor in self.actors:
            try:
                if actor and actor.is_alive:
                    actor.destroy()
                    destroyed += 1
            except:
                pass

        print(f"销毁 {destroyed} 个actor")
        self.actors.clear()
        self.sensors.clear()
        self.infrastructure_cameras.clear()
        self.cooperative_vehicles.clear()


def main():
    parser = argparse.ArgumentParser(
        description='CVIPS v4.0 - 基础设施摄像头与多车协同数据生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
协同感知场景示例:
  # 基础协同场景（1主车 + 1协同车 + 2基础设施摄像头）
  python cvips_infrastructure_collaborative.py

  # 复杂协同场景
  python cvips_infrastructure_collaborative.py --num-coop-vehicles 2 --num-infra-cameras 3 --total-duration 120 --capture-interval 3.0

  # 雨天夜晚协同场景
  python cvips_infrastructure_collaborative.py --weather rainy --time-of-day night --scenario night_rain_collab
        """
    )

    parser.add_argument('--scenario', type=str, default='collaborative_test', help='协同感知场景名称')
    parser.add_argument('--town', type=str, default='Town10HD', choices=['Town03', 'Town04', 'Town05', 'Town10HD'], help='CARLA地图')
    parser.add_argument('--weather', type=str, default='clear', choices=['clear', 'rainy', 'cloudy'], help='天气条件')
    parser.add_argument('--time-of-day', type=str, default='noon', choices=['noon', 'sunset', 'night'], help='时间')
    parser.add_argument('--num-coop-vehicles', type=int, default=1, help='协同车辆数量（不包括主车）')
    parser.add_argument('--num-infra-cameras', type=int, default=2, help='基础设施摄像头数量')
    parser.add_argument('--num-pedestrians', type=int, default=8, help='行人数（主要集中在路口）')
    parser.add_argument('--num-background-vehicles', type=int, default=6, help='背景车辆数')
    parser.add_argument('--total-duration', type=int, default=90, help='总收集时间(秒)')
    parser.add_argument('--capture-interval', type=float, default=2.5, help='图像捕捉间隔(秒)')

    args = parser.parse_args(remaining_argv)

    if args.num_coop_vehicles > 3:
        print("⚠ 警告: 协同车辆过多可能导致性能下降")
        args.num_coop_vehicles = min(args.num_coop_vehicles, 3)

    if args.num_infra_cameras > 4:
        print("⚠ 警告: 基础设施摄像头过多可能导致性能下降")
        args.num_infra_cameras = min(args.num_infra_cameras, 4)

    if args.capture_interval < 2.0:
        print("⚠ 建议: 捕捉间隔建议2.0秒以上，以确保场景有明显变化")
        args.capture_interval = 2.0

    print(f"\n协同感知场景配置:")
    print(f"  场景名称: {args.scenario}")
    print(f"  地图: {args.town}")
    print(f"  天气/时间: {args.weather}/{args.time_of_day}")
    print(f"  车辆配置: 1主车 + {args.num_coop_vehicles}协同车")
    print(f"  基础设施: {args.num_infra_cameras}个摄像头")
    print(f"  行人/背景车: {args.num_pedestrians}/{args.num_background_vehicles}")
    print(f"  总时长: {args.total_duration}秒")
    print(f"  捕捉间隔: {args.capture_interval}秒")

    total_cameras = 4 + (args.num_coop_vehicles * 4) + (args.num_infra_cameras * 4)
    estimated_frames = args.total_duration // args.capture_interval
    estimated_images = total_cameras * estimated_frames

    print(f"  预计: {estimated_frames}批次 × {total_cameras}视角 = {estimated_images}张图像")

    generator = InfrastructureCollaborativeGenerator(args)

    try:
        if not generator.connect_to_server():
            print("\n连接失败，退出")
            return

        ego_vehicle = generator.setup_collaborative_scene()
        if not ego_vehicle:
            print("\n场景设置失败")
            generator.cleanup()
            return

        generator.setup_ego_cameras(ego_vehicle)
        generator.setup_cooperative_cameras()
        generator.collect_collaborative_data()

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n运行出错: {e}")
        traceback.print_exc()
    finally:
        generator.cleanup()

        print("\n" + "=" * 80)
        print("协同感知数据收集完成!")
        print(f"数据保存到: {generator.output_dir}")
        print("=" * 80)

        print("\n后续优化建议:")
        print("1. 检查各视角图像质量")
        print("2. 验证基础设施摄像头视角是否覆盖路口")
        print("3. 调整车辆和摄像头位置以获得更好视角")
        print("4. 可尝试增加/减少协同车辆数量")
        print("5. 调整捕捉间隔以获得不同时间尺度的数据")


if __name__ == "__main__":
    main()