import sys
import os
import time
import random
import argparse
import traceback
from datetime import datetime

print("=" * 80)
print("CVIPS v3.0 - 行人安全数据生成器")
print("=" * 80)

# ============================================================
# 1. 设置CARLA路径
# ============================================================
print("\n[1/5] 初始化CARLA环境...")
CARLA_EGG = r"D:\carla\carla0914\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.14-py3.7-win-amd64.egg"

if os.path.exists(CARLA_EGG):
    sys.path.append(CARLA_EGG)
    print(f"✓ CARLA路径设置成功")
else:
    print(f"✗ 找不到egg文件: {CARLA_EGG}")
    sys.exit(1)

# ============================================================
# 2. 导入CARLA
# ============================================================
print("\n[2/5] 导入CARLA模块...")
try:
    import carla

    print("✓ CARLA模块导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)


# ============================================================
# 3. 行人安全数据生成器类
# ============================================================
class PedestrianSafetyGenerator:
    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.actors = []
        self.sensors = []
        self.frame_count = 0
        self.last_save_time = time.time()

        # 创建输出目录
        self.setup_output_directory()

    def setup_output_directory(self):
        """设置输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_name = f"{self.args.scenario}_ped_safety_{timestamp}"
        self.output_dir = os.path.join("pedestrian_safety_data", scenario_name)
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"输出目录: {self.output_dir}")

    def connect_to_server(self):
        """连接到CARLA服务器"""
        print("\n[3/5] 连接到CARLA服务器...")

        for attempt in range(1, 6):
            try:
                print(f"  尝试 {attempt}/5...")

                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(15.0)

                # 加载指定地图（推荐使用交叉路口丰富的地图）
                if self.args.town:
                    self.world = self.client.load_world(self.args.town)
                else:
                    self.world = self.client.get_world()

                print(f"✓ 连接成功! 地图: {self.world.get_map().name}")

                # 设置异步模式
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
        """设置行人安全场景"""
        print("\n[4/5] 设置行人安全场景...")

        try:
            # 1. 设置高质量渲染（关闭模糊效果）
            self.set_high_quality_settings()

            # 2. 设置行人友好的天气和光照
            self.set_pedestrian_friendly_environment()

            # 3. 等待世界稳定
            time.sleep(2.0)

            # 4. 生成主车辆（在行人过马路区域附近）
            ego_vehicle = self.spawn_ego_vehicle_near_crosswalk()
            if not ego_vehicle:
                print("⚠ 无法生成主车辆")
                return None

            # 5. 生成重点行人（过马路行人）
            crossing_pedestrians = self.spawn_crossing_pedestrians()

            # 6. 生成普通行人和交通
            self.spawn_background_traffic()

            # 7. 等待场景稳定
            print("等待场景稳定...")
            time.sleep(5.0)

            return ego_vehicle

        except Exception as e:
            print(f"设置场景失败: {e}")
            return None

    def set_high_quality_settings(self):
        """设置高质量渲染设置（专门为行人检测优化）"""
        try:
            # 关闭所有可能导致模糊的效果
            quality_settings = {
                'r.MotionBlurQuality': 0,  # 关闭运动模糊
                'r.DepthOfFieldQuality': 0,  # 关闭景深
                'r.BloomQuality': 0,  # 关闭光晕
                'r.LensFlareQuality': 0,  # 关闭镜头光晕
                'r.TonemapperQuality': 0,  # 关闭色调映射
                'r.AmbientOcclusionLevels': 0,  # 关闭环境光遮蔽
                'r.ShadowQuality': 3,  # 高质量阴影
                'r.TextureStreaming': True,  # 启用纹理流
            }

            for key, value in quality_settings.items():
                self.world.get_settings().set(str(key), str(value))

            print("✓ 高质量行人检测设置已应用")

        except Exception as e:
            print(f"设置渲染质量失败: {e}")

    def set_pedestrian_friendly_environment(self):
        """设置行人友好的环境（良好的光照条件）"""
        weather = carla.WeatherParameters()

        # 根据参数设置天气，确保行人可见
        if self.args.weather == 'clear':
            # 晴天最佳能见度
            weather.sun_altitude_angle = 75  # 稍微倾斜的光线，产生更好的阴影
            weather.sun_azimuth_angle = 0
            weather.cloudiness = 5.0  # 少量云增加真实感
            weather.precipitation = 0.0
            weather.precipitation_deposits = 0.0
            weather.wind_intensity = 5.0
            weather.fog_density = 0.0
            weather.wetness = 0.0
            weather.scattering_intensity = 1.0
            weather.mie_scattering_scale = 0.8
            weather.rayleigh_scattering_scale = 1.0

        elif self.args.weather == 'rainy':
            # 雨天但保持能见度
            weather.sun_altitude_angle = 40
            weather.cloudiness = 90.0
            weather.precipitation = 60.0
            weather.precipitation_deposits = 50.0
            weather.wind_intensity = 30.0
            weather.fog_density = 15.0  # 轻微雾气
            weather.wetness = 70.0
            weather.scattering_intensity = 1.5

        elif self.args.weather == 'cloudy':
            # 阴天，均匀光照
            weather.sun_altitude_angle = 60
            weather.cloudiness = 70.0
            weather.precipitation = 0.0
            weather.wind_intensity = 10.0
            weather.fog_density = 5.0
            weather.wetness = 10.0
            weather.scattering_intensity = 1.2

        # 时间设置（确保行人可见）
        if self.args.time_of_day == 'night':
            weather.sun_altitude_angle = -10  # 夜晚但有一定月光
            weather.fog_density = 5.0
            weather.wetness = 10.0
            # 增加路灯照明效果
            weather.scattering_intensity = 1.8
        elif self.args.time_of_day == 'sunset':
            weather.sun_altitude_angle = 5  # 日落时分
            weather.cloudiness = 50.0
            weather.fog_density = 10.0

        self.world.set_weather(weather)
        print(f"✓ 行人友好环境设置: {self.args.weather}, {self.args.time_of_day}")

    def find_crosswalk_locations(self):
        """寻找行人过马路区域"""
        # 获取地图的所有生成点
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            return []

        # 选择适合行人过马路的区域（通常是路口）
        crosswalk_points = []

        # 简单策略：选择前几个生成点（通常在地图的重要位置）
        for i, point in enumerate(spawn_points[:10]):
            crosswalk_points.append(point)

        return crosswalk_points[:3]  # 最多选择3个过马路点

    def spawn_ego_vehicle_near_crosswalk(self):
        """在行人过马路区域附近生成主车辆"""
        blueprint_lib = self.world.get_blueprint_library()

        # 选择视野好的车辆（高底盘，大窗户）
        vehicle_types = [
            'vehicle.tesla.model3',  # 电动车，视野好
            'vehicle.audi.tt',  # 紧凑型，视野好
            'vehicle.mini.cooperst',  # 小型车，适合城市
            'vehicle.nissan.patrol'  # SUV，高视野
        ]

        vehicle_bp = None
        for vtype in vehicle_types:
            if blueprint_lib.filter(vtype):
                vehicle_bp = random.choice(blueprint_lib.filter(vtype))
                break

        if not vehicle_bp:
            vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))

        # 寻找过马路区域
        crosswalk_points = self.find_crosswalk_locations()
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            print("⚠ 没有生成点")
            return None

        # 如果有过马路点，在附近生成；否则随机生成
        if crosswalk_points:
            # 选择第一个过马路点附近
            crosswalk_point = crosswalk_points[0]

            # 在过马路点附近找一个生成点
            nearest_spawn_point = None
            min_distance = float('inf')

            for spawn_point in spawn_points:
                distance = spawn_point.location.distance(crosswalk_point.location)
                if distance < min_distance and distance > 10.0:  # 不要太近也不要太远
                    min_distance = distance
                    nearest_spawn_point = spawn_point

            spawn_point = nearest_spawn_point if nearest_spawn_point else random.choice(spawn_points)
        else:
            spawn_point = random.choice(spawn_points)

        try:
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.actors.append(vehicle)

            # 设置自动驾驶，但速度较慢以便观察行人
            vehicle.set_autopilot(True)

            # 控制速度（行人安全场景需要慢速）
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.2,  # 低速
                brake=0.0,
                steer=0.0,
                hand_brake=False,
                reverse=False
            ))

            print(f"✓ 生成主车辆: {vehicle.type_id}")
            print(f"  位置: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")

            return vehicle

        except Exception as e:
            print(f"生成主车辆失败: {e}")
            return None

    def spawn_crossing_pedestrians(self):
        """生成过马路的行人（重点行人）"""
        blueprint_lib = self.world.get_blueprint_library()

        # 寻找过马路区域
        crosswalk_points = self.find_crosswalk_locations()
        if not crosswalk_points:
            print("⚠ 未找到合适的过马路区域")
            return []

        crossing_pedestrians = []
        num_crossing = min(3, self.args.num_crossing_pedestrians)  # 最多3个重点行人

        print(f"生成 {num_crossing} 个过马路行人...")

        for i in range(num_crossing):
            try:
                # 选择行人类型（多样化）
                ped_types = list(blueprint_lib.filter('walker.pedestrian.*'))
                if not ped_types:
                    continue

                ped_bp = random.choice(ped_types)

                # 在过马路点附近生成
                crosswalk_point = crosswalk_points[i % len(crosswalk_points)]

                # 在过马路点附近稍微偏移
                offset_x = random.uniform(-3.0, 3.0)
                offset_y = random.uniform(-3.0, 3.0)

                location = carla.Location(
                    x=crosswalk_point.location.x + offset_x,
                    y=crosswalk_point.location.y + offset_y,
                    z=crosswalk_point.location.z + 1.0  # 确保在地面上
                )

                spawn_point = carla.Transform(location)

                # 生成行人
                pedestrian = self.world.spawn_actor(ped_bp, spawn_point)
                self.actors.append(pedestrian)

                # 添加AI控制器
                controller_bp = blueprint_lib.find('controller.ai.walker')
                if controller_bp:
                    controller = self.world.spawn_actor(
                        controller_bp,
                        carla.Transform(),
                        attach_to=pedestrian
                    )

                    # 设置过马路行为
                    controller.start()

                    # 计算过马路的目标点（对面）
                    target_location = carla.Location(
                        x=location.x + random.uniform(10.0, 20.0) * (1 if random.random() > 0.5 else -1),
                        y=location.y + random.uniform(10.0, 20.0) * (1 if random.random() > 0.5 else -1),
                        z=location.z
                    )

                    controller.go_to_location(target_location)

                    crossing_pedestrians.append(pedestrian)
                    self.actors.append(controller)

                    print(f"  过马路行人 {i + 1} 已生成")

            except Exception as e:
                print(f"  生成过马路行人失败: {e}")
                continue

        print(f"✓ 生成 {len(crossing_pedestrians)} 个过马路行人")
        return crossing_pedestrians

    def spawn_background_traffic(self):
        """生成背景交通（其他车辆和行人）"""
        blueprint_lib = self.world.get_blueprint_library()

        # 生成其他车辆（数量较少，避免干扰）
        vehicles_spawned = 0
        for i in range(min(5, self.args.num_background_vehicles)):
            try:
                vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))
                spawn_points = self.world.get_map().get_spawn_points()

                if spawn_points and len(spawn_points) > i + 10:  # 避免位置冲突
                    spawn_point = spawn_points[i + 10]
                    vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                    self.actors.append(vehicle)
                    vehicle.set_autopilot(True)
                    vehicles_spawned += 1
            except:
                pass

        # 生成背景行人（不在过马路）
        pedestrians_spawned = 0
        for i in range(min(8, self.args.num_background_pedestrians)):
            try:
                ped_bp = random.choice(blueprint_lib.filter('walker.pedestrian.*'))
                location = self.world.get_random_location_from_navigation()

                if location:
                    location.z += 1.0
                    pedestrian = self.world.spawn_actor(ped_bp, carla.Transform(location))
                    self.actors.append(pedestrian)

                    # 添加控制器
                    controller_bp = blueprint_lib.find('controller.ai.walker')
                    controller = self.world.spawn_actor(
                        controller_bp,
                        carla.Transform(),
                        attach_to=pedestrian
                    )
                    controller.start()

                    # 设置随机目标（非过马路）
                    target_location = self.world.get_random_location_from_navigation()
                    if target_location:
                        controller.go_to_location(target_location)

                    pedestrians_spawned += 1
                    self.actors.append(controller)
            except:
                pass

        print(f"✓ 生成 {vehicles_spawned} 辆背景车辆和 {pedestrians_spawned} 个背景行人")

    def setup_pedestrian_safety_cameras(self, vehicle):
        """设置行人安全专用的摄像头系统"""
        if not vehicle:
            return

        blueprint_lib = self.world.get_blueprint_library()

        print("\n安装行人安全摄像头系统...")

        # 定义四个关键视角的摄像头
        camera_configs = [
            # (名称, 位置, 旋转, 视野角度, 说明)
            ('front_wide',
             carla.Location(x=2.0, z=1.8),
             carla.Rotation(pitch=-3.0),
             100,  # 广角视野，覆盖前方大范围
             "前视广角摄像头 - 检测前方行人"),

            ('front_narrow',
             carla.Location(x=2.0, z=1.6),
             carla.Rotation(pitch=0),
             60,  # 窄角视野，专注前方细节
             "前视窄角摄像头 - 细节识别"),

            ('right_side',
             carla.Location(x=0.5, y=1.0, z=1.5),
             carla.Rotation(pitch=-2.0, yaw=45),
             90,  # 右侧前方视野，检测右侧穿行行人
             "右侧摄像头 - 检测右侧行人"),

            ('left_side',
             carla.Location(x=0.5, y=-1.0, z=1.5),
             carla.Rotation(pitch=-2.0, yaw=-45),
             90,  # 左侧前方视野，检测左侧穿行行人
             "左侧摄像头 - 检测左侧行人")
        ]

        installed_cameras = 0

        for name, location, rotation, fov, description in camera_configs:
            try:
                camera_bp = blueprint_lib.find('sensor.camera.rgb')

                # ========== 行人检测优化设置 ==========
                camera_bp.set_attribute('image_size_x', '1280')  # 高清但不至于过大
                camera_bp.set_attribute('image_size_y', '720')  # 720p
                camera_bp.set_attribute('fov', str(fov))  # 根据配置设置视野
                camera_bp.set_attribute('motion_blur_intensity', '0.0')  # 关闭运动模糊
                camera_bp.set_attribute('motion_blur_max_distortion', '0.0')
                camera_bp.set_attribute('enable_postprocess_effects', 'False')
                camera_bp.set_attribute('gamma', '2.2')
                camera_bp.set_attribute('shutter_speed', '100')  # 较快快门
                camera_bp.set_attribute('iso', '200')
                camera_bp.set_attribute('fstop', '2.0')
                camera_bp.set_attribute('lens_k', '0.0')  # 无镜头畸变
                camera_bp.set_attribute('lens_kcube', '0.0')
                camera_bp.set_attribute('lens_x_size', '0.08')
                camera_bp.set_attribute('lens_y_size', '0.08')
                # ====================================

                transform = carla.Transform(location, rotation)
                camera = self.world.spawn_actor(camera_bp, transform, attach_to=vehicle)

                # 为每个摄像头创建保存目录
                camera_dir = os.path.join(self.output_dir, name)
                os.makedirs(camera_dir, exist_ok=True)

                # 图像保存回调函数（带时间间隔控制）
                def make_save_callback(save_dir, cam_name, cam_desc):
                    def save_image(image):
                        current_time = time.time()

                        # 控制保存间隔（至少2秒一张）
                        if current_time - self.last_save_time >= self.args.capture_interval:
                            self.frame_count += 1
                            self.last_save_time = current_time

                            # 保存图像
                            filename = f"{save_dir}/ped_frame_{self.frame_count:04d}.png"
                            image.save_to_disk(filename, carla.ColorConverter.Raw)

                            # 每10帧打印一次信息
                            if self.frame_count % 10 == 0:
                                print(f"  [{cam_name}] 保存第 {self.frame_count} 帧")

                    return save_image

                camera.listen(make_save_callback(camera_dir, name, description))
                self.actors.append(camera)
                self.sensors.append(camera)

                installed_cameras += 1
                print(f"✓ {description}")

            except Exception as e:
                print(f"  安装{name}摄像头失败: {e}")

        print(f"✓ 总共安装 {installed_cameras} 个行人安全摄像头")

    def collect_pedestrian_safety_data(self):
        """收集行人安全数据"""
        print("\n[5/5] 开始收集行人安全数据...")
        print(f"数据收集模式: 间隔{self.args.capture_interval}秒捕捉")
        print(f"预计总时长: {self.args.total_duration}秒")
        print(f"预计帧数: {self.args.total_duration // self.args.capture_interval}")
        print("\n提示: 正在模拟行人过马路场景...")
        print("按 Ctrl+C 提前结束\n")

        start_time = time.time()
        self.frame_count = 0
        self.last_save_time = start_time

        try:
            # 创建进度显示
            update_interval = 5.0  # 每5秒更新一次进度

            while time.time() - start_time < self.args.total_duration:
                elapsed = time.time() - start_time
                remaining = max(0, self.args.total_duration - elapsed)

                # 显示进度
                if int(elapsed) % update_interval == 0 and elapsed % update_interval < 0.1:
                    progress_percent = (elapsed / self.args.total_duration) * 100

                    print(f"  进度: {elapsed:.0f}/{self.args.total_duration}秒 "
                          f"({progress_percent:.1f}%) | "
                          f"已保存帧数: {self.frame_count} | "
                          f"剩余: {remaining:.0f}秒")

                # 轻微睡眠减少CPU使用
                time.sleep(0.1)

            # 收集完成
            elapsed = time.time() - start_time

            print(f"\n✓ 行人安全数据收集完成!")
            print(f"  总时长: {elapsed:.1f}秒")
            print(f"  保存帧数: {self.frame_count}")
            print(f"  实际帧率: {self.frame_count / elapsed if elapsed > 0 else 0:.2f} FPS")

            # 显示数据摘要
            self.display_data_summary()

        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            print(f"\n数据收集中断，已收集 {self.frame_count} 帧")
            print(f"总时长: {elapsed:.1f}秒")

    def display_data_summary(self):
        """显示数据收集摘要"""
        print("\n" + "-" * 60)
        print("数据收集摘要:")
        print("-" * 60)

        # 检查每个摄像头保存的图像数量
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
        """清理场景（只清理必要部分）"""
        print("\n清理场景...")

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


# ============================================================
# 4. 主函数 - 行人安全数据生成
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='CVIPS v3.0 - 行人安全协同感知数据生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    # 场景参数
    parser.add_argument('--scenario', type=str, default='pedestrian_crossing',
                        help='场景名称')
    parser.add_argument('--town', type=str, default='Town10HD',
                        choices=['Town03', 'Town04', 'Town05', 'Town10HD'],
                        help='CARLA地图（推荐Town10HD行人多）')

    # 环境参数（为行人检测优化）
    parser.add_argument('--weather', type=str, default='clear',
                        choices=['clear', 'rainy', 'cloudy'],
                        help='天气条件（推荐clear）')
    parser.add_argument('--time-of-day', type=str, default='noon',
                        choices=['noon', 'sunset', 'night'],
                        help='时间（推荐noon）')

    # 行人参数（重点）
    parser.add_argument('--num-crossing-pedestrians', type=int, default=3,
                        help='过马路行人数（重点行人）')
    parser.add_argument('--num-background-pedestrians', type=int, default=6,
                        help='背景行人数')
    parser.add_argument('--num-background-vehicles', type=int, default=4,
                        help='背景车辆数')

    # 数据收集参数（优化捕捉间隔）
    parser.add_argument('--total-duration', type=int, default=60,
                        help='总收集时间(秒)')
    parser.add_argument('--capture-interval', type=float, default=2.0,
                        help='图像捕捉间隔(秒) - 建议2.0-5.0秒')

    args = parser.parse_args()

    # 参数验证
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

    # 创建行人安全数据生成器
    generator = PedestrianSafetyGenerator(args)

    try:
        # 1. 连接到服务器
        if not generator.connect_to_server():
            print("\n连接失败，退出")
            return

        # 2. 设置行人安全场景
        ego_vehicle = generator.setup_pedestrian_safety_scene()

        if not ego_vehicle:
            print("\n场景设置失败")
            generator.cleanup()
            return

        # 3. 安装行人安全摄像头系统
        generator.setup_pedestrian_safety_cameras(ego_vehicle)

        if not generator.sensors:
            print("\n摄像头安装失败")
            generator.cleanup()
            return

        # 4. 收集行人安全数据
        generator.collect_pedestrian_safety_data()

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n运行出错: {e}")
        traceback.print_exc()
    finally:
        # 5. 清理场景
        generator.cleanup()

        print("\n" + "=" * 80)
        print("行人安全数据收集完成!")
        print(f"数据保存到: {generator.output_dir}")
        print("=" * 80)

        # 使用提示
        print("\n使用提示:")
        print("1. 检查输出目录中的图像质量")
        print("2. 确保行人清晰可见")
        print("3. 可调整capture-interval参数控制图像间隔")
        print("4. 下次运行可尝试不同天气和时间条件")


# ============================================================
# 5. 程序入口
# ============================================================
if __name__ == "__main__":
    main()