# cvips_clear_images.py
"""
CVIPS 清晰图像版本 - 修复图像模糊问题
"""

import sys
import os
import time
import random
import argparse
import traceback
import json
from datetime import datetime

print("=" * 80)
print("CVIPS 清晰图像数据生成器")
print("=" * 80)

# ============================================================
# 1. 设置CARLA路径
# ============================================================
print("\n[1/5] 设置CARLA路径...")
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
# 3. 高清图像生成器类
# ============================================================
class CVIPSHDGenerator:
    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.actors = []
        self.sensors = []
        self.frame_count = 0

        # 创建输出目录
        self.setup_output()

    def setup_output(self):
        """设置输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"cvips_hd/{self.args.scenario}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        # 创建传感器子目录
        self.sensor_dirs = {}
        for sensor in ['front_hd', 'rear_hd', 'left_hd', 'right_hd']:
            dir_path = os.path.join(self.output_dir, sensor)
            os.makedirs(dir_path, exist_ok=True)
            self.sensor_dirs[sensor] = dir_path

        print(f"高清输出目录: {self.output_dir}")

    def connect_with_retry(self):
        """带重试的连接到CARLA服务器"""
        print("\n[3/5] 连接到CARLA服务器...")

        for attempt in range(1, 6):
            try:
                print(f"  尝试 {attempt}/5...")

                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(10.0)

                # 加载指定地图
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
                print(f"  尝试 {attempt} 失败: {str(e)[:80]}...")
                if attempt < 5:
                    print("  等待3秒后重试...")
                    time.sleep(3)

        print("✗ 连接失败")
        return False

    def setup_scene(self):
        """设置高清场景"""
        print("\n[4/5] 设置高清场景...")

        # 1. 设置高清渲染质量
        self.set_high_quality_settings()

        # 2. 设置天气和时间
        self.set_weather_and_time()

        # 3. 等待世界稳定
        time.sleep(3.0)

        # 4. 生成主车辆
        ego_vehicle = self.spawn_ego_vehicle()
        if not ego_vehicle:
            print("⚠ 无法生成主车辆")
            return None

        # 5. 生成交通
        self.spawn_traffic()

        # 6. 等待交通稳定
        time.sleep(5.0)

        return ego_vehicle

    def set_high_quality_settings(self):
        """设置高质量渲染设置"""
        try:
            # 设置高质量渲染参数
            quality_settings = {
                'epic': {
                    'QualityLevel': 'Epic',
                    'r.ShadowQuality': 3,  # 高质量阴影
                    'r.ReflectionQuality': 3,  # 高质量反射
                    'r.PostProcessAAQuality': 6,  # 高质量抗锯齿
                    'r.TextureStreaming': True,
                    'r.MotionBlurQuality': 0,  # 关闭运动模糊（重要！）
                    'r.DepthOfFieldQuality': 0,  # 关闭景深模糊
                    'r.BloomQuality': 0,  # 关闭光晕效果
                    'r.TonemapperQuality': 0,  # 关闭色调映射
                    'r.LensFlareQuality': 0,  # 关闭镜头光晕
                    'r.SSAOQuality': 0,  # 关闭环境光遮蔽
                }
            }

            # 应用设置
            for key, value in quality_settings['epic'].items():
                self.world.get_settings().set(str(key), str(value))

            print("✓ 高质量渲染设置已应用")
            print("  - 关闭运动模糊")
            print("  - 关闭景深效果")
            print("  - 高质量抗锯齿")

        except Exception as e:
            print(f"设置高质量渲染失败: {e}")

    def set_weather_and_time(self):
        """设置天气和时间（优化版）"""
        weather = carla.WeatherParameters()

        # 天气设置（增加光照强度）
        if self.args.weather == 'clear':
            weather.sun_altitude_angle = 90  # 正午太阳
            weather.sun_azimuth_angle = 0
            weather.cloudiness = 0.0
            weather.precipitation = 0.0
            weather.wind_intensity = 0.0
            weather.fog_density = 0.0
            weather.wetness = 0.0
            weather.scattering_intensity = 1.0  # 增加散射强度
            weather.mie_scattering_scale = 1.0
            weather.rayleigh_scattering_scale = 1.0

        elif self.args.weather == 'rainy':
            weather.sun_altitude_angle = 45  # 较低但仍有光线
            weather.cloudiness = 90.0
            weather.precipitation = 80.0
            weather.precipitation_deposits = 60.0
            weather.wind_intensity = 40.0
            weather.fog_density = 20.0
            weather.wetness = 80.0
            weather.scattering_intensity = 1.5  # 增加散射以补偿阴天

        elif self.args.weather == 'cloudy':
            weather.sun_altitude_angle = 60
            weather.cloudiness = 70.0
            weather.precipitation = 0.0
            weather.wind_intensity = 10.0
            weather.fog_density = 5.0
            weather.wetness = 0.0
            weather.scattering_intensity = 1.2

        # 时间设置
        if self.args.time_of_day == 'night':
            weather.sun_altitude_angle = -15  # 夜晚但有一定月光
            weather.fog_density = 0.1  # 轻微雾气增加真实感
        elif self.args.time_of_day == 'sunset':
            weather.sun_altitude_angle = 0  # 日落
            weather.cloudiness = 40.0  # 晚霞效果

        self.world.set_weather(weather)
        print(f"✓ 天气: {self.args.weather}, 时间: {self.args.time_of_day}")

    def spawn_ego_vehicle(self):
        """生成主车辆"""
        blueprint_lib = self.world.get_blueprint_library()

        # 选择高清模型车辆
        vehicle_types = [
            'vehicle.tesla.model3',  # 特斯拉模型细节丰富
            'vehicle.audi.a2',  # 奥迪模型高清
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

        # 获取生成点（选择光照好的位置）
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            print("⚠ 没有生成点")
            return None

        # 选择一个开阔区域的生成点（避免在阴影中）
        spawn_point = spawn_points[0]  # 通常第一个点位置较好

        try:
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.actors.append(vehicle)

            # 设置自动驾驶，但速度较慢便于清晰拍摄
            vehicle.set_autopilot(True)

            # 限制速度（可选）
            vehicle.apply_control(carla.VehicleControl(throttle=0.3, brake=0.0))

            print(f"✓ 生成主车辆: {vehicle.type_id}")
            return vehicle

        except Exception as e:
            print(f"生成车辆失败: {e}")
            return None

    def spawn_traffic(self):
        """生成交通"""
        blueprint_lib = self.world.get_blueprint_library()

        print(f"生成 {self.args.num_vehicles} 辆车和 {self.args.num_pedestrians} 个行人")

        # 生成其他车辆
        vehicles_spawned = 0
        for i in range(self.args.num_vehicles):
            try:
                vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))
                spawn_points = self.world.get_map().get_spawn_points()

                if spawn_points and len(spawn_points) > i + 5:  # 避免位置冲突
                    spawn_point = spawn_points[i + 5]  # 使用稍远的位置
                    vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                    self.actors.append(vehicle)
                    vehicle.set_autopilot(True)
                    vehicles_spawned += 1
            except:
                pass

        # 生成行人（只在白天生成，夜晚行人看不清）
        pedestrians_spawned = 0
        if self.args.time_of_day != 'night' or self.args.num_pedestrians > 0:
            for i in range(self.args.num_pedestrians):
                try:
                    ped_bp = random.choice(blueprint_lib.filter('walker.pedestrian.*'))
                    location = self.world.get_random_location_from_navigation()

                    if location:
                        location.z += 1.0
                        pedestrian = self.world.spawn_actor(ped_bp, carla.Transform(location))
                        self.actors.append(pedestrian)

                        # 添加控制器
                        controller_bp = blueprint_lib.find('controller.ai.walker')
                        controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=pedestrian)
                        controller.start()
                        self.actors.append(controller)

                        # 设置目标
                        target = self.world.get_random_location_from_navigation()
                        if target:
                            controller.go_to_location(target)

                        pedestrians_spawned += 1
                except:
                    pass

        print(f"✓ 实际生成 {vehicles_spawned} 辆车和 {pedestrians_spawned} 个行人")

    def setup_hd_cameras(self, vehicle):
        """设置高清摄像头"""
        if not vehicle:
            return

        blueprint_lib = self.world.get_blueprint_library()

        # 摄像头位置配置
        camera_configs = [
            ('front_hd', carla.Transform(
                carla.Location(x=2.0, z=1.8),  # 更靠前，更高
                carla.Rotation(pitch=-5.0)  # 稍微向下看
            )),
            ('rear_hd', carla.Transform(
                carla.Location(x=-1.8, z=1.8),
                carla.Rotation(pitch=-5.0, yaw=180)
            )),
            ('left_hd', carla.Transform(
                carla.Location(x=0.0, y=-1.2, z=1.6),
                carla.Rotation(pitch=-3.0, yaw=-90)
            )),
            ('right_hd', carla.Transform(
                carla.Location(x=0.0, y=1.2, z=1.6),
                carla.Rotation(pitch=-3.0, yaw=90)
            ))
        ]

        for name, transform in camera_configs:
            try:
                camera_bp = blueprint_lib.find('sensor.camera.rgb')

                # ========== 关键：高清摄像头设置 ==========
                camera_bp.set_attribute('image_size_x', '1920')  # 全高清宽度
                camera_bp.set_attribute('image_size_y', '1080')  # 全高清高度
                camera_bp.set_attribute('fov', '80')  # 合适的视野
                camera_bp.set_attribute('motion_blur_intensity', '0.0')  # 关闭运动模糊
                camera_bp.set_attribute('motion_blur_max_distortion', '0.0')
                camera_bp.set_attribute('motion_blur_min_object_screen_size', '0.0')
                camera_bp.set_attribute('enable_postprocess_effects', 'False')  # 关闭后期效果
                camera_bp.set_attribute('gamma', '2.2')  # 标准gamma
                camera_bp.set_attribute('shutter_speed', '200')  # 较快快门减少模糊
                camera_bp.set_attribute('iso', '100')  # 低ISO减少噪点
                camera_bp.set_attribute('fstop', '1.8')  # 较大光圈
                camera_bp.set_attribute('lens_circle_multiplier', '0.0')  # 关闭镜头畸变
                camera_bp.set_attribute('lens_circle_falloff', '0.0')
                camera_bp.set_attribute('chromatic_aberration_intensity', '0.0')  # 关闭色差
                camera_bp.set_attribute('chromatic_aberration_offset', '0.0')
                # ========================================

                camera = self.world.spawn_actor(camera_bp, transform, attach_to=vehicle)

                # 图像保存回调函数
                def make_save_callback(save_dir, sensor_name):
                    def save_image(image):
                        self.frame_count += 1

                        # 使用最高质量保存
                        image.save_to_disk(
                            f"{save_dir}/frame_{self.frame_count:06d}.png",
                            carla.ColorConverter.Raw  # 保存为原始格式
                        )

                        # 打印第一帧的信息用于调试
                        if self.frame_count == 1:
                            print(f"  第一帧保存: {sensor_name}, 尺寸: {image.width}x{image.height}")

                    return save_image

                camera.listen(make_save_callback(self.sensor_dirs[name], name))
                self.actors.append(camera)
                self.sensors.append(camera)

                print(f"✓ 安装{name}摄像头 (1920x1080)")

            except Exception as e:
                print(f"安装{name}摄像头失败: {e}")

        print(f"✓ 总共安装 {len(self.sensors)} 个高清摄像头")

    def collect_data(self):
        """收集高清数据"""
        print("\n[5/5] 收集高清数据...")
        print(f"持续时间: {self.args.duration}秒")
        print("提示: 车辆低速行驶，图像更清晰")
        print("按 Ctrl+C 提前结束")

        start_time = time.time()
        initial_count = self.frame_count

        try:
            while time.time() - start_time < self.args.duration:
                elapsed = time.time() - start_time

                # 每5秒显示进度
                if int(elapsed) % 5 == 0 and elapsed % 5 < 0.1:
                    collected = self.frame_count - initial_count
                    remaining = self.args.duration - elapsed
                    fps = collected / elapsed if elapsed > 0 else 0

                    print(f"  进度: {elapsed:.1f}/{self.args.duration}秒 | "
                          f"帧数: {collected} | "
                          f"FPS: {fps:.1f}")

                time.sleep(0.1)

            # 收集完成
            collected = self.frame_count - initial_count
            elapsed = time.time() - start_time
            fps = collected / elapsed if elapsed > 0 else 0

            print(f"\n✓ 高清数据收集完成!")
            print(f"  总帧数: {collected}")
            print(f"  持续时间: {elapsed:.1f}秒")
            print(f"  平均帧率: {fps:.1f} FPS")
            print(f"  分辨率: 1920x1080")

            # 保存统计信息
            self.save_statistics(collected, elapsed, fps)

        except KeyboardInterrupt:
            collected = self.frame_count - initial_count
            print(f"\n数据收集中断，已收集 {collected} 帧高清图像")

    def save_statistics(self, frames, duration, fps):
        """保存统计信息"""
        stats = {
            'total_frames': frames,
            'duration_seconds': duration,
            'average_fps': fps,
            'resolution': '1920x1080',
            'scenario': self.args.scenario,
            'town': self.args.town,
            'weather': self.args.weather,
            'time_of_day': self.args.time_of_day,
            'timestamp': datetime.now().isoformat(),
            'notes': '高清版本，关闭了运动模糊和后期效果'
        }

        stats_file = os.path.join(self.output_dir, 'hd_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        # 也保存一份图像质量说明
        quality_note = """
图像质量设置说明：
1. 分辨率: 1920x1080 (全高清)
2. 关闭了所有运动模糊效果
3. 关闭了景深、光晕等后期效果
4. 优化了曝光和光照设置
5. 使用Raw格式保存，无压缩损失

如果图像仍然模糊，请检查：
1. CARLA图形设置是否为最高质量
2. 显卡驱动是否更新
3. 是否有足够的显存
"""

        note_file = os.path.join(self.output_dir, 'image_quality_note.txt')
        with open(note_file, 'w') as f:
            f.write(quality_note)

        print("✓ 统计信息和质量说明已保存")

    def cleanup(self):
        """清理场景"""
        print("\n清理场景...")

        # 先停止所有传感器
        for sensor in self.sensors:
            try:
                sensor.stop()
            except:
                pass

        # 然后销毁所有actor
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
# 4. 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='CVIPS 高清图像数据生成器')

    # 基本参数
    parser.add_argument('--scenario', type=str, default='hd_test',
                        help='场景名称')
    parser.add_argument('--town', type=str, default='Town10HD',
                        help='CARLA地图（推荐Town10HD细节更丰富）')

    # 环境参数（推荐使用白天晴天获取最清晰图像）
    parser.add_argument('--weather', type=str, default='clear',
                        choices=['clear', 'rainy', 'cloudy'],
                        help='天气条件（推荐clear）')
    parser.add_argument('--time-of-day', type=str, default='noon',
                        choices=['noon', 'sunset', 'night'],
                        help='时间（推荐noon）')

    # 交通参数
    parser.add_argument('--num-vehicles', type=int, default=3,
                        help='交通车辆数（较少减少遮挡）')
    parser.add_argument('--num-pedestrians', type=int, default=5,
                        help='行人数（白天才有效）')

    # 收集参数
    parser.add_argument('--duration', type=int, default=30,
                        help='收集时间(秒)')

    args = parser.parse_args()

    # 创建高清生成器
    generator = CVIPSHDGenerator(args)

    try:
        # 1. 连接
        if not generator.connect_with_retry():
            return

        # 2. 设置高清场景
        ego_vehicle = generator.setup_scene()

        # 3. 设置高清摄像头
        if ego_vehicle:
            generator.setup_hd_cameras(ego_vehicle)

        # 4. 收集高清数据
        generator.collect_data()

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n运行出错: {e}")
        traceback.print_exc()
    finally:
        # 5. 清理
        generator.cleanup()

        print("\n" + "=" * 80)
        print(f"高清数据保存到: {generator.output_dir}")
        print("建议: 打开输出目录查看第一张图像是否清晰")
        print("=" * 80)


if __name__ == "__main__":
    main()