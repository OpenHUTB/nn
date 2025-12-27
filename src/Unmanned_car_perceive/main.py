#!/usr/bin/env python3

import carla
import numpy as np
import time
import random
import sys
import os

# 添加CARLA PythonAPI路径
try:
    # 尝试自动查找CARLA路径
    possible_paths = [
        "D:/CARLA_0.9.10/WindowsNoEditor/PythonAPI/carla",
        "D:/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla",
        "D:/CARLA_0.9.12/WindowsNoEditor/PythonAPI/carla",
        "D:/CARLA_0.9.13/WindowsNoEditor/PythonAPI/carla",
        "D:/CARLA_0.9.14/WindowsNoEditor/PythonAPI/carla",
        "C:/CARLA_0.9.10/WindowsNoEditor/PythonAPI/carla",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            if path not in sys.path:
                sys.path.append(path)
                print(f"✅ 添加CARLA路径: {path}")
            break
    else:
        print("❌ 未找到CARLA路径，请手动设置")
        sys.exit(1)

except Exception as e:
    print(f"⚠️  路径设置警告: {e}")

from drawer import PyGameDrawer
from sync_pygame import SyncPyGame
# 导入障碍物检测器
from obstacle_detector import ObstacleDetector


class Main():
    def __init__(self):
        # 配置参数
        self.CARLA_SERVER = "localhost"
        self.PORT = 2000
        self.VEHICLE_MODEL = "model3"
        self.LIDAR_RANGE = 50

        print("=" * 50)
        print("🚗 CARLA 自动驾驶模拟器")
        print("=" * 50)

        try:
            # 连接Carla服务器
            print("🔄 连接到Carla服务器...")
            self.client = carla.Client(self.CARLA_SERVER, self.PORT)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
            print(f"✅ 已连接，当前地图: {self.map.name}")

            # 初始化Pygame
            print("🎮 初始化Pygame界面...")
            self.game = SyncPyGame(self)

            # 生成主车辆
            print("🚘 生成自动驾驶车辆...")
            self.spawn_vehicle()

            # 添加障碍物检测器
            self.obstacle_detector = ObstacleDetector()

            # 用于存储最新的激光雷达数据
            self.latest_lidar_data = None

            # 安装传感器
            self.setup_lidar()
            self.setup_camera()

            # 初始化绘制器
            self.drawer = PyGameDrawer(self)

            # 开始游戏循环
            print("▶️ 启动自动驾驶...")
            print("📊 车辆速度和位置将显示在屏幕上")
            print("🚧 障碍物检测系统已启用")
            print("ℹ️  按ESC键退出程序")
            print("=" * 50)

            self.game.game_loop(self.world, self.on_tick)

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup()

    def spawn_vehicle(self):
        """生成车辆"""
        try:
            # 获取所有生成点
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                raise Exception("地图中没有可用的生成点")

            # 选择一个生成点
            spawn_point = random.choice(spawn_points)

            # 获取车辆蓝图
            blueprint_lib = self.world.get_blueprint_library()
            vehicle_bp = blueprint_lib.filter(self.VEHICLE_MODEL)

            if not vehicle_bp:
                # 如果指定车型不存在，使用第一个可用的
                vehicle_bp = blueprint_lib.filter("vehicle.*")[0]
                print(f"⚠️  车辆 '{self.VEHICLE_MODEL}' 不存在，使用默认车辆")
            else:
                vehicle_bp = vehicle_bp[0]

            # 生成车辆
            self.ego = self.world.try_spawn_actor(vehicle_bp, spawn_point)

            if not self.ego:
                # 如果生成失败，尝试其他位置
                for point in spawn_points:
                    self.ego = self.world.try_spawn_actor(vehicle_bp, point)
                    if self.ego:
                        spawn_point = point
                        break

            if not self.ego:
                raise Exception("无法生成车辆，请检查地图和生成点")

            print(f"✅ 车辆已生成在位置: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")

            # 启用自动驾驶
            self.ego.set_autopilot(True)
            print("🚦 自动驾驶已启用")

        except Exception as e:
            print(f"❌ 生成车辆失败: {e}")
            raise

    def setup_lidar(self):
        """安装激光雷达传感器"""
        try:
            lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
            lidar_bp.set_attribute("range", str(self.LIDAR_RANGE))
            lidar_bp.set_attribute("points_per_second", "50000")
            lidar_bp.set_attribute("rotation_frequency", "10")
            lidar_bp.set_attribute("channels", "32")

            lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.4))
            self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego)
            # 修改监听函数，存储激光雷达数据
            self.lidar.listen(lambda data: self.process_lidar(data))
            print("✅ 激光雷达已安装")
        except Exception as e:
            print(f"⚠️  安装激光雷达失败: {e}")

    def setup_camera(self):
        """安装摄像头传感器"""
        try:
            camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", "800")
            camera_bp.set_attribute("image_size_y", "600")
            camera_bp.set_attribute("fov", "110")

            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego)
            self.camera.listen(lambda image: self.process_camera(image))
            print("✅ 摄像头已安装")
        except Exception as e:
            print(f"⚠️  安装摄像头失败: {e}")

    def process_lidar(self, data):
        """处理激光雷达数据并存储"""
        try:
            point_cloud = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))

            # 存储最新的激光雷达数据用于障碍物检测
            self.latest_lidar_data = point_cloud

            # 减少控制台输出频率，避免过于频繁
            if random.random() < 0.01:  # 1%的概率输出
                print(f"📡 激光雷达点云: {len(point_cloud)} 个点")

        except Exception as e:
            pass

    def process_camera(self, image):
        """处理摄像头数据"""
        try:
            # 将CARLA图像转换为numpy数组
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))

            # 转换格式：BGRA → RGB，并且调整方向
            # CARLA默认是BGRA，Pygame需要RGB
            array = array[:, :, :3]  # 去掉Alpha通道
            array = array[:, :, ::-1]  # BGR → RGB

            # 将图像数据传递给绘制器
            if hasattr(self, 'drawer'):
                self.drawer.camera_image = array

        except Exception as e:
            print(f"❌ 处理摄像头数据失败: {e}")

    def on_tick(self):
        """每一帧调用的主函数"""
        try:
            # 🆕 帧率计算
            if not hasattr(self, 'frame_count'):
                self.frame_count = 0
                self.last_time = time.time()
                self.fps = 0

            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:  # 每秒钟更新一次
                self.fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time

            # 获取车辆状态
            if hasattr(self, 'ego') and self.ego:
                location = self.ego.get_location()
                velocity = self.ego.get_velocity()

                # 计算速度 (m/s 转换为 km/h)
                speed_m_s = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                speed_kmh = speed_m_s * 3.6

                # 障碍物检测
                if self.latest_lidar_data is not None:
                    obstacles = self.obstacle_detector.detect(self.latest_lidar_data)

                    # 定期输出检测结果（避免控制台太拥挤）
                    if random.random() < 0.05:  # 5%概率输出
                        if self.obstacle_detector.warning_level > 0:
                            print(f"🚧 {self.obstacle_detector.warning_message}")

                # 更新绘制器显示
                self.drawer.display_speed(speed_kmh)
                self.drawer.display_location(location)

                # 显示障碍物警告信息
                self.drawer.display_warning(
                    self.obstacle_detector.warning_message,
                    self.obstacle_detector.get_warning_color(),
                    self.obstacle_detector.warning_level
                )

                # 🆕 显示摄像头图像
                self.drawer.display_camera()

                # 🆕 显示帧率
                self.drawer.display_fps(self.fps)

                # 🆕 新增：显示摄像头图像
                self.drawer.display_camera()

                # 更新观察者视角跟随车辆
                self.update_spectator()

        except Exception as e:
            print(f"⚠️  更新车辆状态失败: {e}")

    def update_spectator(self):
        """更新观察者视角"""
        try:
            spectator = self.world.get_spectator()
            transform = self.ego.get_transform()

            # 计算观察者位置（车辆后方10米，上方5米）
            location = transform.location
            rotation = transform.rotation

            x = location.x - 10 * np.cos(np.radians(rotation.yaw))
            y = location.y - 10 * np.sin(np.radians(rotation.yaw))
            z = location.z + 5

            spectator.set_transform(carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(pitch=-20, yaw=rotation.yaw)
            ))
        except Exception as e:
            pass

    def cleanup(self):
        """清理资源"""
        print("\n🧹 开始清理资源...")

        # 销毁传感器
        if hasattr(self, 'camera') and self.camera:
            try:
                self.camera.destroy()
                print("✅ 摄像头已销毁")
            except:
                pass

        if hasattr(self, 'lidar') and self.lidar:
            try:
                self.lidar.stop()
                self.lidar.destroy()
                print("✅ 激光雷达已销毁")
            except:
                pass

        # 销毁车辆
        if hasattr(self, 'ego') and self.ego:
            try:
                self.ego.destroy()
                print("✅ 车辆已销毁")
            except:
                pass

        print("🧹 资源清理完成！")


if __name__ == '__main__':
    try:
        Main()
    except KeyboardInterrupt:
        print("\n🛑 程序被用户停止")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback

        traceback.print_exc()