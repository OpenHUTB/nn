# abandoned_park_controller.py - AbandonedPark模拟器专用控制器
import airsim
import time
import numpy as np
import cv2
import threading
from queue import Queue
import json
from datetime import datetime


class AbandonedParkController:
    """AbandonedPark模拟器专用控制器"""

    def __init__(self, ip="127.0.0.1", port=41451):
        """
        初始化控制器
        Args:
            ip: 模拟器IP地址（默认为本地）
            port: 端口（AirSim默认41451）
        """
        self.ip = ip
        self.port = port
        self.client = None
        self.is_connected = False
        self.is_drone_mode = False
        self.is_flying = False

        # 状态信息
        self.battery_level = 100.0
        self.position = None
        self.velocity = None
        self.altitude = 0.0

        # 图像捕获队列
        self.image_queue = Queue(maxsize=10)
        self.image_thread = None
        self.capture_running = False

        # 日志
        self.setup_logging()

        print(f"AbandonedPark控制器初始化完成")
        print(f"目标: {ip}:{port}")

    def setup_logging(self):
        """设置日志"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def connect(self, timeout=10):
        """
        连接到AbandonedPark模拟器
        Args:
            timeout: 连接超时时间（秒）
        Returns:
            bool: 连接是否成功
        """
        print(f"正在连接到AbandonedPark模拟器 {self.ip}:{self.port}...")

        try:
            # 创建客户端
            self.client = airsim.MultirotorClient(ip=self.ip, port=self.port)

            # 测试连接
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    ping_time = self.client.ping()
                    print(f"✓ 连接成功！响应时间: {ping_time} ms")
                    self.is_connected = True
                    break
                except:
                    print(".", end="", flush=True)
                    time.sleep(1)

            if not self.is_connected:
                print(f"\n✗ 连接超时 ({timeout}秒)")
                return False

            # 确认连接
            self.client.confirmConnection()
            print("✓ 连接已确认")

            # 获取初始状态
            self.update_status()

            # 检查是否为无人机模式
            try:
                state = self.client.getMultirotorState()
                self.is_drone_mode = True
                print("✓ 无人机模式已检测")
            except:
                print("⚠️ 当前不是无人机模式，请切换模式")
                self.is_drone_mode = False

            return True

        except Exception as e:
            print(f"✗ 连接失败: {e}")
            return False

    def switch_to_drone_mode(self):
        """切换到无人机模式"""
        if not self.is_connected:
            print("未连接，无法切换模式")
            return False

        print("尝试切换到无人机模式...")

        try:
            # 方法1: 通过API切换
            self.client.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)),
                True
            )

            # 方法2: 检查当前模式
            time.sleep(1)
            state = self.client.getMultirotorState()
            self.is_drone_mode = True
            print("✓ 已切换到无人机模式")
            return True

        except Exception as e:
            print(f"切换失败: {e}")
            print("\n请手动切换:")
            print("1. 切换到模拟器窗口")
            print("2. 按 ~ 键打开控制台")
            print("3. 输入: Vehicle change Drone")
            print("4. 按回车确认")

            response = input("切换完成后按回车继续...")

            # 再次检查
            try:
                state = self.client.getMultirotorState()
                self.is_drone_mode = True
                print("✓ 无人机模式已激活")
                return True
            except:
                print("✗ 仍不是无人机模式")
                return False

    def update_status(self):
        """更新状态信息"""
        if not self.is_connected or not self.is_drone_mode:
            return

        try:
            state = self.client.getMultirotorState()

            self.position = state.kinematics_estimated.position
            self.velocity = state.kinematics_estimated.linear_velocity
            self.altitude = -self.position.z_val  # Z轴向下为负

            # 模拟电池消耗
            if self.is_flying:
                self.battery_level -= 0.01  # 每秒消耗0.01%
                if self.battery_level < 0:
                    self.battery_level = 0

            return True
        except Exception as e:
            print(f"更新状态失败: {e}")
            return False

    def get_status(self):
        """获取当前状态"""
        self.update_status()

        status = {
            "connected": self.is_connected,
            "drone_mode": self.is_drone_mode,
            "flying": self.is_flying,
            "battery": self.battery_level,
            "altitude": self.altitude if self.altitude else 0,
            "position": {
                "x": self.position.x_val if self.position else 0,
                "y": self.position.y_val if self.position else 0,
                "z": self.position.z_val if self.position else 0
            },
            "velocity": {
                "x": self.velocity.x_val if self.velocity else 0,
                "y": self.velocity.y_val if self.velocity else 0,
                "z": self.velocity.z_val if self.velocity else 0
            }
        }

        return status

    def print_status(self):
        """打印当前状态"""
        status = self.get_status()

        print("\n" + "=" * 50)
        print("无人机状态")
        print("=" * 50)
        print(f"连接状态: {'✓ 已连接' if status['connected'] else '✗ 未连接'}")
        print(f"模式: {'✓ 无人机模式' if status['drone_mode'] else '✗ 其他模式'}")
        print(f"飞行状态: {'✈️ 飞行中' if status['flying'] else '🛬 已降落'}")
        print(f"电池电量: {status['battery']:.1f}%")
        print(f"高度: {status['altitude']:.1f} m")
        print(
            f"位置: X={status['position']['x']:.1f}, Y={status['position']['y']:.1f}, Z={status['position']['z']:.1f}")
        print(
            f"速度: X={status['velocity']['x']:.1f}, Y={status['velocity']['y']:.1f}, Z={status['velocity']['z']:.1f}")
        print("=" * 50)

    def takeoff(self, altitude=10.0):
        """起飞到指定高度"""
        if not self.is_connected:
            print("未连接，无法起飞")
            return False

        if not self.is_drone_mode:
            print("不是无人机模式，无法起飞")
            return False

        try:
            print(f"起飞到 {altitude} 米高度...")

            # 解锁无人机
            self.client.enableApiControl(True)
            self.client.armDisarm(True)

            # 起飞
            self.client.takeoffAsync().join()
            time.sleep(2)

            # 上升到指定高度
            self.client.moveToZAsync(-altitude, 3).join()

            self.is_flying = True
            print(f"✓ 已起飞到 {altitude} 米高度")
            return True

        except Exception as e:
            print(f"✗ 起飞失败: {e}")
            return False

    def land(self):
        """降落"""
        if not self.is_flying:
            print("未在飞行中")
            return True

        try:
            print("开始降落...")

            self.client.landAsync().join()
            time.sleep(2)

            self.client.armDisarm(False)
            self.client.enableApiControl(False)

            self.is_flying = False
            print("✓ 已安全降落")
            return True

        except Exception as e:
            print(f"✗ 降落失败: {e}")
            return False

    def move_to(self, x, y, z, speed=3.0):
        """移动到指定位置"""
        if not self.is_flying:
            print("未起飞，无法移动")
            return False

        try:
            print(f"移动到位置: ({x}, {y}, {z})")
            self.client.moveToPositionAsync(x, y, z, speed).join()
            print("✓ 移动完成")
            return True
        except Exception as e:
            print(f"✗ 移动失败: {e}")
            return False

    def move_by_velocity(self, vx, vy, vz, duration=1.0):
        """以指定速度移动"""
        if not self.is_flying:
            print("未起飞，无法移动")
            return False

        try:
            print(f"以速度 ({vx}, {vy}, {vz}) 移动 {duration} 秒")
            self.client.moveByVelocityAsync(vx, vy, vz, duration).join()
            print("✓ 移动完成")
            return True
        except Exception as e:
            print(f"✗ 移动失败: {e}")
            return False

    def hover(self, duration=5.0):
        """悬停指定时间"""
        if not self.is_flying:
            print("未起飞，无法悬停")
            return False

        try:
            print(f"悬停 {duration} 秒...")
            self.client.hoverAsync().join()
            time.sleep(duration)
            print("✓ 悬停完成")
            return True
        except Exception as e:
            print(f"✗ 悬停失败: {e}")
            return False

    def start_image_capture(self, camera_id=0, interval=0.5):
        """开始图像捕获"""
        if not self.is_connected:
            print("未连接，无法捕获图像")
            return False

        if self.capture_running:
            print("图像捕获已在运行")
            return True

        self.capture_running = True

        def capture_loop():
            while self.capture_running:
                try:
                    # 捕获图像
                    responses = self.client.simGetImages([
                        airsim.ImageRequest(
                            str(camera_id),
                            airsim.ImageType.Scene,
                            False, False
                        )
                    ])

                    if responses:
                        response = responses[0]
                        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                        img_rgb = img1d.reshape(response.height, response.width, 3)

                        # 添加到队列
                        if not self.image_queue.full():
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            self.image_queue.put({
                                'timestamp': timestamp,
                                'image': img_rgb,
                                'camera_id': camera_id
                            })

                except Exception as e:
                    print(f"图像捕获错误: {e}")

                time.sleep(interval)

        # 启动捕获线程
        self.image_thread = threading.Thread(target=capture_loop, daemon=True)
        self.image_thread.start()

        print(f"✓ 图像捕获已启动 (间隔: {interval}秒)")
        return True

    def stop_image_capture(self):
        """停止图像捕获"""
        if not self.capture_running:
            return True

        self.capture_running = False

        if self.image_thread:
            self.image_thread.join(timeout=2)

        print("✓ 图像捕获已停止")
        return True

    def get_captured_image(self):
        """获取捕获的图像"""
        if self.image_queue.empty():
            return None

        return self.image_queue.get()

    def save_captured_images(self, output_dir="captured_images"):
        """保存所有捕获的图像"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        saved_count = 0

        while not self.image_queue.empty():
            try:
                data = self.image_queue.get()
                filename = f"{output_dir}/{data['timestamp']}_camera{data['camera_id']}.jpg"
                cv2.imwrite(filename, data['image'])
                saved_count += 1
            except Exception as e:
                print(f"保存图像失败: {e}")

        print(f"✓ 已保存 {saved_count} 张图像到 {output_dir}/")
        return saved_count

    def explore_park(self, exploration_time=60):
        """自动探索公园"""
        if not self.is_flying:
            print("请先起飞")
            return False

        print(f"开始自动探索公园 ({exploration_time}秒)...")

        # 定义探索路径
        exploration_actions = [
            ("向前移动", lambda: self.move_by_velocity(3, 0, 0, 3)),
            ("向右移动", lambda: self.move_by_velocity(0, 3, 0, 3)),
            ("向左移动", lambda: self.move_by_velocity(0, -3, 0, 3)),
            ("上升", lambda: self.move_by_velocity(0, 0, -2, 2)),
            ("下降", lambda: self.move_by_velocity(0, 0, 1, 2)),
            ("悬停观察", lambda: self.hover(2))
        ]

        import random

        start_time = time.time()
        action_count = 0

        # 开始图像捕获
        self.start_image_capture(interval=0.3)

        while time.time() - start_time < exploration_time:
            if self.battery_level < 20:
                print("⚠️ 电池电量低，停止探索")
                break

            # 随机选择一个动作
            action_name, action_func = random.choice(exploration_actions)
            print(f"执行动作: {action_name}")

            # 执行动作
            if action_func():
                action_count += 1

            # 更新状态
            self.update_status()
            time.sleep(1)

        # 停止图像捕获
        self.stop_image_capture()

        # 保存图像
        self.save_captured_images()

        print(f"探索完成！执行了 {action_count} 个动作")
        return True

    def disconnect(self):
        """断开连接"""
        print("断开连接...")

        # 如果正在飞行，先降落
        if self.is_flying:
            self.land()

        # 停止图像捕获
        self.stop_image_capture()

        # 重置状态
        self.is_connected = False
        self.is_drone_mode = False
        self.is_flying = False

        print("✓ 已断开连接")


# 测试函数
def test_controller():
    """测试控制器"""
    print("AbandonedPark控制器测试")
    print("=" * 50)

    # 创建控制器
    controller = AbandonedParkController()

    try:
        # 连接模拟器
        if not controller.connect():
            print("连接失败，请检查模拟器是否运行")
            return

        # 检查状态
        controller.print_status()

        # 如果不在无人机模式，尝试切换
        if not controller.is_drone_mode:
            print("\n需要切换到无人机模式...")
            if not controller.switch_to_drone_mode():
                print("无法切换到无人机模式，测试终止")
                return

        # 再次检查状态
        controller.print_status()

        # 起飞
        response = input("\n是否起飞？(y/n): ")
        if response.lower() == 'y':
            if controller.takeoff(altitude=10):
                controller.print_status()

                # 简单移动测试
                response = input("\n是否进行移动测试？(y/n): ")
                if response.lower() == 'y':
                    controller.move_by_velocity(2, 0, 0, 2)
                    controller.hover(2)
                    controller.move_by_velocity(0, 2, 0, 2)
                    controller.hover(2)

                # 图像捕获测试
                response = input("\n是否进行图像捕获测试？(y/n): ")
                if response.lower() == 'y':
                    controller.start_image_capture(interval=0.5)
                    time.sleep(5)  # 捕获5秒
                    controller.stop_image_capture()
                    controller.save_captured_images()

                # 自动探索
                response = input("\n是否进行自动探索？(y/n): ")
                if response.lower() == 'y':
                    exploration_time = int(input("探索时间（秒）: ") or "30")
                    controller.explore_park(exploration_time)

                # 降落
                response = input("\n是否降落？(y/n): ")
                if response.lower() == 'y':
                    controller.land()
                    controller.print_status()

        print("\n测试完成！")

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 断开连接
        controller.disconnect()


if __name__ == "__main__":
    test_controller()
