#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
abandoned_park_controller.py - AbandonedPark 模拟器专用无人机控制器

该模块提供了与 AirSim 模拟器（AbandonedPark 场景）交互的完整控制功能，
包括连接、模式切换、状态监控、飞行控制、图像捕获、自动探索等。

主要特性：
- 稳健的连接与重试机制
- 状态自动更新与电池模拟
- 多线程图像捕获（队列缓冲）
- 丰富的运动控制接口
- 可扩展的自动探索逻辑
"""

import airsim
import time
import numpy as np
import cv2
import threading
import json
import os
import sys
import logging
import random
from queue import Queue, Empty
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List, Callable


class AbandonedParkController:
    """AbandonedPark 模拟器专用无人机控制器类"""

    # 默认配置常量
    DEFAULT_IP = "127.0.0.1"
    DEFAULT_PORT = 41451
    CONNECTION_TIMEOUT = 10          # 连接超时（秒）
    BATTERY_DRAIN_RATE = 0.01        # 飞行时每秒电池消耗（%）
    IMAGE_QUEUE_SIZE = 10             # 图像队列最大长度
    STATUS_UPDATE_INTERVAL = 0.5      # 状态更新最小间隔（秒）

    def __init__(self, ip: str = DEFAULT_IP, port: int = DEFAULT_PORT, auto_connect: bool = True):
        """
        初始化控制器

        :param ip: 模拟器 IP 地址
        :param port: 模拟器端口
        :param auto_connect: 是否自动连接
        """
        self.ip = ip
        self.port = port
        self.client: Optional[airsim.MultirotorClient] = None
        self.is_connected = False
        self.is_drone_mode = False
        self.is_flying = False

        # 状态缓存（减少对模拟器的频繁调用）
        self._state_cache: Optional[airsim.MultirotorState] = None
        self._last_state_update = 0.0
        self.battery_level = 100.0          # 模拟电池电量
        self.position: Optional[airsim.Vector3r] = None
        self.velocity: Optional[airsim.Vector3r] = None
        self.altitude = 0.0

        # 图像捕获线程相关
        self.image_queue: Queue = Queue(maxsize=self.IMAGE_QUEUE_SIZE)
        self.image_thread: Optional[threading.Thread] = None
        self.capture_running = False
        self._capture_lock = threading.Lock()

        # 配置日志
        self._setup_logging()

        if auto_connect:
            self.connect()

        print(f"AbandonedPark 控制器初始化完成，目标: {ip}:{port}")

    # ----------------------------------------------------------------------
    # 私有辅助方法
    # ----------------------------------------------------------------------
    def _setup_logging(self) -> None:
        """配置日志记录器"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _log_info(self, msg: str) -> None:
        """输出信息日志（同时打印到控制台）"""
        self.logger.info(msg)
        print(msg)

    def _log_warning(self, msg: str) -> None:
        """输出警告日志"""
        self.logger.warning(msg)
        print(f"⚠️ {msg}")

    def _log_error(self, msg: str) -> None:
        """输出错误日志"""
        self.logger.error(msg)
        print(f"✗ {msg}")

    def _log_success(self, msg: str) -> None:
        """输出成功信息"""
        print(f"✓ {msg}")

    # ----------------------------------------------------------------------
    # 连接与初始化
    # ----------------------------------------------------------------------
    def connect(self, timeout: float = CONNECTION_TIMEOUT, max_retries: int = 3) -> bool:
        """
        连接到 AirSim 模拟器

        :param timeout: 单次连接超时（秒）
        :param max_retries: 最大重试次数
        :return: 是否连接成功
        """
        self._log_info(f"正在连接到 AbandonedPark 模拟器 {self.ip}:{self.port}...")
        for attempt in range(1, max_retries + 1):
            try:
                self.client = airsim.MultirotorClient(ip=self.ip, port=self.port)
                # 使用 ping 测试连接
                ping_time = self.client.ping()
                if ping_time > 0:
                    self._log_success(f"连接成功！响应时间: {ping_time} ms (尝试 {attempt})")
                    self.is_connected = True
                    break
            except Exception as e:
                self._log_error(f"连接尝试 {attempt} 失败: {e}")
                if attempt < max_retries:
                    time.sleep(2)

        if not self.is_connected:
            self._log_error(f"无法连接到模拟器，请检查模拟器是否运行。")
            return False

        # 确认完整连接
        try:
            self.client.confirmConnection()
            self._log_success("连接已确认")
        except Exception as e:
            self._log_error(f"连接确认失败: {e}")
            self.is_connected = False
            return False

        # 获取初始状态并检测无人机模式
        self.update_status()
        try:
            self.client.getMultirotorState()
            self.is_drone_mode = True
            self._log_success("无人机模式已检测")
        except Exception:
            self._log_warning("当前不是无人机模式，请切换模式")
            self.is_drone_mode = False

        return True

    def switch_to_drone_mode(self, interactive: bool = True) -> bool:
        """
        切换到无人机模式

        :param interactive: 是否在 API 失败时提示用户手动切换
        :return: 是否成功切换
        """
        if not self.is_connected:
            self._log_error("未连接，无法切换模式")
            return False

        self._log_info("尝试切换到无人机模式...")

        # 方法1：通过设置位姿尝试激活无人机模式（不一定有效，但无害）
        try:
            self.client.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)),
                True
            )
            time.sleep(1)
        except Exception:
            pass

        # 检查是否已切换成功
        try:
            self.client.getMultirotorState()
            self.is_drone_mode = True
            self._log_success("无人机模式已激活")
            return True
        except Exception as e:
            self._log_error(f"API 切换失败: {e}")

        if not interactive:
            return False

        # 提供手动切换指导
        print("\n请手动切换到无人机模式：")
        print("1. 切换到模拟器窗口")
        print("2. 按 `~` 键打开控制台")
        print("3. 输入命令: `Vehicle change Drone` 并回车")
        print("4. 等待几秒后按回车继续")

        input("切换完成后按回车继续...")

        # 再次检查
        try:
            self.client.getMultirotorState()
            self.is_drone_mode = True
            self._log_success("无人机模式已激活")
            return True
        except Exception:
            self._log_error("仍不是无人机模式")
            return False

    # ----------------------------------------------------------------------
    # 状态管理
    # ----------------------------------------------------------------------
    def update_status(self, force: bool = False) -> bool:
        """
        更新无人机状态（使用缓存避免频繁调用）

        :param force: 是否强制更新（忽略时间间隔）
        :return: 是否更新成功
        """
        if not self.is_connected or not self.is_drone_mode:
            return False

        now = time.time()
        if not force and (now - self._last_state_update) < self.STATUS_UPDATE_INTERVAL:
            return True  # 使用缓存

        try:
            state = self.client.getMultirotorState()
            self._state_cache = state
            self._last_state_update = now

            self.position = state.kinematics_estimated.position
            self.velocity = state.kinematics_estimated.linear_velocity
            self.altitude = -self.position.z_val  # AirSim Z 向下为负

            # 模拟电池消耗（仅当飞行时）
            if self.is_flying:
                self.battery_level -= self.BATTERY_DRAIN_RATE
                if self.battery_level < 0:
                    self.battery_level = 0

            return True
        except Exception as e:
            self._log_error(f"更新状态失败: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        获取当前完整状态字典

        :return: 包含连接、模式、位置、速度等信息的字典
        """
        self.update_status()
        return {
            "connected": self.is_connected,
            "drone_mode": self.is_drone_mode,
            "flying": self.is_flying,
            "battery": round(self.battery_level, 1),
            "altitude": round(self.altitude, 1),
            "position": {
                "x": round(self.position.x_val, 1) if self.position else 0,
                "y": round(self.position.y_val, 1) if self.position else 0,
                "z": round(self.position.z_val, 1) if self.position else 0,
            },
            "velocity": {
                "x": round(self.velocity.x_val, 1) if self.velocity else 0,
                "y": round(self.velocity.y_val, 1) if self.velocity else 0,
                "z": round(self.velocity.z_val, 1) if self.velocity else 0,
            },
        }

    def print_status(self) -> None:
        """打印当前状态（格式化输出）"""
        s = self.get_status()
        print("\n" + "=" * 50)
        print("无人机状态")
        print("=" * 50)
        print(f"连接状态: {'✓ 已连接' if s['connected'] else '✗ 未连接'}")
        print(f"模式: {'✓ 无人机模式' if s['drone_mode'] else '✗ 其他模式'}")
        print(f"飞行状态: {'✈️ 飞行中' if s['flying'] else '🛬 已降落'}")
        print(f"电池电量: {s['battery']:.1f}%")
        print(f"高度: {s['altitude']} m")
        print(f"位置: X={s['position']['x']}, Y={s['position']['y']}, Z={s['position']['z']}")
        print(f"速度: X={s['velocity']['x']}, Y={s['velocity']['y']}, Z={s['velocity']['z']}")
        print("=" * 50)

    # ----------------------------------------------------------------------
    # 飞行控制
    # ----------------------------------------------------------------------
    def takeoff(self, altitude: float = 10.0, timeout: float = 10) -> bool:
        """
        起飞到指定高度

        :param altitude: 目标高度（米）
        :param timeout: 起飞超时（秒）
        :return: 是否成功
        """
        if not self.is_connected:
            self._log_error("未连接，无法起飞")
            return False
        if not self.is_drone_mode:
            self._log_error("不是无人机模式，无法起飞")
            return False

        self._log_info(f"起飞到 {altitude} 米...")
        try:
            # 启用 API 控制并解锁
            self.client.enableApiControl(True)
            self.client.armDisarm(True)

            # 起飞
            self.client.takeoffAsync().join(timeout=timeout)
            time.sleep(2)

            # 上升到指定高度
            self.client.moveToZAsync(-altitude, 3).join(timeout=timeout)
            time.sleep(1)

            self.is_flying = True
            self._log_success(f"已起飞到 {altitude} 米")
            return True
        except Exception as e:
            self._log_error(f"起飞失败: {e}")
            return False

    def land(self) -> bool:
        """降落无人机"""
        if not self.is_flying:
            self._log_info("未在飞行中")
            return True

        self._log_info("开始降落...")
        try:
            self.client.landAsync().join()
            time.sleep(2)

            self.client.armDisarm(False)
            self.client.enableApiControl(False)

            self.is_flying = False
            self._log_success("已安全降落")
            return True
        except Exception as e:
            self._log_error(f"降落失败: {e}")
            return False

    def move_to(self, x: float, y: float, z: float, speed: float = 3.0) -> bool:
        """
        移动到指定世界坐标

        :param x: X 坐标
        :param y: Y 坐标
        :param z: Z 坐标（注意 AirSim Z 向下，负值越高）
        :param speed: 移动速度 (m/s)
        :return: 是否成功
        """
        if not self.is_flying:
            self._log_error("未起飞，无法移动")
            return False

        try:
            self._log_info(f"移动到 ({x:.1f}, {y:.1f}, {z:.1f}) 速度 {speed} m/s")
            self.client.moveToPositionAsync(x, y, z, speed).join()
            self._log_success("移动完成")
            return True
        except Exception as e:
            self._log_error(f"移动失败: {e}")
            return False

    def move_by_velocity(self, vx: float, vy: float, vz: float, duration: float = 1.0) -> bool:
        """
        以指定速度移动一段时间

        :param vx: X 方向速度 (m/s)
        :param vy: Y 方向速度 (m/s)
        :param vz: Z 方向速度 (m/s，正向下，负向上)
        :param duration: 持续时间（秒）
        :return: 是否成功
        """
        if not self.is_flying:
            self._log_error("未起飞，无法移动")
            return False

        try:
            self._log_info(f"以速度 ({vx:.1f}, {vy:.1f}, {vz:.1f}) 移动 {duration} 秒")
            self.client.moveByVelocityAsync(vx, vy, vz, duration).join()
            self._log_success("移动完成")
            return True
        except Exception as e:
            self._log_error(f"移动失败: {e}")
            return False

    def hover(self, duration: float = 5.0) -> bool:
        """
        悬停指定时间

        :param duration: 悬停时间（秒）
        :return: 是否成功
        """
        if not self.is_flying:
            self._log_error("未起飞，无法悬停")
            return False

        try:
            self._log_info(f"悬停 {duration} 秒...")
            self.client.hoverAsync().join()
            time.sleep(duration)
            self._log_success("悬停完成")
            return True
        except Exception as e:
            self._log_error(f"悬停失败: {e}")
            return False

    # ----------------------------------------------------------------------
    # 图像捕获（多线程）
    # ----------------------------------------------------------------------
    def start_image_capture(self, camera_id: int = 0, interval: float = 0.5) -> bool:
        """
        启动后台线程持续捕获图像

        :param camera_id: 摄像头 ID（默认为前置 0）
        :param interval: 捕获间隔（秒）
        :return: 是否成功启动
        """
        if not self.is_connected:
            self._log_error("未连接，无法捕获图像")
            return False

        with self._capture_lock:
            if self.capture_running:
                self._log_warning("图像捕获已在运行")
                return True

            self.capture_running = True

        def capture_loop():
            while self.capture_running:
                try:
                    responses = self.client.simGetImages([
                        airsim.ImageRequest(str(camera_id), airsim.ImageType.Scene, False, False)
                    ])
                    if responses:
                        response = responses[0]
                        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                        img_rgb = img1d.reshape(response.height, response.width, 3)

                        # 非阻塞添加至队列
                        if not self.image_queue.full():
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            self.image_queue.put_nowait({
                                'timestamp': timestamp,
                                'image': img_rgb,
                                'camera_id': camera_id
                            })
                except Exception as e:
                    self._log_error(f"图像捕获错误: {e}")
                time.sleep(interval)

        self.image_thread = threading.Thread(target=capture_loop, daemon=True)
        self.image_thread.start()
        self._log_success(f"图像捕获已启动 (间隔: {interval}秒)")
        return True

    def stop_image_capture(self) -> bool:
        """停止后台图像捕获"""
        with self._capture_lock:
            if not self.capture_running:
                return True
            self.capture_running = False

        if self.image_thread:
            self.image_thread.join(timeout=2)
        self._log_success("图像捕获已停止")
        return True

    def get_captured_image(self, block: bool = False, timeout: float = 1.0) -> Optional[Dict]:
        """
        从队列中获取一张捕获的图像

        :param block: 是否阻塞等待
        :param timeout: 阻塞超时（秒）
        :return: 图像数据字典或 None
        """
        try:
            if block:
                return self.image_queue.get(timeout=timeout)
            else:
                return self.image_queue.get_nowait()
        except Empty:
            return None

    def save_captured_images(self, output_dir: str = "captured_images") -> int:
        """
        将队列中所有图像保存到指定目录

        :param output_dir: 输出目录
        :return: 保存的图像数量
        """
        os.makedirs(output_dir, exist_ok=True)
        saved = 0
        while not self.image_queue.empty():
            try:
                data = self.image_queue.get_nowait()
                filename = os.path.join(output_dir, f"{data['timestamp']}_camera{data['camera_id']}.jpg")
                # 转换为 BGR 保存（OpenCV 默认）
                cv2.imwrite(filename, cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR))
                saved += 1
            except Empty:
                break
            except Exception as e:
                self._log_error(f"保存图像失败: {e}")
        self._log_success(f"已保存 {saved} 张图像到 {output_dir}/")
        return saved

    # ----------------------------------------------------------------------
    # 自动探索功能
    # ----------------------------------------------------------------------
    def explore_park(self, exploration_time: float = 60, check_battery: bool = True) -> bool:
        """
        自动探索公园（随机选择动作，同时捕获图像）

        :param exploration_time: 探索持续时间（秒）
        :param check_battery: 是否检查电池电量（低于 20% 时停止）
        :return: 是否成功完成
        """
        if not self.is_flying:
            self._log_error("请先起飞")
            return False

        self._log_info(f"开始自动探索公园 ({exploration_time} 秒)...")

        # 定义可执行动作库
        actions: List[Tuple[str, Callable[[], bool]]] = [
            ("向前移动", lambda: self.move_by_velocity(3, 0, 0, 3)),
            ("向右移动", lambda: self.move_by_velocity(0, 3, 0, 3)),
            ("向左移动", lambda: self.move_by_velocity(0, -3, 0, 3)),
            ("上升", lambda: self.move_by_velocity(0, 0, -2, 2)),
            ("下降", lambda: self.move_by_velocity(0, 0, 1, 2)),
            ("悬停观察", lambda: self.hover(2)),
        ]

        start_time = time.time()
        action_count = 0

        # 启动图像捕获
        self.start_image_capture(interval=0.3)

        try:
            while time.time() - start_time < exploration_time:
                if check_battery and self.battery_level < 20:
                    self._log_warning("电池电量低，停止探索")
                    break

                # 随机选择一个动作
                name, func = random.choice(actions)
                self._log_info(f"执行动作: {name}")
                if func():
                    action_count += 1

                self.update_status()
                time.sleep(1)  # 动作间短暂停顿
        finally:
            # 无论成功与否，停止捕获并保存图像
            self.stop_image_capture()
            self.save_captured_images()

        self._log_info(f"探索完成！执行了 {action_count} 个动作")
        return True

    # ----------------------------------------------------------------------
    # 清理与断开
    # ----------------------------------------------------------------------
    def disconnect(self) -> None:
        """断开与模拟器的连接，并清理资源"""
        self._log_info("断开连接...")
        if self.is_flying:
            self.land()
        self.stop_image_capture()
        self.is_connected = False
        self.is_drone_mode = False
        self.is_flying = False
        self.client = None
        self._log_success("已断开连接")


# ----------------------------------------------------------------------
# 测试与演示脚本
# ----------------------------------------------------------------------
def test_controller():
    """交互式测试控制器功能"""
    print("AbandonedPark 控制器测试")
    print("=" * 50)

    controller = AbandonedParkController(auto_connect=True)
    if not controller.is_connected:
        print("无法连接，测试终止。")
        return

    try:
        # 主交互菜单
        while True:
            print("\n" + "-" * 40)
            print("主菜单：")
            print("1. 显示状态")
            print("2. 切换无人机模式")
            print("3. 起飞")
            print("4. 降落")
            print("5. 移动测试")
            print("6. 图像捕获测试")
            print("7. 自动探索")
            print("0. 退出")
            choice = input("请选择: ").strip()

            if choice == '1':
                controller.print_status()
            elif choice == '2':
                controller.switch_to_drone_mode(interactive=True)
            elif choice == '3':
                alt = input("请输入高度（米，默认10）: ").strip()
                alt = float(alt) if alt else 10.0
                controller.takeoff(alt)
            elif choice == '4':
                controller.land()
            elif choice == '5':
                if not controller.is_flying:
                    print("请先起飞！")
                    continue
                print("执行简单移动序列：前进2秒，右移2秒，悬停2秒")
                controller.move_by_velocity(2, 0, 0, 2)
                controller.move_by_velocity(0, 2, 0, 2)
                controller.hover(2)
            elif choice == '6':
                interval = input("捕获间隔（秒，默认0.5）: ").strip()
                interval = float(interval) if interval else 0.5
                duration = input("捕获持续时间（秒，默认5）: ").strip()
                duration = float(duration) if duration else 5.0

                controller.start_image_capture(interval=interval)
                print(f"正在捕获 {duration} 秒...")
                time.sleep(duration)
                controller.stop_image_capture()
                controller.save_captured_images()
            elif choice == '7':
                if not controller.is_flying:
                    print("请先起飞！")
                    continue
                t = input("探索时间（秒，默认30）: ").strip()
                t = float(t) if t else 30.0
                controller.explore_park(t)
            elif choice == '0':
                break
            else:
                print("无效输入，请重新选择。")
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        controller.disconnect()
        print("测试结束。")


if __name__ == "__main__":
    test_controller()