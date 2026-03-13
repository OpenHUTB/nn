# -*- coding: utf-8 -*-
"""
Project: CARLA AutoVision Navigator
Module: Perception - Sensor Management
Version: v1.0.0
Description: 传感器管理模块。实现 RGB 摄像头的挂载、数据监听以及原始图像流向 OpenCV 格式的预处理。
Author: wangadsa
License: MIT License
"""
import carla
import numpy as np
import cv2
import sys
import os

# 导入配置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class SensorManager:
    """
    传感器管理类：负责摄像头的挂载、数据监听与图像预处理
    """

    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.camera = None
        self.image_data = None  # 存储最新的图像帧

    def attach_camera(self):
        """挂载 RGB 摄像头到主车"""
        # 1. 获取摄像头蓝图
        bp_library = self.world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')

        # 2. 根据 config 设置分辨率和视场角
        camera_bp.set_attribute('image_size_x', str(config.CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(config.CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(config.CAMERA_FOV))

        # 3. 设置安装位置（主车前盖上方）
        spawn_point = carla.Transform(carla.Location(x=1.5, z=2.4))

        # 4. 生成并挂载到主车上
        self.camera = self.world.spawn_actor(camera_bp, spawn_point, attach_to=self.vehicle)

        # 5. 设置数据监听回调函数
        self.camera.listen(lambda image: self._on_image_received(image))
        print(f"传感器状态: RGB 摄像头已挂载并开始监听。")

    def _on_image_received(self, image):
        """回调函数：将 CARLA 原始图像转换为 OpenCV 格式"""
        # 将原始数据转换为 RGBA 格式的 numpy 数组
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))

        # 取前三个通道并转为 BGR (OpenCV 格式)
        rgb_image = array[:, :, :3]
        self.image_data = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    def get_current_frame(self):
        """获取当前最新的图像帧"""
        return self.image_data

    def cleanup(self):
        """销毁传感器"""
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
            print("传感器状态: 摄像头已断开连接并销毁。")