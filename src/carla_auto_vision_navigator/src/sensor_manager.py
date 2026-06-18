<<<<<<< HEAD
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
=======
import carla
import cv2
import numpy as np
import open3d as o3d
from config import SENSOR_CONFIG


class SensorManager:
    def __init__(self, carla_client, vehicle):
        self.client = carla_client
        self.vehicle = vehicle
        self.world = carla_client.world
        self.sensors = {}
        self.sensor_data = {
            "rgb": None,
            "lidar": None,
            "radar": None,
            "imu": None
        }

    def _rgb_callback(self, data):
        """RGB相机回调函数"""
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = array.reshape((data.height, data.width, 4))  # RGBA
        array = array[:, :, :3]  # 去掉Alpha通道
        self.sensor_data["rgb"] = array

    def _lidar_callback(self, data):
        """LiDAR回调函数"""
        # 解析点云数据
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        points = points.reshape((-1, 4))[:, :3]  # x,y,z

        # 转换为Open3D格式（可选）
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        self.sensor_data["lidar"] = pcd

    def _radar_callback(self, data):
        """雷达回调函数"""
        # 解析雷达数据
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        points = points.reshape((-1, 4))  # x,y,z,velocity
        self.sensor_data["radar"] = points

    def _imu_callback(self, data):
        """IMU回调函数"""
        self.sensor_data["imu"] = {
            "accelerometer": data.accelerometer,
            "gyroscope": data.gyroscope,
            "compass": data.compass
        }

    def setup_sensors(self):
        """初始化所有多模态传感器"""
        blueprint_library = self.world.get_blueprint_library()

        # 1. RGB相机
        rgb_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_bp.set_attribute("fov", str(SENSOR_CONFIG["rgb_camera"]["fov"]))
        rgb_bp.set_attribute("image_size_x", str(SENSOR_CONFIG["rgb_camera"]["width"]))
        rgb_bp.set_attribute("image_size_y", str(SENSOR_CONFIG["rgb_camera"]["height"]))
        rgb_transform = carla.Transform(
            carla.Location(*SENSOR_CONFIG["rgb_camera"]["position"])
        )
        rgb_sensor = self.world.spawn_actor(rgb_bp, rgb_transform, attach_to=self.vehicle)
        rgb_sensor.listen(self._rgb_callback)
        self.sensors["rgb"] = rgb_sensor

        # 2. LiDAR
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(SENSOR_CONFIG["lidar"]["channels"]))
        lidar_bp.set_attribute("range", str(SENSOR_CONFIG["lidar"]["range"]))
        lidar_bp.set_attribute("points_per_second", str(SENSOR_CONFIG["lidar"]["points_per_second"]))
        lidar_transform = carla.Transform(
            carla.Location(*SENSOR_CONFIG["lidar"]["position"])
        )
        lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        lidar_sensor.listen(self._lidar_callback)
        self.sensors["lidar"] = lidar_sensor

        # 3. 雷达
        radar_bp = blueprint_library.find("sensor.other.radar")
        radar_bp.set_attribute("range", str(SENSOR_CONFIG["radar"]["range"]))
        radar_bp.set_attribute("points_per_second", str(SENSOR_CONFIG["radar"]["points_per_second"]))
        radar_transform = carla.Transform(
            carla.Location(*SENSOR_CONFIG["radar"]["position"])
        )
        radar_sensor = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.vehicle)
        radar_sensor.listen(self._radar_callback)
        self.sensors["radar"] = radar_sensor

        # 4. IMU
        imu_bp = blueprint_library.find("sensor.other.imu")
        imu_bp.set_attribute("frequency", str(SENSOR_CONFIG["imu"]["frequency"]))
        imu_transform = carla.Transform(
            carla.Location(*SENSOR_CONFIG["imu"]["position"])
        )
        imu_sensor = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle)
        imu_sensor.listen(self._imu_callback)
        self.sensors["imu"] = imu_sensor

    def get_sensor_data(self):
        """获取最新传感器数据"""
        return self.sensor_data

    def visualize_data(self):
        """可视化多模态数据（调试用）"""
        # 可视化RGB图像
        if self.sensor_data["rgb"] is not None:
            cv2.imshow("RGB Camera", self.sensor_data["rgb"])
            cv2.waitKey(1)

        # 可视化LiDAR点云（可选）
        if self.sensor_data["lidar"] is not None:
            o3d.visualization.draw_geometries([self.sensor_data["lidar"]], window_name="LiDAR Point Cloud")

    def clean_up(self):
        """销毁所有传感器"""
        for sensor in self.sensors.values():
            sensor.stop()
            sensor.destroy()
        self.sensors = {}
        cv2.destroyAllWindows()
>>>>>>> upstream/main
