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