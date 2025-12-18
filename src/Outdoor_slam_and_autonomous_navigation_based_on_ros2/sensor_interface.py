# sensor_interface.py
import numpy as np
import time
import threading
from queue import Queue
import json
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PointCloud:
    points: np.ndarray  # (N, 3) or (N, 4) for intensity
    timestamp: float
    frame_id: str = "laser"


@dataclass
class IMUData:
    accel: np.ndarray  # (3,) m/s²
    gyro: np.ndarray  # (3,) rad/s
    timestamp: float


@dataclass
class GNSSData:
    lat: float
    lon: float
    alt: float
    cov: np.ndarray  # (3, 3) covariance
    timestamp: float


class SensorInterface:
    """多传感器数据采集与同步接口"""

    def __init__(self, config_path: str = "config/sensors.json"):
        self.load_config(config_path)
        self.pointcloud_queue = Queue(maxsize=10)
        self.imu_queue = Queue(maxsize=50)
        self.gnss_queue = Queue(maxsize=5)
        self.odom_queue = Queue(maxsize=10)
        self.running = False
        self.sync_tolerance = 0.01  # 10ms同步容差

    def load_config(self, config_path: str):
        """加载传感器配置"""
        default_config = {
            "lidar": {
                "type": "velodyne",
                "port": 2368,
                "max_range": 100.0,
                "min_range": 0.5
            },
            "imu": {
                "type": "xsens",
                "port": "/dev/ttyUSB0",
                "baudrate": 115200
            },
            "gnss": {
                "type": "ublox",
                "port": "/dev/ttyACM0",
                "baudrate": 9600,
                "use_rtk": True
            }
        }
        self.config = default_config

    def start_capture(self):
        """启动所有传感器"""
        self.running = True
        self.lidar_thread = threading.Thread(target=self._lidar_capture)
        self.imu_thread = threading.Thread(target=self._imu_capture)
        self.gnss_thread = threading.Thread(target=self._gnss_capture)

        self.lidar_thread.start()
        self.imu_thread.start()
        self.gnss_thread.start()

        print("所有传感器已启动")

    def stop_capture(self):
        """停止传感器采集"""
        self.running = False
        if hasattr(self, 'lidar_thread'):
            self.lidar_thread.join()
        if hasattr(self, 'imu_thread'):
            self.imu_thread.join()
        if hasattr(self, 'gnss_thread'):
            self.gnss_thread.join()

    def _lidar_capture(self):
        """模拟激光雷达数据采集"""
        from sensor_simulators import LidarSimulator

        sim = LidarSimulator()
        while self.running:
            points = sim.generate_pointcloud()
            pc = PointCloud(
                points=points,
                timestamp=time.time()
            )
            if not self.pointcloud_queue.full():
                self.pointcloud_queue.put(pc)
            time.sleep(0.1)  # 10Hz

    def _imu_capture(self):
        """模拟IMU数据采集"""
        from sensor_simulators import IMUSimulator

        sim = IMUSimulator()
        while self.running:
            imu_data = IMUData(
                accel=sim.get_acceleration(),
                gyro=sim.get_gyro(),
                timestamp=time.time()
            )
            if not self.imu_queue.full():
                self.imu_queue.put(imu_data)
            time.sleep(0.01)  # 100Hz

    def _gnss_capture(self):
        """模拟GNSS数据采集"""
        from sensor_simulators import GNSSTrueSimulator

        sim = GNSSTrueSimulator()
        while self.running:
            gnss_data = GNSSData(
                lat=sim.get_latitude(),
                lon=sim.get_longitude(),
                alt=sim.get_altitude(),
                cov=np.eye(3) * 0.1,  # 模拟协方差
                timestamp=time.time()
            )
            if not self.gnss_queue.full():
                self.gnss_queue.put(gnss_data)
            time.sleep(0.2)  # 5Hz

    def get_synchronized_data(self, timeout=1.0) -> Tuple[Optional[PointCloud],
    Optional[IMUData],
    Optional[GNSSData]]:
        """获取时间同步的传感器数据"""
        try:
            # 获取最新的点云
            pc_data = None
            if not self.pointcloud_queue.empty():
                pc_data = self.pointcloud_queue.get(timeout=timeout)

            # 获取同步的IMU数据
            imu_data = None
            target_time = pc_data.timestamp if pc_data else time.time()

            # 从队列中寻找时间最接近的IMU数据
            closest_imu = None
            min_diff = float('inf')

            temp_imu_list = []
            while not self.imu_queue.empty():
                imu = self.imu_queue.get(timeout=0.01)
                temp_imu_list.append(imu)

            for imu in temp_imu_list:
                time_diff = abs(imu.timestamp - target_time)
                if time_diff < min_diff and time_diff < self.sync_tolerance:
                    min_diff = time_diff
                    closest_imu = imu

            # 放回未使用的数据
            for imu in temp_imu_list:
                if imu is not closest_imu and not self.imu_queue.full():
                    self.imu_queue.put(imu)

            imu_data = closest_imu

            # 获取同步的GNSS数据
            gnss_data = None
            if not self.gnss_queue.empty():
                gnss_data = self.gnss_queue.get(timeout=0.1)

            return pc_data, imu_data, gnss_data

        except Exception as e:
            print(f"数据同步失败: {e}")
            return None, None, None