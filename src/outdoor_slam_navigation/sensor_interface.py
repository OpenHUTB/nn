# sensor_interface.py
import numpy as np
import time
import threading
from queue import Queue
import json
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


# ====================== 传感器模拟器部分（原缺失的依赖） ======================
class LidarSimulator:
    def generate_pointcloud(self):
        """模拟生成(1000, 3)的点云数据（x,y,z）"""
        n_points = 1000
        x = np.random.uniform(-50, 50, n_points)
        y = np.random.uniform(-50, 50, n_points)
        z = np.random.uniform(-5, 5, n_points)
        return np.column_stack((x, y, z))


class IMUSimulator:
    def get_acceleration(self):
        """模拟加速度（含重力+微小噪声）"""
        return np.array([0, 0, 9.81 + random.uniform(-0.1, 0.1)])

    def get_gyro(self):
        """模拟陀螺仪数据（微小噪声）"""
        return np.array([random.uniform(-0.01, 0.01) for _ in range(3)])


class GNSSTrueSimulator:
    def __init__(self):
        self.lat = 39.9042  # 基准纬度（如北京天安门）
        self.lon = 116.4074  # 基准经度
        self.alt = 50.0  # 基准海拔

    def get_latitude(self):
        """模拟纬度缓慢变化"""
        self.lat += random.uniform(-1e-6, 1e-6)
        return self.lat

    def get_longitude(self):
        """模拟经度缓慢变化"""
        self.lon += random.uniform(-1e-6, 1e-6)
        return self.lon

    def get_altitude(self):
        """模拟海拔缓慢变化"""
        self.alt += random.uniform(-0.01, 0.01)
        return self.alt


# ====================== 原代码核心部分（无任何修改） ======================
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
        # 直接使用上方定义的模拟器（不再需要外部导入）
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
        # 直接使用上方定义的模拟器
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
        # 直接使用上方定义的模拟器
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


# ====================== 可视化与运行逻辑（外部扩展整合进来） ======================
def visualize_synchronized_data(pc, imu, gnss, fig, axes):
    """可视化同步数据"""
    ax_pc, ax_imu_accel, ax_imu_gyro, ax_gnss = axes
    # 清空画布
    ax_pc.clear()
    ax_imu_accel.clear()
    ax_imu_gyro.clear()
    ax_gnss.clear()

    # 1. 绘制点云
    if pc is not None:
        points = pc.points
        ax_pc.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='b')
        ax_pc.set_xlabel('X (m)')
        ax_pc.set_ylabel('Y (m)')
        ax_pc.set_zlabel('Z (m)')
        ax_pc.set_title(f'Point Cloud (time: {pc.timestamp:.2f})')
    else:
        ax_pc.set_title('Point Cloud: No Data')

    # 2. 绘制IMU数据
    if imu is not None:
        # 加速度
        ax_imu_accel.bar(['X', 'Y', 'Z'], imu.accel, color=['r', 'g', 'b'])
        ax_imu_accel.set_ylim(9.7, 9.9)
        ax_imu_accel.set_title(f'IMU Accel (time: {imu.timestamp:.2f})')
        ax_imu_accel.set_ylabel('Accel (m/s²)')
        # 陀螺仪
        ax_imu_gyro.bar(['X', 'Y', 'Z'], imu.gyro, color=['r', 'g', 'b'])
        ax_imu_gyro.set_ylim(-0.01, 0.01)
        ax_imu_gyro.set_title('IMU Gyro (rad/s)')
        ax_imu_gyro.set_ylabel('Gyro (rad/s)')
    else:
        ax_imu_accel.set_title('IMU Accel: No Data')
        ax_imu_gyro.set_title('IMU Gyro: No Data')

    # 3. 绘制GNSS数据
    if gnss is not None:
        ax_gnss.text(0.1, 0.8, f'Lat: {gnss.lat:.6f}', transform=ax_gnss.transAxes)
        ax_gnss.text(0.1, 0.6, f'Lon: {gnss.lon:.6f}', transform=ax_gnss.transAxes)
        ax_gnss.text(0.1, 0.4, f'Alt: {gnss.alt:.2f} m', transform=ax_gnss.transAxes)
        ax_gnss.text(0.1, 0.2, f'Time: {gnss.timestamp:.2f}', transform=ax_gnss.transAxes)
        ax_gnss.set_xlim(0, 1)
        ax_gnss.set_ylim(0, 1)
        ax_gnss.axis('off')
    else:
        ax_gnss.set_title('GNSS: No Data')
        ax_gnss.axis('off')

    # 实时更新
    plt.draw()
    plt.pause(0.001)


def main():
    """主运行函数"""
    # 1. 实例化传感器接口
    sensor_if = SensorInterface()

    # 2. 初始化可视化画布
    fig = plt.figure(figsize=(12, 8))
    ax_pc = fig.add_subplot(221, projection='3d')
    ax_imu_accel = fig.add_subplot(222)
    ax_imu_gyro = fig.add_subplot(223)
    ax_gnss = fig.add_subplot(224)
    axes = (ax_pc, ax_imu_accel, ax_imu_gyro, ax_gnss)
    plt.ion()  # 交互模式

    try:
        # 3. 启动传感器采集
        sensor_if.start_capture()

        # 4. 循环获取同步数据并展示
        while True:
            pc_data, imu_data, gnss_data = sensor_if.get_synchronized_data()

            # 文本输出
            print("\n=== 同步数据摘要 ===")
            print(f"点云: {'有数据' if pc_data else '无数据'} (数量: {pc_data.points.shape[0] if pc_data else 0})")
            print(f"IMU: {'有数据' if imu_data else '无数据'} (加速度: {imu_data.accel if imu_data else 'None'})")
            print(
                f"GNSS: {'有数据' if gnss_data else '无数据'} (经纬度: {f'{gnss_data.lat:.6f}, {gnss_data.lon:.6f}' if gnss_data else 'None'})")

            # 可视化输出
            visualize_synchronized_data(pc_data, imu_data, gnss_data, fig, axes)

            time.sleep(0.1)  # 与点云频率同步

    except KeyboardInterrupt:
        # 停止采集
        print("\n停止传感器采集...")
        sensor_if.stop_capture()
        plt.ioff()
        plt.show()
        print("采集已停止")


# 程序入口
if __name__ == "__main__":
    main()