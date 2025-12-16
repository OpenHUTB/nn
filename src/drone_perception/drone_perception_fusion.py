"""
drone_perception_fusion.py
无人机感知模块 - 多传感器融合版本
包含视觉、LiDAR和IMU传感器融合
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import time
import os
import json
from dataclasses import dataclass
from enum import Enum
import math
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.spatial import KDTree
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("注意: scipy库未安装，某些功能将使用简化版本")

class SensorType(Enum):
    """传感器类型枚举"""
    CAMERA = "camera"
    LIDAR = "lidar"
    IMU = "imu"
    GPS = "gps"
    SONAR = "sonar"

@dataclass
class SensorData:
    """传感器数据基类"""
    timestamp: float
    sensor_type: SensorType
    data: Dict[str, Any]

@dataclass
class CameraData(SensorData):
    """相机数据"""
    frame: np.ndarray
    camera_matrix: Optional[np.ndarray] = None
    distortion_coeffs: Optional[np.ndarray] = None

@dataclass
class LidarData(SensorData):
    """LiDAR数据"""
    points: np.ndarray  # (N, 3) 点云数据 [x, y, z]
    intensities: Optional[np.ndarray] = None  # (N,) 点云强度
    fov_horizontal: float = 360.0  # 水平视场角(度)
    fov_vertical: float = 30.0  # 垂直视场角(度)
    max_range: float = 100.0  # 最大测量距离(米)

@dataclass
class IMUData(SensorData):
    """IMU数据"""
    acceleration: np.ndarray  # 加速度 (m/s^2) [ax, ay, az]
    angular_velocity: np.ndarray  # 角速度 (rad/s) [wx, wy, wz]
    orientation: Optional[np.ndarray] = None  # 姿态四元数 [qw, qx, qy, qz]
    magnetic_field: Optional[np.ndarray] = None  # 磁场强度 (μT) [mx, my, mz]

@dataclass
class GPSData(SensorData):
    """GPS数据"""
    latitude: float  # 纬度
    longitude: float  # 经度
    altitude: float  # 海拔高度(米)
    velocity: Optional[np.ndarray] = None  # 速度 (m/s) [vx, vy, vz]
    accuracy: float = 1.0  # 位置精度(米)

@dataclass
class FusedObject:
    """融合后的物体"""
    id: int
    position: np.ndarray  # 3D位置 [x, y, z] (米)
    velocity: np.ndarray  # 3D速度 [vx, vy, vz] (m/s)
    size: np.ndarray  # 尺寸 [长, 宽, 高] (米)
    confidence: float  # 置信度 0-1
    object_type: str  # 物体类型
    timestamp: float  # 时间戳
    sensor_sources: List[SensorType]  # 数据来源的传感器
    bbox_2d: Optional[np.ndarray] = None  # 2D边界框 [x1, y1, x2, y2]
    bbox_3d: Optional[np.ndarray] = None  # 3D边界框 [8, 3]

@dataclass
class FusedMap:
    """融合后的环境地图"""
    timestamp: float
    occupancy_grid: Optional[np.ndarray] = None  # 占据栅格地图
    elevation_map: Optional[np.ndarray] = None  # 高程地图
    semantic_map: Optional[np.ndarray] = None  # 语义地图
    objects: List[FusedObject] = None  # 地图中的物体

    def __post_init__(self):
        if self.objects is None:
            self.objects = []

class SensorCalibration:
    """传感器标定类"""

    def __init__(self):
        # 传感器之间的外参变换矩阵
        self.transformations = {}

    def set_transformation(self, from_sensor: SensorType, to_sensor: SensorType,
                          rotation: np.ndarray, translation: np.ndarray):
        """设置传感器之间的变换关系"""
        key = f"{from_sensor.value}_to_{to_sensor.value}"
        self.transformations[key] = {
            'R': rotation.astype(np.float64),  # 3x3 旋转矩阵
            't': translation.astype(np.float64),  # 3x1 平移向量
            'T': np.eye(4, dtype=np.float64)  # 4x4 齐次变换矩阵
        }
        self.transformations[key]['T'][:3, :3] = rotation.astype(np.float64)
        self.transformations[key]['T'][:3, 3] = translation.flatten().astype(np.float64)

    def transform_point(self, point: np.ndarray, from_sensor: SensorType,
                       to_sensor: SensorType) -> np.ndarray:
        """将点从一个传感器坐标系变换到另一个传感器坐标系"""
        if from_sensor == to_sensor:
            return point.astype(np.float64)

        key = f"{from_sensor.value}_to_{to_sensor.value}"
        if key not in self.transformations:
            return point.astype(np.float64)

        T = self.transformations[key]['T']
        point_homogeneous = np.ones(4, dtype=np.float64)
        point_homogeneous[:3] = point.astype(np.float64)
        transformed = T @ point_homogeneous
        return transformed[:3]

    def transform_points(self, points: np.ndarray, from_sensor: SensorType,
                        to_sensor: SensorType) -> np.ndarray:
        """批量变换点云"""
        if from_sensor == to_sensor:
            return points.astype(np.float64)

        key = f"{from_sensor.value}_to_{to_sensor.value}"
        if key not in self.transformations:
            return points.astype(np.float64)

        T = self.transformations[key]['T']
        n = points.shape[0]
        points_homogeneous = np.ones((n, 4), dtype=np.float64)
        points_homogeneous[:, :3] = points.astype(np.float64)
        transformed = (T @ points_homogeneous.T).T
        return transformed[:, :3]

class VirtualLidar:
    """虚拟LiDAR传感器"""

    def __init__(self, num_beams: int = 360, max_range: float = 50.0,
                 fov_horizontal: float = 360.0, fov_vertical: float = 30.0):
        self.num_beams = num_beams
        self.max_range = max_range
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
        self.noise_std = 0.02  # 噪声标准差

        # 生成光束方向
        self.beam_angles_h = np.linspace(-fov_horizontal/2, fov_horizontal/2, num_beams)
        self.beam_angles_v = np.linspace(-fov_vertical/2, fov_vertical/2, num_beams//12)

    def simulate_scan(self, position: np.ndarray, orientation: np.ndarray,
                     environment_objects: List[Dict]) -> LidarData:
        """模拟LiDAR扫描"""
        points = []
        intensities = []

        # 确保position和orientation是浮点数
        position = position.astype(np.float64)

        # 简化模拟：在环境中随机生成点云
        for obj in environment_objects:
            obj_pos = np.array(obj['position'], dtype=np.float64)
            obj_size = np.array(obj['size'], dtype=np.float64)

            # 生成物体表面的点
            n_points_per_obj = np.random.randint(10, 50)
            for _ in range(n_points_per_obj):
                # 在物体边界框内随机生成点
                offset = (np.random.rand(3).astype(np.float64) - 0.5) * obj_size
                point = obj_pos + offset

                # 计算到传感器的距离
                distance = np.linalg.norm(point - position)

                if distance <= self.max_range:
                    # 添加噪声
                    noise = np.random.randn(3).astype(np.float64) * self.noise_std
                    point_noisy = point + noise

                    points.append(point_noisy.astype(np.float64))
                    # 强度基于距离和材料（简化）
                    intensity = max(0.1, 1.0 - distance/self.max_range)
                    intensities.append(float(intensity))

        # 如果没有检测到物体，生成一些随机点（模拟地面和天空）
        if len(points) == 0:
            n_random_points = 100
            for _ in range(n_random_points):
                # 在传感器前方生成随机点
                angle_h = np.random.uniform(-self.fov_horizontal/2, self.fov_horizontal/2)
                angle_v = np.random.uniform(-self.fov_vertical/2, self.fov_vertical/2)
                distance = np.random.uniform(1.0, self.max_range)

                # 球坐标转笛卡尔坐标
                x = distance * np.cos(np.radians(angle_v)) * np.cos(np.radians(angle_h))
                y = distance * np.cos(np.radians(angle_v)) * np.sin(np.radians(angle_h))
                z = distance * np.sin(np.radians(angle_v))

                points.append([x, y, z])
                intensities.append(0.3)

        # 确保数据类型正确
        if len(points) > 0:
            points_array = np.array(points, dtype=np.float64)
            intensities_array = np.array(intensities, dtype=np.float64)
        else:
            points_array = np.zeros((0, 3), dtype=np.float64)
            intensities_array = np.zeros(0, dtype=np.float64)

        return LidarData(
            timestamp=time.time(),
            sensor_type=SensorType.LIDAR,
            data={'position': position, 'orientation': orientation},
            points=points_array,
            intensities=intensities_array,
            fov_horizontal=self.fov_horizontal,
            fov_vertical=self.fov_vertical,
            max_range=self.max_range
        )

class VirtualIMU:
    """虚拟IMU传感器"""

    def __init__(self):
        self.gravity = np.array([0, 0, -9.81], dtype=np.float64)  # 重力加速度
        self.noise_accel = 0.05  # 加速度计噪声
        self.noise_gyro = 0.01  # 陀螺仪噪声

    def simulate_measurement(self, true_acceleration: np.ndarray,
                           true_angular_velocity: np.ndarray,
                           true_orientation: Optional[np.ndarray] = None) -> IMUData:
        """模拟IMU测量"""
        # 确保输入是浮点数
        true_acceleration = true_acceleration.astype(np.float64)
        true_angular_velocity = true_angular_velocity.astype(np.float64)

        # 添加重力到加速度
        accel_with_gravity = true_acceleration - self.gravity

        # 添加噪声
        accel_noise = np.random.randn(3).astype(np.float64) * self.noise_accel
        gyro_noise = np.random.randn(3).astype(np.float64) * self.noise_gyro

        measured_accel = accel_with_gravity + accel_noise
        measured_gyro = true_angular_velocity + gyro_noise

        if true_orientation is not None:
            true_orientation = true_orientation.astype(np.float64)

        return IMUData(
            timestamp=time.time(),
            sensor_type=SensorType.IMU,
            data={},
            acceleration=measured_accel,
            angular_velocity=measured_gyro,
            orientation=true_orientation
        )

class SensorFusion:
    """多传感器融合核心类"""

    def __init__(self, calibration: SensorCalibration):
        self.calibration = calibration
        self.fused_objects = {}  # 融合后的物体字典 {id: FusedObject}
        self.next_object_id = 1

        # 滤波参数
        self.kalman_filters = {}  # 卡尔曼滤波器字典

        # 时间同步
        self.sensor_timestamps = {}

        # 融合权重
        self.sensor_weights = {
            SensorType.CAMERA: 0.4,
            SensorType.LIDAR: 0.5,
            SensorType.IMU: 0.1
        }

    def time_sync(self, sensor_data: Dict[SensorType, SensorData]) -> Dict[SensorType, SensorData]:
        """时间同步 - 将不同传感器数据对齐到同一时间戳"""
        # 简化处理：只返回原始数据
        return sensor_data

    def associate_detections(self, camera_objects: List[Dict],
                            lidar_clusters: List[Dict]) -> List[Dict]:
        """关联不同传感器的检测结果"""
        associations = []

        if not camera_objects or not lidar_clusters:
            return associations

        # 简单的距离关联
        for cam_obj in camera_objects:
            if 'position_3d' not in cam_obj:
                continue

            cam_pos = np.array(cam_obj.get('position_3d', np.zeros(3)), dtype=np.float64)

            best_match = None
            best_distance = float('inf')

            for lidar_cluster in lidar_clusters:
                lidar_pos = np.array(lidar_cluster.get('centroid', np.zeros(3)), dtype=np.float64)

                # 计算3D距离
                distance = np.linalg.norm(cam_pos - lidar_pos)

                # 找到距离最近的匹配
                if distance < 2.0 and distance < best_distance:  # 2米阈值
                    best_distance = distance
                    best_match = lidar_cluster

            if best_match is not None:
                association = {
                    'camera_object': cam_obj,
                    'lidar_cluster': best_match,
                    'distance': best_distance,
                    'confidence': max(cam_obj.get('confidence', 0.5),
                                     best_match.get('confidence', 0.5))
                }
                associations.append(association)

        return associations

    def fuse_object_data(self, associations: List[Dict]) -> List[FusedObject]:
        """融合物体数据"""
        fused_objects = []

        for assoc in associations:
            cam_obj = assoc['camera_object']
            lidar_cluster = assoc['lidar_cluster']

            # 提取位置信息
            cam_pos = np.array(cam_obj.get('position_3d', np.zeros(3)), dtype=np.float64)
            lidar_pos = np.array(lidar_cluster.get('centroid', np.zeros(3)), dtype=np.float64)

            # 加权融合位置
            weight_cam = self.sensor_weights[SensorType.CAMERA]
            weight_lidar = self.sensor_weights[SensorType.LIDAR]

            total_weight = weight_cam + weight_lidar
            fused_pos = (cam_pos * weight_cam + lidar_pos * weight_lidar) / total_weight

            # 提取尺寸信息
            cam_size = np.array(cam_obj.get('size_3d', np.ones(3)), dtype=np.float64)
            lidar_size = np.array(lidar_cluster.get('size', np.ones(3)), dtype=np.float64)

            # 使用LiDAR尺寸（通常更准确）
            fused_size = lidar_size if np.any(lidar_size > 0) else cam_size

            # 提取类型信息
            cam_type = cam_obj.get('class_name', 'unknown')
            fused_type = cam_type  # 优先使用视觉分类

            # 创建融合物体
            fused_object = FusedObject(
                id=self.next_object_id,
                position=fused_pos,
                velocity=np.zeros(3, dtype=np.float64),  # 简化处理
                size=fused_size,
                confidence=float(assoc['confidence']),
                object_type=fused_type,
                timestamp=time.time(),
                sensor_sources=[SensorType.CAMERA, SensorType.LIDAR],
                bbox_2d=cam_obj.get('bbox'),
                bbox_3d=cam_obj.get('bbox_3d')
            )

            fused_objects.append(fused_object)
            self.next_object_id += 1

        return fused_objects

    def update_kalman_filter(self, object_id: int, new_measurement: np.ndarray):
        """更新卡尔曼滤波器"""
        if object_id not in self.kalman_filters:
            # 初始化卡尔曼滤波器
            self.kalman_filters[object_id] = self._init_kalman_filter()

        # 简化处理：直接使用测量值
        pass

    def _init_kalman_filter(self):
        """初始化卡尔曼滤波器"""
        return {}

    def create_occupancy_grid(self, lidar_data: LidarData, grid_size: Tuple[int, int] = (100, 100),
                             grid_resolution: float = 0.5) -> np.ndarray:
        """创建占据栅格地图"""
        width, height = grid_size
        grid = np.zeros((height, width), dtype=np.float32)

        if lidar_data.points.size == 0:
            return grid

        # 将点云投影到2D平面 (x, y)
        points_2d = lidar_data.points[:, :2]

        # 转换到栅格坐标
        grid_center = np.array([width//2, height//2], dtype=np.float32)
        grid_coords = (points_2d / grid_resolution + grid_center).astype(np.int32)

        # 过滤在栅格范围内的点
        valid_x = (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < width)
        valid_y = (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < height)
        valid = valid_x & valid_y

        if np.any(valid):
            grid_coords_valid = grid_coords[valid]

            # 设置占据概率
            for coord in grid_coords_valid:
                x, y = coord
                if 0 <= y < height and 0 <= x < width:
                    grid[y, x] = 1.0

            # 高斯平滑
            if SCIPY_AVAILABLE:
                grid = gaussian_filter(grid, sigma=1.0)
            else:
                # 简化版本：使用OpenCV的模糊
                grid = cv2.GaussianBlur(grid, (5, 5), 1.0)

        return grid

    def fuse_all_sensors(self, sensor_data: Dict[SensorType, SensorData],
                        camera_objects: List[Dict]) -> FusedMap:
        """融合所有传感器数据"""
        # 时间同步
        synced_data = self.time_sync(sensor_data)

        # 提取LiDAR数据并聚类
        lidar_clusters = []
        if SensorType.LIDAR in synced_data:
            lidar_data = synced_data[SensorType.LIDAR]
            lidar_clusters = self.cluster_lidar_points(lidar_data)

        # 关联检测结果
        associations = self.associate_detections(camera_objects, lidar_clusters)

        # 融合物体数据
        fused_objects = self.fuse_object_data(associations)

        # 创建占据栅格地图
        occupancy_grid = None
        if SensorType.LIDAR in synced_data:
            occupancy_grid = self.create_occupancy_grid(synced_data[SensorType.LIDAR])

        # 创建融合地图
        fused_map = FusedMap(
            timestamp=time.time(),
            occupancy_grid=occupancy_grid,
            objects=fused_objects
        )

        return fused_map

    def cluster_lidar_points(self, lidar_data: LidarData,
                            distance_threshold: float = 0.5) -> List[Dict]:
        """对LiDAR点云进行聚类"""
        clusters = []

        if lidar_data.points.size == 0 or len(lidar_data.points) == 0:
            return clusters

        # 使用简单的欧氏距离聚类
        points = lidar_data.points
        n_points = len(points)

        # 如果点太多，进行降采样
        if n_points > 1000:
            indices = np.random.choice(n_points, 1000, replace=False)
            points = points[indices]
            n_points = 1000

        if n_points == 0:
            return clusters

        if SCIPY_AVAILABLE and n_points > 10:
            # 使用KD树进行快速最近邻搜索
            try:
                tree = KDTree(points)
                visited = np.zeros(n_points, dtype=bool)

                for i in range(n_points):
                    if not visited[i]:
                        # 找到所有距离小于阈值的点
                        indices = tree.query_ball_point(points[i], distance_threshold)

                        if len(indices) > 3:  # 至少需要3个点构成一个聚类
                            cluster_points = points[indices]

                            # 计算聚类属性
                            centroid = np.mean(cluster_points, axis=0)
                            bbox_min = np.min(cluster_points, axis=0)
                            bbox_max = np.max(cluster_points, axis=0)
                            size = bbox_max - bbox_min

                            # 计算置信度（基于点数量和紧凑度）
                            n_points_cluster = len(cluster_points)
                            if n_points_cluster > 0:
                                compactness = 1.0 / (np.sum(np.var(cluster_points, axis=0)) + 1e-6)
                                confidence = min(0.2 * n_points_cluster + 0.8 * compactness, 1.0)
                            else:
                                confidence = 0.5

                            cluster = {
                                'points': cluster_points,
                                'centroid': centroid,
                                'size': size,
                                'n_points': n_points_cluster,
                                'confidence': confidence
                            }
                            clusters.append(cluster)

                        visited[indices] = True
            except Exception as e:
                print(f"KDTree聚类失败: {e}")
                # 使用简化聚类方法
                clusters = self._simple_cluster(points, distance_threshold)
        else:
            # 使用简化聚类方法
            clusters = self._simple_cluster(points, distance_threshold)

        return clusters

    def _simple_cluster(self, points: np.ndarray, distance_threshold: float = 0.5) -> List[Dict]:
        """简化聚类方法"""
        clusters = []
        n_points = len(points)

        if n_points == 0:
            return clusters

        # 使用网格聚类
        grid_resolution = distance_threshold
        grid_dict = {}

        for i, point in enumerate(points):
            # 计算网格索引
            grid_idx = tuple((point[:2] / grid_resolution).astype(int))

            if grid_idx not in grid_dict:
                grid_dict[grid_idx] = []
            grid_dict[grid_idx].append(i)

        # 合并相邻网格
        for grid_idx, indices in grid_dict.items():
            if len(indices) > 2:
                cluster_points = points[indices]

                centroid = np.mean(cluster_points, axis=0)
                bbox_min = np.min(cluster_points, axis=0)
                bbox_max = np.max(cluster_points, axis=0)
                size = bbox_max - bbox_min

                n_points_cluster = len(cluster_points)
                confidence = min(0.1 * n_points_cluster, 1.0)

                cluster = {
                    'points': cluster_points,
                    'centroid': centroid,
                    'size': size,
                    'n_points': n_points_cluster,
                    'confidence': confidence
                }
                clusters.append(cluster)

        return clusters

class MultiSensorPerception:
    """多传感器感知主类"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # 初始化传感器标定
        self.calibration = SensorCalibration()
        self._setup_calibration()

        # 初始化传感器融合
        self.sensor_fusion = SensorFusion(self.calibration)

        # 初始化虚拟传感器
        self.virtual_lidar = VirtualLidar()
        self.virtual_imu = VirtualIMU()

        # 视觉感知模块
        self.visual_perception = VisualPerception()

        # 数据缓冲区
        self.sensor_buffer = {}
        self.fusion_history = []

        # 状态变量
        self.current_position = np.array([0.0, 0.0, 5.0], dtype=np.float64)  # 默认高度5米
        self.current_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.current_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # 四元数

        print("多传感器感知模块初始化完成")

    def _setup_calibration(self):
        """设置传感器标定参数"""
        # 相机到LiDAR的变换（假设LiDAR在相机上方0.1米处）
        R_cam_to_lidar = np.eye(3, dtype=np.float64)  # 旋转相同
        t_cam_to_lidar = np.array([0.0, 0.0, 0.1], dtype=np.float64)  # 平移

        self.calibration.set_transformation(
            SensorType.CAMERA, SensorType.LIDAR,
            R_cam_to_lidar, t_cam_to_lidar
        )

        # LiDAR到IMU的变换（假设在同一位置）
        R_lidar_to_imu = np.eye(3, dtype=np.float64)
        t_lidar_to_imu = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.calibration.set_transformation(
            SensorType.LIDAR, SensorType.IMU,
            R_lidar_to_imu, t_lidar_to_imu
        )

    def update_sensor_data(self, sensor_type: SensorType, data: SensorData):
        """更新传感器数据"""
        self.sensor_buffer[sensor_type] = data

    def process_all_sensors(self, camera_frame: np.ndarray) -> Dict:
        """处理所有传感器数据"""
        results = {
            'timestamp': time.time(),
            'sensor_data': {},
            'fused_map': None,
            'objects': [],
            'safety_score': 1.0,
            'visual': None
        }

        try:
            # 1. 视觉感知
            visual_results = self.visual_perception.process_frame(camera_frame)
            results['visual'] = visual_results

            camera_data = CameraData(
                timestamp=time.time(),
                sensor_type=SensorType.CAMERA,
                data={'frame_shape': camera_frame.shape},
                frame=camera_frame
            )
            self.update_sensor_data(SensorType.CAMERA, camera_data)

            # 2. 模拟LiDAR数据（基于视觉检测结果）
            environment_objects = self._create_environment_from_visual(visual_results)
            lidar_data = self.virtual_lidar.simulate_scan(
                self.current_position, self.current_orientation, environment_objects
            )
            self.update_sensor_data(SensorType.LIDAR, lidar_data)

            # 3. 模拟IMU数据
            imu_data = self.virtual_imu.simulate_measurement(
                np.array([0.0, 0.0, 0.0], dtype=np.float64),  # 假设加速度为0
                np.array([0.0, 0.0, 0.0], dtype=np.float64),  # 假设角速度为0
                self.current_orientation
            )
            self.update_sensor_data(SensorType.IMU, imu_data)

            # 4. 传感器融合
            camera_objects_3d = self._estimate_3d_from_2d(visual_results['targets'])
            fused_map = self.sensor_fusion.fuse_all_sensors(
                self.sensor_buffer, camera_objects_3d
            )
            results['fused_map'] = fused_map
            results['objects'] = fused_map.objects

            # 5. 计算安全分数
            results['safety_score'] = self._calculate_safety_score(
                fused_map, visual_results
            )

            # 保存到历史
            self.fusion_history.append({
                'timestamp': time.time(),
                'fused_map': fused_map,
                'position': self.current_position.copy(),
                'orientation': self.current_orientation.copy()
            })

            # 限制历史长度
            if len(self.fusion_history) > 100:
                self.fusion_history = self.fusion_history[-100:]

        except Exception as e:
            results['error'] = str(e)
            print(f"传感器融合出错: {e}")
            import traceback
            traceback.print_exc()

        return results

    def _create_environment_from_visual(self, visual_results: Dict) -> List[Dict]:
        """从视觉结果创建环境对象（用于模拟LiDAR）"""
        environment = []

        # 添加检测到的目标
        for target in visual_results.get('targets', []):
            if 'bbox' in target:
                # 简化：假设目标在地面上，高度为1米
                bbox = target['bbox']
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0

                # 将2D位置映射到3D（简化）
                position = np.array([
                    (center_x - 320) * 0.01,  # 假设每像素0.01米
                    (center_y - 240) * 0.01,
                    1.0  # 高度1米
                ], dtype=np.float64)

                # 加上当前无人机位置
                position += self.current_position

                obj = {
                    'position': position,
                    'size': np.array([0.5, 0.5, 1.0], dtype=np.float64),  # 假设尺寸
                    'type': target.get('class_name', 'unknown')
                }
                environment.append(obj)

        # 添加地面
        ground_pos = self.current_position.copy()
        ground_pos[2] = 0.0  # 地面高度为0
        ground = {
            'position': ground_pos,
            'size': np.array([20.0, 20.0, 0.1], dtype=np.float64),  # 地面平面
            'type': 'ground'
        }
        environment.append(ground)

        return environment

    def _estimate_3d_from_2d(self, targets_2d: List[Dict]) -> List[Dict]:
        """从2D检测估计3D信息"""
        objects_3d = []

        for target in targets_2d:
            if 'bbox' not in target:
                continue

            bbox = target['bbox']

            # 计算2D边界框中心
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0

            # 计算2D边界框大小
            width_2d = float(bbox[2] - bbox[0])
            height_2d = float(bbox[3] - bbox[1])

            # 简化3D估计：假设目标在地面上
            focal_length = 920.0  # 假设焦距
            object_height_real = 1.0  # 假设真实高度1米

            # 估计距离
            if height_2d > 0:
                distance = (focal_length * object_height_real) / height_2d
            else:
                distance = 10.0  # 默认距离

            # 计算3D位置（相机坐标系）
            x_3d = (center_x - 320.0) * distance / focal_length
            y_3d = (center_y - 240.0) * distance / focal_length
            z_3d = distance

            position_3d = np.array([x_3d, y_3d, z_3d], dtype=np.float64)

            # 估计3D尺寸
            if height_2d > 0:
                width_3d = (width_2d / height_2d) * object_height_real
            else:
                width_3d = 1.0
            size_3d = np.array([width_3d, object_height_real, 0.5], dtype=np.float64)  # 深度假设0.5米

            # 创建3D边界框（简化）
            bbox_3d = self._create_3d_bbox(position_3d, size_3d)

            obj_3d = {
                'bbox_2d': bbox,
                'position_3d': position_3d,
                'size_3d': size_3d,
                'bbox_3d': bbox_3d,
                'class_name': target.get('class_name', 'unknown'),
                'confidence': float(target.get('confidence', 0.5))
            }
            objects_3d.append(obj_3d)

        return objects_3d

    def _create_3d_bbox(self, center: np.ndarray, size: np.ndarray) -> np.ndarray:
        """创建3D边界框"""
        # 创建轴对齐的3D边界框
        half_size = size / 2.0

        # 8个顶点
        vertices = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ], dtype=np.float64)

        # 缩放和平移
        bbox = vertices * half_size + center
        return bbox

    def _calculate_safety_score(self, fused_map: FusedMap, visual_results: Dict) -> float:
        """计算综合安全分数"""
        safety_score = 1.0

        # 1. 基于障碍物距离
        for obj in fused_map.objects:
            distance = np.linalg.norm(obj.position - self.current_position)
            if distance < 5.0:  # 5米内危险
                distance_factor = max(0.0, 1.0 - distance/5.0)
                safety_score *= (1.0 - distance_factor * 0.3)

        # 2. 基于视觉安全分数
        visual_safety = visual_results.get('safety_score', 1.0)
        safety_score *= visual_safety

        # 3. 基于速度（速度越快安全分数越低）
        speed = np.linalg.norm(self.current_velocity)
        if speed > 5.0:  # 5m/s以上
            speed_factor = min(1.0, (speed - 5.0) / 10.0)
            safety_score *= (1.0 - speed_factor * 0.2)

        return round(max(0.0, min(1.0, safety_score)), 2)

    def visualize_fusion(self, camera_frame: np.ndarray, results: Dict) -> np.ndarray:
        """可视化融合结果"""
        if results.get('visual') is None:
            return camera_frame

        # 可视化视觉结果
        vis_frame = self.visual_perception.visualize(camera_frame, results['visual'])

        # 添加融合物体信息
        fused_objects = results.get('objects', [])

        for obj in fused_objects:
            # 显示3D位置
            pos_text = f"({obj.position[0]:.1f}, {obj.position[1]:.1f}, {obj.position[2]:.1f})"

            # 在图像上添加文本
            if obj.bbox_2d is not None:
                x1, y1, x2, y2 = map(int, obj.bbox_2d)
                cv2.putText(vis_frame, f"Fused: {obj.object_type}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 0), 2)
                cv2.putText(vis_frame, pos_text,
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (255, 255, 0), 1)
            else:
                # 如果没有2D边界框，在图像底部显示信息
                y_offset = 180 + len(fused_objects) * 20
                info_text = f"{obj.object_type}: {pos_text}"
                cv2.putText(vis_frame, info_text,
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 0), 1)

        # 显示无人机状态
        status_text = f"Pos: ({self.current_position[0]:.1f}, {self.current_position[1]:.1f}, {self.current_position[2]:.1f})"
        cv2.putText(vis_frame, status_text,
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)

        speed = np.linalg.norm(self.current_velocity)
        speed_text = f"Speed: {speed:.1f} m/s"
        cv2.putText(vis_frame, speed_text,
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)

        # 显示融合安全分数
        safety_score = results.get('safety_score', 1.0)
        if safety_score > 0.7:
            safety_color = (0, 255, 0)  # 绿色
        elif safety_score > 0.4:
            safety_color = (0, 255, 255)  # 黄色
        else:
            safety_color = (0, 0, 255)  # 红色

        cv2.putText(vis_frame, f"Fused Safety: {safety_score}",
                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, safety_color, 2)

        # 显示传感器状态
        sensor_status = f"Sensors: {len(self.sensor_buffer)} active"
        cv2.putText(vis_frame, sensor_status,
                   (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)

        return vis_frame

    def save_fusion_results(self, camera_frame: np.ndarray, results: Dict,
                           output_dir: str = "output_fusion"):
        """保存融合结果"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        try:
            # 1. 保存可视化图像
            vis_frame = self.visualize_fusion(camera_frame, results)
            image_path = os.path.join(output_dir, f"fusion_{timestamp}.jpg")
            cv2.imwrite(image_path, vis_frame)

            # 2. 保存占据栅格地图
            if results.get('fused_map') and results['fused_map'].occupancy_grid is not None:
                grid = results['fused_map'].occupancy_grid
                if grid.size > 0:
                    # 归一化到0-255
                    grid_normalized = cv2.normalize(grid, None, 0, 255, cv2.NORM_MINMAX)
                    grid_display = grid_normalized.astype(np.uint8)
                    grid_colored = cv2.applyColorMap(grid_display, cv2.COLORMAP_JET)
                    grid_path = os.path.join(output_dir, f"occupancy_grid_{timestamp}.jpg")
                    cv2.imwrite(grid_path, grid_colored)

            # 3. 保存数据到JSON
            data_to_save = {
                'timestamp': timestamp,
                'position': self.current_position.tolist(),
                'orientation': self.current_orientation.tolist(),
                'safety_score': results.get('safety_score', 1.0),
                'visual_safety': results.get('visual', {}).get('safety_score', 1.0),
                'objects': []
            }

            for obj in results.get('objects', []):
                obj_data = {
                    'id': obj.id,
                    'type': obj.object_type,
                    'position': obj.position.tolist(),
                    'size': obj.size.tolist(),
                    'confidence': obj.confidence
                }
                data_to_save['objects'].append(obj_data)

            json_path = os.path.join(output_dir, f"data_{timestamp}.json")
            with open(json_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)

            print(f"融合结果已保存到: {output_dir}")
            return image_path

        except Exception as e:
            print(f"保存结果时出错: {e}")
            return None

class VisualPerception:
    """视觉感知模块"""

    def __init__(self):
        self.camera_params = {
            'fx': 920.0, 'fy': 920.0,
            'cx': 640.0, 'cy': 360.0,
            'width': 1280, 'height': 720
        }

        self.target_detector = TargetDetector()
        self.depth_estimator = DepthEstimator()

        self.obstacles = []
        self.targets = []

    def process_frame(self, frame: np.ndarray) -> dict:
        """处理视觉帧"""
        results = {
            'timestamp': time.time(),
            'targets': [],
            'obstacles': [],
            'depth_map': None,
            'safety_score': 1.0
        }

        try:
            # 目标检测
            self.targets = self.target_detector.detect(frame)
            results['targets'] = self.targets

            # 深度估计
            depth_map = self.depth_estimator.estimate(frame)
            results['depth_map'] = depth_map

            # 障碍物检测
            self.obstacles = self.detect_obstacles(frame, depth_map)
            results['obstacles'] = self.obstacles

            # 安全分数
            results['safety_score'] = self.analyze_safety(self.obstacles)

        except Exception as e:
            results['error'] = str(e)
            print(f"视觉处理出错: {e}")

        return results

    def detect_obstacles(self, frame, depth_map):
        """检测障碍物"""
        obstacles = []

        if depth_map is not None and depth_map.size > 0:
            try:
                # 检测深度突变区域作为障碍物
                depth_grad_x = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
                depth_grad_y = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
                depth_grad_mag = np.sqrt(depth_grad_x**2 + depth_grad_y**2)

                # 阈值处理
                _, obstacles_mask = cv2.threshold(depth_grad_mag, 0.1, 1.0, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(
                    obstacles_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours[:5]:
                    area = cv2.contourArea(contour)
                    if area > 100:
                        x, y, w, h = cv2.boundingRect(contour)

                        # 估计距离
                        roi_depth = depth_map[y:y+h, x:x+w]
                        if roi_depth.size > 0:
                            median_depth = np.median(roi_depth)

                            obstacle = {
                                'bbox': [int(x), int(y), int(x+w), int(y+h)],
                                'estimated_distance': float(median_depth),
                                'area': float(area)
                            }
                            obstacles.append(obstacle)
            except Exception as e:
                print(f"障碍物检测出错: {e}")

        return obstacles

    def analyze_safety(self, obstacles):
        """分析安全性"""
        if not obstacles:
            return 1.0

        danger_score = 0.0
        for obs in obstacles:
            if 'estimated_distance' in obs:
                dist = obs['estimated_distance']
                if dist < 0.5:
                    danger_score += 0.3
                elif dist < 1.0:
                    danger_score += 0.1

        return round(max(0.0, min(1.0, 1.0 - danger_score)), 2)

    def visualize(self, frame, results):
        """可视化视觉结果"""
        vis_frame = frame.copy()

        # 绘制目标
        for target in results.get('targets', []):
            if 'bbox' in target:
                x1, y1, x2, y2 = map(int, target['bbox'])
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if 'class_name' in target:
                    cv2.putText(vis_frame, target['class_name'],
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (0, 255, 0), 2)

        # 绘制障碍物
        for obstacle in results.get('obstacles', []):
            if 'bbox' in obstacle:
                x1, y1, x2, y2 = map(int, obstacle['bbox'])
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

                if 'estimated_distance' in obstacle:
                    dist_text = f"{obstacle['estimated_distance']:.1f}m"
                    cv2.putText(vis_frame, dist_text,
                               (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX,
                               0.4, (0, 0, 255), 1)

        # 安全分数
        safety_score = results.get('safety_score', 1.0)
        safety_color = (0, 255, 0) if safety_score > 0.5 else (0, 0, 255)
        cv2.putText(vis_frame, f"Visual Safety: {safety_score}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, safety_color, 2)

        # 目标数量
        target_count = len(results.get('targets', []))
        cv2.putText(vis_frame, f"Targets: {target_count}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1)

        return vis_frame

class TargetDetector:
    """目标检测器"""

    def __init__(self):
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255])
        }

    def detect(self, frame):
        targets = []

        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for color_name, (lower, upper) in self.color_ranges.items():
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)

                mask = cv2.inRange(hsv, lower_np, upper_np)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:
                        x, y, w, h = cv2.boundingRect(contour)

                        target = {
                            'bbox': [int(x), int(y), int(x+w), int(y+h)],
                            'class_name': color_name,
                            'confidence': 0.8,
                            'area': float(area)
                        }
                        targets.append(target)
        except Exception as e:
            print(f"目标检测出错: {e}")

        return targets

class DepthEstimator:
    """深度估计器"""

    def estimate(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth_map = cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0)

            if depth_map.max() > depth_map.min():
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

            h, w = depth_map.shape
            y_coords, x_coords = np.ogrid[:h, :w]
            center_x, center_y = w // 2, h // 2

            center_mask = 1.0 - np.sqrt(((x_coords - center_x) / max(center_x, 1))**2 +
                                       ((y_coords - center_y) / max(center_y, 1))**2)
            center_mask = np.clip(center_mask, 0, 1)

            depth_map = 0.7 * depth_map + 0.3 * center_mask

            return depth_map
        except Exception as e:
            print(f"深度估计出错: {e}")
            return np.zeros(frame.shape[:2], dtype=np.float32)

def test_multi_sensor_fusion():
    """测试多传感器融合"""
    print("=== 测试多传感器融合模块 ===")

    if not SCIPY_AVAILABLE:
        print("注意: scipy库未安装，某些功能将受限")
        print("安装命令: pip install scipy")

    # 初始化多传感器感知模块
    perception = MultiSensorPerception()

    # 创建测试图像
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 添加测试目标
    cv2.rectangle(test_frame, (100, 100), (200, 200), (0, 255, 0), -1)  # 绿色
    cv2.rectangle(test_frame, (400, 150), (500, 250), (0, 0, 255), -1)  # 红色
    cv2.rectangle(test_frame, (300, 300), (400, 400), (255, 0, 0), -1)  # 蓝色

    # 添加一些障碍物纹理
    cv2.rectangle(test_frame, (50, 50), (80, 200), (100, 100, 100), -1)
    cv2.rectangle(test_frame, (550, 300), (600, 400), (150, 150, 150), -1)

    # 添加一些噪声（模拟真实环境）
    noise = np.random.randint(0, 30, test_frame.shape[:2], dtype=np.uint8)
    for i in range(3):
        test_frame[:,:,i] = cv2.add(test_frame[:,:,i], noise[:,:,np.newaxis])

    print("开始处理传感器数据...")

    # 处理所有传感器
    results = perception.process_all_sensors(test_frame)

    print(f"\n传感器融合结果:")
    print(f"检测到融合物体数量: {len(results['objects'])}")

    if results.get('visual'):
        print(f"视觉安全分数: {results['visual'].get('safety_score', 'N/A')}")
        print(f"视觉目标数量: {len(results['visual'].get('targets', []))}")

    print(f"融合安全分数: {results['safety_score']}")

    if results.get('fused_map'):
        if results['fused_map'].occupancy_grid is not None:
            grid_shape = results['fused_map'].occupancy_grid.shape
            print(f"占据栅格地图尺寸: {grid_shape}")
        else:
            print("占据栅格地图: 未生成")

    if results.get('error'):
        print(f"错误信息: {results['error']}")

    # 保存结果
    output_dir = "output_fusion"
    saved_file = perception.save_fusion_results(test_frame, results, output_dir)

    if saved_file:
        print(f"可视化结果已保存: {saved_file}")

    # 显示融合物体详情
    for i, obj in enumerate(results['objects'][:5]):  # 只显示前5个
        print(f"\n融合物体 {i+1}:")
        print(f"  类型: {obj.object_type}")
        print(f"  位置: ({obj.position[0]:.2f}, {obj.position[1]:.2f}, {obj.position[2]:.2f})")
        print(f"  尺寸: ({obj.size[0]:.2f}, {obj.size[1]:.2f}, {obj.size[2]:.2f})")
        print(f"  置信度: {obj.confidence:.2f}")

    # 显示无人机状态
    print(f"\n无人机状态:")
    print(f"位置: [{perception.current_position[0]:.2f}, {perception.current_position[1]:.2f}, {perception.current_position[2]:.2f}]")
    print(f"速度: [{perception.current_velocity[0]:.2f}, {perception.current_velocity[1]:.2f}, {perception.current_velocity[2]:.2f}] m/s")

    print(f"\n测试完成! 结果保存在 {output_dir} 目录")

    return True

def run_simple_demo():
    """运行简化演示"""
    print("=== 多传感器融合简化演示 ===")

    # 创建感知模块
    perception = MultiSensorPerception()

    # 创建简单的测试场景
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 添加几个彩色目标
    colors = [
        ((0, 255, 0), "绿色目标", (100, 100)),
        ((0, 0, 255), "红色目标", (400, 200)),
        ((255, 0, 0), "蓝色目标", (300, 350))
    ]

    for color, label, pos in colors:
        x, y = pos
        cv2.rectangle(frame, (x, y), (x+50, y+50), color, -1)
        cv2.putText(frame, label, (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 处理传感器数据
    results = perception.process_all_sensors(frame)

    # 打印结果
    print(f"视觉检测目标: {len(results.get('visual', {}).get('targets', []))}")
    print(f"融合物体: {len(results.get('objects', []))}")
    print(f"安全分数: {results.get('safety_score', 0.0)}")

    # 保存结果
    perception.save_fusion_results(frame, results, "demo_output")

    print("演示完成!")

if __name__ == "__main__":
    try:
        # 运行测试
        success = test_multi_sensor_fusion()

        # 如果测试成功，可以运行简化演示
        if success:
            print("\n" + "="*50)
            run_simple_demo()

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()