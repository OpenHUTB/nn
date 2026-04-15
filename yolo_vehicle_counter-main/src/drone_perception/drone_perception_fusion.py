"""
drone_perception_fusion.py
无人机感知模块 - 多传感器融合版本
修复版本
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

# 检查scipy是否可用
try:
    from scipy.spatial import KDTree
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("注意: scipy库未安装，使用简化聚类算法")

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
    objects: List[FusedObject] = None  # 地图中的物体

    def __post_init__(self):
        if self.objects is None:
            self.objects = []

class SensorCalibration:
    """传感器标定类"""

    def __init__(self):
        self.transformations = {}

    def set_transformation(self, from_sensor: SensorType, to_sensor: SensorType,
                          rotation: np.ndarray, translation: np.ndarray):
        """设置传感器之间的变换关系"""
        key = f"{from_sensor.value}_to_{to_sensor.value}"
        self.transformations[key] = {
            'R': rotation.astype(np.float64),
            't': translation.astype(np.float64),
            'T': np.eye(4, dtype=np.float64)
        }
        self.transformations[key]['T'][:3, :3] = rotation.astype(np.float64)
        self.transformations[key]['T'][:3, 3] = translation.flatten().astype(np.float64)

class VirtualLidar:
    """虚拟LiDAR传感器"""

    def __init__(self, num_beams: int = 360, max_range: float = 50.0,
                 fov_horizontal: float = 360.0, fov_vertical: float = 30.0):
        self.num_beams = num_beams
        self.max_range = max_range
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
        self.noise_std = 0.02

    def simulate_scan(self, position: np.ndarray, orientation: np.ndarray,
                     environment_objects: List[Dict]) -> LidarData:
        """模拟LiDAR扫描"""
        points = []
        intensities = []

        position = position.astype(np.float64)

        # 生成一些随机点云模拟LiDAR扫描
        n_points = 500
        for _ in range(n_points):
            # 随机角度
            angle_h = np.random.uniform(-self.fov_horizontal/2, self.fov_horizontal/2)
            angle_v = np.random.uniform(-self.fov_vertical/2, self.fov_vertical/2)
            distance = np.random.uniform(1.0, self.max_range)

            # 球坐标转笛卡尔坐标
            x = distance * np.cos(np.radians(angle_v)) * np.cos(np.radians(angle_h))
            y = distance * np.cos(np.radians(angle_v)) * np.sin(np.radians(angle_h))
            z = distance * np.sin(np.radians(angle_v))

            point = np.array([x, y, z], dtype=np.float64)

            # 检查是否与物体相交
            hit_object = False
            for obj in environment_objects:
                obj_pos = np.array(obj['position'], dtype=np.float64)
                obj_size = np.array(obj['size'], dtype=np.float64)

                # 简单碰撞检测
                if (np.abs(point[0] - obj_pos[0]) < obj_size[0]/2 and
                    np.abs(point[1] - obj_pos[1]) < obj_size[1]/2 and
                    np.abs(point[2] - obj_pos[2]) < obj_size[2]/2):
                    hit_object = True
                    # 添加噪声
                    noise = np.random.randn(3).astype(np.float64) * self.noise_std
                    point = obj_pos + (point - obj_pos) * 0.9 + noise  # 稍微向内移动
                    intensity = 0.8
                    break

            if not hit_object:
                intensity = 0.3

            points.append(point)
            intensities.append(intensity)

        points_array = np.array(points, dtype=np.float64)
        intensities_array = np.array(intensities, dtype=np.float64)

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
        self.gravity = np.array([0, 0, -9.81], dtype=np.float64)
        self.noise_accel = 0.05
        self.noise_gyro = 0.01

    def simulate_measurement(self, true_acceleration: np.ndarray,
                           true_angular_velocity: np.ndarray,
                           true_orientation: Optional[np.ndarray] = None) -> IMUData:
        """模拟IMU测量"""
        true_acceleration = true_acceleration.astype(np.float64)
        true_angular_velocity = true_angular_velocity.astype(np.float64)

        accel_with_gravity = true_acceleration - self.gravity
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
        self.fused_objects = {}
        self.next_object_id = 1
        self.kalman_filters = {}
        self.sensor_timestamps = {}

        # 调整融合权重
        self.sensor_weights = {
            SensorType.CAMERA: 0.6,  # 提高视觉权重
            SensorType.LIDAR: 0.4,   # 降低LiDAR权重
        }

    def associate_detections(self, camera_objects: List[Dict],
                            lidar_clusters: List[Dict]) -> List[Dict]:
        """关联不同传感器的检测结果"""
        associations = []

        if not camera_objects or not lidar_clusters:
            # 如果没有LiDAR聚类，也使用视觉检测
            for cam_obj in camera_objects:
                association = {
                    'camera_object': cam_obj,
                    'lidar_cluster': None,
                    'distance': 0.0,
                    'confidence': cam_obj.get('confidence', 0.5)
                }
                associations.append(association)
            return associations

        # 关联算法
        for cam_obj in camera_objects:
            if 'position_3d' not in cam_obj:
                continue

            cam_pos = np.array(cam_obj.get('position_3d', np.zeros(3)), dtype=np.float64)

            best_match = None
            best_distance = float('inf')

            for lidar_cluster in lidar_clusters:
                lidar_pos = np.array(lidar_cluster.get('centroid', np.zeros(3)), dtype=np.float64)
                distance = np.linalg.norm(cam_pos - lidar_pos)

                if distance < 5.0 and distance < best_distance:  # 增大阈值到5米
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
            else:
                # 即使没有LiDAR匹配，也使用视觉检测
                association = {
                    'camera_object': cam_obj,
                    'lidar_cluster': None,
                    'distance': 0.0,
                    'confidence': cam_obj.get('confidence', 0.5) * 0.8  # 降低置信度
                }

            associations.append(association)

        return associations

    def fuse_object_data(self, associations: List[Dict]) -> List[FusedObject]:
        """融合物体数据"""
        fused_objects = []

        for assoc in associations:
            cam_obj = assoc['camera_object']
            lidar_cluster = assoc['lidar_cluster']

            # 位置融合
            cam_pos = np.array(cam_obj.get('position_3d', np.zeros(3)), dtype=np.float64)

            if lidar_cluster is not None:
                lidar_pos = np.array(lidar_cluster.get('centroid', np.zeros(3)), dtype=np.float64)
                weight_cam = self.sensor_weights[SensorType.CAMERA]
                weight_lidar = self.sensor_weights[SensorType.LIDAR]
                fused_pos = (cam_pos * weight_cam + lidar_pos * weight_lidar) / (weight_cam + weight_lidar)

                # 尺寸融合
                cam_size = np.array(cam_obj.get('size_3d', np.ones(3)), dtype=np.float64)
                lidar_size = np.array(lidar_cluster.get('size', np.ones(3)), dtype=np.float64)
                fused_size = lidar_size if np.any(lidar_size > 0) else cam_size
            else:
                fused_pos = cam_pos
                fused_size = np.array(cam_obj.get('size_3d', np.ones(3)), dtype=np.float64)

            # 创建融合物体
            fused_object = FusedObject(
                id=self.next_object_id,
                position=fused_pos,
                velocity=np.zeros(3, dtype=np.float64),
                size=fused_size,
                confidence=float(assoc['confidence']),
                object_type=cam_obj.get('class_name', 'unknown'),
                timestamp=time.time(),
                sensor_sources=[SensorType.CAMERA] + ([SensorType.LIDAR] if lidar_cluster else []),
                bbox_2d=cam_obj.get('bbox_2d'),
                bbox_3d=cam_obj.get('bbox_3d')
            )

            fused_objects.append(fused_object)
            self.next_object_id += 1

        return fused_objects

    def cluster_lidar_points(self, lidar_data: LidarData,
                            distance_threshold: float = 1.0) -> List[Dict]:
        """对LiDAR点云进行聚类"""
        clusters = []

        if lidar_data.points.size == 0 or len(lidar_data.points) == 0:
            return clusters

        points = lidar_data.points
        n_points = len(points)

        # 降采样以提高性能
        if n_points > 1000:
            indices = np.random.choice(n_points, 1000, replace=False)
            points = points[indices]
            n_points = 1000

        # 网格聚类（简化版）
        grid_size = distance_threshold
        grid_dict = {}

        for i, point in enumerate(points):
            grid_x = int(point[0] / grid_size)
            grid_y = int(point[1] / grid_size)
            grid_z = int(point[2] / grid_size)
            key = (grid_x, grid_y, grid_z)

            if key not in grid_dict:
                grid_dict[key] = []
            grid_dict[key].append(i)

        # 合并相邻网格
        for key, indices in grid_dict.items():
            if len(indices) >= 3:  # 至少3个点
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

    def create_occupancy_grid(self, lidar_data: LidarData, grid_size: Tuple[int, int] = (100, 100),
                             grid_resolution: float = 0.5) -> np.ndarray:
        """创建占据栅格地图"""
        width, height = grid_size
        grid = np.zeros((height, width), dtype=np.float32)

        if lidar_data.points.size == 0:
            return grid

        points_2d = lidar_data.points[:, :2]
        grid_center = np.array([width//2, height//2], dtype=np.float32)
        grid_coords = (points_2d / grid_resolution + grid_center).astype(np.int32)

        valid_x = (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < width)
        valid_y = (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < height)
        valid = valid_x & valid_y

        if np.any(valid):
            grid_coords_valid = grid_coords[valid]
            for coord in grid_coords_valid:
                x, y = coord
                if 0 <= y < height and 0 <= x < width:
                    grid[y, x] = 1.0

            # 模糊处理
            grid = cv2.GaussianBlur(grid, (5, 5), 1.0)

        return grid

    def fuse_all_sensors(self, sensor_data: Dict[SensorType, SensorData],
                        camera_objects: List[Dict]) -> FusedMap:
        """融合所有传感器数据"""
        # 提取LiDAR数据并聚类
        lidar_clusters = []
        if SensorType.LIDAR in sensor_data:
            lidar_data = sensor_data[SensorType.LIDAR]
            lidar_clusters = self.cluster_lidar_points(lidar_data)

        # 关联检测结果
        associations = self.associate_detections(camera_objects, lidar_clusters)

        # 融合物体数据
        fused_objects = self.fuse_object_data(associations)

        # 创建占据栅格地图
        occupancy_grid = None
        if SensorType.LIDAR in sensor_data:
            occupancy_grid = self.create_occupancy_grid(sensor_data[SensorType.LIDAR])

        # 创建融合地图
        fused_map = FusedMap(
            timestamp=time.time(),
            occupancy_grid=occupancy_grid,
            objects=fused_objects
        )

        return fused_map

class MultiSensorPerception:
    """多传感器感知主类"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

        self.calibration = SensorCalibration()
        self._setup_calibration()
        self.sensor_fusion = SensorFusion(self.calibration)

        self.virtual_lidar = VirtualLidar()
        self.virtual_imu = VirtualIMU()
        self.visual_perception = VisualPerception()

        self.sensor_buffer = {}
        self.fusion_history = []

        # 初始化位置和姿态
        self.current_position = np.array([0.0, 0.0, 5.0], dtype=np.float64)  # 默认高度5米
        self.current_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.current_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        print("多传感器感知模块初始化完成")

    def _setup_calibration(self):
        """设置传感器标定参数"""
        R_cam_to_lidar = np.eye(3, dtype=np.float64)
        t_cam_to_lidar = np.array([0.0, 0.0, 0.1], dtype=np.float64)

        self.calibration.set_transformation(
            SensorType.CAMERA, SensorType.LIDAR,
            R_cam_to_lidar, t_cam_to_lidar
        )

    def update_sensor_data(self, sensor_type: SensorType, data: SensorData):
        """更新传感器数据"""
        self.sensor_buffer[sensor_type] = data

    def process_all_sensors(self, camera_frame: np.ndarray) -> Dict:
        """处理所有传感器数据"""
        results = {
            'timestamp': time.time(),
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

            # 2. 从视觉结果创建模拟环境
            environment_objects = self._create_environment_from_visual(visual_results)

            # 3. 模拟LiDAR扫描
            lidar_data = self.virtual_lidar.simulate_scan(
                self.current_position, self.current_orientation, environment_objects
            )
            self.update_sensor_data(SensorType.LIDAR, lidar_data)

            # 4. 模拟IMU数据
            imu_data = self.virtual_imu.simulate_measurement(
                np.array([0.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, 0.0], dtype=np.float64),
                self.current_orientation
            )
            self.update_sensor_data(SensorType.IMU, imu_data)

            # 5. 从2D检测估计3D信息
            camera_objects_3d = self._estimate_3d_from_2d(visual_results['targets'])

            # 6. 传感器融合
            fused_map = self.sensor_fusion.fuse_all_sensors(
                self.sensor_buffer, camera_objects_3d
            )
            results['fused_map'] = fused_map
            results['objects'] = fused_map.objects

            # 7. 计算安全分数（基于视觉和融合结果）
            results['safety_score'] = self._calculate_safety_score(fused_map, visual_results)

            # 保存历史
            self.fusion_history.append({
                'timestamp': time.time(),
                'fused_map': fused_map,
                'position': self.current_position.copy(),
                'orientation': self.current_orientation.copy()
            })

            if len(self.fusion_history) > 100:
                self.fusion_history = self.fusion_history[-100:]

        except Exception as e:
            results['error'] = str(e)
            print(f"传感器融合出错: {e}")

        return results

    def _create_environment_from_visual(self, visual_results: Dict) -> List[Dict]:
        """从视觉结果创建环境对象"""
        environment = []

        # 添加检测到的目标
        for target in visual_results.get('targets', []):
            if 'bbox' in target:
                bbox = target['bbox']
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0

                # 假设目标在地面上（z=0）
                position = np.array([
                    (center_x - 320) * 0.05,  # 缩放因子
                    (center_y - 240) * 0.05,
                    0.0  # 地面
                ], dtype=np.float64)

                position += self.current_position

                obj = {
                    'position': position,
                    'size': np.array([1.0, 1.0, 2.0], dtype=np.float64),  # 假设尺寸
                    'type': target.get('class_name', 'unknown')
                }
                environment.append(obj)

        # 添加地面
        ground_pos = self.current_position.copy()
        ground_pos[2] = 0.0
        ground = {
            'position': ground_pos,
            'size': np.array([50.0, 50.0, 0.1], dtype=np.float64),
            'type': 'ground'
        }
        environment.append(ground)

        return environment

    def _estimate_3d_from_2d(self, targets_2d: List[Dict]) -> List[Dict]:
        """从2D检测估计3D信息（简化版本）"""
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

            # 简化3D估计
            focal_length = 920.0
            if height_2d > 0:
                # 假设目标高度为2米
                distance = (focal_length * 2.0) / height_2d
            else:
                distance = 10.0

            # 计算3D位置（相机坐标系）
            x_3d = (center_x - 320.0) * distance / focal_length
            y_3d = (center_y - 240.0) * distance / focal_length
            z_3d = distance

            # 转换到世界坐标系（简化）
            position_3d = np.array([x_3d, y_3d, z_3d], dtype=np.float64)
            position_3d += self.current_position

            # 估计3D尺寸
            if height_2d > 0:
                width_3d = (width_2d / height_2d) * 2.0
            else:
                width_3d = 1.0
            size_3d = np.array([width_3d, 2.0, 1.0], dtype=np.float64)

            obj_3d = {
                'bbox_2d': bbox,
                'position_3d': position_3d,
                'size_3d': size_3d,
                'class_name': target.get('class_name', 'unknown'),
                'confidence': float(target.get('confidence', 0.5))
            }
            objects_3d.append(obj_3d)

        return objects_3d

    def _calculate_safety_score(self, fused_map: FusedMap, visual_results: Dict) -> float:
        """计算综合安全分数"""
        # 使用视觉安全分数作为基础
        visual_safety = visual_results.get('safety_score', 1.0)

        # 基于融合物体调整
        adjustment = 1.0
        for obj in fused_map.objects:
            distance = np.linalg.norm(obj.position - self.current_position)
            if distance < 10.0:
                adjustment *= 0.9  # 近距离物体降低安全分数

        safety_score = visual_safety * adjustment
        return round(max(0.0, min(1.0, safety_score)), 2)

    def visualize_fusion(self, camera_frame: np.ndarray, results: Dict) -> np.ndarray:
        """可视化融合结果"""
        if results.get('visual') is None:
            return camera_frame

        # 可视化视觉结果
        vis_frame = self.visual_perception.visualize(camera_frame, results['visual'])

        # 添加融合物体信息
        fused_objects = results.get('objects', [])

        for i, obj in enumerate(fused_objects[:5]):  # 只显示前5个
            pos_text = f"({obj.position[0]:.1f}, {obj.position[1]:.1f}, {obj.position[2]:.1f})"
            y_offset = 180 + i * 20
            info_text = f"{obj.object_type}: {pos_text}"

            cv2.putText(vis_frame, info_text,
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 0), 1)

        # 显示状态信息
        status_texts = [
            f"Pos: ({self.current_position[0]:.1f}, {self.current_position[1]:.1f}, {self.current_position[2]:.1f})",
            f"Targets: {len(results.get('objects', []))}",
            f"Visual Safety: {results['visual'].get('safety_score', 0.0)}",
            f"Fused Safety: {results.get('safety_score', 0.0)}"
        ]

        for i, text in enumerate(status_texts):
            cv2.putText(vis_frame, text,
                       (10, 100 + i*20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1)

        return vis_frame

    def save_fusion_results(self, camera_frame: np.ndarray, results: Dict,
                           output_dir: str = "output_fusion"):
        """保存融合结果"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        try:
            # 保存可视化图像
            vis_frame = self.visualize_fusion(camera_frame, results)
            image_path = os.path.join(output_dir, f"fusion_{timestamp}.jpg")
            cv2.imwrite(image_path, vis_frame)

            # 保存数据
            data_to_save = {
                'timestamp': timestamp,
                'position': self.current_position.tolist(),
                'safety_score': results.get('safety_score', 1.0),
                'visual_safety': results.get('visual', {}).get('safety_score', 1.0),
                'objects_count': len(results.get('objects', []))
            }

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
        self.target_detector = TargetDetector()
        self.depth_estimator = DepthEstimator()

    def process_frame(self, frame: np.ndarray) -> dict:
        """处理视觉帧"""
        results = {
            'timestamp': time.time(),
            'targets': [],
            'obstacles': [],
            'depth_map': None,
            'safety_score': 1.0  # 默认安全分数
        }

        try:
            # 目标检测
            targets = self.target_detector.detect(frame)
            results['targets'] = targets

            # 深度估计
            depth_map = self.depth_estimator.estimate(frame)
            results['depth_map'] = depth_map

            # 安全分数（基于目标数量）
            if len(targets) > 0:
                # 有目标不一定危险，给一个中等分数
                results['safety_score'] = 0.7
            else:
                results['safety_score'] = 1.0

        except Exception as e:
            results['error'] = str(e)
            print(f"视觉处理出错: {e}")

        return results

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

        # 安全分数
        safety_score = results.get('safety_score', 1.0)
        safety_color = (0, 255, 0) if safety_score > 0.5 else (0, 0, 255)

        cv2.putText(vis_frame, f"Safety: {safety_score}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, safety_color, 2)

        cv2.putText(vis_frame, f"Targets: {len(results.get('targets', []))}",
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            depth_map = cv2.GaussianBlur(gray, (15, 15), 0)

            if depth_map.max() > depth_map.min():
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

            return depth_map
        except Exception as e:
            print(f"深度估计出错: {e}")
            return np.zeros(frame.shape[:2], dtype=np.float32)

def test_multi_sensor_fusion():
    """测试多传感器融合"""
    print("=== 测试多传感器融合模块 ===")

    # 初始化多传感器感知模块
    perception = MultiSensorPerception()

    # 创建测试图像
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 添加测试目标
    cv2.rectangle(test_frame, (100, 100), (200, 200), (0, 255, 0), -1)  # 绿色
    cv2.rectangle(test_frame, (400, 150), (500, 250), (0, 0, 255), -1)  # 红色
    cv2.rectangle(test_frame, (300, 300), (400, 400), (255, 0, 0), -1)  # 蓝色

    # 添加一些纹理
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

    print(f"融合安全分数: {results.get('safety_score', 0.0)}")

    # 显示融合物体详情
    for i, obj in enumerate(results['objects'][:5]):
        print(f"\n融合物体 {i+1}:")
        print(f"  类型: {obj.object_type}")
        print(f"  位置: ({obj.position[0]:.2f}, {obj.position[1]:.2f}, {obj.position[2]:.2f})")
        print(f"  置信度: {obj.confidence:.2f}")

    # 保存结果
    output_dir = "output_fusion"
    saved_file = perception.save_fusion_results(test_frame, results, output_dir)

    if saved_file:
        print(f"\n可视化结果已保存: {saved_file}")

    print(f"\n测试完成! 结果保存在 {output_dir} 目录")
    return len(results['objects']) > 0

def run_improved_demo():
    """运行改进的演示"""
    print("\n" + "="*50)
    print("=== 多传感器融合改进演示 ===")

    perception = MultiSensorPerception()

    # 创建更真实的测试场景
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 添加背景（模拟地面）
    cv2.rectangle(frame, (0, 400), (640, 480), (50, 50, 50), -1)  # 地面

    # 添加彩色目标
    targets = [
        ((0, 255, 0), "绿色目标", (150, 150, 200, 250)),  # (x1, y1, x2, y2)
        ((0, 0, 255), "红色目标", (450, 200, 550, 300)),
        ((255, 0, 0), "蓝色目标", (300, 300, 400, 400))
    ]

    for color, label, bbox in targets:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 添加一些随机点模拟纹理
    for _ in range(100):
        x = np.random.randint(0, 640)
        y = np.random.randint(0, 480)
        cv2.circle(frame, (x, y), 1, (200, 200, 200), -1)

    # 处理传感器数据
    results = perception.process_all_sensors(frame)

    # 打印结果
    print(f"\n视觉检测目标: {len(results.get('visual', {}).get('targets', []))}")
    print(f"融合物体: {len(results.get('objects', []))}")
    print(f"视觉安全分数: {results.get('visual', {}).get('safety_score', 0.0)}")
    print(f"融合安全分数: {results.get('safety_score', 0.0)}")

    # 保存结果
    output_dir = "demo_output"
    perception.save_fusion_results(frame, results, output_dir)

    print(f"\n演示完成! 结果保存在 {output_dir} 目录")

if __name__ == "__main__":
    try:
        # 运行测试
        print("无人机多传感器融合感知系统")
        print("="*50)

        success = test_multi_sensor_fusion()

        if success:
            print("\n✅ 传感器融合测试成功!")
        else:
            print("\n⚠️  传感器融合测试完成，但未检测到融合物体")

        # 运行改进演示
        run_improved_demo()

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
