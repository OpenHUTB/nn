import numpy as np
import struct
import os
import json
import threading
import gc
from datetime import datetime
import zlib
import pickle
import gzip
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib


class LidarProcessor:
    """LiDAR处理器（增强版）"""

    def __init__(self, output_dir, config=None):
        self.output_dir = output_dir
        self.lidar_dir = os.path.join(output_dir, "lidar")
        os.makedirs(self.lidar_dir, exist_ok=True)

        self.calibration_dir = os.path.join(output_dir, "calibration")
        os.makedirs(self.calibration_dir, exist_ok=True)

        self.frame_counter = 0
        self.data_lock = threading.RLock()
        self.processing_lock = threading.Lock()

        # 配置参数
        self.config = config or {}
        self.batch_size = self.config.get('batch_size', 8)
        self.point_cloud_batch = []
        self.enable_compression = self.config.get('enable_compression', True)
        self.compression_level = self.config.get('compression_level', 4)

        self.max_points_per_frame = self.config.get('max_points_per_frame', 60000)
        self.enable_downsampling = self.config.get('enable_downsampling', True)
        self.downsample_ratio = self.config.get('downsample_ratio', 0.3)

        self.memory_warning_threshold = self.config.get('memory_warning_threshold', 400)
        self.max_batch_memory_mb = self.config.get('max_batch_memory_mb', 60)
        self.v2x_save_interval = self.config.get('v2x_save_interval', 5)

        # 性能监控
        self.processing_stats = {
            'total_frames': 0,
            'total_points': 0,
            'avg_points_per_frame': 0,
            'processing_times': [],
            'compression_ratios': [],
            'errors': 0
        }

        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        # 行人检测相关
        self.pedestrian_detection = self.config.get('pedestrian_detection', True)
        self.pedestrian_points = []

        self._init_calibration_files()

    def _init_calibration_files(self):
        """初始化标定文件（增强版）"""
        # LiDAR内参
        lidar_intrinsic = {
            "sensor_type": "lidar",
            "model": "velodyne_vls_128",
            "channels": 32,
            "range": 120.0,
            "points_per_second": 100000,
            "rotation_frequency": 10.0,
            "horizontal_fov": 360.0,
            "vertical_fov": 30.0,
            "upper_fov": 15.0,
            "lower_fov": -25.0,
            "angular_resolution": 0.1,
            "timestamp": datetime.now().isoformat(),
            "pedestrian_detection": self.pedestrian_detection,
            "height_filter": [-0.5, 2.5]
        }

        # LiDAR外参（车辆坐标系）
        lidar_extrinsic = {
            "sensor_id": "lidar_main",
            "vehicle_id": "ego_vehicle_1",
            "translation": [0.0, 0.0, 2.5],
            "rotation": [0.0, 0.0, 0.0],
            "matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            "timestamp": datetime.now().isoformat(),
            "coordinate_system": "carla_vehicle"
        }

        # 相机-LiDAR标定
        camera_lidar_calib = {
            "camera_id": "front_wide",
            "lidar_id": "lidar_main",
            "projection_matrix": [
                [7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00],
                [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00],
                [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]
            ],
            "rectification_matrix": [
                [9.999239e-01, 9.837760e-03, -7.445048e-03],
                [-9.869795e-03, 9.999421e-01, -4.278459e-03],
                [7.402527e-03, 4.351614e-03, 9.999631e-01]
            ],
            "translation": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0]
        }

        intrinsic_file = os.path.join(self.calibration_dir, "lidar_intrinsic.json")
        extrinsic_file = os.path.join(self.calibration_dir, "lidar_extrinsic.json")
        camera_lidar_file = os.path.join(self.calibration_dir, "camera_lidar_calibration.json")

        with open(intrinsic_file, 'w', encoding='utf-8') as f:
            json.dump(lidar_intrinsic, f, indent=2)

        with open(extrinsic_file, 'w', encoding='utf-8') as f:
            json.dump(lidar_extrinsic, f, indent=2)

        with open(camera_lidar_file, 'w', encoding='utf-8') as f:
            json.dump(camera_lidar_calib, f, indent=2)

    def process_lidar_data(self, lidar_data, frame_num):
        """处理LiDAR数据（增强版）"""
        start_time = time.time()

        with self.processing_lock:
            try:
                self.frame_counter = frame_num

                # 转换数据格式
                points = self._carla_lidar_to_numpy(lidar_data)

                if points is None or points.shape[0] == 0:
                    return None

                # 检查内存使用
                if self._check_memory_usage():
                    return None

                original_count = points.shape[0]

                # 预处理：过滤地面和过高点
                points = self._preprocess_point_cloud(points)

                # 行人检测
                if self.pedestrian_detection:
                    pedestrian_clusters = self._detect_pedestrians(points)
                    if pedestrian_clusters:
                        self._save_pedestrian_detections(frame_num, pedestrian_clusters)

                # 下采样
                if self.enable_downsampling and points.shape[0] > self.max_points_per_frame:
                    points = self._downsample_point_cloud(points)

                # 检查批处理内存
                batch_memory_mb = self._estimate_batch_memory_mb()
                if batch_memory_mb > self.max_batch_memory_mb:
                    self._save_batch_async()

                # 添加到批处理
                with self.data_lock:
                    self.point_cloud_batch.append((frame_num, points))
                    self.processing_stats['total_frames'] += 1
                    self.processing_stats['total_points'] += points.shape[0]

                # 异步保存
                if len(self.point_cloud_batch) >= self.batch_size:
                    self._save_batch_async()

                # 保存单个文件
                save_results = self._save_point_cloud_files(points, frame_num)

                # 每N帧保存V2X格式
                v2xformer_path = None
                if frame_num % self.v2x_save_interval == 0:
                    try:
                        v2xformer_path = self._save_as_v2xformer_format(points, frame_num)
                    except Exception as e:
                        v2xformer_path = None

                # 生成元数据
                metadata = self._generate_metadata(points, frame_num, save_results, v2xformer_path, original_count)

                # 定期清理
                if frame_num % 20 == 0:
                    gc.collect()

                # 更新统计
                processing_time = time.time() - start_time
                self.processing_stats['processing_times'].append(processing_time)

                # 计算平均点数
                if self.processing_stats['total_frames'] > 0:
                    self.processing_stats['avg_points_per_frame'] = \
                        self.processing_stats['total_points'] / self.processing_stats['total_frames']

                return metadata

            except Exception as e:
                self.processing_stats['errors'] += 1
                return None

    def _carla_lidar_to_numpy(self, lidar_data):
        """CARLA LiDAR数据转numpy（增强版）"""
        try:
            # 方法1：从原始数据解析
            points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
            points = np.reshape(points, (int(points.shape[0] / 4), 4))

            # 分离坐标和强度
            coordinates = points[:, :3]
            intensities = points[:, 3]

            # 过滤无效点
            valid_mask = ~np.any(np.isnan(coordinates), axis=1) & ~np.any(np.isinf(coordinates), axis=1)
            coordinates = coordinates[valid_mask]
            intensities = intensities[valid_mask]

            # 添加强度信息
            points_with_intensity = np.column_stack([coordinates, intensities])

            del lidar_data
            gc.collect()

            return points_with_intensity

        except Exception as e:
            try:
                # 方法2：从点列表解析
                points = []
                intensities = []

                # 假设lidar_data是点列表
                for point in lidar_data:
                    points.append([point.x, point.y, point.z])
                    intensities.append(point.intensity if hasattr(point, 'intensity') else 1.0)

                points_array = np.array(points, dtype=np.float32)
                intensities_array = np.array(intensities, dtype=np.float32)

                return np.column_stack([points_array, intensities_array])

            except:
                return None

    def _preprocess_point_cloud(self, points):
        """预处理点云"""
        if points.shape[1] < 3:
            return points

        # 过滤地面点（z坐标接近0）
        ground_mask = points[:, 2] > 0.2  # 保留地面以上20cm的点

        # 过滤过高点（高于建筑物）
        height_mask = points[:, 2] < 50.0

        # 过滤无效点
        valid_mask = ground_mask & height_mask

        return points[valid_mask]

    def _detect_pedestrians(self, points):
        """检测行人"""
        if points.shape[0] < 100:
            return []

        # 高度过滤：行人通常在0.5m到2.5m之间
        height_mask = (points[:, 2] > 0.5) & (points[:, 2] < 2.5)
        candidate_points = points[height_mask]

        if candidate_points.shape[0] < 10:
            return []

        # 简单聚类：基于空间距离
        from sklearn.cluster import DBSCAN

        # 使用DBSCAN聚类
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(candidate_points[:, :3])
        labels = clustering.labels_

        clusters = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # 噪声点
                continue

            cluster_points = candidate_points[labels == label]
            if cluster_points.shape[0] < 10:  # 点数太少
                continue

            # 计算聚类中心
            center = np.mean(cluster_points[:, :3], axis=0)
            bbox_min = np.min(cluster_points[:, :3], axis=0)
            bbox_max = np.max(cluster_points[:, :3], axis=0)

            # 计算大小
            size = bbox_max - bbox_min

            # 判断是否为行人（基于大小和形状）
            if (size[0] < 1.0 and size[1] < 1.0 and size[2] > 0.5 and size[2] < 2.0):
                clusters.append({
                    'center': center.tolist(),
                    'size': size.tolist(),
                    'point_count': cluster_points.shape[0],
                    'bbox': {
                        'min': bbox_min.tolist(),
                        'max': bbox_max.tolist()
                    }
                })

        return clusters

    def _save_pedestrian_detections(self, frame_num, clusters):
        """保存行人检测结果"""
        detection_data = {
            'frame_id': frame_num,
            'timestamp': datetime.now().isoformat(),
            'pedestrian_count': len(clusters),
            'clusters': clusters
        }

        detection_file = os.path.join(self.lidar_dir, f"pedestrian_detections_{frame_num:06d}.json")

        try:
            with open(detection_file, 'w', encoding='utf-8') as f:
                json.dump(detection_data, f, indent=2)
        except:
            pass

    def _downsample_point_cloud(self, points):
        """下采样点云（增强版）"""
        if points.shape[0] <= self.max_points_per_frame:
            return points

        # 使用体素网格下采样
        try:
            from sklearn.neighbors import NearestNeighbors

            # 创建体素网格
            voxel_size = 0.1  # 10cm体素

            # 计算每个点所属的体素
            voxel_indices = np.floor(points[:, :3] / voxel_size).astype(int)

            # 找到唯一体素
            unique_voxels, indices = np.unique(voxel_indices, axis=0, return_index=True)

            # 从每个体素中随机选择一个点
            if len(indices) > self.max_points_per_frame:
                # 如果需要进一步下采样，随机选择
                selected_indices = np.random.choice(
                    indices,
                    size=self.max_points_per_frame,
                    replace=False
                )
                downsampled = points[selected_indices]
            else:
                downsampled = points[indices]

            return downsampled

        except:
            # 如果体素下采样失败，使用随机下采样
            indices = np.random.choice(
                points.shape[0],
                size=self.max_points_per_frame,
                replace=False
            )
            return points[indices]

    def _estimate_batch_memory_mb(self):
        """估计批处理内存使用"""
        if not self.point_cloud_batch:
            return 0

        total_points = 0
        for _, points in self.point_cloud_batch:
            total_points += points.shape[0]

        # 估计内存：每个点4个float（x,y,z,intensity），加上开销
        memory_bytes = total_points * 4 * 4 * 1.2
        return memory_bytes / (1024 * 1024)

    def _check_memory_usage(self):
        """检查内存使用"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            if memory_mb > self.memory_warning_threshold:
                return True
            return False
        except:
            return False

    def _save_batch_async(self):
        """异步保存批处理数据"""
        if not self.point_cloud_batch:
            return

        # 复制当前批处理数据
        with self.data_lock:
            batch_to_save = self.point_cloud_batch.copy()
            self.point_cloud_batch.clear()

        # 异步保存
        self.thread_pool.submit(self._save_batch_sync, batch_to_save)

    def _save_batch_sync(self, batch_data):
        """同步保存批处理数据"""
        if not batch_data:
            return

        batch_list = []
        for frame_num, points in batch_data:
            batch_list.append({
                'frame_num': frame_num,
                'points': points.tolist(),
                'num_points': points.shape[0],
                'timestamp': datetime.now().timestamp()
            })

        min_frame = min([item[0] for item in batch_data])
        max_frame = max([item[0] for item in batch_data])
        batch_filename = f"lidar_batch_{min_frame:06d}_{max_frame:06d}.json"
        batch_path = os.path.join(self.lidar_dir, batch_filename)

        try:
            if self.enable_compression:
                json_str = json.dumps(batch_list)
                compressed_data = zlib.compress(
                    json_str.encode('utf-8'),
                    level=self.compression_level
                )

                # 计算压缩率
                original_size = len(json_str.encode('utf-8'))
                compressed_size = len(compressed_data)
                compression_ratio = compressed_size / max(1, original_size)
                self.processing_stats['compression_ratios'].append(compression_ratio)

                with open(batch_path, 'wb') as f:
                    f.write(compressed_data)
            else:
                with open(batch_path, 'w', encoding='utf-8') as f:
                    json.dump(batch_list, f)

        except Exception as e:
            pass

        gc.collect()

    def _save_point_cloud_files(self, points, frame_num):
        """保存点云文件（多种格式）"""
        save_results = {
            'bin': None,
            'npy': None,
            'pcd': None
        }

        # 1. 保存为.bin文件（KITTI格式）
        bin_path = os.path.join(self.lidar_dir, f"lidar_{frame_num:06d}.bin")
        try:
            # KITTI格式：x,y,z,intensity
            points_with_intensity = np.zeros((points.shape[0], 4), dtype=np.float32)
            if points.shape[1] >= 4:
                points_with_intensity[:, :] = points[:, :4]
            else:
                points_with_intensity[:, :3] = points[:, :3]
                points_with_intensity[:, 3] = 1.0

            points_with_intensity.tofile(bin_path)
            save_results['bin'] = bin_path
        except:
            save_results['bin'] = None

        # 2. 保存为.npy文件
        npy_path = os.path.join(self.lidar_dir, f"lidar_{frame_num:06d}.npy")
        try:
            np.save(npy_path, points)
            save_results['npy'] = npy_path
        except:
            save_results['npy'] = None

        # 3. 尝试保存为.pcd文件（点云库格式）
        pcd_path = os.path.join(self.lidar_dir, f"lidar_{frame_num:06d}.pcd")
        try:
            self._save_as_pcd(points, pcd_path)
            save_results['pcd'] = pcd_path
        except:
            save_results['pcd'] = None

        return save_results

    def _save_as_pcd(self, points, filepath):
        """保存为PCD格式"""
        try:
            with open(filepath, 'w') as f:
                f.write("# .PCD v0.7 - Point Cloud Data file format\n")
                f.write("VERSION 0.7\n")
                f.write("FIELDS x y z intensity\n")
                f.write("SIZE 4 4 4 4\n")
                f.write("TYPE F F F F\n")
                f.write("COUNT 1 1 1 1\n")
                f.write(f"WIDTH {points.shape[0]}\n")
                f.write("HEIGHT 1\n")
                f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {points.shape[0]}\n")
                f.write("DATA ascii\n")

                for i in range(points.shape[0]):
                    if points.shape[1] >= 4:
                        f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} {points[i, 3]:.6f}\n")
                    else:
                        f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} 1.0\n")
        except:
            pass

    def _save_as_v2xformer_format(self, points, frame_num):
        """保存为V2XFormer格式"""
        v2x_dir = os.path.join(self.output_dir, "v2xformer_format", "point_cloud")
        os.makedirs(v2x_dir, exist_ok=True)

        try:
            # 创建V2XFormer格式数据
            v2x_data = {
                'frame_id': frame_num,
                'timestamp': datetime.now().timestamp(),
                'sensor_id': 'lidar_main',
                'point_cloud': {
                    'points': points[:, :3].tolist(),
                    'intensities': points[:, 3].tolist() if points.shape[1] >= 4 else [1.0] * points.shape[0],
                    'num_points': points.shape[0],
                    'bounding_box': {
                        'min': [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 2].min())],
                        'max': [float(points[:, 0].max()), float(points[:, 1].max()), float(points[:, 2].max())]
                    }
                },
                'metadata': {
                    'sensor_type': 'lidar',
                    'format_version': '2.0',
                    'coordinate_system': 'carla_world',
                    'frame_rate': 10.0,
                    'pedestrian_detection': self.pedestrian_detection,
                    'compression': 'none'
                },
                'pedestrian_detections': self.pedestrian_points[-10:] if self.pedestrian_points else []
            }

            # 使用gzip压缩保存
            filename = f"{frame_num:06d}.pkl.gz"
            filepath = os.path.join(v2x_dir, filename)

            with gzip.open(filepath, 'wb') as f:
                pickle.dump(v2x_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            return filepath

        except Exception as e:
            return None

    def _generate_metadata(self, points, frame_num, save_results, v2xformer_path, original_count):
        """生成元数据（增强版）"""
        metadata = {
            'frame_id': frame_num,
            'timestamp': datetime.now().isoformat(),
            'processing_info': {
                'original_points': original_count,
                'processed_points': int(points.shape[0]),
                'downsampling_applied': self.enable_downsampling and original_count > self.max_points_per_frame,
                'downsample_ratio': self.downsample_ratio if self.enable_downsampling else 1.0,
                'compression_enabled': self.enable_compression,
                'compression_level': self.compression_level,
                'v2x_saved': v2xformer_path is not None,
                'pedestrian_detection': self.pedestrian_detection
            },
            'file_paths': {
                'bin': os.path.basename(save_results['bin']) if save_results['bin'] else None,
                'npy': os.path.basename(save_results['npy']) if save_results['npy'] else None,
                'pcd': os.path.basename(save_results['pcd']) if save_results['pcd'] else None,
                'v2xformer': os.path.basename(v2xformer_path) if v2xformer_path else None
            },
            'statistics': {
                'point_count': int(points.shape[0]),
                'x_range': [float(points[:, 0].min()), float(points[:, 0].max())],
                'y_range': [float(points[:, 1].min()), float(points[:, 1].max())],
                'z_range': [float(points[:, 2].min()), float(points[:, 2].max())],
                'mean_position': [float(points[:, 0].mean()), float(points[:, 1].mean()), float(points[:, 2].mean())],
                'std_position': [float(points[:, 0].std()), float(points[:, 1].std()), float(points[:, 2].std())],
                'density': points.shape[0] / max(1, (points[:, 0].max() - points[:, 0].min()) *
                                                 (points[:, 1].max() - points[:, 1].min()))
            }
        }

        # 如果有强度信息，添加强度统计
        if points.shape[1] >= 4:
            metadata['statistics']['intensity'] = {
                'min': float(points[:, 3].min()),
                'max': float(points[:, 3].max()),
                'mean': float(points[:, 3].mean()),
                'std': float(points[:, 3].std())
            }

        # 保存元数据文件
        meta_path = os.path.join(self.lidar_dir, f"lidar_meta_{frame_num:06d}.json")
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            pass

        return metadata

    def flush_batch(self):
        """刷新批处理数据"""
        if self.point_cloud_batch:
            self._save_batch_sync(self.point_cloud_batch.copy())
            self.point_cloud_batch.clear()

    def generate_lidar_summary(self):
        """生成LiDAR数据摘要（增强版）"""
        if not os.path.exists(self.lidar_dir):
            return None

        import glob

        # 查找各种文件
        bin_files = glob.glob(os.path.join(self.lidar_dir, "*.bin"))
        npy_files = glob.glob(os.path.join(self.lidar_dir, "*.npy"))
        pcd_files = glob.glob(os.path.join(self.lidar_dir, "*.pcd"))
        json_files = glob.glob(os.path.join(self.lidar_dir, "*.json"))
        batch_files = glob.glob(os.path.join(self.lidar_dir, "*batch*.json"))

        # V2XFormer格式文件
        v2x_dir = os.path.join(self.output_dir, "v2xformer_format", "point_cloud")
        v2x_files = glob.glob(os.path.join(v2x_dir, "*.pkl.gz")) if os.path.exists(v2x_dir) else []

        # 计算总点数和文件大小
        total_points = 0
        total_size = 0

        # 抽样检查文件
        sample_files = bin_files[:min(10, len(bin_files))]
        for bin_file in sample_files:
            try:
                file_size = os.path.getsize(bin_file)
                total_size += file_size

                # 估计点数：每个点4个float32 = 16字节
                points_in_file = file_size // 16
                total_points += points_in_file
            except:
                continue

        # 计算统计
        if sample_files and len(bin_files) > 0:
            avg_points_per_file = total_points / max(len(sample_files), 1)
            total_points_estimated = avg_points_per_file * len(bin_files)
        else:
            total_points_estimated = 0
            avg_points_per_file = 0

        # 处理统计
        avg_processing_time = np.mean(self.processing_stats['processing_times']) if self.processing_stats[
            'processing_times'] else 0
        avg_compression_ratio = np.mean(self.processing_stats['compression_ratios']) if self.processing_stats[
            'compression_ratios'] else 1.0

        summary = {
            'summary_time': datetime.now().isoformat(),
            'file_statistics': {
                'total_frames': len(bin_files),
                'bin_files': len(bin_files),
                'npy_files': len(npy_files),
                'pcd_files': len(pcd_files),
                'json_files': len(json_files),
                'batch_files': len(batch_files),
                'v2xformer_files': len(v2x_files)
            },
            'point_statistics': {
                'total_points_estimated': int(total_points_estimated),
                'average_points_per_frame': int(avg_points_per_file),
                'total_points_processed': self.processing_stats['total_points']
            },
            'size_statistics': {
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'average_file_size_kb': round((total_size / max(1, len(sample_files))) / 1024, 2),
                'compression_ratio': round(avg_compression_ratio, 3)
            },
            'processing_statistics': {
                'total_frames_processed': self.processing_stats['total_frames'],
                'average_processing_time_ms': round(avg_processing_time * 1000, 2),
                'error_count': self.processing_stats['errors'],
                'success_rate': round((self.processing_stats['total_frames'] - self.processing_stats['errors']) /
                                      max(1, self.processing_stats['total_frames']) * 100, 1)
            },
            'pedestrian_detection': {
                'enabled': self.pedestrian_detection,
                'detection_count': len(self.pedestrian_points)
            }
        }

        # 保存摘要
        summary_path = os.path.join(self.output_dir, "metadata", "lidar_summary.json")
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            pass

        return summary

    def cleanup(self):
        """清理资源"""
        self.flush_batch()
        self.thread_pool.shutdown(wait=True)
        gc.collect()


class MultiSensorFusion:
    """多传感器融合（增强版）"""

    def __init__(self, output_dir, config=None):
        self.output_dir = output_dir
        self.fusion_dir = os.path.join(output_dir, "fusion")
        os.makedirs(self.fusion_dir, exist_ok=True)

        self.calibration_data = {}
        self.fusion_cache = {}
        self.cache_size = config.get('fusion_cache_size', 100) if config else 100

        self.fusion_stats = {
            'total_fusions': 0,
            'cache_hits': 0,
            'processing_times': []
        }

        self._load_calibration()

    def _load_calibration(self):
        """加载标定数据（增强版）"""
        calibration_dir = os.path.join(self.output_dir, "calibration")

        if not os.path.exists(calibration_dir):
            return

        calibration_files = []
        for root, dirs, files in os.walk(calibration_dir):
            for file in files:
                if file.endswith('.json'):
                    calibration_files.append(os.path.join(root, file))

        for file in calibration_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                sensor_name = os.path.basename(file).replace('.json', '')
                self.calibration_data[sensor_name] = data

                # 解析传感器类型
                if 'sensor_type' in data:
                    sensor_type = data['sensor_type']
                    if sensor_type not in self.calibration_data:
                        self.calibration_data[sensor_type] = []
                    self.calibration_data[sensor_type].append(data)

            except Exception as e:
                pass

    def create_synchronization_file(self, frame_num, sensor_data):
        """创建同步文件（增强版）"""
        start_time = time.time()

        # 生成缓存键
        cache_key = f"{frame_num}_{hash(str(sorted(sensor_data.items())))}"

        if cache_key in self.fusion_cache:
            self.fusion_stats['cache_hits'] += 1
            return self.fusion_cache[cache_key]

        sync_data = {
            'frame_id': frame_num,
            'timestamp': datetime.now().timestamp(),
            'iso_timestamp': datetime.now().isoformat(),
            'sensors': {},
            'transformations': {},
            'synchronization_quality': 'good',
            'missing_sensors': []
        }

        # 处理每个传感器的数据
        for sensor_type, data_path in sensor_data.items():
            if data_path and os.path.exists(data_path):
                file_info = self._get_file_info(data_path)
                sync_data['sensors'][sensor_type] = file_info

                # 添加标定变换
                if sensor_type in self.calibration_data:
                    calib_data = self.calibration_data[sensor_type]
                    sync_data['transformations'][sensor_type] = {
                        'matrix': calib_data.get('matrix',
                                                 [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                        'calibration_source': 'loaded'
                    }
                else:
                    # 使用默认变换
                    sync_data['transformations'][sensor_type] = {
                        'matrix': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                        'calibration_source': 'default'
                    }
            else:
                sync_data['missing_sensors'].append(sensor_type)

        # 计算同步质量
        if len(sync_data['missing_sensors']) > 2:
            sync_data['synchronization_quality'] = 'poor'
        elif len(sync_data['missing_sensors']) > 0:
            sync_data['synchronization_quality'] = 'fair'

        # 添加传感器间的时间偏差估计
        sync_data['temporal_alignment'] = self._estimate_temporal_alignment(sync_data['sensors'])

        # 保存同步文件
        sync_file = os.path.join(self.fusion_dir, f"sync_{frame_num:06d}.json")

        try:
            # 根据数据大小选择是否压缩
            json_str = json.dumps(sync_data, separators=(',', ':'))

            if len(json_str) > 5000:  # 大于5KB时压缩
                compressed_data = zlib.compress(
                    json_str.encode('utf-8'),
                    level=3
                )
                sync_file = sync_file.replace('.json', '.json.gz')
                with open(sync_file, 'wb') as f:
                    f.write(compressed_data)
            else:
                with open(sync_file, 'w', encoding='utf-8') as f:
                    json.dump(sync_data, f, indent=2)

        except Exception as e:
            # 保存失败时使用简单格式
            try:
                with open(sync_file, 'w') as f:
                    f.write(json.dumps({'frame_id': frame_num, 'error': str(e)}))
            except:
                sync_file = None

        # 更新缓存
        if sync_file and len(self.fusion_cache) >= self.cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self.fusion_cache))
            del self.fusion_cache[oldest_key]

        if sync_file:
            self.fusion_cache[cache_key] = sync_file

        # 更新统计
        self.fusion_stats['total_fusions'] += 1
        processing_time = time.time() - start_time
        self.fusion_stats['processing_times'].append(processing_time)

        return sync_file

    def _get_file_info(self, filepath):
        """获取文件信息（增强版）"""
        file_info = {
            'file_path': os.path.basename(filepath),
            'file_size': os.path.getsize(filepath),
            'modified_time': os.path.getmtime(filepath),
            'file_type': os.path.splitext(filepath)[1][1:]
        }

        # 图像文件额外信息
        if filepath.endswith(('.png', '.jpg', '.jpeg')):
            try:
                import cv2
                img = cv2.imread(filepath)
                if img is not None:
                    file_info['dimensions'] = img.shape[:2]
                    file_info['channels'] = img.shape[2] if len(img.shape) > 2 else 1
                    file_info['dtype'] = str(img.dtype)
            except:
                pass

        # LiDAR文件额外信息
        elif filepath.endswith('.bin'):
            try:
                file_size = os.path.getsize(filepath)
                # 估计点数：每个点4个float32 = 16字节
                estimated_points = file_size // 16
                file_info['estimated_points'] = estimated_points
                file_info['point_format'] = 'xyz-intensity'
            except:
                pass

        return file_info

    def _estimate_temporal_alignment(self, sensors_info):
        """估计时间对齐"""
        alignment = {
            'max_time_difference': 0,
            'average_time_difference': 0,
            'alignment_score': 1.0
        }

        if len(sensors_info) < 2:
            return alignment

        # 获取修改时间
        mod_times = [info['modified_time'] for info in sensors_info.values()]

        if mod_times:
            max_diff = max(mod_times) - min(mod_times)
            avg_diff = np.mean([abs(t - np.mean(mod_times)) for t in mod_times])

            alignment['max_time_difference'] = max_diff
            alignment['average_time_difference'] = avg_diff

            # 计算对齐分数（差异越小分数越高）
            if max_diff > 1.0:  # 超过1秒
                alignment['alignment_score'] = 0.3
            elif max_diff > 0.1:  # 超过0.1秒
                alignment['alignment_score'] = 0.7
            else:
                alignment['alignment_score'] = 1.0

        return alignment

    def generate_fusion_report(self):
        """生成融合报告（增强版）"""
        if not os.path.exists(self.fusion_dir):
            return None

        import glob
        sync_files = glob.glob(os.path.join(self.fusion_dir, "*.json*"))

        frame_ids = []
        file_sizes = []
        compression_stats = {'compressed': 0, 'uncompressed': 0}

        for sync_file in sync_files[:50]:  # 只检查前50个文件
            try:
                filename = os.path.basename(sync_file)
                if filename.startswith('sync_'):
                    # 提取帧ID
                    frame_id_str = filename.split('_')[1].split('.')[0]
                    if frame_id_str.isdigit():
                        frame_id = int(frame_id_str)
                        frame_ids.append(frame_id)

                # 统计文件大小
                file_size = os.path.getsize(sync_file)
                file_sizes.append(file_size)

                # 统计压缩情况
                if sync_file.endswith('.gz'):
                    compression_stats['compressed'] += 1
                else:
                    compression_stats['uncompressed'] += 1

            except:
                continue

        # 计算处理统计
        avg_processing_time = np.mean(self.fusion_stats['processing_times']) if self.fusion_stats[
            'processing_times'] else 0
        cache_hit_rate = self.fusion_stats['cache_hits'] / max(1, self.fusion_stats['total_fusions'])

        report = {
            'report_time': datetime.now().isoformat(),
            'fusion_statistics': {
                'total_sync_frames': len(sync_files),
                'calibration_data_count': len(self.calibration_data),
                'sensor_types': list(set([data.get('sensor_type', 'unknown')
                                          for data in self.calibration_data.values()
                                          if isinstance(data, dict)])),
                'frame_range': [min(frame_ids), max(frame_ids)] if frame_ids else [],
                'cache_hit_rate': round(cache_hit_rate, 3),
                'average_processing_time_ms': round(avg_processing_time * 1000, 2)
            },
            'file_statistics': {
                'total_files': len(sync_files),
                'total_size_mb': round(sum(file_sizes) / (1024 * 1024), 2),
                'average_file_size_kb': round(np.mean(file_sizes) / 1024, 2) if file_sizes else 0,
                'compression_ratio': round(
                    sum([s for f, s in zip(sync_files, file_sizes) if f.endswith('.gz')]) /
                    max(1, sum([s for f, s in zip(sync_files, file_sizes) if not f.endswith('.gz')])), 2
                ) if sync_files else 1.0,
                'compression_stats': compression_stats
            },
            'calibration_summary': {
                'total_calibrations': len(self.calibration_data),
                'by_type': {}
            }
        }

        # 按类型统计标定数据
        for data in self.calibration_data.values():
            if isinstance(data, dict) and 'sensor_type' in data:
                sensor_type = data['sensor_type']
                report['calibration_summary']['by_type'][sensor_type] = \
                    report['calibration_summary']['by_type'].get(sensor_type, 0) + 1

        # 保存报告
        report_path = os.path.join(self.output_dir, "metadata", "fusion_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        return report

    def cleanup(self):
        """清理资源"""
        self.fusion_cache.clear()
        gc.collect()