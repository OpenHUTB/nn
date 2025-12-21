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


class LidarProcessor:

    def __init__(self, output_dir, config=None):
        self.output_dir = output_dir
        self.lidar_dir = os.path.join(output_dir, "lidar")
        os.makedirs(self.lidar_dir, exist_ok=True)

        self.calibration_dir = os.path.join(output_dir, "calibration")
        os.makedirs(self.calibration_dir, exist_ok=True)

        self.frame_counter = 0
        self.data_lock = threading.Lock()

        # 性能优化：批量处理设置
        self.batch_size = config.get('batch_size', 10) if config else 10
        self.point_cloud_batch = []
        self.enable_compression = config.get('enable_compression', True) if config else True
        self.compression_level = config.get('compression_level', 3) if config else 3

        # 内存管理 - 优化点3：更保守的内存设置
        self.max_points_per_frame = config.get('max_points_per_frame', 50000) if config else 50000  # 减少到5万点
        self.enable_downsampling = config.get('enable_downsampling', True) if config else True
        self.downsample_ratio = config.get('downsample_ratio', 0.3) if config else 0.3  # 增加下采样比例

        # 新增：内存监控和限制
        self.memory_warning_threshold = config.get('memory_warning_threshold', 300) if config else 300  # MB
        self.max_batch_memory_mb = config.get('max_batch_memory_mb', 50) if config else 50  # 批次最大内存
        self.v2x_save_interval = config.get('v2x_save_interval', 5) if config else 5  # 每5帧保存一次V2X格式

        self._init_calibration_files()

    def _init_calibration_files(self):
        lidar_intrinsic = {
            "channels": 32,
            "range": 100.0,
            "points_per_second": 56000,
            "rotation_frequency": 10.0,
            "horizontal_fov": 360.0,
            "vertical_fov": 30.0,
            "upper_fov": 10.0,
            "lower_fov": -20.0
        }

        lidar_extrinsic = {
            "translation": [0.0, 0.0, 2.5],
            "rotation": [0.0, 0.0, 0.0],
            "matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        }

        intrinsic_file = os.path.join(self.calibration_dir, "lidar_intrinsic.json")
        extrinsic_file = os.path.join(self.calibration_dir, "lidar_extrinsic.json")

        with open(intrinsic_file, 'w') as f:
            json.dump(lidar_intrinsic, f, indent=2)

        with open(extrinsic_file, 'w') as f:
            json.dump(lidar_extrinsic, f, indent=2)

    def process_lidar_data(self, lidar_data, frame_num):
        with self.data_lock:
            try:
                self.frame_counter = frame_num

                points = self._carla_lidar_to_numpy(lidar_data)

                if points is None or points.shape[0] == 0:
                    print(f"警告: LiDAR数据为空或无效 (帧 {frame_num})")
                    return None

                # 检查内存使用情况
                if self._check_memory_usage():
                    print(f"警告: 内存使用过高，跳过LiDAR处理 (帧 {frame_num})")
                    return None

                # 下采样以减少内存使用
                original_count = points.shape[0]
                if self.enable_downsampling and points.shape[0] > self.max_points_per_frame:
                    points = self._downsample_point_cloud(points)
                    print(f"LiDAR下采样: {original_count} -> {points.shape[0]} 点 (帧 {frame_num})")

                # 检查批次内存使用
                batch_memory_mb = self._estimate_batch_memory_mb()
                if batch_memory_mb > self.max_batch_memory_mb:
                    print(f"批处理内存过高 ({batch_memory_mb:.1f}MB)，强制保存当前批次")
                    self._save_batch()

                # 添加到批处理
                self.point_cloud_batch.append((frame_num, points))

                # 如果达到批处理大小，保存批处理数据
                if len(self.point_cloud_batch) >= self.batch_size:
                    self._save_batch()

                # 保存单个文件（向后兼容）
                bin_path = self._save_as_bin(points, frame_num)
                npy_path = self._save_as_npy(points, frame_num)

                # 生成V2XFormer兼容格式 - 优化点：每v2x_save_interval帧保存一次
                v2xformer_path = None
                if frame_num % self.v2x_save_interval == 0:
                    try:
                        v2xformer_path = self._save_as_v2xformer_format(points, frame_num)
                    except Exception as e:
                        print(f"警告: V2XFormer格式保存失败: {e}")
                        v2xformer_path = None

                metadata = self._generate_metadata(points, bin_path, npy_path, v2xformer_path)

                # 定期清理内存
                if frame_num % 20 == 0:
                    gc.collect()

                return metadata

            except Exception as e:
                print(f"处理LiDAR数据失败: {e}")
                # 强制清理内存
                gc.collect()
                return None

    def _carla_lidar_to_numpy(self, lidar_data):
        """高效转换LiDAR数据"""
        try:
            # 使用内存视图减少内存分配
            points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
            points = np.reshape(points, (int(points.shape[0] / 4), 4))

            # 只保留位置信息（x, y, z），忽略强度
            points = points[:, :3]

            # 清理临时数组
            del lidar_data
            gc.collect()

            return points

        except Exception as e:
            print(f"LiDAR数据转换失败: {e}")

            # 备选转换方法
            try:
                points = []
                for i in range(0, len(lidar_data), 4):
                    point = lidar_data[i:i + 4]
                    points.append([point.x, point.y, point.z])

                return np.array(points, dtype=np.float32)
            except:
                return None

    def _downsample_point_cloud(self, points):
        """下采样点云以减少内存使用"""
        if points.shape[0] <= self.max_points_per_frame:
            return points

        # 随机下采样
        indices = np.random.choice(points.shape[0],
                                   int(points.shape[0] * self.downsample_ratio),
                                   replace=False)
        downsampled = points[indices]

        # 清理内存
        del points
        gc.collect()

        return downsampled

    def _estimate_batch_memory_mb(self):
        """估计批处理内存使用"""
        if not self.point_cloud_batch:
            return 0

        total_points = 0
        for _, points in self.point_cloud_batch:
            total_points += points.shape[0]

        # 每个点3个float32 (12字节)，加上一些额外开销
        memory_bytes = total_points * 12 * 1.2  # 20%额外开销
        return memory_bytes / (1024 * 1024)

    def _check_memory_usage(self):
        """检查内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            if memory_mb > self.memory_warning_threshold:
                print(f"警告: 进程内存使用过高: {memory_mb:.1f}MB")
                return True
            return False
        except:
            return False

    def _save_batch(self):
        """批量保存点云数据（提高效率）"""
        if not self.point_cloud_batch:
            return

        batch_data = []
        for frame_num, points in self.point_cloud_batch:
            # 只保存点的坐标，不保存完整对象
            batch_data.append({
                'frame_num': frame_num,
                'points': points.tolist(),
                'num_points': points.shape[0]
            })

        # 生成批处理文件名
        min_frame = min([item[0] for item in self.point_cloud_batch])
        max_frame = max([item[0] for item in self.point_cloud_batch])
        batch_filename = f"lidar_batch_{min_frame:06d}_{max_frame:06d}.json"
        batch_path = os.path.join(self.lidar_dir, batch_filename)

        # 压缩保存
        if self.enable_compression:
            try:
                json_str = json.dumps(batch_data)
                compressed_data = zlib.compress(
                    json_str.encode('utf-8'),
                    level=self.compression_level
                )
                with open(batch_path, 'wb') as f:
                    f.write(compressed_data)
            except Exception as e:
                print(f"压缩保存批处理数据失败: {e}")
                # 尝试非压缩保存
                try:
                    with open(batch_path, 'w') as f:
                        json.dump(batch_data, f)
                except Exception as e2:
                    print(f"非压缩保存批处理数据失败: {e2}")
        else:
            try:
                with open(batch_path, 'w') as f:
                    json.dump(batch_data, f)
            except Exception as e:
                print(f"保存批处理数据失败: {e}")

        # 清理批处理缓存
        self.point_cloud_batch.clear()
        gc.collect()

        print(f"批量保存LiDAR数据: {batch_filename}")

    def _save_as_bin(self, points, frame_num):
        """保存为二进制格式（兼容KITTI）"""
        bin_path = os.path.join(self.lidar_dir, f"lidar_{frame_num:06d}.bin")

        # 添加强度信息（全为1）
        points_with_intensity = np.zeros((points.shape[0], 4), dtype=np.float32)
        points_with_intensity[:, :3] = points
        points_with_intensity[:, 3] = 1.0  # 强度值

        try:
            points_with_intensity.tofile(bin_path)
        except Exception as e:
            print(f"保存BIN文件失败: {e}")
            return None

        return bin_path

    def _save_as_npy(self, points, frame_num):
        """保存为numpy格式"""
        npy_path = os.path.join(self.lidar_dir, f"lidar_{frame_num:06d}.npy")

        try:
            # 使用内存映射减少内存使用
            memmap_array = np.lib.format.open_memmap(
                npy_path,
                mode='w+',
                dtype=np.float32,
                shape=points.shape
            )
            memmap_array[:] = points[:]
            memmap_array.flush()
        except Exception as e:
            print(f"保存NPY文件失败: {e}")
            # 备选保存方式
            try:
                np.save(npy_path, points)
            except Exception as e2:
                print(f"备选保存NPY文件失败: {e2}")
                return None

        return npy_path

    def _save_as_v2xformer_format(self, points, frame_num):
        """保存为V2XFormer兼容格式 - 优化点：减少保存频率"""
        v2x_dir = os.path.join(self.output_dir, "v2xformer_format")
        os.makedirs(v2x_dir, exist_ok=True)

        try:
            # 创建V2XFormer格式的数据结构
            v2x_data = {
                'frame_id': frame_num,
                'timestamp': datetime.now().timestamp(),
                'point_cloud': {
                    'points': points.tolist(),
                    'num_points': points.shape[0],
                    'range': [float(points[:, 0].min()), float(points[:, 0].max()),
                              float(points[:, 1].min()), float(points[:, 1].max()),
                              float(points[:, 2].min()), float(points[:, 2].max())]
                },
                'metadata': {
                    'sensor_type': 'lidar',
                    'format_version': '1.0',
                    'coordinate_system': 'carla_world',
                    'frame_rate': 10.0
                }
            }

            # 保存为压缩的JSON格式
            filename = f"{frame_num:06d}.pkl.gz"
            filepath = os.path.join(v2x_dir, filename)

            # 使用pickle+gzip压缩保存
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(v2x_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"保存V2XFormer格式: {filepath} ({file_size:.2f} KB)")

            return filepath

        except Exception as e:
            print(f"警告: 保存V2XFormer格式失败: {e}")
            return None

    def _generate_metadata(self, points, bin_path, npy_path, v2xformer_path):
        metadata = {
            'frame_id': self.frame_counter,
            'timestamp': datetime.now().isoformat(),
            'point_count': int(points.shape[0]),
            'file_paths': {
                'bin': os.path.basename(bin_path) if bin_path else None,
                'npy': os.path.basename(npy_path) if npy_path else None,
                'v2xformer': os.path.basename(v2xformer_path) if v2xformer_path else None
            },
            'statistics': {
                'x_range': [float(points[:, 0].min()), float(points[:, 0].max())],
                'y_range': [float(points[:, 1].min()), float(points[:, 1].max())],
                'z_range': [float(points[:, 2].min()), float(points[:, 2].max())],
                'mean': [float(points[:, 0].mean()), float(points[:, 1].mean()), float(points[:, 2].mean())],
                'std': [float(points[:, 0].std()), float(points[:, 1].std()), float(points[:, 2].std())]
            },
            'processing_info': {
                'downsampled': self.enable_downsampling and points.shape[0] > self.max_points_per_frame,
                'downsample_ratio': self.downsample_ratio if self.enable_downsampling else 1.0,
                'compression_enabled': self.enable_compression,
                'compression_level': self.compression_level,
                'v2x_saved': v2xformer_path is not None
            }
        }

        meta_path = os.path.join(self.lidar_dir, f"lidar_meta_{self.frame_counter:06d}.json")
        try:
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"保存LiDAR元数据失败: {e}")

        return metadata

    def flush_batch(self):
        """强制刷新批处理数据"""
        if self.point_cloud_batch:
            self._save_batch()

    def generate_lidar_summary(self):
        """生成LiDAR数据摘要（优化版）"""
        if not os.path.exists(self.lidar_dir):
            return None

        # 快速统计文件
        import glob
        bin_files = glob.glob(os.path.join(self.lidar_dir, "*.bin"))
        npy_files = glob.glob(os.path.join(self.lidar_dir, "*.npy"))
        batch_files = glob.glob(os.path.join(self.lidar_dir, "*batch*.json"))
        v2x_files = glob.glob(os.path.join(self.output_dir, "v2xformer_format", "*.pkl.gz"))

        total_points = 0
        total_size = 0

        # 采样统计，避免读取所有文件
        sample_files = bin_files[:5] if len(bin_files) > 5 else bin_files  # 减少采样数量
        for bin_file in sample_files:
            if os.path.exists(bin_file):
                try:
                    file_size = os.path.getsize(bin_file)
                    total_size += file_size
                    # 估算点数
                    points_in_file = file_size // (4 * 4)  # 4个float32，每个4字节
                    total_points += points_in_file
                except:
                    continue

        # 根据采样估算总数
        if sample_files and len(bin_files) > 0:
            avg_points_per_file = total_points / max(len(sample_files), 1)
            total_points_estimated = avg_points_per_file * len(bin_files)
        else:
            total_points_estimated = 0

        summary = {
            'total_frames': len(bin_files),
            'total_points_estimated': int(total_points_estimated),
            'average_points_per_frame': int(avg_points_per_file) if sample_files else 0,
            'file_types': {
                'bin': len(bin_files),
                'npy': len(npy_files),
                'batch': len(batch_files),
                'v2xformer': len(v2x_files)
            },
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'compression_ratio': round(sum([os.path.getsize(f) for f in batch_files]) /
                                       max(1, sum([os.path.getsize(f) for f in bin_files])), 2)
            if bin_files and batch_files else 1.0
        }

        summary_path = os.path.join(self.output_dir, "metadata", "lidar_summary.json")
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"保存LiDAR摘要失败: {e}")

        return summary


class MultiSensorFusion:
    """多传感器融合（优化版）"""

    def __init__(self, output_dir, config=None):
        self.output_dir = output_dir
        self.fusion_dir = os.path.join(output_dir, "fusion")
        os.makedirs(self.fusion_dir, exist_ok=True)

        self.calibration_data = {}
        self.fusion_cache = {}  # 融合缓存
        self.cache_size = config.get('fusion_cache_size', 50) if config else 50  # 减少缓存大小

        self._load_calibration()

    def _load_calibration(self):
        calibration_dir = os.path.join(self.output_dir, "calibration")

        if os.path.exists(calibration_dir):
            # 批量读取校准文件
            calibration_files = []
            for root, dirs, files in os.walk(calibration_dir):
                for file in files:
                    if file.endswith('.json'):
                        calibration_files.append(os.path.join(root, file))

            # 限制同时读取的文件数量
            for file in calibration_files[:10]:  # 最多读取10个文件
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    sensor_name = os.path.basename(file).replace('.json', '')
                    self.calibration_data[sensor_name] = data
                except Exception as e:
                    print(f"加载校准文件 {file} 失败: {e}")

    def create_synchronization_file(self, frame_num, sensor_data):
        """创建同步文件（优化版）"""
        # 检查缓存
        cache_key = f"{frame_num}_{hash(str(sensor_data))}"
        if cache_key in self.fusion_cache:
            return self.fusion_cache[cache_key]

        sync_data = {
            'frame_id': frame_num,
            'timestamp': datetime.now().timestamp(),  # 使用时间戳而不是ISO格式
            'sensors': {},
            'transformations': {}
        }

        # 批量处理传感器数据
        for sensor_type, data_path in sensor_data.items():
            if data_path and os.path.exists(data_path):
                # 获取文件信息（不加载完整文件）
                file_info = {
                    'file_path': os.path.basename(data_path),
                    'file_size': os.path.getsize(data_path),
                    'modified_time': os.path.getmtime(data_path)
                }

                # 如果是图像，获取尺寸信息
                if data_path.endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        import cv2
                        img = cv2.imread(data_path)
                        if img is not None:
                            file_info['dimensions'] = img.shape[:2]
                    except:
                        pass

                sync_data['sensors'][sensor_type] = file_info

                # 添加变换信息（如果可用）
                if sensor_type in self.calibration_data:
                    sync_data['transformations'][sensor_type] = {
                        'matrix': self.calibration_data[sensor_type].get('matrix',
                                                                         [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                                                          [0, 0, 0, 1]])
                    }

        # 压缩保存
        sync_file = os.path.join(self.fusion_dir, f"sync_{frame_num:06d}.json")

        # 使用压缩的JSON
        if len(str(sync_data)) > 1000:  # 如果数据较大，压缩保存
            compressed_data = zlib.compress(
                json.dumps(sync_data).encode('utf-8'),
                level=3
            )
            sync_file = sync_file.replace('.json', '.json.gz')
            with open(sync_file, 'wb') as f:
                f.write(compressed_data)
        else:
            with open(sync_file, 'w') as f:
                json.dump(sync_data, f, separators=(',', ':'))  # 紧凑格式

        # 更新缓存
        if len(self.fusion_cache) >= self.cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self.fusion_cache))
            del self.fusion_cache[oldest_key]

        self.fusion_cache[cache_key] = sync_file

        return sync_file

    def generate_fusion_report(self):
        """生成融合报告（优化版）"""
        if not os.path.exists(self.fusion_dir):
            return None

        # 快速扫描文件
        import glob
        sync_files = glob.glob(os.path.join(self.fusion_dir, "*.json*"))

        # 解析文件名获取帧范围（避免读取所有文件）
        frame_ids = []
        for sync_file in sync_files[:20]:  # 只检查前20个文件
            try:
                # 从文件名提取帧号
                filename = os.path.basename(sync_file)
                if filename.startswith('sync_'):
                    frame_id = int(filename.split('_')[1].split('.')[0])
                    frame_ids.append(frame_id)
            except:
                continue

        report = {
            'total_sync_frames': len(sync_files),
            'calibration_data_count': len(self.calibration_data),
            'sensor_types': list(set([f.split('_')[0] for f in self.calibration_data.keys()])),
            'frame_range': [min(frame_ids), max(frame_ids)] if frame_ids else [],
            'file_sizes': {
                'total_mb': round(sum([os.path.getsize(f) for f in sync_files]) / (1024 * 1024), 2),
                'average_kb': round(np.mean([os.path.getsize(f) / 1024 for f in sync_files]), 2) if sync_files else 0
            },
            'compression_info': {
                'compressed_files': len([f for f in sync_files if f.endswith('.gz')]),
                'compression_ratio': round(
                    sum([os.path.getsize(f) for f in sync_files if f.endswith('.gz')]) /
                    max(1, sum([os.path.getsize(f) for f in sync_files if not f.endswith('.gz')])),
                    2
                ) if sync_files else 1.0
            }
        }

        report_path = os.path.join(self.output_dir, "metadata", "fusion_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report