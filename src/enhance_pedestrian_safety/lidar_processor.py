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

        self.batch_size = config.get('batch_size', 10) if config else 10
        self.point_cloud_batch = []
        self.enable_compression = config.get('enable_compression', True) if config else True
        self.compression_level = config.get('compression_level', 3) if config else 3

        self.max_points_per_frame = config.get('max_points_per_frame', 50000) if config else 50000
        self.enable_downsampling = config.get('enable_downsampling', True) if config else True
        self.downsample_ratio = config.get('downsample_ratio', 0.3) if config else 0.3

        self.memory_warning_threshold = config.get('memory_warning_threshold', 300) if config else 300
        self.max_batch_memory_mb = config.get('max_batch_memory_mb', 50) if config else 50
        self.v2x_save_interval = config.get('v2x_save_interval', 5) if config else 5

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
                    return None

                if self._check_memory_usage():
                    return None

                original_count = points.shape[0]
                if self.enable_downsampling and points.shape[0] > self.max_points_per_frame:
                    points = self._downsample_point_cloud(points)

                batch_memory_mb = self._estimate_batch_memory_mb()
                if batch_memory_mb > self.max_batch_memory_mb:
                    self._save_batch()

                self.point_cloud_batch.append((frame_num, points))

                if len(self.point_cloud_batch) >= self.batch_size:
                    self._save_batch()

                bin_path = self._save_as_bin(points, frame_num)
                npy_path = self._save_as_npy(points, frame_num)

                v2xformer_path = None
                if frame_num % self.v2x_save_interval == 0:
                    try:
                        v2xformer_path = self._save_as_v2xformer_format(points, frame_num)
                    except Exception as e:
                        v2xformer_path = None

                metadata = self._generate_metadata(points, bin_path, npy_path, v2xformer_path)

                if frame_num % 20 == 0:
                    gc.collect()

                return metadata

            except Exception as e:
                gc.collect()
                return None

    def _carla_lidar_to_numpy(self, lidar_data):
        try:
            points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            points = points[:, :3]

            del lidar_data
            gc.collect()

            return points

        except Exception as e:
            try:
                points = []
                for i in range(0, len(lidar_data), 4):
                    point = lidar_data[i:i + 4]
                    points.append([point.x, point.y, point.z])
                return np.array(points, dtype=np.float32)
            except:
                return None

    def _downsample_point_cloud(self, points):
        if points.shape[0] <= self.max_points_per_frame:
            return points

        indices = np.random.choice(points.shape[0],
                                   int(points.shape[0] * self.downsample_ratio),
                                   replace=False)
        downsampled = points[indices]

        del points
        gc.collect()

        return downsampled

    def _estimate_batch_memory_mb(self):
        if not self.point_cloud_batch:
            return 0

        total_points = 0
        for _, points in self.point_cloud_batch:
            total_points += points.shape[0]

        memory_bytes = total_points * 12 * 1.2
        return memory_bytes / (1024 * 1024)

    def _check_memory_usage(self):
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            if memory_mb > self.memory_warning_threshold:
                return True
            return False
        except:
            return False

    def _save_batch(self):
        if not self.point_cloud_batch:
            return

        batch_data = []
        for frame_num, points in self.point_cloud_batch:
            batch_data.append({
                'frame_num': frame_num,
                'points': points.tolist(),
                'num_points': points.shape[0]
            })

        min_frame = min([item[0] for item in self.point_cloud_batch])
        max_frame = max([item[0] for item in self.point_cloud_batch])
        batch_filename = f"lidar_batch_{min_frame:06d}_{max_frame:06d}.json"
        batch_path = os.path.join(self.lidar_dir, batch_filename)

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
                try:
                    with open(batch_path, 'w') as f:
                        json.dump(batch_data, f)
                except Exception as e2:
                    pass
        else:
            try:
                with open(batch_path, 'w') as f:
                    json.dump(batch_data, f)
            except Exception as e:
                pass

        self.point_cloud_batch.clear()
        gc.collect()

    def _save_as_bin(self, points, frame_num):
        bin_path = os.path.join(self.lidar_dir, f"lidar_{frame_num:06d}.bin")

        points_with_intensity = np.zeros((points.shape[0], 4), dtype=np.float32)
        points_with_intensity[:, :3] = points
        points_with_intensity[:, 3] = 1.0

        try:
            points_with_intensity.tofile(bin_path)
        except Exception as e:
            return None

        return bin_path

    def _save_as_npy(self, points, frame_num):
        npy_path = os.path.join(self.lidar_dir, f"lidar_{frame_num:06d}.npy")

        try:
            memmap_array = np.lib.format.open_memmap(
                npy_path,
                mode='w+',
                dtype=np.float32,
                shape=points.shape
            )
            memmap_array[:] = points[:]
            memmap_array.flush()
        except Exception as e:
            try:
                np.save(npy_path, points)
            except Exception as e2:
                return None

        return npy_path

    def _save_as_v2xformer_format(self, points, frame_num):
        v2x_dir = os.path.join(self.output_dir, "v2xformer_format")
        os.makedirs(v2x_dir, exist_ok=True)

        try:
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

            filename = f"{frame_num:06d}.pkl.gz"
            filepath = os.path.join(v2x_dir, filename)

            with gzip.open(filepath, 'wb') as f:
                pickle.dump(v2x_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            return filepath

        except Exception as e:
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
            pass

        return metadata

    def flush_batch(self):
        if self.point_cloud_batch:
            self._save_batch()

    def generate_lidar_summary(self):
        if not os.path.exists(self.lidar_dir):
            return None

        import glob
        bin_files = glob.glob(os.path.join(self.lidar_dir, "*.bin"))
        npy_files = glob.glob(os.path.join(self.lidar_dir, "*.npy"))
        batch_files = glob.glob(os.path.join(self.lidar_dir, "*batch*.json"))
        v2x_files = glob.glob(os.path.join(self.output_dir, "v2xformer_format", "*.pkl.gz"))

        total_points = 0
        total_size = 0

        sample_files = bin_files[:5] if len(bin_files) > 5 else bin_files
        for bin_file in sample_files:
            if os.path.exists(bin_file):
                try:
                    file_size = os.path.getsize(bin_file)
                    total_size += file_size
                    points_in_file = file_size // (4 * 4)
                    total_points += points_in_file
                except:
                    continue

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
            pass

        return summary


class MultiSensorFusion:

    def __init__(self, output_dir, config=None):
        self.output_dir = output_dir
        self.fusion_dir = os.path.join(output_dir, "fusion")
        os.makedirs(self.fusion_dir, exist_ok=True)

        self.calibration_data = {}
        self.fusion_cache = {}
        self.cache_size = config.get('fusion_cache_size', 50) if config else 50

        self._load_calibration()

    def _load_calibration(self):
        calibration_dir = os.path.join(self.output_dir, "calibration")

        if os.path.exists(calibration_dir):
            calibration_files = []
            for root, dirs, files in os.walk(calibration_dir):
                for file in files:
                    if file.endswith('.json'):
                        calibration_files.append(os.path.join(root, file))

            for file in calibration_files[:10]:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    sensor_name = os.path.basename(file).replace('.json', '')
                    self.calibration_data[sensor_name] = data
                except Exception as e:
                    pass

    def create_synchronization_file(self, frame_num, sensor_data):
        cache_key = f"{frame_num}_{hash(str(sensor_data))}"
        if cache_key in self.fusion_cache:
            return self.fusion_cache[cache_key]

        sync_data = {
            'frame_id': frame_num,
            'timestamp': datetime.now().timestamp(),
            'sensors': {},
            'transformations': {}
        }

        for sensor_type, data_path in sensor_data.items():
            if data_path and os.path.exists(data_path):
                file_info = {
                    'file_path': os.path.basename(data_path),
                    'file_size': os.path.getsize(data_path),
                    'modified_time': os.path.getmtime(data_path)
                }

                if data_path.endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        import cv2
                        img = cv2.imread(data_path)
                        if img is not None:
                            file_info['dimensions'] = img.shape[:2]
                    except:
                        pass

                sync_data['sensors'][sensor_type] = file_info

                if sensor_type in self.calibration_data:
                    sync_data['transformations'][sensor_type] = {
                        'matrix': self.calibration_data[sensor_type].get('matrix',
                                                                         [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                                                          [0, 0, 0, 1]])
                    }

        sync_file = os.path.join(self.fusion_dir, f"sync_{frame_num:06d}.json")

        if len(str(sync_data)) > 1000:
            compressed_data = zlib.compress(
                json.dumps(sync_data).encode('utf-8'),
                level=3
            )
            sync_file = sync_file.replace('.json', '.json.gz')
            with open(sync_file, 'wb') as f:
                f.write(compressed_data)
        else:
            with open(sync_file, 'w') as f:
                json.dump(sync_data, f, separators=(',', ':'))

        if len(self.fusion_cache) >= self.cache_size:
            oldest_key = next(iter(self.fusion_cache))
            del self.fusion_cache[oldest_key]

        self.fusion_cache[cache_key] = sync_file

        return sync_file

    def generate_fusion_report(self):
        if not os.path.exists(self.fusion_dir):
            return None

        import glob
        sync_files = glob.glob(os.path.join(self.fusion_dir, "*.json*"))

        frame_ids = []
        for sync_file in sync_files[:20]:
            try:
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