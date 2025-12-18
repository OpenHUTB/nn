import numpy as np
import struct
import os
import json
import threading
from datetime import datetime


class LidarProcessor:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.lidar_dir = os.path.join(output_dir, "lidar")
        os.makedirs(self.lidar_dir, exist_ok=True)

        self.calibration_dir = os.path.join(output_dir, "calibration")
        os.makedirs(self.calibration_dir, exist_ok=True)

        self.frame_counter = 0
        self.data_lock = threading.Lock()

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
            self.frame_counter = frame_num

            points = self._carla_lidar_to_numpy(lidar_data)

            if points is None or points.shape[0] == 0:
                return None

            bin_path = self._save_as_bin(points, frame_num)
            npy_path = self._save_as_npy(points, frame_num)
            csv_path = self._save_as_csv(points, frame_num)

            metadata = self._generate_metadata(points, bin_path, npy_path, csv_path)

            return metadata

    def _carla_lidar_to_numpy(self, lidar_data):
        try:
            points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
            points = np.reshape(points, (int(points.shape[0] / 4), 4))

            points = points[:, :3]

            return points

        except Exception as e:
            print(f"LiDAR数据转换失败: {e}")

            try:
                points = []
                for point in lidar_data:
                    points.append([point.x, point.y, point.z])

                return np.array(points, dtype=np.float32)
            except:
                return None

    def _save_as_bin(self, points, frame_num):
        bin_path = os.path.join(self.lidar_dir, f"lidar_{frame_num:06d}.bin")

        points_with_intensity = np.zeros((points.shape[0], 4), dtype=np.float32)
        points_with_intensity[:, :3] = points
        points_with_intensity[:, 3] = 0.0

        points_with_intensity.tofile(bin_path)

        return bin_path

    def _save_as_npy(self, points, frame_num):
        npy_path = os.path.join(self.lidar_dir, f"lidar_{frame_num:06d}.npy")
        np.save(npy_path, points)
        return npy_path

    def _save_as_csv(self, points, frame_num):
        csv_path = os.path.join(self.lidar_dir, f"lidar_{frame_num:06d}.csv")

        sample_points = points[:min(1000, points.shape[0])]

        header = "x,y,z\n"

        with open(csv_path, 'w') as f:
            f.write(header)
            np.savetxt(f, sample_points, delimiter=',', fmt='%.6f')

        return csv_path

    def _generate_metadata(self, points, bin_path, npy_path, csv_path):
        metadata = {
            'frame_id': self.frame_counter,
            'timestamp': datetime.now().isoformat(),
            'point_count': int(points.shape[0]),
            'file_paths': {
                'bin': os.path.basename(bin_path),
                'npy': os.path.basename(npy_path),
                'csv': os.path.basename(csv_path)
            },
            'statistics': {
                'x_range': [float(points[:, 0].min()), float(points[:, 0].max())],
                'y_range': [float(points[:, 1].min()), float(points[:, 1].max())],
                'z_range': [float(points[:, 2].min()), float(points[:, 2].max())],
                'mean': [float(points[:, 0].mean()), float(points[:, 1].mean()), float(points[:, 2].mean())],
                'std': [float(points[:, 0].std()), float(points[:, 1].std()), float(points[:, 2].std())]
            },
            'file_sizes': {
                'bin': os.path.getsize(bin_path) if os.path.exists(bin_path) else 0,
                'npy': os.path.getsize(npy_path) if os.path.exists(npy_path) else 0,
                'csv': os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
            }
        }

        meta_path = os.path.join(self.lidar_dir, f"lidar_meta_{self.frame_counter:06d}.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def generate_lidar_summary(self):
        if not os.path.exists(self.lidar_dir):
            return None

        bin_files = [f for f in os.listdir(self.lidar_dir) if f.endswith('.bin')]
        npy_files = [f for f in os.listdir(self.lidar_dir) if f.endswith('.npy')]
        meta_files = [f for f in os.listdir(self.lidar_dir) if f.endswith('.json')]

        total_points = 0
        for bin_file in bin_files:
            bin_path = os.path.join(self.lidar_dir, bin_file)
            if os.path.exists(bin_path):
                file_size = os.path.getsize(bin_path)
                points_in_file = file_size // (4 * 4)
                total_points += points_in_file

        summary = {
            'total_frames': len(bin_files),
            'total_points': total_points,
            'average_points_per_frame': total_points // max(len(bin_files), 1),
            'file_types': {
                'bin': len(bin_files),
                'npy': len(npy_files),
                'meta': len(meta_files)
            },
            'total_size_mb': round(sum(os.path.getsize(os.path.join(self.lidar_dir, f))
                                       for f in bin_files + npy_files + meta_files) / (1024 * 1024), 2)
        }

        summary_path = os.path.join(self.output_dir, "metadata", "lidar_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary


class MultiSensorFusion:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.fusion_dir = os.path.join(output_dir, "fusion")
        os.makedirs(self.fusion_dir, exist_ok=True)

        self.calibration_data = {}
        self._load_calibration()

    def _load_calibration(self):
        calibration_dir = os.path.join(self.output_dir, "calibration")

        if os.path.exists(calibration_dir):
            for file in os.listdir(calibration_dir):
                if file.endswith('.json'):
                    filepath = os.path.join(calibration_dir, file)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        sensor_name = file.replace('.json', '')
                        self.calibration_data[sensor_name] = data
                    except:
                        pass

    def create_synchronization_file(self, frame_num, sensor_data):
        sync_data = {
            'frame_id': frame_num,
            'timestamp': datetime.now().isoformat(),
            'sensors': {}
        }

        for sensor_type, data_path in sensor_data.items():
            if data_path and os.path.exists(data_path):
                sync_data['sensors'][sensor_type] = {
                    'file_path': os.path.basename(data_path),
                    'file_size': os.path.getsize(data_path),
                    'timestamp': datetime.fromtimestamp(os.path.getctime(data_path)).isoformat()
                }

        sync_file = os.path.join(self.fusion_dir, f"sync_{frame_num:06d}.json")
        with open(sync_file, 'w') as f:
            json.dump(sync_data, f, indent=2)

        return sync_file

    def generate_fusion_report(self):
        if not os.path.exists(self.fusion_dir):
            return None

        sync_files = [f for f in os.listdir(self.fusion_dir) if f.endswith('.json')]

        report = {
            'total_sync_frames': len(sync_files),
            'calibration_data': list(self.calibration_data.keys()),
            'sensor_types': set(),
            'frame_range': []
        }

        if sync_files:
            frame_ids = []
            for sync_file in sync_files:
                try:
                    frame_id = int(sync_file.split('_')[1].split('.')[0])
                    frame_ids.append(frame_id)

                    filepath = os.path.join(self.fusion_dir, sync_file)
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    for sensor_type in data.get('sensors', {}).keys():
                        report['sensor_types'].add(sensor_type)

                except:
                    continue

            if frame_ids:
                report['frame_range'] = [min(frame_ids), max(frame_ids)]

        report['sensor_types'] = list(report['sensor_types'])

        report_path = os.path.join(self.output_dir, "metadata", "fusion_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report