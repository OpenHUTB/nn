import json
import os
import numpy as np
from datetime import datetime


class DataValidator:
    """数据验证器（增强版）"""

    @staticmethod
    def validate_dataset(data_dir, detailed=False):
        """验证数据集（增强版）"""
        print(f"\n{'=' * 60}")
        print(f"验证数据集: {data_dir}")
        print(f"{'=' * 60}")

        validation_results = {
            'validation_time': datetime.now().isoformat(),
            'dataset_path': data_dir,
            'directory_structure': DataValidator._check_directory_structure(data_dir),
            'raw_images': DataValidator._validate_raw_images(data_dir),
            'stitched_images': DataValidator._validate_stitched_images(data_dir),
            'annotations': DataValidator._validate_annotations(data_dir),
            'metadata': DataValidator._validate_metadata(data_dir),
            'lidar_data': DataValidator._validate_lidar_data(data_dir),
            'cooperative_data': DataValidator._validate_cooperative_data(data_dir),
            'fusion_data': DataValidator._validate_fusion_data(data_dir),
            'safety_data': DataValidator._validate_safety_data(data_dir),
            'enhanced_data': DataValidator._validate_enhanced_data(data_dir),
            'calibration_data': DataValidator._validate_calibration_data(data_dir),
            'timestamps': DataValidator._validate_timestamps(data_dir)
        }

        # 计算安全指标
        validation_results['safety_metrics'] = DataValidator._calculate_safety_metrics(data_dir)

        validation_results['overall_score'] = DataValidator._calculate_score(validation_results)
        validation_results['health_status'] = DataValidator._get_health_status(validation_results['overall_score'])

        # 详细验证
        if detailed:
            validation_results['detailed_analysis'] = DataValidator._detailed_analysis(data_dir)

        DataValidator._save_validation_report(data_dir, validation_results)
        DataValidator._print_validation_report(validation_results)

        return validation_results

    @staticmethod
    def _check_directory_structure(data_dir):
        """检查目录结构（增强版）"""
        required_dirs = [
            "raw/vehicle_1",
            "raw/infrastructure",
            "stitched",
            "metadata",
            "cooperative/v2x_messages",
            "cooperative/shared_perception",
            "fusion",
            "annotations"
        ]

        optional_dirs = [
            "lidar",
            "calibration",
            "safety_reports",
            "enhanced",
            "v2xformer_format",
            "kitti_format",
            "risk_maps",
            "emergency_events"
        ]

        missing_dirs = []
        for dir_path in required_dirs:
            full_path = os.path.join(data_dir, dir_path)
            if not os.path.exists(full_path):
                missing_dirs.append(dir_path)

        missing_optional = []
        for dir_path in optional_dirs:
            full_path = os.path.join(data_dir, dir_path)
            if not os.path.exists(full_path):
                missing_optional.append(dir_path)

        # 检查文件数量
        dir_stats = {}
        total_files = 0
        for root, dirs, files in os.walk(data_dir):
            if root.startswith(os.path.join(data_dir, '.')):
                continue  # 跳过隐藏目录

            rel_path = os.path.relpath(root, data_dir)
            if rel_path == '.':
                rel_path = 'root'

            dir_stats[rel_path] = len(files)
            total_files += len(files)

        status = 'PASS' if len(missing_dirs) == 0 else 'FAIL'

        result = {
            'status': status,
            'missing_directories': missing_dirs,
            'missing_optional_directories': missing_optional,
            'directory_stats': dir_stats,
            'total_files': total_files,
            'required_directories': required_dirs,
            'optional_directories': optional_dirs
        }

        return result

    @staticmethod
    def _validate_raw_images(data_dir):
        """验证原始图像（增强版）"""
        raw_path = os.path.join(data_dir, "raw")

        if not os.path.exists(raw_path):
            return {'vehicle': {'status': 'MISSING', 'count': 0, 'errors': [], 'sizes': []},
                    'infrastructure': {'status': 'MISSING', 'count': 0, 'errors': [], 'sizes': []}}

        raw_dirs = [d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))]
        results = {}

        for raw_dir in raw_dirs:
            path = os.path.join(data_dir, "raw", raw_dir)

            camera_dirs = []
            if os.path.exists(path):
                camera_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

            total_images = 0
            errors = []
            image_sizes = []

            for camera_dir in camera_dirs:
                camera_path = os.path.join(path, camera_dir)
                images = [f for f in os.listdir(camera_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

                for img_file in images:
                    img_path = os.path.join(camera_path, img_file)
                    try:
                        file_size = os.path.getsize(img_path)
                        if file_size == 0:
                            errors.append(f"空文件: {img_file}")
                        else:
                            image_sizes.append(file_size)
                    except:
                        errors.append(f"文件访问失败: {img_file}")

                total_images += len(images)

            if len(errors) == 0 and total_images > 0:
                status = 'PASS'
            elif len(errors) < 5 and total_images > 0:
                status = 'WARNING'
            else:
                status = 'FAIL'

            # 计算统计信息
            stats = {}
            if image_sizes:
                stats = {
                    'min_size_kb': min(image_sizes) / 1024,
                    'max_size_kb': max(image_sizes) / 1024,
                    'avg_size_kb': np.mean(image_sizes) / 1024,
                    'total_size_mb': sum(image_sizes) / (1024 * 1024)
                }

            results[raw_dir] = {
                'status': status,
                'count': total_images,
                'camera_count': len(camera_dirs),
                'errors': errors,
                'statistics': stats
            }

        return results

    @staticmethod
    def _validate_stitched_images(data_dir):
        """验证拼接图像（增强版）"""
        stitched_dir = os.path.join(data_dir, "stitched")

        if not os.path.exists(stitched_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': [], 'statistics': {}}

        images = [f for f in os.listdir(stitched_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        errors = []
        image_sizes = []

        for img_file in images:
            img_path = os.path.join(stitched_dir, img_file)
            try:
                file_size = os.path.getsize(img_path)
                if file_size == 0:
                    errors.append(f"空文件: {img_file}")
                else:
                    image_sizes.append(file_size)
            except:
                errors.append(f"文件访问失败: {img_file}")

        if len(errors) == 0 and len(images) > 0:
            status = 'PASS'
        elif len(errors) < 3 and len(images) > 0:
            status = 'WARNING'
        else:
            status = 'FAIL'

        # 计算统计信息
        stats = {}
        if image_sizes:
            stats = {
                'min_size_kb': min(image_sizes) / 1024,
                'max_size_kb': max(image_sizes) / 1024,
                'avg_size_kb': np.mean(image_sizes) / 1024,
                'total_size_mb': sum(image_sizes) / (1024 * 1024)
            }

        return {
            'status': status,
            'count': len(images),
            'errors': errors,
            'statistics': stats
        }

    @staticmethod
    def _validate_annotations(data_dir):
        """验证标注文件（增强版）"""
        annotations_dir = os.path.join(data_dir, "annotations")

        if not os.path.exists(annotations_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': [], 'frame_range': []}

        json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
        errors = []
        valid_files = 0
        frame_ids = []

        for json_file in json_files:
            json_path = os.path.join(annotations_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # 检查基本结构
                required_keys = ['frame_id', 'objects', 'safety_info']
                for key in required_keys:
                    if key not in data:
                        errors.append(f"缺失必要键: {key} in {json_file}")

                # 检查安全信息
                if 'safety_info' in data:
                    safety_info = data['safety_info']
                    if 'pedestrian_count' not in safety_info or 'vehicle_count' not in safety_info:
                        errors.append(f"缺失安全统计信息: {json_file}")

                # 记录帧ID
                if 'frame_id' in data:
                    frame_ids.append(data['frame_id'])

                valid_files += 1
            except Exception as e:
                errors.append(f"无效的JSON文件: {json_file} - {str(e)}")

        if len(errors) == 0 and valid_files > 0:
            status = 'PASS'
        elif len(errors) < 5 and valid_files > 0:
            status = 'WARNING'
        else:
            status = 'FAIL'

        # 计算帧范围
        frame_range = []
        if frame_ids:
            frame_range = [min(frame_ids), max(frame_ids)]

        return {
            'status': status,
            'count': valid_files,
            'total_files': len(json_files),
            'frame_range': frame_range,
            'errors': errors
        }

    @staticmethod
    def _validate_metadata(data_dir):
        """验证元数据（增强版）"""
        metadata_dir = os.path.join(data_dir, "metadata")

        if not os.path.exists(metadata_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': [], 'files': []}

        json_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
        errors = []
        valid_files = 0
        file_details = []

        required_files = ['collection_info.json', 'scene_description.json']
        missing_files = []

        for req_file in required_files:
            if req_file not in json_files:
                missing_files.append(req_file)

        for json_file in json_files:
            json_path = os.path.join(metadata_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # 检查文件大小
                file_size = os.path.getsize(json_path)

                file_details.append({
                    'file': json_file,
                    'size_kb': file_size / 1024,
                    'keys': list(data.keys())
                })

                valid_files += 1
            except Exception as e:
                errors.append(f"无效的JSON文件: {json_file} - {str(e)}")

        if len(errors) == 0 and valid_files > 0:
            status = 'PASS'
        elif len(errors) < 3 and valid_files > 0:
            status = 'WARNING'
        else:
            status = 'FAIL'

        if missing_files:
            errors.extend([f"缺失必要元数据文件: {f}" for f in missing_files])

        return {
            'status': status,
            'count': valid_files,
            'files': file_details,
            'missing_files': missing_files,
            'errors': errors
        }

    @staticmethod
    def _validate_lidar_data(data_dir):
        """验证LiDAR数据（增强版）"""
        lidar_dir = os.path.join(data_dir, "lidar")

        if not os.path.exists(lidar_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': [], 'statistics': {}}

        bin_files = [f for f in os.listdir(lidar_dir) if f.endswith('.bin')]
        npy_files = [f for f in os.listdir(lidar_dir) if f.endswith('.npy')]
        json_files = [f for f in os.listdir(lidar_dir) if f.endswith('.json')]

        errors = []
        valid_bin_files = 0
        file_sizes = []

        for bin_file in bin_files:
            bin_path = os.path.join(lidar_dir, bin_file)
            try:
                file_size = os.path.getsize(bin_path)
                if file_size > 0:
                    valid_bin_files += 1
                    file_sizes.append(file_size)
                else:
                    errors.append(f"空文件: {bin_file}")
            except:
                errors.append(f"文件访问失败: {bin_file}")

        total_files = len(bin_files) + len(npy_files) + len(json_files)

        if len(errors) == 0 and valid_bin_files > 0:
            status = 'PASS'
        elif len(errors) < 5 and valid_bin_files > 0:
            status = 'WARNING'
        else:
            status = 'FAIL'

        # 计算统计信息
        stats = {}
        if file_sizes:
            stats = {
                'min_size_mb': min(file_sizes) / (1024 * 1024),
                'max_size_mb': max(file_sizes) / (1024 * 1024),
                'avg_size_mb': np.mean(file_sizes) / (1024 * 1024),
                'total_size_gb': sum(file_sizes) / (1024 * 1024 * 1024)
            }

        return {
            'status': status,
            'count': total_files,
            'bin_files': len(bin_files),
            'npy_files': len(npy_files),
            'json_files': len(json_files),
            'valid_bin_files': valid_bin_files,
            'errors': errors,
            'statistics': stats
        }

    @staticmethod
    def _validate_cooperative_data(data_dir):
        """验证协同数据（增强版）"""
        coop_dir = os.path.join(data_dir, "cooperative")

        if not os.path.exists(coop_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': [], 'v2x_messages': 0, 'shared_perception': 0}

        errors = []

        # 检查V2X消息目录
        v2x_dir = os.path.join(coop_dir, "v2x_messages")
        v2x_files = []
        v2x_stats = {'total': 0, 'types': {}}

        if os.path.exists(v2x_dir):
            v2x_files = [f for f in os.listdir(v2x_dir) if f.endswith('.json')]
            v2x_stats['total'] = len(v2x_files)

            for v2x_file in v2x_files[:min(10, len(v2x_files))]:
                try:
                    with open(os.path.join(v2x_dir, v2x_file), 'r') as f:
                        data = json.load(f)

                    # 检查必要字段
                    required_keys = ['message', 'recipients', 'transmission_time']
                    for key in required_keys:
                        if key not in data:
                            errors.append(f"V2X消息缺失字段 {key}: {v2x_file}")

                    # 统计消息类型
                    if 'message' in data and 'message_type' in data['message']:
                        msg_type = data['message']['message_type']
                        v2x_stats['types'][msg_type] = v2x_stats['types'].get(msg_type, 0) + 1

                except Exception as e:
                    errors.append(f"V2X消息文件无效: {v2x_file} - {str(e)}")
        else:
            errors.append("V2X消息目录不存在")

        # 检查共享感知目录
        perception_dir = os.path.join(coop_dir, "shared_perception")
        perception_files = []
        perception_stats = {'total': 0, 'avg_objects': 0}

        if os.path.exists(perception_dir):
            perception_files = [f for f in os.listdir(perception_dir) if f.endswith('.json')]
            perception_stats['total'] = len(perception_files)

            object_counts = []
            for perception_file in perception_files[:min(10, len(perception_files))]:
                try:
                    with open(os.path.join(perception_dir, perception_file), 'r') as f:
                        data = json.load(f)

                    # 检查必要字段
                    required_keys = ['frame_id', 'timestamp', 'shared_objects']
                    for key in required_keys:
                        if key not in data:
                            errors.append(f"共享感知文件缺失字段 {key}: {perception_file}")

                    # 统计对象数量
                    if 'shared_objects' in data:
                        object_counts.append(len(data['shared_objects']))

                except Exception as e:
                    errors.append(f"共享感知文件无效: {perception_file} - {str(e)}")

            if object_counts:
                perception_stats['avg_objects'] = np.mean(object_counts)
        else:
            errors.append("共享感知目录不存在")

        # 检查协同摘要
        summary_file = os.path.join(coop_dir, "cooperative_summary.json")
        summary_valid = False
        if not os.path.exists(summary_file):
            errors.append("协同摘要文件不存在")
        else:
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)

                # 检查摘要字段
                required_keys = ['total_vehicles', 'ego_vehicles', 'cooperative_vehicles', 'v2x_stats']
                for key in required_keys:
                    if key not in data:
                        errors.append(f"协同摘要缺失字段: {key}")

                summary_valid = True
            except Exception as e:
                errors.append(f"协同摘要文件无效: {str(e)}")

        total_files = len(v2x_files) + len(perception_files)

        if len(errors) == 0 and total_files > 0 and summary_valid:
            status = 'PASS'
        elif len(errors) < 5 and total_files > 0:
            status = 'WARNING'
        else:
            status = 'FAIL'

        return {
            'status': status,
            'count': total_files,
            'v2x_messages': v2x_stats,
            'shared_perception': perception_stats,
            'summary_valid': summary_valid,
            'errors': errors
        }

    @staticmethod
    def _validate_fusion_data(data_dir):
        """验证融合数据（增强版）"""
        fusion_dir = os.path.join(data_dir, "fusion")

        if not os.path.exists(fusion_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': [], 'statistics': {}}

        json_files = [f for f in os.listdir(fusion_dir) if f.endswith('.json')]
        gz_files = [f for f in os.listdir(fusion_dir) if f.endswith('.gz')]
        all_files = json_files + gz_files

        errors = []
        valid_files = 0
        sensor_types = set()

        for json_file in all_files[:min(10, len(all_files))]:
            json_path = os.path.join(fusion_dir, json_file)
            try:
                # 处理压缩文件
                if json_file.endswith('.gz'):
                    import gzip
                    with gzip.open(json_path, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                # 检查必要字段
                required_keys = ['frame_id', 'timestamp', 'sensors']
                for key in required_keys:
                    if key not in data:
                        errors.append(f"融合文件缺失字段 {key}: {json_file}")

                # 收集传感器类型
                if 'sensors' in data:
                    sensor_types.update(data['sensors'].keys())

                valid_files += 1
            except Exception as e:
                errors.append(f"融合文件无效: {json_file} - {str(e)}")

        if len(errors) == 0 and valid_files > 0:
            status = 'PASS'
        elif len(errors) < 5 and valid_files > 0:
            status = 'WARNING'
        else:
            status = 'FAIL'

        return {
            'status': status,
            'count': len(all_files),
            'json_files': len(json_files),
            'gz_files': len(gz_files),
            'valid_files': valid_files,
            'sensor_types': list(sensor_types),
            'errors': errors
        }

    @staticmethod
    def _validate_safety_data(data_dir):
        """验证安全数据（增强版）"""
        safety_dir = os.path.join(data_dir, "safety_reports")

        if not os.path.exists(safety_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': [], 'risk_stats': {}}

        json_files = [f for f in os.listdir(safety_dir) if f.endswith('.json')]
        errors = []
        valid_files = 0
        risk_stats = {'high': 0, 'medium': 0, 'low': 0, 'critical': 0}

        for json_file in json_files[:min(10, len(json_files))]:
            json_path = os.path.join(safety_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # 检查必要字段
                required_keys = ['timestamp', 'total_interactions']
                for key in required_keys:
                    if key not in data:
                        errors.append(f"安全报告缺失字段 {key}: {json_file}")

                # 统计风险级别
                if 'risk_level' in data:
                    risk_level = data['risk_level']
                    if risk_level in risk_stats:
                        risk_stats[risk_level] += 1

                valid_files += 1
            except Exception as e:
                errors.append(f"安全报告无效: {json_file} - {str(e)}")

        if len(errors) == 0 and valid_files > 0:
            status = 'PASS'
        elif len(errors) < 5 and valid_files > 0:
            status = 'WARNING'
        else:
            status = 'FAIL'

        return {
            'status': status,
            'count': len(json_files),
            'valid_files': valid_files,
            'risk_stats': risk_stats,
            'errors': errors
        }

    @staticmethod
    def _validate_enhanced_data(data_dir):
        """验证增强数据"""
        enhanced_dir = os.path.join(data_dir, "enhanced")

        if not os.path.exists(enhanced_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': []}

        # 查找所有增强图像
        enhanced_files = []
        for root, dirs, files in os.walk(enhanced_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    enhanced_files.append(os.path.join(root, file))

        errors = []

        # 检查样本文件
        sample_files = enhanced_files[:min(5, len(enhanced_files))]
        for file_path in sample_files:
            try:
                if os.path.getsize(file_path) == 0:
                    errors.append(f"空增强文件: {os.path.basename(file_path)}")
            except:
                errors.append(f"文件访问失败: {os.path.basename(file_path)}")

        if len(errors) == 0 and len(enhanced_files) > 0:
            status = 'PASS'
        elif len(errors) < 3 and len(enhanced_files) > 0:
            status = 'WARNING'
        else:
            status = 'FAIL'

        return {
            'status': status,
            'count': len(enhanced_files),
            'errors': errors
        }

    @staticmethod
    def _validate_calibration_data(data_dir):
        """验证标定数据"""
        calib_dir = os.path.join(data_dir, "calibration")

        if not os.path.exists(calib_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': []}

        json_files = [f for f in os.listdir(calib_dir) if f.endswith('.json')]
        errors = []
        valid_files = 0

        for json_file in json_files:
            json_path = os.path.join(calib_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                valid_files += 1
            except Exception as e:
                errors.append(f"标定文件无效: {json_file} - {str(e)}")

        if len(errors) == 0 and valid_files > 0:
            status = 'PASS'
        elif len(errors) < 3 and valid_files > 0:
            status = 'WARNING'
        else:
            status = 'FAIL'

        return {
            'status': status,
            'count': valid_files,
            'errors': errors
        }

    @staticmethod
    def _validate_timestamps(data_dir):
        """验证时间戳连续性"""
        annotations_dir = os.path.join(data_dir, "annotations")

        if not os.path.exists(annotations_dir):
            return {'status': 'SKIPPED', 'errors': ['标注目录不存在']}

        json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
        if len(json_files) < 2:
            return {'status': 'SKIPPED', 'errors': ['标注文件不足']}

        timestamps = []
        errors = []

        # 按文件名排序
        json_files.sort()

        for json_file in json_files[:min(100, len(json_files))]:
            json_path = os.path.join(annotations_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                if 'timestamp' in data:
                    timestamps.append(data['timestamp'])
                elif 'frame_id' in data:
                    timestamps.append(data['frame_id'])
            except:
                continue

        if len(timestamps) < 2:
            return {'status': 'SKIPPED', 'errors': ['有效时间戳不足']}

        # 检查时间戳是否递增
        for i in range(1, len(timestamps)):
            if isinstance(timestamps[i], (int, float)) and isinstance(timestamps[i - 1], (int, float)):
                if timestamps[i] <= timestamps[i - 1]:
                    errors.append(f"时间戳不递增: {timestamps[i - 1]} -> {timestamps[i]}")

        if len(errors) == 0:
            status = 'PASS'
        elif len(errors) < 3:
            status = 'WARNING'
        else:
            status = 'FAIL'

        return {
            'status': status,
            'timestamp_count': len(timestamps),
            'errors': errors
        }

    @staticmethod
    def _calculate_safety_metrics(data_dir):
        """计算安全指标"""
        safety_dir = os.path.join(data_dir, "safety_reports")

        if not os.path.exists(safety_dir):
            return {'status': 'MISSING', 'metrics': {}}

        json_files = [f for f in os.listdir(safety_dir) if f.endswith('.json')]

        total_high_risk = 0
        total_interactions = 0
        safety_scores = []

        for json_file in json_files[:min(20, len(json_files))]:
            json_path = os.path.join(safety_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                if 'high_risk_cases' in data:
                    total_high_risk += data['high_risk_cases']
                if 'total_interactions' in data:
                    total_interactions += data['total_interactions']
                if 'safety_score' in data:
                    safety_scores.append(data['safety_score'])
            except:
                continue

        metrics = {}
        if total_interactions > 0:
            metrics['high_risk_ratio'] = total_high_risk / total_interactions
            metrics['interaction_density'] = total_interactions / len(json_files) if json_files else 0

        if safety_scores:
            metrics['avg_safety_score'] = np.mean(safety_scores)
            metrics['min_safety_score'] = min(safety_scores)
            metrics['max_safety_score'] = max(safety_scores)

        return {'status': 'CALCULATED', 'metrics': metrics}

    @staticmethod
    def _calculate_score(results):
        """计算总体评分（增强版）"""
        weights = {
            'directory_structure': 0.10,
            'raw_images': 0.15,
            'stitched_images': 0.05,
            'annotations': 0.10,
            'metadata': 0.08,
            'lidar_data': 0.10,
            'cooperative_data': 0.10,
            'fusion_data': 0.08,
            'safety_data': 0.12,
            'enhanced_data': 0.04,
            'calibration_data': 0.03,
            'timestamps': 0.03,
            'safety_metrics': 0.02
        }

        score = 0
        for key, weight in weights.items():
            if key not in results:
                score += 30 * weight
                continue

            result = results[key]
            if 'status' not in result:
                score += 30 * weight
                continue

            if result['status'] == 'PASS':
                score += 100 * weight
            elif result['status'] == 'WARNING':
                score += 70 * weight
            elif result['status'] == 'FAIL':
                score += 30 * weight
            elif result['status'] == 'MISSING':
                score += 20 * weight
            elif result['status'] == 'SKIPPED':
                score += 50 * weight
            else:
                score += 50 * weight

        return round(min(100, max(0, score)), 1)

    @staticmethod
    def _get_health_status(score):
        """获取健康状态"""
        if score >= 90:
            return 'EXCELLENT'
        elif score >= 75:
            return 'GOOD'
        elif score >= 60:
            return 'FAIR'
        elif score >= 40:
            return 'POOR'
        else:
            return 'CRITICAL'

    @staticmethod
    def _detailed_analysis(data_dir):
        """详细分析"""
        analysis = {
            'file_size_distribution': DataValidator._analyze_file_sizes(data_dir),
            'data_consistency': DataValidator._check_data_consistency(data_dir),
            'completeness': DataValidator._check_completeness(data_dir)
        }
        return analysis

    @staticmethod
    def _analyze_file_sizes(data_dir):
        """分析文件大小分布"""
        size_ranges = {
            'tiny': 0,  # < 1KB
            'small': 0,  # 1KB - 100KB
            'medium': 0,  # 100KB - 1MB
            'large': 0,  # 1MB - 10MB
            'huge': 0  # > 10MB
        }

        total_size = 0
        file_count = 0

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    total_size += size
                    file_count += 1

                    if size < 1024:
                        size_ranges['tiny'] += 1
                    elif size < 1024 * 100:
                        size_ranges['small'] += 1
                    elif size < 1024 * 1024:
                        size_ranges['medium'] += 1
                    elif size < 1024 * 1024 * 10:
                        size_ranges['large'] += 1
                    else:
                        size_ranges['huge'] += 1
                except:
                    continue

        return {
            'total_files': file_count,
            'total_size_gb': total_size / (1024 ** 3),
            'avg_file_size_kb': (total_size / max(1, file_count)) / 1024,
            'size_distribution': size_ranges
        }

    @staticmethod
    def _check_data_consistency(data_dir):
        """检查数据一致性"""
        # 检查标注文件和图像文件数量是否匹配
        annotations_dir = os.path.join(data_dir, "annotations")
        raw_dir = os.path.join(data_dir, "raw")

        annotation_count = 0
        image_count = 0

        if os.path.exists(annotations_dir):
            annotation_count = len([f for f in os.listdir(annotations_dir) if f.endswith('.json')])

        if os.path.exists(raw_dir):
            for root, dirs, files in os.walk(raw_dir):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        image_count += 1

        consistency = abs(annotation_count - image_count) <= max(annotation_count, image_count) * 0.1

        return {
            'annotation_count': annotation_count,
            'image_count': image_count,
            'consistency': consistency,
            'difference': abs(annotation_count - image_count)
        }

    @staticmethod
    def _check_completeness(data_dir):
        """检查数据完整性"""
        completeness = {}

        # 检查必要目录
        required_dirs = ['raw', 'annotations', 'metadata']
        for dir_name in required_dirs:
            dir_path = os.path.join(data_dir, dir_name)
            completeness[dir_name] = os.path.exists(dir_path) and len(os.listdir(dir_path)) > 0

        # 检查必要文件
        metadata_dir = os.path.join(data_dir, "metadata")
        if os.path.exists(metadata_dir):
            required_files = ['collection_info.json']
            for file_name in required_files:
                file_path = os.path.join(metadata_dir, file_name)
                completeness[file_name] = os.path.exists(file_path)

        return completeness

    @staticmethod
    def _save_validation_report(data_dir, results):
        """保存验证报告"""
        report_path = os.path.join(data_dir, "metadata", "validation_report.json")

        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n验证报告已保存: {report_path}")

    @staticmethod
    def _print_validation_report(results):
        """打印验证报告"""
        print("\n" + "=" * 60)
        print("数据集验证报告")
        print("=" * 60)

        # 总体信息
        print(f"\n📊 总体信息:")
        print(f"  数据集路径: {results['dataset_path']}")
        print(f"  验证时间: {results['validation_time']}")
        print(f"  总体评分: {results['overall_score']}/100")
        print(f"  健康状态: {results['health_status']}")

        # 各模块验证结果
        print(f"\n🔍 详细验证结果:")

        for key, result in results.items():
            if key in ['overall_score', 'health_status', 'dataset_path', 'validation_time',
                       'safety_metrics', 'detailed_analysis']:
                continue

            print(f"\n{key.replace('_', ' ').title()}:")

            if 'status' in result:
                status_icon = '✓' if result['status'] == 'PASS' else '⚠' if result['status'] in ['WARNING',
                                                                                                 'SKIPPED'] else '✗'
                print(f"  状态: {status_icon} {result['status']}")

            if 'count' in result:
                if isinstance(result['count'], dict):
                    print(f"  统计: {json.dumps(result['count'], indent=2)}")
                else:
                    print(f"  数量: {result['count']}")

            # 特定类型数据的额外信息
            if key == 'raw_images' and isinstance(result, dict):
                for subkey, subresult in result.items():
                    if isinstance(subresult, dict) and 'count' in subresult:
                        print(f"    {subkey}: {subresult['count']} 图像")
                        if 'statistics' in subresult and subresult['statistics']:
                            stats = subresult['statistics']
                            print(f"      大小: {stats.get('avg_size_kb', 0):.1f} KB/张")

            if key == 'safety_data' and isinstance(result, dict):
                if 'risk_stats' in result:
                    print(f"    风险统计: {json.dumps(result['risk_stats'], indent=2)}")

            if 'errors' in result and result['errors']:
                print(f"  错误 ({len(result['errors'])}):")
                for error in result['errors'][:3]:
                    print(f"    - {error}")
                if len(result['errors']) > 3:
                    print(f"    ... 还有 {len(result['errors']) - 3} 个错误")

        # 安全指标
        if 'safety_metrics' in results and results['safety_metrics']['status'] != 'MISSING':
            print(f"\n🛡️ 安全指标:")
            metrics = results['safety_metrics']['metrics']
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

        # 建议
        print(f"\n💡 建议:")
        overall_score = results.get('overall_score', 0)
        if overall_score >= 90:
            print("  ✓ 数据集质量优秀，可直接使用")
        elif overall_score >= 75:
            print("  ✓ 数据集质量良好，建议进行少量优化")
        elif overall_score >= 60:
            print("  ⚠ 数据集质量一般，建议进行优化")
            if results.get('directory_structure', {}).get('missing_directories'):
                print("    - 补全缺失的必要目录")
            if results.get('raw_images', {}).get('vehicle', {}).get('count', 0) < 10:
                print("    - 增加车辆图像数量")
        else:
            print("  ✗ 数据集质量需要重大改进")
            print("    - 检查数据采集过程")
            print("    - 验证传感器配置")
            print("    - 重新收集关键数据")

        print("\n" + "=" * 60)