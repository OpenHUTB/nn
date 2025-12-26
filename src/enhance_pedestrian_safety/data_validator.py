import json
import os


class DataValidator:

    @staticmethod
    def validate_dataset(data_dir):
        print(f"验证数据集: {data_dir}")

        validation_results = {
            'directory_structure': DataValidator._check_directory_structure(data_dir),
            'raw_images': DataValidator._validate_raw_images(data_dir),
            'stitched_images': DataValidator._validate_stitched_images(data_dir),
            'annotations': DataValidator._validate_annotations(data_dir),
            'metadata': DataValidator._validate_metadata(data_dir),
            'lidar_data': DataValidator._validate_lidar_data(data_dir),
            'cooperative_data': DataValidator._validate_cooperative_data(data_dir),
            'fusion_data': DataValidator._validate_fusion_data(data_dir),
            'safety_data': DataValidator._validate_safety_data(data_dir)
        }

        validation_results['overall_score'] = DataValidator._calculate_score(validation_results)

        DataValidator._save_validation_report(data_dir, validation_results)
        DataValidator._print_validation_report(validation_results)

        return validation_results

    @staticmethod
    def _check_directory_structure(data_dir):
        required_dirs = [
            "raw/vehicle",
            "raw/infrastructure",
            "stitched",
            "metadata",
            "cooperative",
            "cooperative/v2x_messages",
            "cooperative/shared_perception",
            "fusion"
        ]

        optional_dirs = [
            "lidar",
            "calibration",
            "annotations",
            "safety_reports"
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

        status = 'PASS' if len(missing_dirs) == 0 else 'FAIL'

        result = {
            'status': status,
            'missing_directories': missing_dirs,
            'missing_optional_directories': missing_optional,
            'required_directories': required_dirs,
            'optional_directories': optional_dirs
        }

        return result

    @staticmethod
    def _validate_raw_images(data_dir):
        raw_path = os.path.join(data_dir, "raw")

        if not os.path.exists(raw_path):
            return {'vehicle': {'status': 'MISSING', 'count': 0, 'errors': []},
                    'infrastructure': {'status': 'MISSING', 'count': 0, 'errors': []}}

        raw_dirs = [d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))]
        results = {}

        for raw_dir in raw_dirs:
            path = os.path.join(data_dir, "raw", raw_dir)

            camera_dirs = []
            if os.path.exists(path):
                camera_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

            total_images = 0
            errors = []

            for camera_dir in camera_dirs:
                camera_path = os.path.join(path, camera_dir)
                images = [f for f in os.listdir(camera_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

                for img_file in images:
                    img_path = os.path.join(camera_path, img_file)
                    try:
                        if os.path.getsize(img_path) == 0:
                            errors.append(f"空文件: {img_file}")
                    except:
                        errors.append(f"文件访问失败: {img_file}")

                total_images += len(images)

            if len(errors) == 0 and total_images > 0:
                status = 'PASS'
            elif len(errors) < 3 and total_images > 0:
                status = 'WARNING'
            else:
                status = 'FAIL'

            results[raw_dir] = {
                'status': status,
                'count': total_images,
                'errors': errors
            }

        return results

    @staticmethod
    def _validate_stitched_images(data_dir):
        stitched_dir = os.path.join(data_dir, "stitched")

        if not os.path.exists(stitched_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': []}

        images = [f for f in os.listdir(stitched_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        errors = []

        for img_file in images:
            img_path = os.path.join(stitched_dir, img_file)
            try:
                if os.path.getsize(img_path) == 0:
                    errors.append(f"空文件: {img_file}")
            except:
                errors.append(f"文件访问失败: {img_file}")

        if len(errors) == 0 and len(images) > 0:
            status = 'PASS'
        elif len(errors) < 3 and len(images) > 0:
            status = 'WARNING'
        else:
            status = 'FAIL'

        return {
            'status': status,
            'count': len(images),
            'errors': errors
        }

    @staticmethod
    def _validate_annotations(data_dir):
        annotations_dir = os.path.join(data_dir, "annotations")

        if not os.path.exists(annotations_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': []}

        json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
        errors = []
        valid_files = 0

        for json_file in json_files:
            json_path = os.path.join(annotations_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # 检查基本结构
                required_keys = ['frame_id', 'objects']
                for key in required_keys:
                    if key not in data:
                        errors.append(f"缺失必要键: {key} in {json_file}")

                valid_files += 1
            except Exception as e:
                errors.append(f"无效的JSON文件: {json_file} - {str(e)}")

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
    def _validate_metadata(data_dir):
        metadata_dir = os.path.join(data_dir, "metadata")

        if not os.path.exists(metadata_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': []}

        json_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
        errors = []
        valid_files = 0

        for json_file in json_files:
            json_path = os.path.join(metadata_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                valid_files += 1
            except Exception as e:
                errors.append(f"无效的JSON文件: {json_file} - {str(e)}")

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
    def _validate_lidar_data(data_dir):
        lidar_dir = os.path.join(data_dir, "lidar")

        if not os.path.exists(lidar_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': []}

        bin_files = [f for f in os.listdir(lidar_dir) if f.endswith('.bin')]
        npy_files = [f for f in os.listdir(lidar_dir) if f.endswith('.npy')]
        json_files = [f for f in os.listdir(lidar_dir) if f.endswith('.json')]

        errors = []
        valid_bin_files = 0

        for bin_file in bin_files:
            bin_path = os.path.join(lidar_dir, bin_file)
            try:
                if os.path.getsize(bin_path) > 0:
                    valid_bin_files += 1
                else:
                    errors.append(f"空文件: {bin_file}")
            except:
                errors.append(f"文件访问失败: {bin_file}")

        total_files = len(bin_files) + len(npy_files) + len(json_files)

        if len(errors) == 0 and valid_bin_files > 0:
            status = 'PASS'
        elif len(errors) < 3 and valid_bin_files > 0:
            status = 'WARNING'
        else:
            status = 'FAIL'

        return {
            'status': status,
            'count': total_files,
            'bin_files': len(bin_files),
            'npy_files': len(npy_files),
            'json_files': len(json_files),
            'errors': errors
        }

    @staticmethod
    def _validate_cooperative_data(data_dir):
        """验证协同数据"""
        coop_dir = os.path.join(data_dir, "cooperative")

        if not os.path.exists(coop_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': [], 'v2x_messages': 0, 'shared_perception': 0}

        errors = []

        # 检查V2X消息目录
        v2x_dir = os.path.join(coop_dir, "v2x_messages")
        v2x_files = []
        if os.path.exists(v2x_dir):
            v2x_files = [f for f in os.listdir(v2x_dir) if f.endswith('.json')]
            for v2x_file in v2x_files[:min(5, len(v2x_files))]:  # 检查前5个文件
                try:
                    with open(os.path.join(v2x_dir, v2x_file), 'r') as f:
                        data = json.load(f)

                    # 检查必要字段
                    required_keys = ['message', 'recipients', 'transmission_time']
                    for key in required_keys:
                        if key not in data:
                            errors.append(f"V2X消息缺失字段 {key}: {v2x_file}")
                except Exception as e:
                    errors.append(f"V2X消息文件无效: {v2x_file} - {str(e)}")
        else:
            errors.append("V2X消息目录不存在")

        # 检查共享感知目录
        perception_dir = os.path.join(coop_dir, "shared_perception")
        perception_files = []
        if os.path.exists(perception_dir):
            perception_files = [f for f in os.listdir(perception_dir) if f.endswith('.json')]
            for perception_file in perception_files[:min(5, len(perception_files))]:
                try:
                    with open(os.path.join(perception_dir, perception_file), 'r') as f:
                        data = json.load(f)

                    # 检查必要字段
                    required_keys = ['frame_id', 'timestamp', 'shared_objects']
                    for key in required_keys:
                        if key not in data:
                            errors.append(f"共享感知文件缺失字段 {key}: {perception_file}")
                except Exception as e:
                    errors.append(f"共享感知文件无效: {perception_file} - {str(e)}")
        else:
            errors.append("共享感知目录不存在")

        # 检查协同摘要
        summary_file = os.path.join(coop_dir, "cooperative_summary.json")
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
            except Exception as e:
                errors.append(f"协同摘要文件无效: {str(e)}")

        total_files = len(v2x_files) + len(perception_files)

        if len(errors) == 0 and total_files > 0:
            status = 'PASS'
        elif len(errors) < 3 and total_files > 0:
            status = 'WARNING'
        else:
            status = 'FAIL'

        return {
            'status': status,
            'count': total_files,
            'v2x_messages': len(v2x_files),
            'shared_perception': len(perception_files),
            'errors': errors
        }

    @staticmethod
    def _validate_fusion_data(data_dir):
        """验证融合数据"""
        fusion_dir = os.path.join(data_dir, "fusion")

        if not os.path.exists(fusion_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': []}

        json_files = [f for f in os.listdir(fusion_dir) if f.endswith('.json')]
        errors = []
        valid_files = 0

        for json_file in json_files[:min(5, len(json_files))]:  # 检查前5个文件
            json_path = os.path.join(fusion_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # 检查必要字段
                required_keys = ['frame_id', 'timestamp', 'sensors']
                for key in required_keys:
                    if key not in data:
                        errors.append(f"融合文件缺失字段 {key}: {json_file}")

                valid_files += 1
            except Exception as e:
                errors.append(f"融合文件无效: {json_file} - {str(e)}")

        if len(errors) == 0 and valid_files > 0:
            status = 'PASS'
        elif len(errors) < 3 and valid_files > 0:
            status = 'WARNING'
        else:
            status = 'FAIL'

        return {
            'status': status,
            'count': len(json_files),
            'errors': errors
        }

    @staticmethod
    def _validate_safety_data(data_dir):
        """验证安全数据"""
        safety_dir = os.path.join(data_dir, "safety_reports")

        if not os.path.exists(safety_dir):
            return {'status': 'MISSING', 'count': 0, 'errors': []}

        json_files = [f for f in os.listdir(safety_dir) if f.endswith('.json')]
        errors = []
        valid_files = 0

        for json_file in json_files[:min(5, len(json_files))]:
            json_path = os.path.join(safety_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # 检查必要字段
                required_keys = ['timestamp', 'total_interactions']
                for key in required_keys:
                    if key not in data:
                        errors.append(f"安全报告缺失字段 {key}: {json_file}")

                valid_files += 1
            except Exception as e:
                errors.append(f"安全报告无效: {json_file} - {str(e)}")

        if len(errors) == 0 and valid_files > 0:
            status = 'PASS'
        elif len(errors) < 3 and valid_files > 0:
            status = 'WARNING'
        else:
            status = 'FAIL'

        return {
            'status': status,
            'count': len(json_files),
            'errors': errors
        }

    @staticmethod
    def _calculate_score(results):
        weights = {
            'directory_structure': 0.10,
            'raw_images': 0.15,
            'stitched_images': 0.05,
            'annotations': 0.08,
            'metadata': 0.08,
            'lidar_data': 0.12,
            'cooperative_data': 0.12,
            'fusion_data': 0.10,
            'safety_data': 0.20
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
            elif result['status'] in ['FAIL', 'MISSING']:
                score += 30 * weight
            else:
                score += 50 * weight

        return round(score, 1)

    @staticmethod
    def _save_validation_report(data_dir, results):
        report_path = os.path.join(data_dir, "metadata", "validation_report.json")

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _print_validation_report(results):
        print("\n" + "=" * 60)
        print("数据集验证报告")
        print("=" * 60)

        for key, result in results.items():
            if key == 'overall_score':
                continue

            print(f"\n{key.replace('_', ' ').title()}:")

            if 'status' in result:
                status_icon = '✓' if result['status'] == 'PASS' else '⚠' if result['status'] == 'WARNING' else '✗'
                print(f"  状态: {status_icon} {result['status']}")
            else:
                print(f"  状态: 未知")

            if 'count' in result:
                print(f"  数量: {result['count']}")

            # 特定类型数据的额外信息
            if key == 'raw_images' and isinstance(result, dict):
                for subkey, subresult in result.items():
                    if isinstance(subresult, dict) and 'count' in subresult:
                        print(f"    {subkey}: {subresult['count']} 图像")

            if key == 'cooperative_data' and isinstance(result, dict):
                if 'v2x_messages' in result:
                    print(f"    V2X消息: {result['v2x_messages']}")
                if 'shared_perception' in result:
                    print(f"    共享感知: {result['shared_perception']}")

            if key == 'lidar_data' and isinstance(result, dict):
                if 'bin_files' in result:
                    print(f"    BIN文件: {result['bin_files']}")
                if 'npy_files' in result:
                    print(f"    NPY文件: {result['npy_files']}")
                if 'json_files' in result:
                    print(f"    JSON文件: {result['json_files']}")

            if key == 'safety_data' and isinstance(result, dict):
                if 'count' in result:
                    print(f"    安全报告: {result['count']} 个")

            if 'errors' in result and result['errors']:
                print(f"  错误 ({len(result['errors'])}):")
                for error in result['errors'][:3]:
                    print(f"    - {error}")
                if len(result['errors']) > 3:
                    print(f"    ... 还有 {len(result['errors']) - 3} 个错误")

        print(f"\n总体评分: {results.get('overall_score', 0)}/100")

        overall_score = results.get('overall_score', 0)
        if overall_score >= 90:
            print("✓ 数据集质量优秀")
        elif overall_score >= 70:
            print("✓ 数据集质量良好")
        elif overall_score >= 50:
            print("⚠ 数据集质量一般")
        else:
            print("✗ 数据集质量需要改进")

        print("=" * 60)