import os
import json


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
            'lidar_data': DataValidator._validate_lidar_data(data_dir)
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
            "metadata"
        ]

        missing_dirs = []
        for dir_path in required_dirs:
            full_path = os.path.join(data_dir, dir_path)
            if not os.path.exists(full_path):
                missing_dirs.append(dir_path)

        status = 'PASS' if len(missing_dirs) == 0 else 'FAIL'

        result = {
            'status': status,
            'missing_directories': missing_dirs,
            'required_directories': required_dirs
        }

        return result

    @staticmethod
    def _validate_raw_images(data_dir):
        raw_dirs = ["vehicle", "infrastructure"]
        results = {}

        for raw_dir in raw_dirs:
            path = os.path.join(data_dir, "raw", raw_dir)
            if not os.path.exists(path):
                results[raw_dir] = {'status': 'MISSING', 'count': 0, 'errors': []}
                continue

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
        errors = []
        valid_files = 0

        for bin_file in bin_files:
            bin_path = os.path.join(lidar_dir, bin_file)
            try:
                if os.path.getsize(bin_path) > 0:
                    valid_files += 1
                else:
                    errors.append(f"空文件: {bin_file}")
            except:
                errors.append(f"文件访问失败: {bin_file}")

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
    def _calculate_score(results):
        weights = {
            'directory_structure': 0.15,
            'raw_images': 0.25,
            'stitched_images': 0.15,
            'annotations': 0.1,
            'metadata': 0.15,
            'lidar_data': 0.2
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
                print(f"  状态: {result['status']}")
            else:
                print(f"  状态: 未知")

            if 'count' in result:
                print(f"  数量: {result['count']}")

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