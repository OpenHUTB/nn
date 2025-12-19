import os
import json
import numpy as np
from collections import defaultdict


class DataAnalyzer:
    """数据分析器 - 生成数据集统计信息"""

    @staticmethod
    def analyze_dataset(data_dir):
        """分析数据集并生成详细报告"""
        print(f"分析数据集: {data_dir}")

        analysis = {
            'basic_stats': DataAnalyzer._get_basic_stats(data_dir),
            'file_distribution': DataAnalyzer._analyze_file_distribution(data_dir),
            'object_statistics': DataAnalyzer._analyze_objects(data_dir),
            'temporal_analysis': DataAnalyzer._analyze_temporal(data_dir),

            'cooperative_data': DataAnalyzer._analyze_cooperative_data(data_dir),


            'quality_metrics': DataAnalyzer._calculate_quality_metrics(data_dir)
        }

        # 生成评分
        analysis['overall_score'] = DataAnalyzer._calculate_overall_score(analysis)

        # 保存分析结果
        DataAnalyzer._save_analysis_report(data_dir, analysis)

        # 打印摘要
        DataAnalyzer._print_analysis_summary(analysis)

        return analysis

    @staticmethod
    def _get_basic_stats(data_dir):
        """获取基本统计信息"""
        stats = {
            'total_size_mb': DataAnalyzer._get_directory_size(data_dir),
            'file_count': 0,
            'directory_count': 0,
            'data_types': defaultdict(int)
        }

        for root, dirs, files in os.walk(data_dir):
            stats['directory_count'] += len(dirs)
            stats['file_count'] += len(files)

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in ['.png', '.jpg', '.jpeg']:
                    stats['data_types']['images'] += 1
                elif ext == '.json':
                    stats['data_types']['json'] += 1
                elif ext == '.txt':
                    stats['data_types']['text'] += 1
                elif ext == '.bin':
                    stats['data_types']['binary'] += 1
                else:
                    stats['data_types']['other'] += 1

        return stats

    @staticmethod
    def _get_directory_size(path):
        """计算目录大小"""
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return round(total / (1024 * 1024), 2)

    @staticmethod
    def _analyze_file_distribution(data_dir):
        """分析文件分布"""
        distribution = {}

        # 分析原始图像

        raw_dirs = []
        raw_path = os.path.join(data_dir, "raw")
        if os.path.exists(raw_path):
            raw_dirs = [d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))]

        for raw_dir in raw_dirs:
            path = os.path.join(data_dir, "raw", raw_dir)
            camera_stats = {}
            for camera_dir in os.listdir(path):
                camera_path = os.path.join(path, camera_dir)
                if os.path.isdir(camera_path):
                    images = [f for f in os.listdir(camera_path)
                              if f.endswith(('.png', '.jpg', '.jpeg'))]
                    camera_stats[camera_dir] = len(images)
            distribution[f'raw_{raw_dir}'] = camera_stats

        raw_dirs = ['vehicle', 'infrastructure']
        for raw_dir in raw_dirs:
            path = os.path.join(data_dir, "raw", raw_dir)
            if os.path.exists(path):
                camera_stats = {}
                for camera_dir in os.listdir(path):
                    camera_path = os.path.join(path, camera_dir)
                    if os.path.isdir(camera_path):
                        images = [f for f in os.listdir(camera_path)
                                  if f.endswith(('.png', '.jpg', '.jpeg'))]
                        camera_stats[camera_dir] = len(images)
                distribution[f'raw_{raw_dir}'] = camera_stats


        # 分析拼接图像
        stitched_dir = os.path.join(data_dir, "stitched")
        if os.path.exists(stitched_dir):
            stitched_images = [f for f in os.listdir(stitched_dir)
                               if f.endswith(('.jpg', '.jpeg', '.png'))]
            distribution['stitched'] = len(stitched_images)

        # 分析标注文件
        annotations_dir = os.path.join(data_dir, "annotations")
        if os.path.exists(annotations_dir):
            json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
            distribution['annotations'] = len(json_files)


        # 分析LiDAR数据
        lidar_dir = os.path.join(data_dir, "lidar")
        if os.path.exists(lidar_dir):
            bin_files = [f for f in os.listdir(lidar_dir) if f.endswith('.bin')]
            npy_files = [f for f in os.listdir(lidar_dir) if f.endswith('.npy')]
            distribution['lidar'] = {'bin': len(bin_files), 'npy': len(npy_files)}

        # 分析融合数据
        fusion_dir = os.path.join(data_dir, "fusion")
        if os.path.exists(fusion_dir):
            json_files = [f for f in os.listdir(fusion_dir) if f.endswith('.json')]
            distribution['fusion'] = len(json_files)



        return distribution

    @staticmethod
    def _analyze_objects(data_dir):
        """分析物体统计"""
        annotations_dir = os.path.join(data_dir, "annotations")

        if not os.path.exists(annotations_dir):

            return {'total_objects': 0, 'by_class': {}, 'by_frame': {}, 'class_distribution': {}}

            return {'total_objects': 0, 'by_class': {}, 'by_frame': {}}


        object_stats = {
            'total_objects': 0,
            'by_class': defaultdict(int),
            'by_frame': defaultdict(int),
            'class_distribution': {}
        }

        json_files = [f for f in os.listdir(annotations_dir)
                      if f.endswith('.json') and f.startswith('frame_')]

        for json_file in json_files:
            try:
                with open(os.path.join(annotations_dir, json_file), 'r') as f:
                    data = json.load(f)

                frame_id = data.get('frame_id', 0)
                objects = data.get('objects', [])

                object_stats['by_frame'][frame_id] = len(objects)
                object_stats['total_objects'] += len(objects)

                for obj in objects:
                    obj_class = obj.get('class', 'unknown')
                    object_stats['by_class'][obj_class] += 1

            except Exception as e:
                print(f"分析标注文件 {json_file} 失败: {e}")

        # 计算类分布百分比
        if object_stats['total_objects'] > 0:
            for obj_class, count in object_stats['by_class'].items():
                object_stats['class_distribution'][obj_class] = round(
                    count / object_stats['total_objects'] * 100, 2)

        return object_stats

    @staticmethod
    def _analyze_temporal(data_dir):
        """分析时间分布"""
        temporal_stats = {
            'frame_intervals': [],
            'total_duration': 0,
            'frame_rate': 0
        }

        metadata_file = os.path.join(data_dir, "metadata", "collection_info.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                collection_stats = metadata.get('collection_stats', {})
                temporal_stats['total_duration'] = collection_stats.get('duration_seconds', 0)
                temporal_stats['frame_rate'] = collection_stats.get('average_fps', 0)

            except Exception as e:
                print(f"分析元数据失败: {e}")

        return temporal_stats

    @staticmethod

    def _analyze_cooperative_data(data_dir):
        """分析协同数据"""
        coop_dir = os.path.join(data_dir, "cooperative")

        if not os.path.exists(coop_dir):
            return {
                'v2x_messages': 0,
                'shared_perception': 0,
                'vehicles_count': 0,
                'communication_stats': {}
            }

        # V2X消息统计
        v2x_dir = os.path.join(coop_dir, "v2x_messages")
        v2x_files = []
        if os.path.exists(v2x_dir):
            v2x_files = [f for f in os.listdir(v2x_dir) if f.endswith('.json')]

        # 共享感知统计
        perception_dir = os.path.join(coop_dir, "shared_perception")
        perception_files = []
        if os.path.exists(perception_dir):
            perception_files = [f for f in os.listdir(perception_dir) if f.endswith('.json')]

        # 读取协同摘要
        coop_summary = {}
        summary_file = os.path.join(coop_dir, "cooperative_summary.json")
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    coop_summary = json.load(f)
            except:
                pass

        # 分析V2X消息内容
        v2x_stats = {
            'total_messages': len(v2x_files),
            'message_types': defaultdict(int),
            'average_message_size': 0
        }

        if v2x_files:
            total_size = 0
            for v2x_file in v2x_files[:min(10, len(v2x_files))]:  # 抽样分析
                try:
                    with open(os.path.join(v2x_dir, v2x_file), 'r') as f:
                        data = json.load(f)
                    message_type = data.get('message', {}).get('message_type', 'unknown')
                    v2x_stats['message_types'][message_type] += 1

                    file_size = os.path.getsize(os.path.join(v2x_dir, v2x_file))
                    total_size += file_size
                except:
                    pass

            v2x_stats['average_message_size'] = round(total_size / max(len(v2x_files), 1), 2)

        analysis = {
            'v2x_messages': len(v2x_files),
            'shared_perception_frames': len(perception_files),
            'total_vehicles': coop_summary.get('total_vehicles', 0),
            'ego_vehicles': coop_summary.get('ego_vehicles', 0),
            'cooperative_vehicles': coop_summary.get('cooperative_vehicles', 0),
            'v2x_stats': v2x_stats,
            'shared_objects_count': coop_summary.get('shared_objects_count', 0),
            'communication_range': coop_summary.get('communication_range', 0),
            'collaborative_detections': coop_summary.get('v2x_stats', {}).get('collaborative_detections', 0)
        }

        return analysis

    @staticmethod


    def _calculate_quality_metrics(data_dir):
        """计算质量指标"""
        quality_metrics = {
            'completeness_score': 0,
            'consistency_score': 0,
            'diversity_score': 0,

            'cooperative_score': 0,


            'issues_found': []
        }

        # 检查完整性

        required_dirs = ["raw/vehicle", "raw/infrastructure", "stitched", "metadata", "cooperative"]
        missing_dirs = []

        for dir_path in required_dirs:
            full_path = os.path.join(data_dir, dir_path)
            if not os.path.exists(full_path):

        required_dirs = ["raw/vehicle", "raw/infrastructure", "stitched", "metadata"]
        missing_dirs = []

        for dir_path in required_dirs:
            if not os.path.exists(os.path.join(data_dir, dir_path)):

                missing_dirs.append(dir_path)

        if missing_dirs:
            quality_metrics['issues_found'].append(f"缺失目录: {missing_dirs}")

            quality_metrics['completeness_score'] = 100 - (len(missing_dirs) * 20)

            quality_metrics['completeness_score'] = 50

        else:
            quality_metrics['completeness_score'] = 100

        # 检查一致性（图像数量）
        raw_vehicle = os.path.join(data_dir, "raw", "vehicle")
        if os.path.exists(raw_vehicle):
            camera_counts = []
            for camera_dir in os.listdir(raw_vehicle):
                camera_path = os.path.join(raw_vehicle, camera_dir)
                if os.path.isdir(camera_path):
                    images = [f for f in os.listdir(camera_path)
                              if f.endswith('.png')]
                    camera_counts.append(len(images))

            if camera_counts:

                max_diff = max(camera_counts) - min(camera_counts) if camera_counts else 0

                max_diff = max(camera_counts) - min(camera_counts)

                if max_diff > 5:
                    quality_metrics['issues_found'].append(f"摄像头图像数量不一致: 差异{max_diff}")
                    quality_metrics['consistency_score'] = 70
                else:
                    quality_metrics['consistency_score'] = 95

        # 多样性评分（基于物体类别）
        object_stats = DataAnalyzer._analyze_objects(data_dir)
        num_classes = len(object_stats.get('by_class', {}))

        if num_classes >= 5:
            quality_metrics['diversity_score'] = 90
        elif num_classes >= 3:
            quality_metrics['diversity_score'] = 70
        else:
            quality_metrics['diversity_score'] = 50
            quality_metrics['issues_found'].append(f"物体类别较少: {num_classes}类")


        # 协同评分
        cooperative_data = DataAnalyzer._analyze_cooperative_data(data_dir)
        if cooperative_data['v2x_messages'] > 10 and cooperative_data['shared_perception_frames'] > 5:
            quality_metrics['cooperative_score'] = 90
        elif cooperative_data['v2x_messages'] > 0 or cooperative_data['shared_perception_frames'] > 0:
            quality_metrics['cooperative_score'] = 60
        else:
            quality_metrics['cooperative_score'] = 30
            quality_metrics['issues_found'].append("协同数据较少")



        return quality_metrics

    @staticmethod
    def _calculate_overall_score(analysis):
        """计算总体评分"""
        weights = {

            'completeness': 0.25,
            'consistency': 0.20,
            'diversity': 0.15,
            'cooperative': 0.20,
            'temporal': 0.20

            'completeness': 0.3,
            'consistency': 0.25,
            'diversity': 0.25,
            'temporal': 0.2

        }

        quality = analysis['quality_metrics']

        score = (
                quality['completeness_score'] * weights['completeness'] +
                quality['consistency_score'] * weights['consistency'] +

                quality['diversity_score'] * weights['diversity'] +
                quality['cooperative_score'] * weights['cooperative']

                quality['diversity_score'] * weights['diversity']

        )

        # 时间因素
        temporal = analysis['temporal_analysis']
        if temporal['frame_rate'] >= 2.0:
            score += 20 * weights['temporal']
        elif temporal['frame_rate'] >= 1.0:
            score += 15 * weights['temporal']
        else:
            score += 5 * weights['temporal']


        return round(min(score, 100), 1)

        return round(score, 1)


    @staticmethod
    def _save_analysis_report(data_dir, analysis):
        """保存分析报告"""
        report_file = os.path.join(data_dir, "metadata", "dataset_analysis.json")

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        print(f"数据集分析报告保存: {report_file}")

    @staticmethod
    def _print_analysis_summary(analysis):
        """打印分析摘要"""
        print("\n" + "=" * 60)
        print("数据集分析摘要")
        print("=" * 60)

        basic = analysis['basic_stats']
        print(f"\n基本统计:")
        print(f"  总大小: {basic['total_size_mb']} MB")
        print(f"  文件数: {basic['file_count']}")
        print(f"  目录数: {basic['directory_count']}")

        print(f"  数据类型:")
        for data_type, count in basic['data_types'].items():
            print(f"    {data_type}: {count}")

        distribution = analysis['file_distribution']
        print(f"\n文件分布:")
        for key, value in distribution.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():

                    if isinstance(subvalue, dict):
                        for subsubkey, subsubvalue in subvalue.items():
                            print(f"    {subsubkey}: {subsubvalue}")
                    else:
                        print(f"    {subkey}: {subvalue}")

                    print(f"    {subkey}: {subvalue}")

            else:
                print(f"  {key}: {value}")

        objects = analysis['object_statistics']
        print(f"\n物体统计:")
        print(f"  总物体数: {objects['total_objects']}")
        if objects['by_class']:
            print(f"  类别分布:")
            for obj_class, count in objects['by_class'].items():
                percentage = objects['class_distribution'].get(obj_class, 0)
                print(f"    {obj_class}: {count} ({percentage}%)")


        # 协同数据分析
        cooperative = analysis['cooperative_data']
        print(f"\n协同数据分析:")
        print(f"  V2X消息: {cooperative['v2x_messages']}")
        print(f"  共享感知帧: {cooperative['shared_perception_frames']}")
        print(f"  车辆总数: {cooperative['total_vehicles']}")
        print(f"    主车: {cooperative['ego_vehicles']}")
        print(f"    协同车: {cooperative['cooperative_vehicles']}")
        print(f"  共享对象数: {cooperative['shared_objects_count']}")
        print(f"  协作检测数: {cooperative['collaborative_detections']}")

        if cooperative['v2x_stats']['message_types']:
            print(f"  V2X消息类型:")
            for msg_type, count in cooperative['v2x_stats']['message_types'].items():
                print(f"    {msg_type}: {count}")



        temporal = analysis['temporal_analysis']
        print(f"\n时间分析:")
        print(f"  总时长: {temporal['total_duration']:.1f}秒")
        print(f"  平均帧率: {temporal['frame_rate']:.2f} FPS")

        quality = analysis['quality_metrics']
        print(f"\n质量指标:")
        print(f"  完整性: {quality['completeness_score']}/100")
        print(f"  一致性: {quality['consistency_score']}/100")
        print(f"  多样性: {quality['diversity_score']}/100")

        print(f"  协同性: {quality['cooperative_score']}/100")



        if quality['issues_found']:
            print(f"  发现的问题 ({len(quality['issues_found'])}):")
            for issue in quality['issues_found'][:3]:
                print(f"    - {issue}")
            if len(quality['issues_found']) > 3:
                print(f"    ... 还有 {len(quality['issues_found']) - 3} 个问题")

        print(f"\n总体评分: {analysis['overall_score']}/100")


        overall_score = analysis['overall_score']
        if overall_score >= 90:
            print("✓ 数据集质量优秀")
        elif overall_score >= 75:
            print("✓ 数据集质量良好")
        elif overall_score >= 60:

        if analysis['overall_score'] >= 90:
            print("✓ 数据集质量优秀")
        elif analysis['overall_score'] >= 75:
            print("✓ 数据集质量良好")
        elif analysis['overall_score'] >= 60:

            print("⚠ 数据集质量一般")
        else:
            print("✗ 数据集质量需要改进")

        print("=" * 60)