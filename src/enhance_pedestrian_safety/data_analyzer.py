import os
import json
import numpy as np
from collections import defaultdict
import hashlib
import pickle
import time


class DataAnalyzer:
    """数据分析器 - 生成数据集统计信息（优化版）"""

    # 添加缓存机制
    _cache_dir = ".analysis_cache"
    _cache_enabled = True

    @staticmethod
    def _get_cache_key(data_dir):
        """生成缓存键"""
        # 使用目录结构和文件修改时间生成哈希
        file_times = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_times.append(str(os.path.getmtime(file_path)))
                except:
                    pass

        content = data_dir + "".join(sorted(file_times))
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def _load_from_cache(cache_key):
        """从缓存加载"""
        if not DataAnalyzer._cache_enabled:
            return None

        cache_file = os.path.join(DataAnalyzer._cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_time, analysis = pickle.load(f)
                # 缓存有效期1小时
                if time.time() - cached_time < 3600:
                    print(
                        f"使用缓存分析结果 (缓存时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cached_time))})")
                    return analysis
            except:
                pass
        return None

    @staticmethod
    def _save_to_cache(cache_key, analysis):
        """保存到缓存"""
        if not DataAnalyzer._cache_enabled:
            return

        os.makedirs(DataAnalyzer._cache_dir, exist_ok=True)
        cache_file = os.path.join(DataAnalyzer._cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((time.time(), analysis), f)
        except:
            pass

    @staticmethod
    def analyze_dataset(data_dir, force_refresh=False):
        """分析数据集并生成详细报告（带缓存）"""
        print(f"分析数据集: {data_dir}")

        # 检查缓存
        cache_key = DataAnalyzer._get_cache_key(data_dir)
        if not force_refresh:
            cached = DataAnalyzer._load_from_cache(cache_key)
            if cached:
                return cached

        analysis_start = time.time()

        # 并行执行分析任务
        analysis = {
            'basic_stats': DataAnalyzer._get_basic_stats(data_dir),
            'file_distribution': DataAnalyzer._analyze_file_distribution(data_dir),
            'object_statistics': DataAnalyzer._analyze_objects(data_dir),
            'temporal_analysis': DataAnalyzer._analyze_temporal(data_dir),
            'cooperative_data': DataAnalyzer._analyze_cooperative_data(data_dir),
            'quality_metrics': DataAnalyzer._calculate_quality_metrics(data_dir),
            'safety_analysis': DataAnalyzer._analyze_safety_data(data_dir)
        }

        # 生成评分
        analysis['overall_score'] = DataAnalyzer._calculate_overall_score(analysis)

        # 添加分析元数据
        analysis['metadata'] = {
            'analysis_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_duration': round(time.time() - analysis_start, 2),
            'cache_key': cache_key
        }

        # 保存分析结果
        DataAnalyzer._save_analysis_report(data_dir, analysis)

        # 保存到缓存
        DataAnalyzer._save_to_cache(cache_key, analysis)

        # 打印摘要
        DataAnalyzer._print_analysis_summary(analysis)

        return analysis

    @staticmethod
    def _get_basic_stats(data_dir):
        """获取基本统计信息（优化版）"""
        stats = {
            'total_size_mb': 0,
            'file_count': 0,
            'directory_count': 0,
            'data_types': defaultdict(int),
            'largest_files': [],
            'oldest_newest_files': {}
        }

        file_sizes = []
        file_times = []

        for root, dirs, files in os.walk(data_dir):
            stats['directory_count'] += len(dirs)
            stats['file_count'] += len(files)

            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # 文件大小
                    file_size = os.path.getsize(file_path)
                    stats['total_size_mb'] += file_size

                    # 文件修改时间
                    mtime = os.path.getmtime(file_path)
                    file_times.append((file_path, mtime))

                    # 记录大文件
                    if file_size > 10 * 1024 * 1024:  # 10MB以上
                        file_sizes.append((file_path, file_size))

                    # 文件类型统计
                    ext = os.path.splitext(file)[1].lower()
                    if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                        stats['data_types']['images'] += 1
                    elif ext == '.json':
                        stats['data_types']['json'] += 1
                    elif ext in ['.txt', '.csv', '.log']:
                        stats['data_types']['text'] += 1
                    elif ext in ['.bin', '.pcd']:
                        stats['data_types']['binary'] += 1
                    elif ext in ['.pkl', '.pickle']:
                        stats['data_types']['pickle'] += 1
                    elif ext == '.gz':
                        stats['data_types']['compressed'] += 1
                    else:
                        stats['data_types']['other'] += 1

                except Exception as e:
                    print(f"处理文件 {file_path} 失败: {e}")

        # 转换为MB
        stats['total_size_mb'] = round(stats['total_size_mb'] / (1024 * 1024), 2)

        # 找出最大的5个文件
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        stats['largest_files'] = [
            {'path': os.path.relpath(path, data_dir), 'size_mb': round(size / (1024 * 1024), 2)}
            for path, size in file_sizes[:5]
        ]

        # 找出最旧和最新的文件
        if file_times:
            file_times.sort(key=lambda x: x[1])
            oldest = file_times[0]
            newest = file_times[-1]
            stats['oldest_newest_files'] = {
                'oldest': {
                    'path': os.path.relpath(oldest[0], data_dir),
                    'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(oldest[1]))
                },
                'newest': {
                    'path': os.path.relpath(newest[0], data_dir),
                    'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(newest[1]))
                }
            }

        return stats

    @staticmethod
    def _analyze_file_distribution(data_dir):
        """分析文件分布（优化版）"""
        distribution = {}

        # 快速扫描目录结构
        if os.path.exists(data_dir):
            # 使用os.scandir提高效率
            with os.scandir(data_dir) as entries:
                for entry in entries:
                    if entry.is_dir():
                        dir_name = entry.name
                        if dir_name == "raw":
                            distribution.update(DataAnalyzer._analyze_raw_data(data_dir))
                        elif dir_name == "stitched":
                            stitched_dir = os.path.join(data_dir, "stitched")
                            if os.path.exists(stitched_dir):
                                stitched_images = [f for f in os.listdir(stitched_dir)
                                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                                distribution['stitched'] = {
                                    'total': len(stitched_images),
                                    'formats': defaultdict(int)
                                }
                                for img in stitched_images:
                                    ext = os.path.splitext(img)[1].lower()
                                    distribution['stitched']['formats'][ext] += 1
                        elif dir_name == "annotations":
                            annotations_dir = os.path.join(data_dir, "annotations")
                            if os.path.exists(annotations_dir):
                                json_files = [f for f in os.listdir(annotations_dir)
                                              if f.lower().endswith('.json')]
                                distribution['annotations'] = len(json_files)
                        elif dir_name == "lidar":
                            lidar_dir = os.path.join(data_dir, "lidar")
                            if os.path.exists(lidar_dir):
                                distribution['lidar'] = DataAnalyzer._analyze_lidar_data(lidar_dir)
                        elif dir_name == "fusion":
                            fusion_dir = os.path.join(data_dir, "fusion")
                            if os.path.exists(fusion_dir):
                                distribution['fusion'] = DataAnalyzer._analyze_fusion_data(fusion_dir)
                        elif dir_name == "safety_reports":
                            safety_dir = os.path.join(data_dir, "safety_reports")
                            if os.path.exists(safety_dir):
                                distribution['safety_reports'] = DataAnalyzer._analyze_safety_reports(safety_dir)

        return distribution

    @staticmethod
    def _analyze_raw_data(data_dir):
        """分析原始数据"""
        distribution = {}
        raw_path = os.path.join(data_dir, "raw")

        if not os.path.exists(raw_path):
            return distribution

        # 分析原始图像
        for raw_dir in os.listdir(raw_path):
            full_path = os.path.join(raw_path, raw_dir)
            if os.path.isdir(full_path):
                camera_stats = {}
                total_images = 0

                for camera_dir in os.listdir(full_path):
                    camera_path = os.path.join(full_path, camera_dir)
                    if os.path.isdir(camera_path):
                        images = [f for f in os.listdir(camera_path)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        camera_stats[camera_dir] = len(images)
                        total_images += len(images)

                distribution[f'raw_{raw_dir}'] = {
                    'cameras': camera_stats,
                    'total_images': total_images,
                    'camera_count': len(camera_stats)
                }

        return distribution

    @staticmethod
    def _analyze_lidar_data(lidar_dir):
        """分析LiDAR数据"""
        lidar_stats = {
            'bin': 0,
            'npy': 0,
            'json': 0,
            'batch': 0,
            'pcd': 0,
            'total_size_mb': 0
        }

        total_size = 0
        for root, dirs, files in os.walk(lidar_dir):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                try:
                    file_size = os.path.getsize(file_path)
                    total_size += file_size

                    if ext == '.bin':
                        lidar_stats['bin'] += 1
                    elif ext == '.npy':
                        lidar_stats['npy'] += 1
                    elif ext == '.json':
                        lidar_stats['json'] += 1
                    elif 'batch' in file:
                        lidar_stats['batch'] += 1
                    elif ext == '.pcd':
                        lidar_stats['pcd'] += 1
                except:
                    pass

        lidar_stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        return lidar_stats

    @staticmethod
    def _analyze_fusion_data(fusion_dir):
        """分析融合数据"""
        fusion_stats = {
            'sync_files': 0,
            'calibration_files': 0,
            'total_size_mb': 0,
            'formats': defaultdict(int)
        }

        total_size = 0
        for root, dirs, files in os.walk(fusion_dir):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                try:
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    fusion_stats['formats'][ext] += 1

                    if 'sync' in file:
                        fusion_stats['sync_files'] += 1
                    elif 'calib' in file or 'intrinsic' in file or 'extrinsic' in file:
                        fusion_stats['calibration_files'] += 1
                except:
                    pass

        fusion_stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        return fusion_stats

    @staticmethod
    def _analyze_safety_reports(safety_dir):
        """分析安全报告数据"""
        safety_stats = {
            'reports': 0,
            'high_risk': 0,
            'medium_risk': 0,
            'low_risk': 0,
            'total_interactions': 0
        }

        json_files = [f for f in os.listdir(safety_dir) if f.lower().endswith('.json')]
        safety_stats['reports'] = len(json_files)

        if json_files:
            # 采样分析几个文件
            sample_files = json_files[:min(5, len(json_files))]
            for json_file in sample_files:
                try:
                    with open(os.path.join(safety_dir, json_file), 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if 'high_risk_cases' in data:
                        safety_stats['high_risk'] += data['high_risk_cases']
                    if 'medium_risk_cases' in data:
                        safety_stats['medium_risk'] += data['medium_risk_cases']
                    if 'low_risk_cases' in data:
                        safety_stats['low_risk_cases'] += data['low_risk_cases']
                    if 'total_interactions' in data:
                        safety_stats['total_interactions'] += data['total_interactions']
                except:
                    pass

        return safety_stats

    @staticmethod
    def _analyze_objects(data_dir):
        """分析物体统计（优化版）"""
        annotations_dir = os.path.join(data_dir, "annotations")

        if not os.path.exists(annotations_dir):
            return {
                'total_objects': 0,
                'by_class': {},
                'by_frame': {},
                'class_distribution': {},
                'object_density': 0,
                'frames_with_objects': 0
            }

        object_stats = {
            'total_objects': 0,
            'by_class': defaultdict(int),
            'by_frame': defaultdict(int),
            'class_distribution': {},
            'object_density': 0,
            'frames_with_objects': 0,
            'objects_per_frame_stats': {},
            'class_combinations': set()
        }

        json_files = [f for f in os.listdir(annotations_dir)
                      if f.lower().endswith('.json') and f.startswith('frame_')]

        if not json_files:
            return object_stats

        # 采样分析，避免处理所有文件
        sample_size = min(50, len(json_files))
        sample_files = random.sample(json_files, sample_size) if len(json_files) > 50 else json_files

        objects_per_frame = []

        for json_file in sample_files:
            try:
                with open(os.path.join(annotations_dir, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                frame_id = data.get('frame_id', 0)
                objects = data.get('objects', [])

                object_stats['by_frame'][frame_id] = len(objects)
                object_stats['total_objects'] += len(objects)
                objects_per_frame.append(len(objects))

                if objects:
                    object_stats['frames_with_objects'] += 1

                # 统计类别和组合
                frame_classes = set()
                for obj in objects:
                    obj_class = obj.get('class', 'unknown')
                    object_stats['by_class'][obj_class] += 1
                    frame_classes.add(obj_class)

                if frame_classes:
                    object_stats['class_combinations'].add(tuple(sorted(frame_classes)))

            except Exception as e:
                print(f"分析标注文件 {json_file} 失败: {e}")

        # 估算总数
        if sample_files:
            avg_objects_per_file = object_stats['total_objects'] / len(sample_files)
            object_stats['total_objects'] = int(avg_objects_per_file * len(json_files))
            object_stats['frames_with_objects'] = int(
                (object_stats['frames_with_objects'] / len(sample_files)) * len(json_files))

        # 计算类分布百分比
        if object_stats['total_objects'] > 0:
            total = sum(object_stats['by_class'].values())
            for obj_class, count in object_stats['by_class'].items():
                object_stats['class_distribution'][obj_class] = round(
                    count / total * 100, 2
                )

        # 计算物体密度统计
        if objects_per_frame:
            object_stats['objects_per_frame_stats'] = {
                'min': min(objects_per_frame),
                'max': max(objects_per_frame),
                'mean': round(np.mean(objects_per_frame), 2),
                'median': round(np.median(objects_per_frame), 2),
                'std': round(np.std(objects_per_frame), 2)
            }
            object_stats['object_density'] = round(np.mean(objects_per_frame), 2)

        # 转换组合为可序列化的列表
        object_stats['class_combinations'] = [
            list(combo) for combo in object_stats['class_combinations']
        ]

        return object_stats

    @staticmethod
    def _analyze_temporal(data_dir):
        """分析时间分布（增强版）"""
        temporal_stats = {
            'frame_intervals': [],
            'total_duration': 0,
            'frame_rate': 0,
            'temporal_coverage': 0,
            'frame_consistency': 100,
            'time_range': {}
        }

        metadata_file = os.path.join(data_dir, "metadata", "collection_info.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                collection_stats = metadata.get('collection_stats', {})
                temporal_stats['total_duration'] = collection_stats.get('duration_seconds', 0)
                temporal_stats['frame_rate'] = collection_stats.get('frame_rate', 0)

                # 计算时间覆盖
                if 'performance' in metadata:
                    perf = metadata['performance']
                    if 'total_runtime' in perf and temporal_stats['total_duration'] > 0:
                        temporal_stats['temporal_coverage'] = round(
                            min(100, temporal_stats['total_duration'] / perf['total_runtime'] * 100), 1
                        )

                # 获取时间范围
                if 'collection' in metadata:
                    coll = metadata['collection']
                    temporal_stats['time_range'] = {
                        'start_time': coll.get('start_time', 'unknown'),
                        'end_time': coll.get('end_time', 'unknown'),
                        'duration_hours': round(coll.get('duration', 0) / 3600, 2)
                    }

            except Exception as e:
                print(f"分析元数据失败: {e}")

        # 从文件时间推断时间范围
        try:
            all_files = []
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg', '.json', '.bin')):
                        file_path = os.path.join(root, file)
                        try:
                            mtime = os.path.getmtime(file_path)
                            all_files.append((file_path, mtime))
                        except:
                            pass

            if all_files:
                all_files.sort(key=lambda x: x[1])
                oldest = all_files[0][1]
                newest = all_files[-1][1]

                if not temporal_stats['time_range']:
                    temporal_stats['time_range'] = {
                        'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(oldest)),
                        'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(newest)),
                        'duration_hours': round((newest - oldest) / 3600, 2)
                    }
        except Exception as e:
            print(f"分析文件时间失败: {e}")

        return temporal_stats

    @staticmethod
    def _analyze_cooperative_data(data_dir):
        """分析协同数据（增强版）"""
        coop_dir = os.path.join(data_dir, "cooperative")

        if not os.path.exists(coop_dir):
            return {
                'v2x_messages': 0,
                'shared_perception': 0,
                'vehicles_count': 0,
                'communication_stats': {},
                'cooperation_level': 'none',
                'data_quality': {}
            }

        analysis = {
            'v2x_messages': 0,
            'shared_perception_frames': 0,
            'total_vehicles': 0,
            'ego_vehicles': 0,
            'cooperative_vehicles': 0,
            'v2x_stats': {
                'total_messages': 0,
                'message_types': defaultdict(int),
                'average_message_size': 0,
                'message_frequency': 0
            },
            'shared_objects_count': 0,
            'communication_range': 0,
            'collaborative_detections': 0,
            'cooperation_level': 'low',
            'data_quality': {
                'message_completeness': 0,
                'perception_consistency': 0,
                'temporal_alignment': 0
            }
        }

        # V2X消息统计
        v2x_dir = os.path.join(coop_dir, "v2x_messages")
        v2x_files = []
        if os.path.exists(v2x_dir):
            v2x_files = [f for f in os.listdir(v2x_dir) if f.lower().endswith('.json')]
            analysis['v2x_messages'] = len(v2x_files)

        # 共享感知统计
        perception_dir = os.path.join(coop_dir, "shared_perception")
        perception_files = []
        if os.path.exists(perception_dir):
            perception_files = [f for f in os.listdir(perception_dir) if f.lower().endswith('.json')]
            analysis['shared_perception_frames'] = len(perception_files)

        # 读取协同摘要
        coop_summary = {}
        summary_file = os.path.join(coop_dir, "cooperative_summary.json")
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    coop_summary = json.load(f)
            except:
                pass

        # 分析V2X消息内容
        if v2x_files:
            total_size = 0
            valid_messages = 0
            sample_size = min(20, len(v2x_files))

            for v2x_file in v2x_files[:sample_size]:
                try:
                    with open(os.path.join(v2x_dir, v2x_file), 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    message = data.get('message', {})
                    message_type = message.get('message_type', 'unknown')
                    analysis['v2x_stats']['message_types'][message_type] += 1

                    file_size = os.path.getsize(os.path.join(v2x_dir, v2x_file))
                    total_size += file_size
                    valid_messages += 1

                    # 检查消息完整性
                    required_fields = ['sender_id', 'message_type', 'timestamp']
                    completeness = sum(1 for field in required_fields if field in message) / len(required_fields)
                    analysis['data_quality']['message_completeness'] += completeness

                except:
                    pass

            if valid_messages > 0:
                analysis['v2x_stats']['average_message_size'] = round(total_size / valid_messages, 2)
                analysis['data_quality']['message_completeness'] = round(
                    analysis['data_quality']['message_completeness'] / valid_messages * 100, 1
                )

        # 分析共享感知数据
        if perception_files:
            consistent_frames = 0
            sample_size = min(10, len(perception_files))

            for perception_file in perception_files[:sample_size]:
                try:
                    with open(os.path.join(perception_dir, perception_file), 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 检查数据一致性
                    if 'shared_objects' in data and 'timestamp' in data and 'frame_id' in data:
                        consistent_frames += 1
                except:
                    pass

            if sample_size > 0:
                analysis['data_quality']['perception_consistency'] = round(
                    consistent_frames / sample_size * 100, 1
                )

        # 从摘要中获取数据
        analysis.update({
            'total_vehicles': coop_summary.get('total_vehicles', 0),
            'ego_vehicles': coop_summary.get('ego_vehicles', 0),
            'cooperative_vehicles': coop_summary.get('cooperative_vehicles', 0),
            'shared_objects_count': coop_summary.get('shared_objects_count', 0),
            'communication_range': coop_summary.get('communication_range', 0),
            'collaborative_detections': coop_summary.get('v2x_stats', {}).get('collaborative_detections', 0)
        })

        # 计算合作水平
        cooperation_score = 0
        if analysis['v2x_messages'] > 50 and analysis['shared_perception_frames'] > 20:
            cooperation_score = 90
            analysis['cooperation_level'] = 'high'
        elif analysis['v2x_messages'] > 10 and analysis['shared_perception_frames'] > 5:
            cooperation_score = 60
            analysis['cooperation_level'] = 'medium'
        elif analysis['v2x_messages'] > 0 or analysis['shared_perception_frames'] > 0:
            cooperation_score = 30
            analysis['cooperation_level'] = 'low'

        analysis['cooperation_score'] = cooperation_score

        return analysis

    @staticmethod
    def _analyze_safety_data(data_dir):
        """分析安全数据"""
        safety_dir = os.path.join(data_dir, "safety_reports")

        if not os.path.exists(safety_dir):
            return {
                'total_reports': 0,
                'risk_levels': {'high': 0, 'medium': 0, 'low': 0},
                'safety_score': 0,
                'pedestrian_interactions': 0,
                'average_distance': 0
            }

        json_files = [f for f in os.listdir(safety_dir) if f.lower().endswith('.json')]

        safety_data = {
            'total_reports': len(json_files),
            'risk_levels': {'high': 0, 'medium': 0, 'low': 0},
            'safety_score': 0,
            'pedestrian_interactions': 0,
            'average_distance': 0,
            'near_misses': 0,
            'safety_warnings': 0
        }

        if json_files:
            distances = []
            for json_file in json_files[:min(10, len(json_files))]:
                try:
                    with open(os.path.join(safety_dir, json_file), 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if 'high_risk_cases' in data:
                        safety_data['risk_levels']['high'] += data['high_risk_cases']
                    if 'medium_risk_cases' in data:
                        safety_data['risk_levels']['medium'] += data['medium_risk_cases']
                    if 'low_risk_cases' in data:
                        safety_data['risk_levels']['low'] += data['low_risk_cases']
                    if 'total_interactions' in data:
                        safety_data['pedestrian_interactions'] += data['total_interactions']
                    if 'average_distance' in data:
                        distances.append(data['average_distance'])
                    if 'near_misses' in data:
                        safety_data['near_misses'] += data['near_misses']
                    if 'safety_warnings' in data:
                        safety_data['safety_warnings'] += data['safety_warnings']

                except Exception as e:
                    print(f"分析安全报告 {json_file} 失败: {e}")

            if distances:
                safety_data['average_distance'] = round(np.mean(distances), 2)

            # 计算安全评分
            total_risks = sum(safety_data['risk_levels'].values())
            if total_risks > 0:
                high_risk_ratio = safety_data['risk_levels']['high'] / total_risks
                safety_data['safety_score'] = max(0, 100 - high_risk_ratio * 100)
            else:
                safety_data['safety_score'] = 100

        return safety_data

    @staticmethod
    def _calculate_quality_metrics(data_dir):
        """计算质量指标（增强版）"""
        quality_metrics = {
            'completeness_score': 0,
            'consistency_score': 0,
            'diversity_score': 0,
            'cooperative_score': 0,
            'temporal_score': 0,
            'structural_score': 0,
            'safety_score': 0,
            'issues_found': [],
            'recommendations': []
        }

        # 1. 检查完整性
        required_dirs = [
            "raw/vehicle",
            "raw/infrastructure",
            "stitched",
            "metadata",
            "cooperative"
        ]

        optional_dirs = [
            "lidar",
            "fusion",
            "annotations",
            "calibration",
            "safety_reports"
        ]

        missing_required = []
        missing_optional = []

        for dir_path in required_dirs:
            full_path = os.path.join(data_dir, dir_path)
            if not os.path.exists(full_path):
                missing_required.append(dir_path)

        for dir_path in optional_dirs:
            full_path = os.path.join(data_dir, dir_path)
            if not os.path.exists(full_path):
                missing_optional.append(dir_path)

        if missing_required:
            quality_metrics['issues_found'].append(f"缺失必要目录: {missing_required}")
            quality_metrics['completeness_score'] = 100 - (len(missing_required) * 20)
        else:
            quality_metrics['completeness_score'] = 100

        # 2. 结构评分
        structure_score = 100
        if missing_optional:
            structure_score -= len(missing_optional) * 5
            quality_metrics['recommendations'].append(f"建议添加可选目录: {missing_optional[:3]}")

        quality_metrics['structural_score'] = max(0, structure_score)

        # 3. 检查一致性（图像数量）
        raw_vehicle = os.path.join(data_dir, "raw", "vehicle")
        if os.path.exists(raw_vehicle):
            camera_counts = []
            for camera_dir in os.listdir(raw_vehicle):
                camera_path = os.path.join(raw_vehicle, camera_dir)
                if os.path.isdir(camera_path):
                    images = [f for f in os.listdir(camera_path)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    camera_counts.append(len(images))

            if camera_counts:
                max_diff = max(camera_counts) - min(camera_counts) if camera_counts else 0
                if max_diff > 10:
                    quality_metrics['issues_found'].append(f"摄像头图像数量严重不一致: 最大差异{max_diff}张")
                    quality_metrics['consistency_score'] = 60
                elif max_diff > 5:
                    quality_metrics['issues_found'].append(f"摄像头图像数量不一致: 差异{max_diff}张")
                    quality_metrics['consistency_score'] = 75
                else:
                    quality_metrics['consistency_score'] = 95
            else:
                quality_metrics['consistency_score'] = 70
                quality_metrics['issues_found'].append("车辆摄像头无图像数据")
        else:
            quality_metrics['consistency_score'] = 50

        # 4. 多样性评分（基于物体类别）
        object_stats = DataAnalyzer._analyze_objects(data_dir)
        num_classes = len(object_stats.get('by_class', {}))
        class_distribution = object_stats.get('class_distribution', {})

        if num_classes >= 8:
            quality_metrics['diversity_score'] = 95
        elif num_classes >= 5:
            quality_metrics['diversity_score'] = 80
        elif num_classes >= 3:
            quality_metrics['diversity_score'] = 65
            quality_metrics['issues_found'].append(f"物体类别较少: {num_classes}类")
        else:
            quality_metrics['diversity_score'] = 40
            quality_metrics['issues_found'].append(f"物体类别过少: {num_classes}类")

        # 检查类别分布是否均衡
        if class_distribution:
            values = list(class_distribution.values())
            if max(values) > 70:  # 某个类别占比超过70%
                quality_metrics['diversity_score'] *= 0.8  # 降低分数
                quality_metrics['recommendations'].append("数据集类别分布不均衡，建议收集更多样化的场景")

        # 5. 协同评分
        cooperative_data = DataAnalyzer._analyze_cooperative_data(data_dir)
        quality_metrics['cooperative_score'] = cooperative_data.get('cooperation_score', 0)

        if quality_metrics['cooperative_score'] < 50:
            quality_metrics['issues_found'].append("协同数据较少或质量不高")
            quality_metrics['recommendations'].append("增加V2X消息和共享感知数据的生成")

        # 6. 时间评分
        temporal_data = DataAnalyzer._analyze_temporal(data_dir)
        frame_rate = temporal_data.get('frame_rate', 0)
        duration = temporal_data.get('total_duration', 0)

        if frame_rate >= 5.0:
            quality_metrics['temporal_score'] = 95
        elif frame_rate >= 2.0:
            quality_metrics['temporal_score'] = 80
        elif frame_rate >= 1.0:
            quality_metrics['temporal_score'] = 60
        else:
            quality_metrics['temporal_score'] = 30
            quality_metrics['issues_found'].append(f"帧率较低: {frame_rate:.2f} FPS")

        if duration < 30:
            quality_metrics['temporal_score'] *= 0.8  # 时长不足，降低分数
            quality_metrics['recommendations'].append("建议增加数据收集时长以获得更完整的时间序列")

        # 7. 安全评分
        safety_data = DataAnalyzer._analyze_safety_data(data_dir)
        quality_metrics['safety_score'] = safety_data.get('safety_score', 0)

        if quality_metrics['safety_score'] < 80:
            quality_metrics['issues_found'].append(f"安全评分较低: {quality_metrics['safety_score']}")
            quality_metrics['recommendations'].append("建议增加行人安全相关的场景和数据收集")

        # 限制分数在0-100之间
        for key in ['completeness_score', 'consistency_score', 'diversity_score',
                    'cooperative_score', 'temporal_score', 'structural_score', 'safety_score']:
            quality_metrics[key] = max(0, min(100, quality_metrics[key]))

        return quality_metrics

    @staticmethod
    def _calculate_overall_score(analysis):
        """计算总体评分（增强版）"""
        weights = {
            'completeness': 0.15,  # 完整性
            'consistency': 0.12,  # 一致性
            'temporal': 0.12,  # 时间性
            'structural': 0.08,  # 结构性
            'diversity': 0.12,  # 多样性
            'cooperative': 0.12,  # 协同性
            'safety': 0.19,  # 安全性
            'quality_bonus': 0.10  # 质量加成
        }

        quality = analysis['quality_metrics']

        # 基础分数
        base_score = (
                quality['completeness_score'] * weights['completeness'] +
                quality['consistency_score'] * weights['consistency'] +
                quality['temporal_score'] * weights['temporal'] +
                quality['structural_score'] * weights['structural'] +
                quality['diversity_score'] * weights['diversity'] +
                quality['cooperative_score'] * weights['cooperative'] +
                quality['safety_score'] * weights['safety']
        )

        # 质量加成（基于问题数量）
        issues_count = len(quality.get('issues_found', []))
        quality_bonus = max(0, 100 - issues_count * 5) * weights['quality_bonus']

        total_score = base_score + quality_bonus

        # 额外加成（如果数据集特别优秀）
        if (quality['completeness_score'] >= 95 and
                quality['consistency_score'] >= 90 and
                quality['diversity_score'] >= 85 and
                quality['safety_score'] >= 90):
            total_score += 5

        return round(min(total_score, 100), 1)

    @staticmethod
    def _save_analysis_report(data_dir, analysis):
        """保存分析报告（优化版）"""
        metadata_dir = os.path.join(data_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)

        report_file = os.path.join(metadata_dir, "dataset_analysis.json")

        # 保存完整报告
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        # 保存摘要报告（轻量版）
        summary = {
            'overall_score': analysis['overall_score'],
            'quality_metrics': analysis['quality_metrics'],
            'basic_stats': {
                'total_size_mb': analysis['basic_stats']['total_size_mb'],
                'file_count': analysis['basic_stats']['file_count'],
                'directory_count': analysis['basic_stats']['directory_count']
            },
            'object_statistics': {
                'total_objects': analysis['object_statistics']['total_objects'],
                'num_classes': len(analysis['object_statistics']['by_class'])
            },
            'safety_data': analysis.get('safety_analysis', {}),
            'analysis_metadata': analysis.get('metadata', {})
        }

        summary_file = os.path.join(metadata_dir, "dataset_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"数据集分析报告保存: {report_file}")
        print(f"数据集摘要保存: {summary_file}")

    @staticmethod
    def _print_analysis_summary(analysis):
        """打印分析摘要（增强版）"""
        print("\n" + "=" * 70)
        print("数据集分析摘要")
        print("=" * 70)

        # 基本统计
        basic = analysis['basic_stats']
        print(f"\n📊 基本统计:")
        print(f"  总大小: {basic['total_size_mb']} MB")
        print(f"  文件数: {basic['file_count']:,}")
        print(f"  目录数: {basic['directory_count']}")
        print(f"  数据类型分布:")
        for data_type, count in basic['data_types'].items():
            print(f"    {data_type}: {count:,}")

        if basic['largest_files']:
            print(f"  最大的文件:")
            for file_info in basic['largest_files'][:3]:
                print(f"    {file_info['path']}: {file_info['size_mb']} MB")

        # 文件分布
        distribution = analysis['file_distribution']
        print(f"\n📁 文件分布:")
        for key, value in distribution.items():
            if isinstance(value, dict):
                if 'total' in value:
                    print(f"  {key}: {value['total']:,}")
                    if 'formats' in value:
                        for fmt, count in value['formats'].items():
                            print(f"    {fmt}: {count:,}")
                else:
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue:,}")
            else:
                print(f"  {key}: {value:,}")

        # 物体统计
        objects = analysis['object_statistics']
        print(f"\n🎯 物体统计:")
        print(f"  总物体数: {objects['total_objects']:,}")
        print(f"  有物体的帧数: {objects['frames_with_objects']:,}")
        print(f"  平均每帧物体数: {objects['object_density']:.2f}")

        if objects['by_class']:
            print(f"  类别分布:")
            for obj_class, count in sorted(objects['by_class'].items(), key=lambda x: x[1], reverse=True)[:5]:
                percentage = objects['class_distribution'].get(obj_class, 0)
                print(f"    {obj_class}: {count:,} ({percentage}%)")

        if objects['objects_per_frame_stats']:
            stats = objects['objects_per_frame_stats']
            print(f"  每帧物体数统计:")
            print(f"    最小: {stats['min']}, 最大: {stats['max']}, 平均: {stats['mean']}, 中位数: {stats['median']}")

        # 安全数据分析
        if 'safety_analysis' in analysis:
            safety = analysis['safety_analysis']
            print(f"\n🚸 安全数据分析:")
            print(f"  安全评分: {safety.get('safety_score', 0)}/100")
            print(f"  风险等级分布:")
            print(f"    高风险: {safety.get('risk_levels', {}).get('high', 0)}")
            print(f"    中风险: {safety.get('risk_levels', {}).get('medium', 0)}")
            print(f"    低风险: {safety.get('risk_levels', {}).get('low', 0)}")
            print(f"  行人交互次数: {safety.get('pedestrian_interactions', 0)}")
            print(f"  平均距离: {safety.get('average_distance', 0):.2f}米")
            print(f"  近距离事件: {safety.get('near_misses', 0)}")
            print(f"  安全警告: {safety.get('safety_warnings', 0)}")

        # 协同数据分析
        cooperative = analysis['cooperative_data']
        print(f"\n🤝 协同数据分析:")
        print(f"  合作水平: {cooperative['cooperation_level'].upper()}")
        print(f"  V2X消息: {cooperative['v2x_messages']:,}")
        print(f"  共享感知帧: {cooperative['shared_perception_frames']:,}")
        print(f"  车辆总数: {cooperative['total_vehicles']}")
        print(f"    ├ 主车: {cooperative['ego_vehicles']}")
        print(f"    └ 协同车: {cooperative['cooperative_vehicles']}")
        print(f"  共享对象数: {cooperative['shared_objects_count']:,}")
        print(f"  协作检测数: {cooperative['collaborative_detections']:,}")

        if cooperative['v2x_stats']['message_types']:
            print(f"  V2X消息类型:")
            for msg_type, count in cooperative['v2x_stats']['message_types'].items():
                print(f"    {msg_type}: {count}")

        # 时间分析
        temporal = analysis['temporal_analysis']
        print(f"\n⏰ 时间分析:")
        print(f"  总时长: {temporal['total_duration']:.1f}秒 ({temporal['total_duration'] / 60:.1f}分钟)")
        print(f"  平均帧率: {temporal['frame_rate']:.2f} FPS")
        print(f"  时间覆盖率: {temporal['temporal_coverage']}%")

        if temporal['time_range']:
            tr = temporal['time_range']
            print(f"  时间范围: {tr.get('start_time', 'N/A')} 到 {tr.get('end_time', 'N/A')}")
            if 'duration_hours' in tr:
                print(f"  持续时间: {tr['duration_hours']:.1f}小时")

        # 质量指标
        quality = analysis['quality_metrics']
        print(f"\n📈 质量指标:")
        metrics = [
            ('完整性', quality['completeness_score']),
            ('一致性', quality['consistency_score']),
            ('结构性', quality['structural_score']),
            ('时间性', quality['temporal_score']),
            ('多样性', quality['diversity_score']),
            ('协同性', quality['cooperative_score']),
            ('安全性', quality['safety_score'])
        ]

        for name, score in metrics:
            bar = "█" * int(score / 5)
            print(f"  {name:8s}: {score:3.0f}/100 {bar}")

        if quality['issues_found']:
            print(f"\n⚠️  发现的问题 ({len(quality['issues_found'])}):")
            for i, issue in enumerate(quality['issues_found'][:5], 1):
                print(f"    {i}. {issue}")
            if len(quality['issues_found']) > 5:
                print(f"    ... 还有 {len(quality['issues_found']) - 5} 个问题")

        if quality['recommendations']:
            print(f"\n💡 改进建议:")
            for i, rec in enumerate(quality['recommendations'][:3], 1):
                print(f"    {i}. {rec}")

        print(f"\n⭐ 总体评分: {analysis['overall_score']}/100")

        overall_score = analysis['overall_score']
        if overall_score >= 90:
            print("🎉 数据集质量优秀 - 可直接用于模型训练")
        elif overall_score >= 80:
            print("👍 数据集质量良好 - 建议进行少量数据增强")
        elif overall_score >= 70:
            print("⚠️  数据集质量一般 - 建议进行数据清洗和增强")
        elif overall_score >= 60:
            print("🔧 数据集质量需要改进 - 建议补充缺失数据")
        else:
            print("🚨 数据集质量较差 - 需要大规模改进")

        # 分析元数据
        if 'metadata' in analysis:
            meta = analysis['metadata']
            print(f"\n📝 分析信息:")
            print(f"  分析时间: {meta.get('analysis_time', 'N/A')}")
            print(f"  分析耗时: {meta.get('analysis_duration', 0):.1f}秒")

        print("=" * 70)

    @staticmethod
    def generate_comparison_report(data_dirs, output_file=None):
        """生成多个数据集的比较报告"""
        comparisons = {}

        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                analysis = DataAnalyzer.analyze_dataset(data_dir)
                comparisons[os.path.basename(data_dir)] = {
                    'overall_score': analysis['overall_score'],
                    'basic_stats': analysis['basic_stats'],
                    'quality_metrics': analysis['quality_metrics'],
                    'object_statistics': {
                        'total_objects': analysis['object_statistics']['total_objects'],
                        'num_classes': len(analysis['object_statistics']['by_class'])
                    },
                    'safety_analysis': analysis.get('safety_analysis', {})
                }

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comparisons, f, indent=2, ensure_ascii=False)
            print(f"比较报告保存到: {output_file}")

        return comparisons