import json
import os
import argparse
import copy
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigValidator:
    """配置验证器（增强版）"""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        warnings = []

        # 基本结构验证
        required_sections = ['scenario', 'sensors', 'output']
        for section in required_sections:
            if section not in config:
                errors.append(f"缺失必要配置节: {section}")

        # 场景配置验证
        if 'scenario' in config:
            scenario = config['scenario']
            if 'duration' in scenario:
                if scenario['duration'] <= 0:
                    errors.append("场景时长必须大于0")
                elif scenario['duration'] > 3600:
                    warnings.append("场景时长超过1小时，可能导致内存不足")
            if 'town' not in scenario:
                errors.append("场景配置中缺失地图名称")
            if 'pedestrian_safety_mode' not in scenario:
                warnings.append("建议启用行人安全模式以增强数据收集")

        # 传感器配置验证
        if 'sensors' in config:
            sensors = config['sensors']
            if 'capture_interval' in sensors:
                if sensors['capture_interval'] <= 0:
                    errors.append("采集间隔必须大于0")
                elif sensors['capture_interval'] < 0.5:
                    warnings.append("采集间隔过小可能导致性能问题")
            if 'image_size' in sensors:
                if len(sensors['image_size']) != 2:
                    errors.append("图像尺寸必须为[宽度, 高度]格式")
                elif any(dim <= 0 for dim in sensors['image_size']):
                    errors.append("图像尺寸必须大于0")
                elif sensors['image_size'][0] > 1920 or sensors['image_size'][1] > 1080:
                    warnings.append("图像尺寸较大，可能影响处理速度")

        # 行人安全配置验证
        if 'traffic' in config:
            traffic = config['traffic']
            if 'pedestrians' in traffic:
                if traffic['pedestrians'] < 0:
                    errors.append("行人数量不能为负数")
                elif traffic['pedestrians'] > 50:
                    warnings.append("行人数量过多，可能导致场景混乱")
            if 'speed_limit' in traffic:
                if traffic['speed_limit'] > 120:
                    warnings.append("车速限制过高，影响行人安全评估")

        # 性能配置验证
        if 'performance' in config:
            perf = config['performance']
            if 'batch_size' in perf:
                if perf['batch_size'] <= 0:
                    errors.append("批处理大小必须大于0")
                elif perf['batch_size'] > 100:
                    warnings.append("批处理大小过大，可能导致内存溢出")
            if 'frame_rate_limit' in perf:
                if perf['frame_rate_limit'] > 60:
                    warnings.append("帧率限制过高，可能导致性能问题")

        # V2X配置验证
        if 'v2x' in config and config['v2x'].get('enabled', False):
            v2x = config['v2x']
            if 'communication_range' in v2x and v2x['communication_range'] > 1000:
                warnings.append("V2X通信范围过大，可能导致不切实际的模拟")
            if 'update_interval' in v2x and v2x['update_interval'] < 0.1:
                warnings.append("V2X更新间隔过小，可能导致通信拥塞")

        return len(errors) == 0, errors, warnings

    @staticmethod
    def suggest_optimizations(config: Dict[str, Any]) -> List[str]:
        suggestions = []

        # LiDAR优化建议
        if config.get('sensors', {}).get('lidar_sensors', 0) > 0:
            lidar_config = config['sensors'].get('lidar_config', {})
            max_points = lidar_config.get('max_points_per_frame', 50000)
            if max_points > 50000:
                suggestions.append(f"LiDAR最大点数({max_points})较高，建议降低到50000以下以减少内存使用")

            channels = lidar_config.get('channels', 32)
            if channels > 64:
                suggestions.append(f"LiDAR通道数({channels})较高，建议降低到64以下以提高性能")

        # 采集间隔建议
        capture_interval = config['sensors'].get('capture_interval', 2.0)
        if capture_interval < 0.5:
            suggestions.append(f"采集间隔({capture_interval}s)过短，可能导致高负载，建议增加到0.5s以上")
        elif capture_interval > 5.0:
            suggestions.append(f"采集间隔({capture_interval}s)较长，可能丢失重要数据，建议降低到5.0s以下")

        # 输出配置建议
        output = config.get('output', {})
        enabled_outputs = [k for k, v in output.items() if isinstance(v, bool) and v]
        if len(enabled_outputs) > 8:
            suggestions.append(f"启用的输出类型过多({len(enabled_outputs)})，可能影响性能，建议只启用必要的输出")

        # 行人安全建议
        pedestrians = config.get('traffic', {}).get('pedestrians', 0)
        if pedestrians < 5:
            suggestions.append("行人数量较少，建议增加行人数量以更好地测试行人安全")
        elif pedestrians > 30:
            suggestions.append("行人数量较多，建议减少以保持场景清晰度")

        if not config.get('v2x', {}).get('enabled', False):
            suggestions.append("V2X通信未启用，建议启用以支持行人安全预警")

        # 协同感知建议
        if config.get('cooperative', {}).get('num_coop_vehicles', 0) > 5:
            suggestions.append("协同车辆数量过多，建议减少到5辆以下以降低复杂度")

        return suggestions


class ConfigOptimizer:
    """配置优化器（增强版）"""

    @staticmethod
    def optimize_for_memory(config: Dict[str, Any]) -> Dict[str, Any]:
        """内存优化配置"""
        optimized = copy.deepcopy(config)

        # 传感器优化
        optimized['sensors'].update({
            'image_size': [640, 480],
            'capture_interval': 3.0,
            'lidar_sensors': 0,
            'radar_sensors': 0,
            'infrastructure_cameras': 2
        })

        # 性能优化
        perf = optimized.setdefault('performance', {})
        perf.update({
            'batch_size': 2,
            'enable_compression': True,
            'compression_level': 6,
            'enable_memory_cache': True,
            'max_cache_size': 20,
            'frame_rate_limit': 2.0,
            'enable_async_processing': False,
            'max_workers': 1,
            'memory_management': {
                'gc_interval': 20,
                'max_memory_mb': 200,
                'early_stop_threshold': 180
            }
        })

        # 输出优化
        output = optimized['output']
        output.update({
            'save_raw': True,
            'save_stitched': False,
            'save_annotations': True,
            'save_lidar': False,
            'save_fusion': False,
            'save_cooperative': False,
            'save_v2x_messages': False,
            'save_enhanced': False,
            'save_safety_reports': True,
            'compression_enabled': True
        })

        return optimized

    @staticmethod
    def optimize_for_quality(config: Dict[str, Any]) -> Dict[str, Any]:
        """质量优化配置"""
        optimized = copy.deepcopy(config)

        # 传感器优化
        sensors = optimized['sensors']
        sensors.update({
            'image_size': [1920, 1080],
            'capture_interval': 0.5,
            'lidar_sensors': 2,
            'radar_sensors': 1,
            'infrastructure_cameras': 6,
            'vehicle_cameras': 4
        })

        # LiDAR配置
        sensors['lidar_config'].update({
            'channels': 64,
            'range': 200.0,
            'points_per_second': 200000,
            'max_points_per_frame': 120000,
            'downsample_ratio': 0.05
        })

        # 输出配置
        output = optimized['output']
        output.update({
            'save_raw': True,
            'save_stitched': True,
            'save_annotations': True,
            'save_lidar': True,
            'save_fusion': True,
            'save_cooperative': True,
            'save_v2x_messages': True,
            'save_enhanced': True,
            'save_safety_reports': True,
            'run_quality_check': True,
            'validate_data': True,
            'generate_safety_summary': True
        })

        # 增强配置
        enhanced = optimized.setdefault('enhancement', {})
        enhanced.update({
            'enabled': True,
            'enable_random': True,
            'quality_check': True,
            'save_original': True,
            'save_enhanced': True,
            'calibration_generation': True,
            'enhanced_dir_name': 'enhanced_quality',
            'methods': ['normalize', 'contrast', 'brightness', 'sharpness', 'noise'],
            'weather_effects': True,
            'augmentation_level': 'high',
            'pedestrian_safety_mode': True
        })

        return optimized

    @staticmethod
    def optimize_for_speed(config: Dict[str, Any]) -> Dict[str, Any]:
        """速度优化配置"""
        optimized = copy.deepcopy(config)

        # 传感器优化
        sensors = optimized['sensors']
        sensors.update({
            'image_size': [640, 360],
            'capture_interval': 5.0,
            'lidar_sensors': 0,
            'radar_sensors': 0,
            'infrastructure_cameras': 2,
            'vehicle_cameras': 2
        })

        # 性能优化
        perf = optimized.setdefault('performance', {})
        perf.update({
            'batch_size': 20,
            'enable_compression': True,
            'compression_level': 1,
            'enable_memory_cache': False,
            'max_cache_size': 10,
            'frame_rate_limit': 15.0,
            'enable_async_processing': True,
            'max_workers': 4,
            'memory_management': {
                'gc_interval': 100,
                'max_memory_mb': 300,
                'early_stop_threshold': 280
            }
        })

        # 输出优化
        output = optimized['output']
        output.update({
            'save_raw': True,
            'save_stitched': False,
            'save_annotations': False,
            'save_lidar': False,
            'save_fusion': False,
            'save_cooperative': False,
            'save_v2x_messages': False,
            'save_enhanced': False,
            'save_safety_reports': True,
            'compression_enabled': True
        })

        return optimized

    @staticmethod
    def optimize_for_safety(config: Dict[str, Any]) -> Dict[str, Any]:
        """行人安全优化配置（增强版）"""
        optimized = copy.deepcopy(config)

        # 增加行人密度和多样性
        traffic = optimized['traffic']
        traffic.update({
            'pedestrians': 20,  # 增加行人数量
            'pedestrian_types': [
                'walker.pedestrian.0001',
                'walker.pedestrian.0002',
                'walker.pedestrian.0003',
                'walker.pedestrian.0004',
                'walker.pedestrian.0005',
                'walker.pedestrian.0006',
                'walker.pedestrian.0007',
                'walker.pedestrian.0008'
            ],
            'pedestrian_behaviors': ['walking', 'crossing', 'waiting', 'running', 'jogging'],
            'speed_limit': 30.0,
            'pedestrian_safety_zones': True,
            'crosswalk_density': 'high'
        })

        # 优化传感器配置以更好地检测行人
        sensors = optimized['sensors']
        sensors.update({
            'image_size': [1920, 1080],
            'capture_interval': 1.0,  # 更频繁地捕获
            'vehicle_cameras': 4,
            'infrastructure_cameras': 6,
            'camera_config': {
                'fov': 120.0,  # 更宽的视野
                'post_processing': 'semantic',  # 语义分割
                'exposure_mode': 'auto',
                'motion_blur': 0.0,
                'pedestrian_detection_mode': True
            }
        })

        # 启用LiDAR以检测行人
        sensors['lidar_sensors'] = 2
        sensors['lidar_config'].update({
            'channels': 64,  # 更多通道以检测行人
            'range': 150.0,
            'points_per_second': 150000,
            'max_points_per_frame': 100000,
            'downsample_ratio': 0.1,
            'pedestrian_detection': True,
            'height_filter': [-0.5, 2.5]  # 过滤地面和过高点
        })

        # 启用V2X和协同感知
        v2x = optimized.setdefault('v2x', {})
        v2x.update({
            'enabled': True,
            'communication_range': 500.0,
            'update_interval': 0.5,  # 更频繁地更新
            'enable_safety_warnings': True,
            'pedestrian_warning_threshold': 15.0,
            'emergency_brake_warning': True,
            'collision_prediction': True,
            'vulnerable_road_user_protection': True
        })

        # 协同感知配置
        coop = optimized.setdefault('cooperative', {})
        coop.update({
            'num_coop_vehicles': 3,
            'enable_shared_perception': True,
            'enable_traffic_warnings': True,
            'enable_pedestrian_warnings': True,
            'enable_emergency_brake_assist': True,
            'enable_maneuver_coordination': True,
            'data_fusion_interval': 0.3,  # 更频繁地融合
            'max_shared_objects': 200,
            'object_matching_threshold': 2.0,  # 更严格的对象匹配
            'pedestrian_tracking': True,
            'intention_prediction': True
        })

        # 性能优化
        perf = optimized.setdefault('performance', {})
        perf.update({
            'batch_size': 8,
            'enable_compression': True,
            'compression_level': 4,
            'enable_memory_cache': True,
            'max_cache_size': 60,
            'frame_rate_limit': 10.0,
            'safety_monitoring_interval': 0.5,  # 安全监控间隔
            'emergency_response_time': 0.1,
            'memory_management': {
                'gc_interval': 30,
                'max_memory_mb': 600,
                'early_stop_threshold': 550
            }
        })

        # 输出配置
        output = optimized['output']
        output.update({
            'save_raw': True,
            'save_stitched': True,
            'save_annotations': True,
            'save_lidar': True,
            'save_fusion': True,
            'save_cooperative': True,
            'save_v2x_messages': True,
            'save_enhanced': True,
            'save_safety_reports': True,
            'save_risk_maps': True,
            'save_emergency_events': True,
            'validate_data': True,
            'run_analysis': True,
            'run_quality_check': True,
            'generate_summary': True,
            'generate_safety_summary': True,
            'generate_risk_assessment': True
        })

        # 增强配置
        enhanced = optimized.setdefault('enhancement', {})
        enhanced.update({
            'enabled': True,
            'enable_random': True,
            'quality_check': True,
            'save_original': True,
            'save_enhanced': True,
            'calibration_generation': True,
            'enhanced_dir_name': 'enhanced_safety',
            'methods': ['normalize', 'contrast', 'brightness', 'pedestrian_highlight',
                        'safety_warning', 'risk_visualization', 'attention_heatmap'],
            'weather_effects': True,
            'augmentation_level': 'high',
            'pedestrian_safety_mode': True,
            'vulnerable_user_protection': True,
            'emergency_scenario_simulation': True
        })

        # 场景配置
        scenario = optimized['scenario']
        scenario.update({
            'pedestrian_safety_mode': True,
            'emergency_scenarios': ['crosswalk', 'school_zone', 'blind_spot'],
            'risk_assessment': True,
            'safety_metrics_collection': True
        })

        return optimized

    @staticmethod
    def optimize_for_research(config: Dict[str, Any]) -> Dict[str, Any]:
        """研究优化配置"""
        optimized = copy.deepcopy(config)

        # 增加数据多样性
        optimized['scenario'].update({
            'weather_variations': ['clear', 'rainy', 'cloudy', 'foggy', 'wet'],
            'time_variations': ['noon', 'sunset', 'night', 'dawn'],
            'random_seed': -1,  # 随机种子
            'scenario_variations': 10
        })

        # 传感器配置
        sensors = optimized['sensors']
        sensors.update({
            'image_size': [1280, 720],
            'capture_interval': 1.0,
            'lidar_sensors': 1,
            'radar_sensors': 1,
            'vehicle_cameras': 3,
            'infrastructure_cameras': 4
        })

        # 输出配置
        output = optimized['output']
        output.update({
            'save_raw': True,
            'save_stitched': True,
            'save_annotations': True,
            'save_lidar': True,
            'save_fusion': True,
            'save_cooperative': True,
            'save_v2x_messages': True,
            'save_enhanced': True,
            'save_safety_reports': True,
            'save_metadata': True,
            'save_calibration': True
        })

        return optimized


class ConfigManager:
    """配置管理器（增强版）"""

    PRESET_CONFIGS = {
        'balanced': {
            'description': '平衡配置 - 兼顾性能和质量',
            'optimization': 'balanced'
        },
        'high_quality': {
            'description': '高质量配置 - 优先数据质量',
            'optimization': 'quality'
        },
        'fast_collection': {
            'description': '快速采集配置 - 优先处理速度',
            'optimization': 'speed'
        },
        'pedestrian_safety': {
            'description': '行人安全配置 - 优化行人检测和安全评估（增强版）',
            'optimization': 'safety'
        },
        'memory_efficient': {
            'description': '内存高效配置 - 优化内存使用',
            'optimization': 'memory'
        },
        'research_ready': {
            'description': '研究就绪配置 - 适用于学术研究',
            'optimization': 'research'
        },
        'v2x_focused': {
            'description': 'V2X重点配置 - 优化协同数据采集',
            'optimization': 'custom',
            'settings': {
                'v2x': {'enabled': True, 'update_interval': 0.5},
                'cooperative': {'num_coop_vehicles': 4, 'enable_shared_perception': True},
                'output': {'save_cooperative': True, 'save_v2x_messages': True}
            }
        },
        'lidar_focused': {
            'description': 'LiDAR重点配置 - 优化点云数据采集',
            'optimization': 'custom',
            'settings': {
                'sensors': {'lidar_sensors': 2, 'lidar_config': {'channels': 64, 'range': 200}},
                'output': {'save_lidar': True, 'save_fusion': True}
            }
        },
        'urban_testing': {
            'description': '城市测试配置 - 针对城市环境',
            'optimization': 'custom',
            'settings': {
                'traffic': {'pedestrians': 25, 'background_vehicles': 15},
                'scenario': {'town': 'Town10HD'}
            }
        },
        'emergency_testing': {
            'description': '紧急情况测试 - 测试安全系统响应',
            'optimization': 'custom',
            'settings': {
                'scenario': {'emergency_scenarios': ['pedestrian_crossing', 'child_darting']},
                'v2x': {'emergency_brake_warning': True},
                'performance': {'emergency_response_time': 0.05}
            }
        }
    }

    @staticmethod
    def load_config(config_file: Optional[str] = None,
                    preset: Optional[str] = None,
                    validate: bool = True) -> Dict[str, Any]:
        """加载配置（增强版）"""
        print("\n" + "=" * 50)
        print("加载配置")
        print("=" * 50)

        config = ConfigManager._get_default_config()

        if preset:
            config = ConfigManager._apply_preset(config, preset)

        if config_file:
            if os.path.exists(config_file):
                config = ConfigManager._load_config_file(config_file, config)
                print(f"✓ 从文件加载配置: {config_file}")
            else:
                print(f"⚠ 配置文件不存在: {config_file}")
                # 尝试在预设目录中查找
                preset_dirs = ['configs', 'config', 'presets']
                for preset_dir in preset_dirs:
                    preset_path = os.path.join(preset_dir, config_file)
                    if os.path.exists(preset_path):
                        config = ConfigManager._load_config_file(preset_path, config)
                        print(f"✓ 从预设目录加载配置: {preset_path}")
                        break

        if validate:
            is_valid, errors, warnings = ConfigValidator.validate_config(config)

            if warnings:
                print("\n配置警告:")
                for warning in warnings:
                    print(f"  ⚠ {warning}")

            if not is_valid:
                print("\n配置验证错误:")
                for error in errors:
                    print(f"  ✗ {error}")
                raise ValueError("配置验证失败")
            else:
                print("✓ 配置验证通过")

        suggestions = ConfigValidator.suggest_optimizations(config)
        if suggestions:
            print("\n配置优化建议:")
            for suggestion in suggestions:
                print(f"  💡 {suggestion}")

        # 确保必要目录存在
        config['output']['data_dir'] = os.path.abspath(config['output']['data_dir'])
        os.makedirs(config['output']['data_dir'], exist_ok=True)

        print(f"输出目录: {config['output']['data_dir']}")
        print("=" * 50)

        return config

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """获取默认配置（增强版）"""
        return {
            'scenario': {
                'name': 'pedestrian_safety_enhanced',
                'description': '行人安全增强数据采集场景',
                'town': 'Town10HD',
                'weather': 'clear',
                'time_of_day': 'noon',
                'duration': 120,
                'seed': 42,
                'timeout': 600,
                'retry_attempts': 5,
                'pedestrian_safety_mode': True,
                'emergency_scenarios': ['crosswalk', 'school_zone'],
                'risk_assessment': True,
                'safety_metrics_collection': True
            },
            'traffic': {
                'ego_vehicles': 1,
                'background_vehicles': 12,
                'pedestrians': 15,
                'bicycles': 3,
                'motorcycles': 2,
                'traffic_lights': True,
                'batch_spawn': True,
                'max_spawn_attempts': 10,
                'vehicle_types': [
                    'vehicle.tesla.model3',
                    'vehicle.audi.tt',
                    'vehicle.nissan.patrol',
                    'vehicle.bmw.grandtourer',
                    'vehicle.mercedes.coupe',
                    'vehicle.ford.mustang',
                    'vehicle.lincoln.mkz2017',
                    'vehicle.chevrolet.impala'
                ],
                'pedestrian_types': [
                    'walker.pedestrian.0001',
                    'walker.pedestrian.0002',
                    'walker.pedestrian.0003',
                    'walker.pedestrian.0004',
                    'walker.pedestrian.0005',
                    'walker.pedestrian.0006',
                    'walker.pedestrian.0007',
                    'walker.pedestrian.0008'
                ],
                'pedestrian_behaviors': ['walking', 'crossing', 'waiting', 'running'],
                'speed_limit': 40.0,
                'pedestrian_safety_zones': True,
                'crosswalk_density': 'medium',
                'vulnerable_user_protection': True
            },
            'sensors': {
                'vehicle_cameras': 4,
                'infrastructure_cameras': 4,
                'lidar_sensors': 1,
                'radar_sensors': 0,
                'gps_sensors': 1,
                'imu_sensors': 1,
                'image_size': [1280, 720],
                'capture_interval': 2.0,
                'sensor_placement': 'optimized',
                'lidar_config': {
                    'channels': 32,
                    'range': 120.0,
                    'points_per_second': 100000,
                    'rotation_frequency': 10.0,
                    'horizontal_fov': 360.0,
                    'vertical_fov': 30.0,
                    'upper_fov': 15.0,
                    'lower_fov': -25.0,
                    'max_points_per_frame': 60000,
                    'downsample_ratio': 0.3,
                    'memory_warning_threshold': 400,
                    'max_batch_memory_mb': 60,
                    'v2x_save_interval': 5,
                    'compression_format': 'bin',
                    'pedestrian_detection': True,
                    'height_filter': [-0.5, 2.5]
                },
                'camera_config': {
                    'fov': 90.0,
                    'post_processing': 'semantic',
                    'exposure_mode': 'auto',
                    'motion_blur': 0.0,
                    'pedestrian_detection_mode': True,
                    'dynamic_range': 'high'
                },
                'radar_config': {
                    'range': 200.0,
                    'points_per_second': 2000,
                    'horizontal_fov': 60.0,
                    'vertical_fov': 20.0
                }
            },
            'v2x': {
                'enabled': True,
                'communication_range': 300.0,
                'bandwidth': 20.0,
                'latency_mean': 0.02,
                'latency_std': 0.005,
                'packet_loss_rate': 0.005,
                'message_types': ['bsm', 'spat', 'map', 'rsm', 'perception',
                                  'warning', 'pedestrian_warning', 'emergency_brake'],
                'update_interval': 1.0,
                'security_enabled': True,
                'encryption_level': 'basic',
                'qos_policy': 'priority_based',
                'enable_safety_warnings': True,
                'pedestrian_warning_threshold': 10.0,
                'emergency_brake_warning': True,
                'collision_prediction': True,
                'vulnerable_road_user_protection': True
            },
            'cooperative': {
                'num_coop_vehicles': 2,
                'enable_shared_perception': True,
                'enable_traffic_warnings': True,
                'enable_pedestrian_warnings': True,
                'enable_emergency_brake_assist': True,
                'enable_maneuver_coordination': True,
                'data_fusion_interval': 1.0,
                'max_shared_objects': 100,
                'object_matching_threshold': 3.0,
                'data_retention_time': 15.0,
                'consensus_method': 'weighted',
                'pedestrian_tracking': True,
                'intention_prediction': True,
                'risk_assessment': True
            },
            'enhancement': {
                'enabled': True,
                'enable_random': True,
                'quality_check': True,
                'save_original': True,
                'save_enhanced': True,
                'calibration_generation': True,
                'enhanced_dir_name': 'enhanced',
                'methods': ['normalize', 'contrast', 'brightness', 'pedestrian_highlight',
                            'safety_warning', 'risk_visualization'],
                'weather_effects': True,
                'augmentation_level': 'medium',
                'pedestrian_safety_mode': True,
                'vulnerable_user_protection': True,
                'emergency_scenario_simulation': True
            },
            'performance': {
                'batch_size': 8,
                'enable_compression': True,
                'compression_level': 4,
                'enable_downsampling': True,
                'enable_memory_cache': True,
                'max_cache_size': 80,
                'enable_async_processing': True,
                'max_workers': 4,
                'image_processing': {
                    'compress_images': True,
                    'compression_quality': 90,
                    'resize_images': False,
                    'resize_dimensions': [640, 480],
                    'format': 'jpg',
                    'progressive_encoding': True
                },
                'lidar_processing': {
                    'batch_size': 15,
                    'enable_compression': True,
                    'enable_downsampling': True,
                    'max_points_per_frame': 60000,
                    'memory_warning_threshold': 450,
                    'max_batch_memory_mb': 70,
                    'v2x_save_interval': 5,
                    'compression_method': 'zlib',
                    'parallel_processing': True
                },
                'fusion': {
                    'fusion_cache_size': 150,
                    'enable_cache': True,
                    'compression_enabled': True,
                    'real_time_fusion': True
                },
                'sensor_cleanup_timeout': 0.3,
                'frame_rate_limit': 8.0,
                'safety_monitoring_interval': 0.5,
                'emergency_response_time': 0.1,
                'memory_management': {
                    'gc_interval': 40,
                    'max_memory_mb': 500,
                    'early_stop_threshold': 450,
                    'memory_monitoring': True,
                    'leak_detection': True
                }
            },
            'output': {
                'data_dir': 'cvips_dataset_enhanced',
                'output_format': 'standard',
                'save_raw': True,
                'save_stitched': True,
                'save_annotations': True,
                'save_lidar': True,
                'save_fusion': True,
                'save_cooperative': True,
                'save_v2x_messages': True,
                'save_enhanced': True,
                'save_safety_reports': True,
                'save_risk_maps': True,
                'save_emergency_events': True,
                'validate_data': True,
                'run_analysis': True,
                'run_quality_check': True,
                'generate_summary': True,
                'generate_safety_summary': True,
                'generate_risk_assessment': True,
                'compression_enabled': True,
                'file_naming': 'timestamp',
                'backup_original': True,
                'metadata_inclusion': 'full'
            },
            'monitoring': {
                'enable_logging': True,
                'log_level': 'INFO',
                'log_file': 'cvips_enhanced.log',
                'log_rotation': True,
                'max_log_size': 100,
                'enable_performance_monitor': True,
                'performance_log_interval': 5.0,
                'enable_progress_bar': True,
                'enable_real_time_stats': True,
                'stats_update_interval': 2.0,
                'enable_safety_monitor': True,
                'safety_log_interval': 1.0,
                'enable_memory_monitor': True,
                'memory_log_interval': 10.0
            },
            'debug': {
                'enable_debug_mode': False,
                'save_debug_data': False,
                'debug_dir': 'debug',
                'print_config': False,
                'validate_sensors': True,
                'test_mode': False,
                'profiling': False,
                'traceback_limit': 5
            },
            'metadata': {
                'version': '2.0.0',
                'author': 'CVIPS System - Enhanced',
                'description': '行人安全增强数据采集配置',
                'created': datetime.now().isoformat(),
                'modified': '',
                'pedestrian_safety_features': True,
                'compatibility': {
                    'carla_version': '>=0.9.13',
                    'python_version': '>=3.7'
                }
            }
        }

    @staticmethod
    def _apply_preset(config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
        """应用预设配置"""
        if preset_name not in ConfigManager.PRESET_CONFIGS:
            print(f"⚠ 未知的预设配置: {preset_name}")
            return config

        preset = ConfigManager.PRESET_CONFIGS[preset_name]
        print(f"应用预设配置: {preset_name} - {preset['description']}")

        optimization = preset.get('optimization', 'balanced')
        if optimization == 'memory':
            config = ConfigOptimizer.optimize_for_memory(config)
        elif optimization == 'quality':
            config = ConfigOptimizer.optimize_for_quality(config)
        elif optimization == 'speed':
            config = ConfigOptimizer.optimize_for_speed(config)
        elif optimization == 'safety':
            config = ConfigOptimizer.optimize_for_safety(config)
        elif optimization == 'research':
            config = ConfigOptimizer.optimize_for_research(config)
        elif optimization == 'balanced':
            # 平衡配置：中等质量，中等性能
            pass  # 使用默认配置
        elif optimization == 'custom' and 'settings' in preset:
            config = ConfigManager._deep_update(config, preset['settings'])

        return config

    @staticmethod
    def _load_config_file(config_file: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if (config_file.endswith('.yaml') or config_file.endswith('.yml')) and YAML_AVAILABLE:
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)

            print(f"✓ 加载配置文件: {config_file}")
            return ConfigManager._deep_update(base_config, user_config)

        except json.JSONDecodeError as e:
            print(f"✗ JSON解析错误: {e}")
            return base_config
        except yaml.YAMLError as e:
            print(f"✗ YAML解析错误: {e}")
            return base_config
        except Exception as e:
            print(f"✗ 配置文件加载错误: {e}")
            return base_config

    @staticmethod
    def _deep_update(original: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """深度更新配置字典"""
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                ConfigManager._deep_update(original[key], value)
            else:
                original[key] = value
        return original

    @staticmethod
    def merge_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        """合并命令行参数"""
        arg_mappings = {
            'scenario': ('scenario', 'name'),
            'town': ('scenario', 'town'),
            'weather': ('scenario', 'weather'),
            'time_of_day': ('scenario', 'time_of_day'),
            'duration': ('scenario', 'duration'),
            'seed': ('scenario', 'seed'),
            'num_vehicles': ('traffic', 'background_vehicles'),
            'num_pedestrians': ('traffic', 'pedestrians'),
            'num_coop_vehicles': ('cooperative', 'num_coop_vehicles'),
            'capture_interval': ('sensors', 'capture_interval'),
            'batch_size': ('performance', 'batch_size'),
            'output_format': ('output', 'output_format'),
        }

        bool_mappings = {
            'enable_v2x': ('v2x', 'enabled'),
            'enable_enhancement': ('enhancement', 'enabled'),
            'enable_lidar': ('sensors', 'lidar_sensors'),
            'enable_fusion': ('output', 'save_fusion'),
            'enable_cooperative': ('output', 'save_cooperative'),
            'enable_annotations': ('output', 'save_annotations'),
            'enable_safety_monitor': ('monitoring', 'enable_safety_monitor'),
            'enable_compression': ('performance', 'enable_compression'),
            'enable_downsampling': ('performance', 'enable_downsampling'),
            'skip_validation': ('output', 'validate_data'),
            'skip_quality_check': ('output', 'run_quality_check'),
            'run_analysis': ('output', 'run_analysis'),
        }

        # 处理普通参数
        for arg_name, (section, key) in arg_mappings.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                if section in config and key in config[section]:
                    config[section][key] = getattr(args, arg_name)

        # 处理布尔参数
        for arg_name, (section, key) in bool_mappings.items():
            if hasattr(args, arg_name):
                if arg_name.startswith('skip_'):
                    config[section][key] = not getattr(args, arg_name)
                elif arg_name.startswith('enable_'):
                    value = getattr(args, arg_name)
                    if section == 'sensors' and key == 'lidar_sensors':
                        config[section][key] = 1 if value else 0
                    else:
                        config[section][key] = value

        # 特殊处理
        if hasattr(args, 'enable_lidar') and args.enable_lidar:
            config['output']['save_lidar'] = True

        if hasattr(args, 'enable_fusion'):
            config['output']['save_fusion'] = args.enable_fusion

        if hasattr(args, 'enable_cooperative'):
            config['output']['save_cooperative'] = args.enable_cooperative

        if hasattr(args, 'enable_safety_monitor'):
            config['monitoring']['enable_safety_monitor'] = args.enable_safety_monitor

        return config

    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str, format: str = 'json'):
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 更新修改时间
            if 'metadata' in config:
                config['metadata']['modified'] = datetime.now().isoformat()

            if format.lower() == 'yaml' and YAML_AVAILABLE:
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                print(f"✓ 配置保存为YAML: {output_path}")
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=False)
                print(f"✓ 配置保存为JSON: {output_path}")

            return True
        except Exception as e:
            print(f"✗ 保存配置失败: {e}")
            return False

    @staticmethod
    def generate_config_template(output_path: str, preset: Optional[str] = None):
        """生成配置模板"""
        config = ConfigManager.load_config(preset=preset, validate=False)
        config['metadata']['created'] = 'template'
        config['metadata']['description'] = f'配置模板 - {preset if preset else "通用"}'
        config['metadata']['modified'] = ''

        return ConfigManager.save_config(config, output_path)

    @staticmethod
    def print_config_summary(config: Dict[str, Any]):
        """打印配置摘要"""
        print("\n" + "=" * 60)
        print("配置摘要")
        print("=" * 60)

        scenario = config['scenario']
        print(f"\n📋 场景:")
        print(f"  名称: {scenario['name']}")
        print(f"  地图: {scenario['town']}")
        print(f"  天气/时间: {scenario['weather']}/{scenario['time_of_day']}")
        print(f"  时长: {scenario['duration']}秒")
        print(f"  行人安全模式: {'启用' if scenario.get('pedestrian_safety_mode', False) else '禁用'}")
        print(f"  随机种子: {scenario.get('seed', '随机')}")

        traffic = config['traffic']
        print(f"\n🚗 交通:")
        print(f"  主车: {traffic['ego_vehicles']}")
        print(f"  背景车辆: {traffic['background_vehicles']}")
        print(f"  行人: {traffic['pedestrians']}")
        print(f"  自行车: {traffic.get('bicycles', 0)}")
        print(f"  摩托车: {traffic.get('motorcycles', 0)}")
        print(f"  车速限制: {traffic.get('speed_limit', '无')} km/h")
        print(f"  行人安全区域: {'启用' if traffic.get('pedestrian_safety_zones', False) else '禁用'}")

        sensors = config['sensors']
        print(f"\n📷 传感器:")
        print(f"  车辆摄像头: {sensors['vehicle_cameras']}")
        print(f"  基础设施摄像头: {sensors['infrastructure_cameras']}")
        print(f"  LiDAR: {sensors['lidar_sensors']} (通道: {sensors['lidar_config']['channels']})")
        print(f"  采集间隔: {sensors['capture_interval']}秒")
        print(f"  图像尺寸: {sensors['image_size'][0]}x{sensors['image_size'][1]}")
        print(
            f"  行人检测模式: {'启用' if sensors['camera_config'].get('pedestrian_detection_mode', False) else '禁用'}")

        v2x = config['v2x']
        print(f"\n📡 V2X通信:")
        print(f"  状态: {'启用' if v2x['enabled'] else '禁用'}")
        if v2x['enabled']:
            print(f"  通信范围: {v2x['communication_range']}米")
            print(f"  更新间隔: {v2x['update_interval']}秒")
            print(f"  安全警告: {'启用' if v2x.get('enable_safety_warnings', False) else '禁用'}")
            print(f"  紧急制动: {'启用' if v2x.get('emergency_brake_warning', False) else '禁用'}")

        coop = config['cooperative']
        print(f"\n🤝 协同感知:")
        print(f"  协同车辆: {coop['num_coop_vehicles']}")
        print(f"  共享感知: {'启用' if coop['enable_shared_perception'] else '禁用'}")
        print(f"  行人警告: {'启用' if coop.get('enable_pedestrian_warnings', False) else '禁用'}")
        print(f"  紧急制动辅助: {'启用' if coop.get('enable_emergency_brake_assist', False) else '禁用'}")

        perf = config['performance']
        print(f"\n⚡ 性能:")
        print(f"  批处理大小: {perf['batch_size']}")
        print(f"  压缩: {'启用' if perf['enable_compression'] else '禁用'}")
        print(f"  下采样: {'启用' if perf['enable_downsampling'] else '禁用'}")
        print(f"  帧率限制: {perf['frame_rate_limit']} FPS")
        print(f"  安全监控间隔: {perf.get('safety_monitoring_interval', 1.0)}秒")
        print(f"  紧急响应时间: {perf.get('emergency_response_time', 0.1)}秒")

        output = config['output']
        print(f"\n💾 输出:")
        print(f"  输出目录: {output['data_dir']}")
        print(f"  输出格式: {output['output_format']}")
        enabled_outputs = [k.replace('save_', '') for k, v in output.items()
                           if isinstance(v, bool) and v and k.startswith('save_')]
        print(f"  启用输出: {', '.join(enabled_outputs[:8])}")
        if len(enabled_outputs) > 8:
            print(f"            {', '.join(enabled_outputs[8:])}")

        print(f"\n🛡️ 行人安全增强:")
        print(f"  安全监控: {'启用' if config['monitoring'].get('enable_safety_monitor', False) else '禁用'}")
        print(f"  增强安全模式: {'启用' if config['enhancement'].get('pedestrian_safety_mode', False) else '禁用'}")
        print(f"  脆弱用户保护: {'启用' if config['enhancement'].get('vulnerable_user_protection', False) else '禁用'}")

        print("\n📊 统计:")
        total_objects = (traffic['ego_vehicles'] + traffic['background_vehicles'] +
                         traffic['pedestrians'] + traffic.get('bicycles', 0) +
                         traffic.get('motorcycles', 0))
        print(f"  场景总对象数: {total_objects}")
        print(f"  预计数据量: {ConfigManager._estimate_data_size(config)}")

        print("=" * 60)

    @staticmethod
    def _estimate_data_size(config: Dict[str, Any]) -> str:
        """估计数据大小"""
        duration = config['scenario']['duration']
        interval = config['sensors']['capture_interval']
        frames = int(duration / interval)

        # 计算每帧数据大小
        image_size = config['sensors']['image_size']
        image_pixels = image_size[0] * image_size[1]

        # 粗略估计
        total_size_mb = frames * image_pixels * 3 / (1024 * 1024) * 0.1  # 假设压缩率

        if total_size_mb < 1024:
            return f"{total_size_mb:.1f} MB"
        else:
            return f"{total_size_mb / 1024:.1f} GB"

    @staticmethod
    def list_presets():
        """列出可用预设"""
        print("\n可用预设配置:")
        print("-" * 50)
        for name, preset in ConfigManager.PRESET_CONFIGS.items():
            print(f"  {name:20s} - {preset['description']}")
        print("-" * 50)


def load_config(config_file=None, preset=None):
    """加载配置（兼容函数）"""
    return ConfigManager.load_config(config_file, preset)


def merge_args(config, args):
    """合并参数（兼容函数）"""
    return ConfigManager.merge_args(config, args)