import json
import os
import argparse
import copy
from typing import Dict, Any, Optional, List, Tuple

# 尝试导入yaml，如果不存在则跳过
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("警告: PyYAML未安装，将使用JSON格式")


class ConfigValidator:
    """配置验证器"""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证配置的有效性"""
        errors = []

        # 验证必填字段
        required_sections = ['scenario', 'sensors', 'output']
        for section in required_sections:
            if section not in config:
                errors.append(f"缺失必要配置节: {section}")

        # 验证场景配置
        if 'scenario' in config:
            scenario = config['scenario']
            if 'duration' in scenario and scenario['duration'] <= 0:
                errors.append("场景时长必须大于0")
            if 'town' not in scenario:
                errors.append("场景配置中缺失地图名称")

        # 验证传感器配置
        if 'sensors' in config:
            sensors = config['sensors']
            if 'capture_interval' in sensors and sensors['capture_interval'] <= 0:
                errors.append("采集间隔必须大于0")
            if 'image_size' in sensors:
                if len(sensors['image_size']) != 2:
                    errors.append("图像尺寸必须为[宽度, 高度]格式")
                elif any(dim <= 0 for dim in sensors['image_size']):
                    errors.append("图像尺寸必须大于0")

        # 验证性能配置
        if 'performance' in config:
            perf = config['performance']
            if 'batch_size' in perf and perf['batch_size'] <= 0:
                errors.append("批处理大小必须大于0")

        return len(errors) == 0, errors

    @staticmethod
    def suggest_optimizations(config: Dict[str, Any]) -> List[str]:
        """根据配置提供优化建议"""
        suggestions = []

        # 内存使用建议
        if config.get('sensors', {}).get('lidar_sensors', 0) > 0:
            lidar_config = config['sensors'].get('lidar_config', {})
            max_points = lidar_config.get('max_points_per_frame', 50000)
            if max_points > 50000:
                suggestions.append(f"LiDAR最大点数({max_points})较高，建议降低到50000以下以减少内存使用")

        # 性能建议
        capture_interval = config['sensors'].get('capture_interval', 2.0)
        if capture_interval < 1.0:
            suggestions.append(f"采集间隔({capture_interval}s)较短，可能导致高负载，建议增加到1.0s以上")

        # 输出建议
        output = config.get('output', {})
        enabled_outputs = [k for k, v in output.items() if isinstance(v, bool) and v]
        if len(enabled_outputs) > 5:
            suggestions.append(f"启用的输出类型过多({len(enabled_outputs)})，可能影响性能，建议只启用必要的输出")

        return suggestions


class ConfigOptimizer:
    """配置优化器"""

    @staticmethod
    def optimize_for_memory(config: Dict[str, Any]) -> Dict[str, Any]:
        """为内存使用优化配置"""
        optimized = copy.deepcopy(config)

        # 调整LiDAR设置
        if optimized['sensors'].get('lidar_sensors', 0) > 0:
            lidar_config = optimized['sensors'].setdefault('lidar_config', {})
            lidar_config.update({
                'max_points_per_frame': 30000,
                'downsample_ratio': 0.4,
                'memory_warning_threshold': 200,
                'max_batch_memory_mb': 30
            })

        # 调整性能设置
        perf = optimized.setdefault('performance', {})
        perf.update({
            'batch_size': 3,
            'enable_compression': True,
            'compression_level': 4,
            'enable_memory_cache': True,
            'max_cache_size': 30,
            'frame_rate_limit': 3.0
        })

        # 调整图像处理
        perf['image_processing'] = {
            'compress_images': True,
            'compression_quality': 80,
            'resize_images': False
        }

        return optimized

    @staticmethod
    def optimize_for_quality(config: Dict[str, Any]) -> Dict[str, Any]:
        """为数据质量优化配置"""
        optimized = copy.deepcopy(config)

        # 调整传感器设置
        sensors = optimized['sensors']
        sensors.update({
            'image_size': [1920, 1080],
            'capture_interval': 1.0,
            'lidar_sensors': 1,
            'lidar_config': {
                'channels': 64,
                'range': 150.0,
                'points_per_second': 120000,
                'max_points_per_frame': 100000,
                'downsample_ratio': 0.1
            }
        })

        # 调整输出设置
        output = optimized['output']
        output.update({
            'save_annotations': True,
            'save_fusion': True,
            'save_cooperative': True,
            'save_enhanced': True,
            'run_quality_check': True
        })

        # 调整增强设置
        enhanced = optimized.setdefault('enhancement', {})
        enhanced.update({
            'enabled': True,
            'enable_random': True,
            'quality_check': True,
            'methods': ['normalize', 'contrast', 'sharpness', 'noise']
        })

        return optimized

    @staticmethod
    def optimize_for_speed(config: Dict[str, Any]) -> Dict[str, Any]:
        """为处理速度优化配置"""
        optimized = copy.deepcopy(config)

        # 调整传感器设置
        sensors = optimized['sensors']
        sensors.update({
            'image_size': [640, 480],
            'capture_interval': 3.0,
            'lidar_sensors': 0,  # 禁用LiDAR
            'radar_sensors': 0  # 禁用雷达
        })

        # 调整性能设置
        perf = optimized.setdefault('performance', {})
        perf.update({
            'batch_size': 10,
            'enable_compression': True,
            'compression_level': 1,  # 快速压缩
            'enable_downsampling': True,
            'enable_async_processing': True,
            'max_cache_size': 20,
            'frame_rate_limit': 10.0
        })

        # 简化输出
        output = optimized['output']
        output.update({
            'save_raw': True,
            'save_stitched': False,  # 不拼接图像
            'save_annotations': False,
            'save_lidar': False,
            'save_fusion': False,
            'save_cooperative': False
        })

        return optimized


class ConfigManager:
    """配置管理器（增强版）"""

    # 预定义配置模板
    PRESET_CONFIGS = {
        'balanced': {
            'description': '平衡配置 - 兼顾性能和质量',
            'optimization': 'memory'
        },
        'high_quality': {
            'description': '高质量配置 - 优先数据质量',
            'optimization': 'quality'
        },
        'fast_collection': {
            'description': '快速采集配置 - 优先处理速度',
            'optimization': 'speed'
        },
        'v2x_focused': {
            'description': 'V2X重点配置 - 优化协同数据采集',
            'optimization': 'custom',
            'settings': {
                'v2x': {'enabled': True, 'update_interval': 1.0},
                'cooperative': {'num_coop_vehicles': 3, 'enable_shared_perception': True},
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
        }
    }

    @staticmethod
    def load_config(config_file: Optional[str] = None, preset: Optional[str] = None) -> Dict[str, Any]:
        """
        加载配置

        Args:
            config_file: 配置文件路径
            preset: 预设配置名称

        Returns:
            配置字典
        """
        # 基础配置
        config = ConfigManager._get_default_config()

        # 应用预设配置
        if preset:
            config = ConfigManager._apply_preset(config, preset)

        # 加载用户配置文件
        if config_file:
            if os.path.exists(config_file):
                config = ConfigManager._load_config_file(config_file, config)
            else:
                print(f"警告: 配置文件不存在: {config_file}")

        # 验证配置
        is_valid, errors = ConfigValidator.validate_config(config)
        if not is_valid:
            print("配置验证错误:")
            for error in errors:
                print(f"  - {error}")
            raise ValueError("配置验证失败")

        # 提供优化建议
        suggestions = ConfigValidator.suggest_optimizations(config)
        if suggestions:
            print("配置优化建议:")
            for suggestion in suggestions:
                print(f"  ⚡ {suggestion}")

        return config

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """获取默认配置（增强版）"""
        return {
            'scenario': {
                'name': 'multi_sensor_scene',
                'description': '多传感器协同数据采集场景',
                'town': 'Town10HD',
                'weather': 'clear',
                'time_of_day': 'noon',
                'duration': 60,
                'seed': 42,
                'timeout': 300,  # 超时时间（秒）
                'retry_attempts': 3  # 重试次数
            },
            'traffic': {
                'ego_vehicles': 1,
                'background_vehicles': 8,
                'pedestrians': 6,
                'traffic_lights': True,
                'batch_spawn': True,
                'max_spawn_attempts': 5,
                'vehicle_types': [
                    'vehicle.tesla.model3',
                    'vehicle.audi.tt',
                    'vehicle.nissan.patrol',
                    'vehicle.bmw.grandtourer'
                ],
                'pedestrian_types': [
                    'walker.pedestrian.0001',
                    'walker.pedestrian.0002'
                ]
            },
            'sensors': {
                'vehicle_cameras': 4,
                'infrastructure_cameras': 4,
                'lidar_sensors': 1,
                'radar_sensors': 0,
                'gps_sensors': 0,
                'imu_sensors': 0,
                'image_size': [1280, 720],
                'capture_interval': 2.0,
                'sensor_placement': 'default',
                'lidar_config': {
                    'channels': 32,
                    'range': 100.0,
                    'points_per_second': 56000,
                    'rotation_frequency': 10.0,
                    'horizontal_fov': 360.0,
                    'vertical_fov': 30.0,
                    'upper_fov': 10.0,
                    'lower_fov': -20.0,
                    'max_points_per_frame': 50000,
                    'downsample_ratio': 0.3,
                    'memory_warning_threshold': 300,
                    'max_batch_memory_mb': 50,
                    'v2x_save_interval': 5,
                    'compression_format': 'bin'  # bin, npy, pcd
                },
                'camera_config': {
                    'fov': 90.0,
                    'post_processing': 'default',
                    'exposure_mode': 'auto',
                    'motion_blur': 0.0
                }
            },
            'v2x': {
                'enabled': True,
                'communication_range': 300.0,
                'bandwidth': 10.0,
                'latency_mean': 0.05,
                'latency_std': 0.01,
                'packet_loss_rate': 0.01,
                'message_types': ['bsm', 'spat', 'map', 'rsm', 'perception', 'warning'],
                'update_interval': 2.0,
                'security_enabled': False,
                'encryption_level': 'none',
                'qos_policy': 'best_effort'
            },
            'cooperative': {
                'num_coop_vehicles': 2,
                'enable_shared_perception': True,
                'enable_traffic_warnings': True,
                'enable_maneuver_coordination': False,
                'data_fusion_interval': 1.0,
                'max_shared_objects': 50,
                'object_matching_threshold': 5.0,
                'data_retention_time': 10.0,
                'consensus_method': 'simple'
            },
            'enhancement': {
                'enabled': True,
                'enable_random': True,
                'quality_check': True,
                'save_original': True,
                'save_enhanced': True,
                'calibration_generation': True,
                'enhanced_dir_name': 'enhanced',
                'methods': ['normalize', 'contrast', 'brightness'],
                'weather_effects': True,
                'augmentation_level': 'medium'
            },
            'performance': {
                'batch_size': 5,
                'enable_compression': True,
                'compression_level': 3,
                'enable_downsampling': True,
                'enable_memory_cache': True,
                'max_cache_size': 50,
                'enable_async_processing': True,
                'max_workers': 2,
                'image_processing': {
                    'compress_images': True,
                    'compression_quality': 85,
                    'resize_images': False,
                    'resize_dimensions': [640, 480],
                    'format': 'jpg'  # jpg, png
                },
                'lidar_processing': {
                    'batch_size': 10,
                    'enable_compression': True,
                    'enable_downsampling': True,
                    'max_points_per_frame': 50000,
                    'memory_warning_threshold': 350,
                    'max_batch_memory_mb': 50,
                    'v2x_save_interval': 5,
                    'compression_method': 'zlib'  # zlib, lz4, none
                },
                'fusion': {
                    'fusion_cache_size': 100,
                    'enable_cache': True,
                    'compression_enabled': True
                },
                'sensor_cleanup_timeout': 0.5,
                'frame_rate_limit': 5.0,
                'memory_management': {
                    'gc_interval': 50,
                    'max_memory_mb': 500,
                    'early_stop_threshold': 400
                }
            },
            'output': {
                'data_dir': 'cvips_dataset',
                'output_format': 'standard',  # standard, v2xformer, kitti, coco
                'save_raw': True,
                'save_stitched': True,
                'save_annotations': False,
                'save_lidar': True,
                'save_fusion': True,
                'save_cooperative': True,
                'save_v2x_messages': True,
                'save_enhanced': True,
                'validate_data': True,
                'run_analysis': False,
                'run_quality_check': True,
                'generate_summary': True,
                'compression_enabled': True,
                'file_naming': 'sequential',  # sequential, timestamp
                'backup_original': False
            },
            'monitoring': {
                'enable_logging': True,
                'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
                'log_file': 'cvips.log',
                'enable_performance_monitor': True,
                'performance_log_interval': 10.0,
                'enable_progress_bar': True,
                'enable_real_time_stats': True,
                'stats_update_interval': 5.0
            },
            'debug': {
                'enable_debug_mode': False,
                'save_debug_data': False,
                'debug_dir': 'debug',
                'print_config': False,
                'validate_sensors': True,
                'test_mode': False
            },
            'metadata': {
                'version': '1.0.0',
                'author': 'CVIPS System',
                'description': '多传感器协同数据采集配置',
                'created': '',
                'modified': ''
            }
        }

    @staticmethod
    def _apply_preset(config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
        """应用预设配置"""
        if preset_name not in ConfigManager.PRESET_CONFIGS:
            print(f"警告: 未知的预设配置: {preset_name}")
            return config

        preset = ConfigManager.PRESET_CONFIGS[preset_name]
        print(f"应用预设配置: {preset_name} - {preset['description']}")

        # 根据优化类型应用配置
        optimization = preset.get('optimization', 'balanced')
        if optimization == 'memory':
            config = ConfigOptimizer.optimize_for_memory(config)
        elif optimization == 'quality':
            config = ConfigOptimizer.optimize_for_quality(config)
        elif optimization == 'speed':
            config = ConfigOptimizer.optimize_for_speed(config)
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

            print(f"加载配置文件: {config_file}")
            return ConfigManager._deep_update(base_config, user_config)

        except Exception as e:
            print(f"配置文件加载错误: {e}")
            return base_config

    @staticmethod
    def _deep_update(original: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """深度更新字典"""
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                ConfigManager._deep_update(original[key], value)
            else:
                original[key] = value
        return original

    @staticmethod
    def merge_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        """合并命令行参数到配置"""
        # 场景参数
        if hasattr(args, 'scenario') and args.scenario:
            config['scenario']['name'] = args.scenario
        if hasattr(args, 'town') and args.town:
            config['scenario']['town'] = args.town
        if hasattr(args, 'weather') and args.weather:
            config['scenario']['weather'] = args.weather
        if hasattr(args, 'time_of_day') and args.time_of_day:
            config['scenario']['time_of_day'] = args.time_of_day
        if hasattr(args, 'duration') and args.duration:
            config['scenario']['duration'] = args.duration
        if hasattr(args, 'seed') and args.seed:
            config['scenario']['seed'] = args.seed

        # 交通参数
        if hasattr(args, 'num_vehicles') and args.num_vehicles:
            config['traffic']['background_vehicles'] = args.num_vehicles
        if hasattr(args, 'num_pedestrians') and args.num_pedestrians:
            config['traffic']['pedestrians'] = args.num_pedestrians

        # 协同参数
        if hasattr(args, 'num_coop_vehicles') and args.num_coop_vehicles:
            config['cooperative']['num_coop_vehicles'] = args.num_coop_vehicles

        # 传感器参数
        if hasattr(args, 'capture_interval') and args.capture_interval:
            config['sensors']['capture_interval'] = args.capture_interval

        # V2X参数
        if hasattr(args, 'enable_v2x'):
            config['v2x']['enabled'] = args.enable_v2x

        # 增强参数
        if hasattr(args, 'enable_enhancement'):
            config['enhancement']['enabled'] = args.enable_enhancement

        # LiDAR参数
        if hasattr(args, 'enable_lidar'):
            config['sensors']['lidar_sensors'] = 1 if args.enable_lidar else 0
            config['output']['save_lidar'] = args.enable_lidar

        # 融合参数
        if hasattr(args, 'enable_fusion'):
            config['output']['save_fusion'] = args.enable_fusion

        # 协同参数
        if hasattr(args, 'enable_cooperative'):
            config['output']['save_cooperative'] = args.enable_cooperative

        # 标注参数
        if hasattr(args, 'enable_annotations'):
            config['output']['save_annotations'] = args.enable_annotations

        # 验证参数
        if hasattr(args, 'skip_validation'):
            config['output']['validate_data'] = not args.skip_validation

        # 质量检查参数
        if hasattr(args, 'skip_quality_check'):
            config['output']['run_quality_check'] = not args.skip_quality_check

        # 分析参数
        if hasattr(args, 'run_analysis'):
            config['output']['run_analysis'] = args.run_analysis

        # 性能参数
        if hasattr(args, 'batch_size') and args.batch_size:
            config['performance']['batch_size'] = args.batch_size

        if hasattr(args, 'enable_compression'):
            config['performance']['enable_compression'] = args.enable_compression

        if hasattr(args, 'enable_downsampling'):
            config['performance']['enable_downsampling'] = args.enable_downsampling
            if args.enable_downsampling:
                config['sensors']['lidar_config']['downsample_ratio'] = 0.3

        # 输出格式
        if hasattr(args, 'output_format') and args.output_format:
            config['output']['output_format'] = args.output_format

        return config

    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str, format: str = 'json'):
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if format.lower() == 'yaml' and YAML_AVAILABLE:
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

            print(f"配置保存到: {output_path}")
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False

    @staticmethod
    def generate_config_template(output_path: str, preset: Optional[str] = None):
        """生成配置模板"""
        config = ConfigManager.load_config(preset=preset)
        config['metadata']['created'] = 'template'
        config['metadata']['description'] = f'配置模板 - {preset if preset else "通用"}'

        return ConfigManager.save_config(config, output_path)

    @staticmethod
    def print_config_summary(config: Dict[str, Any]):
        """打印配置摘要"""
        print("\n" + "=" * 60)
        print("配置摘要")
        print("=" * 60)

        # 场景信息
        scenario = config['scenario']
        print(f"\n📋 场景:")
        print(f"  名称: {scenario['name']}")
        print(f"  地图: {scenario['town']}")
        print(f"  天气/时间: {scenario['weather']}/{scenario['time_of_day']}")
        print(f"  时长: {scenario['duration']}秒")
        print(f"  随机种子: {scenario.get('seed', '随机')}")

        # 交通信息
        traffic = config['traffic']
        print(f"\n🚗 交通:")
        print(f"  主车: {traffic['ego_vehicles']}")
        print(f"  背景车辆: {traffic['background_vehicles']}")
        print(f"  行人: {traffic['pedestrians']}")
        print(f"  交通灯: {'启用' if traffic['traffic_lights'] else '禁用'}")

        # 传感器信息
        sensors = config['sensors']
        print(f"\n📷 传感器:")
        print(f"  车辆摄像头: {sensors['vehicle_cameras']}")
        print(f"  基础设施摄像头: {sensors['infrastructure_cameras']}")
        print(f"  LiDAR: {sensors['lidar_sensors']} (通道: {sensors['lidar_config']['channels']})")
        print(f"  采集间隔: {sensors['capture_interval']}秒")
        print(f"  图像尺寸: {sensors['image_size'][0]}x{sensors['image_size'][1]}")

        # V2X信息
        v2x = config['v2x']
        print(f"\n📡 V2X通信:")
        print(f"  状态: {'启用' if v2x['enabled'] else '禁用'}")
        if v2x['enabled']:
            print(f"  通信范围: {v2x['communication_range']}米")
            print(f"  更新间隔: {v2x['update_interval']}秒")

        # 协同信息
        coop = config['cooperative']
        print(f"\n🤝 协同感知:")
        print(f"  协同车辆: {coop['num_coop_vehicles']}")
        print(f"  共享感知: {'启用' if coop['enable_shared_perception'] else '禁用'}")
        print(f"  交通警告: {'启用' if coop['enable_traffic_warnings'] else '禁用'}")

        # 性能信息
        perf = config['performance']
        print(f"\n⚡ 性能:")
        print(f"  批处理大小: {perf['batch_size']}")
        print(f"  压缩: {'启用' if perf['enable_compression'] else '禁用'}")
        print(f"  下采样: {'启用' if perf['enable_downsampling'] else '禁用'}")
        print(f"  帧率限制: {perf['frame_rate_limit']} FPS")

        # 输出信息
        output = config['output']
        print(f"\n💾 输出:")
        print(f"  输出目录: {output['data_dir']}")
        print(f"  输出格式: {output['output_format']}")
        enabled_outputs = [k.replace('save_', '') for k, v in output.items()
                           if isinstance(v, bool) and v and k.startswith('save_')]
        print(f"  启用输出: {', '.join(enabled_outputs)}")

        print("=" * 60)

    @staticmethod
    def list_presets():
        """列出所有预设配置"""
        print("\n可用预设配置:")
        print("-" * 40)
        for name, preset in ConfigManager.PRESET_CONFIGS.items():
            print(f"  {name:15s} - {preset['description']}")
        print("-" * 40)


# 兼容旧版本的接口
def load_config(config_file=None):
    """兼容旧版本的加载配置函数"""
    return ConfigManager.load_config(config_file)


def merge_args(config, args):
    """兼容旧版本的合并参数函数"""
    return ConfigManager.merge_args(config, args)