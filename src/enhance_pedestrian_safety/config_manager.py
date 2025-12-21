import json
import os
import argparse


class ConfigManager:

    @staticmethod
    def load_config(config_file=None):
        config = {
            'scenario': {
                'name': 'multi_sensor_scene',
                'town': 'Town10HD',
                'weather': 'clear',
                'time_of_day': 'noon',
                'duration': 60,
                'seed': 42
            },
            'traffic': {
                'ego_vehicles': 1,
                'background_vehicles': 8,
                'pedestrians': 6,
                'traffic_lights': True,
                'batch_spawn': True,
                'max_spawn_attempts': 5
            },
            'sensors': {
                'vehicle_cameras': 4,
                'infrastructure_cameras': 4,
                'lidar_sensors': 1,
                'radar_sensors': 0,
                'image_size': [1280, 720],
                'capture_interval': 2.0,
                'lidar_config': {
                    'channels': 32,
                    'range': 100,
                    'points_per_second': 56000,
                    'rotation_frequency': 10,
                    'max_points_per_frame': 50000,  # 优化点6：减少最大点数
                    'downsample_ratio': 0.3,  # 优化点6：增加下采样比例
                    'memory_warning_threshold': 350,  # 新增：内存警告阈值(MB)
                    'max_batch_memory_mb': 50,  # 新增：批次最大内存
                    'v2x_save_interval': 5  # 新增：V2X格式保存间隔
                }
            },
            'v2x': {
                'enabled': True,
                'communication_range': 300.0,
                'bandwidth': 10.0,
                'latency_mean': 0.05,
                'latency_std': 0.01,
                'packet_loss_rate': 0.01,
                'message_types': ['bsm', 'spat', 'map', 'rsm'],
                'update_interval': 2.0  # 优化点6：V2X更新间隔
            },
            'cooperative': {
                'num_coop_vehicles': 2,
                'enable_shared_perception': True,
                'enable_traffic_warnings': True,
                'enable_maneuver_coordination': False,
                'data_fusion_interval': 1.0,
                'max_shared_objects': 50
            },
            'enhancement': {
                'enabled': True,
                'enable_random': True,
                'quality_check': True,
                'save_original': True,
                'save_enhanced': True,
                'calibration_generation': True,
                'enhanced_dir_name': 'enhanced'
            },
            'performance': {
                'batch_size': 5,
                'enable_compression': True,
                'compression_level': 3,
                'enable_downsampling': True,
                'enable_memory_cache': True,
                'max_cache_size': 50,
                'image_processing': {
                    'compress_images': True,
                    'compression_quality': 85
                },
                'lidar_processing': {
                    'batch_size': 10,
                    'enable_compression': True,
                    'enable_downsampling': True,
                    'max_points_per_frame': 50000,  # 与sensors配置保持一致
                    'memory_warning_threshold': 350,
                    'max_batch_memory_mb': 50,
                    'v2x_save_interval': 5
                },
                'fusion': {
                    'fusion_cache_size': 100
                },
                'sensor_cleanup_timeout': 0.5,  # 新增：传感器清理超时
                'frame_rate_limit': 5.0  # 新增：帧率限制
            },
            'output': {
                'data_dir': 'cvips_dataset',
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
                'output_format': 'standard'
            }
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                ConfigManager._deep_update(config, user_config)
            except:
                pass

        return config

    @staticmethod
    def _deep_update(original, update):
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                ConfigManager._deep_update(original[key], value)
            else:
                original[key] = value

    @staticmethod
    def merge_args(config, args):
        if args.scenario:
            config['scenario']['name'] = args.scenario
        if args.town:
            config['scenario']['town'] = args.town
        if args.weather:
            config['scenario']['weather'] = args.weather
        if args.time_of_day:
            config['scenario']['time_of_day'] = args.time_of_day
        if args.duration:
            config['scenario']['duration'] = args.duration
        if args.seed:
            config['scenario']['seed'] = args.seed

        if args.num_vehicles:
            config['traffic']['background_vehicles'] = args.num_vehicles
        if args.num_pedestrians:
            config['traffic']['pedestrians'] = args.num_pedestrians

        if hasattr(args, 'num_coop_vehicles') and args.num_coop_vehicles:
            config['cooperative']['num_coop_vehicles'] = args.num_coop_vehicles

        if hasattr(args, 'capture_interval') and args.capture_interval:
            config['sensors']['capture_interval'] = args.capture_interval

        if hasattr(args, 'enable_v2x') and args.enable_v2x:
            config['v2x']['enabled'] = True
            # 如果通过命令行启用V2X，使用更保守的更新间隔
            if 'update_interval' not in config['v2x']:
                config['v2x']['update_interval'] = 2.0

        if hasattr(args, 'enable_enhancement') and args.enable_enhancement:
            config['enhancement']['enabled'] = True

        if hasattr(args, 'enable_lidar') and args.enable_lidar:
            config['sensors']['lidar_sensors'] = 1
            config['output']['save_lidar'] = True
            # 优化：如果启用LiDAR，使用更保守的设置
            config['sensors']['lidar_config']['max_points_per_frame'] = 50000
            config['sensors']['lidar_config']['downsample_ratio'] = 0.3

        if hasattr(args, 'enable_fusion') and args.enable_fusion:
            config['output']['save_fusion'] = True

        if hasattr(args, 'enable_cooperative') and args.enable_cooperative:
            config['output']['save_cooperative'] = True

        if hasattr(args, 'enable_annotations') and args.enable_annotations:
            config['output']['save_annotations'] = True

        if hasattr(args, 'skip_validation') and args.skip_validation:
            config['output']['validate_data'] = False

        if hasattr(args, 'skip_quality_check') and args.skip_quality_check:
            config['output']['run_quality_check'] = False

        if hasattr(args, 'run_analysis') and args.run_analysis:
            config['output']['run_analysis'] = True

        # 性能参数
        if hasattr(args, 'batch_size') and args.batch_size:
            config['performance']['batch_size'] = args.batch_size

        if hasattr(args, 'enable_compression') and args.enable_compression:
            config['performance']['enable_compression'] = True

        if hasattr(args, 'enable_downsampling') and args.enable_downsampling:
            config['performance']['enable_downsampling'] = True
            # 如果启用下采样，使用更保守的设置
            config['sensors']['lidar_config']['downsample_ratio'] = 0.3
            config['performance']['lidar_processing']['max_points_per_frame'] = 50000

        if hasattr(args, 'output_format') and args.output_format:
            config['output']['output_format'] = args.output_format

        return config