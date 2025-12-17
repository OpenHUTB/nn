import json
import os
import argparse


class ConfigManager:
    """配置管理器"""

    @staticmethod
    def load_config(config_file=None):
        """加载配置文件"""
        config = {
            'scenario': {
                'name': 'intersection',
                'town': 'Town10HD',
                'weather': 'clear',
                'time_of_day': 'noon',
                'duration': 60,
                'seed': 42
            },
            'traffic': {
                'ego_vehicles': 1,
                'background_vehicles': 8,
                'pedestrians': 10,
                'traffic_lights': True,
                'pedestrian_behavior': {
                    'crossing_probability': 0.3,
                    'walking_speed': 1.2
                }
            },
            'sensors': {
                'vehicle_cameras': 4,
                'infrastructure_cameras': 4,
                'image_size': [1280, 720],
                'capture_interval': 2.0,
                'fov': 90
            },
            'output': {
                'data_dir': 'cvips_dataset',
                'save_raw': True,
                'save_stitched': True,
                'save_metadata': True,
                'save_annotations': False,
                'compression': False
            }
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    if config_file.endswith('.json'):
                        user_config = json.load(f)
                    else:
                        # 尝试导入yaml，如果失败则跳过
                        try:
                            import yaml
                            user_config = yaml.safe_load(f)
                        except ImportError:
                            print("警告: PyYAML未安装，无法加载YAML配置文件")
                            print("请安装: pip install pyyaml")
                            user_config = {}
                ConfigManager._deep_update(config, user_config)
            except Exception as e:
                print(f"配置文件加载失败: {e}")

        return config

    @staticmethod
    def _deep_update(original, update):
        """深度更新配置字典"""
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                ConfigManager._deep_update(original[key], value)
            else:
                original[key] = value

    @staticmethod
    def save_config(config, filepath):
        """保存配置到文件"""
        with open(filepath, 'w') as f:
            if filepath.endswith('.json'):
                json.dump(config, f, indent=2)
            else:
                try:
                    import yaml
                    yaml.dump(config, f, default_flow_style=False)
                except ImportError:
                    print("警告: PyYAML未安装，无法保存YAML配置文件")
                    json.dump(config, f, indent=2)

    @staticmethod
    def merge_args(config, args):
        """合并命令行参数到配置"""
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
        if hasattr(args, 'seed') and args.seed:
            config['scenario']['seed'] = args.seed

        if args.num_vehicles:
            config['traffic']['background_vehicles'] = args.num_vehicles
        if args.num_pedestrians:
            config['traffic']['pedestrians'] = args.num_pedestrians
        if hasattr(args, 'ego_vehicles') and args.ego_vehicles:
            config['traffic']['ego_vehicles'] = args.ego_vehicles
        if args.capture_interval:
            config['sensors']['capture_interval'] = args.capture_interval

        return config