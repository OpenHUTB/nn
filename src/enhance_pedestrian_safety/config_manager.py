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
                'traffic_lights': True
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
                    'rotation_frequency': 10
                }
            },

            'v2x': {
                'enabled': True,
                'communication_range': 300.0,
                'bandwidth': 10.0,
                'latency_mean': 0.05,
                'latency_std': 0.01,
                'packet_loss_rate': 0.01,
                'message_types': ['bsm', 'spat', 'map', 'rsm']
            },
            'cooperative': {
                'num_coop_vehicles': 2,
                'enable_shared_perception': True,
                'enable_traffic_warnings': True,
                'enable_maneuver_coordination': False,
                'data_fusion_interval': 1.0,
                'max_shared_objects': 50
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


                'validate_data': True,
                'run_analysis': False
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


        if args.num_coop_vehicles:
            config['cooperative']['num_coop_vehicles'] = args.num_coop_vehicles

        if args.capture_interval:
            config['sensors']['capture_interval'] = args.capture_interval

        if args.enable_v2x:
            config['v2x']['enabled'] = True


        if args.capture_interval:
            config['sensors']['capture_interval'] = args.capture_interval


        if args.enable_lidar:
            config['sensors']['lidar_sensors'] = 1
            config['output']['save_lidar'] = True

        if args.enable_fusion:
            config['output']['save_fusion'] = True


        if args.enable_cooperative:
            config['output']['save_cooperative'] = True



        if args.enable_annotations:
            config['output']['save_annotations'] = True

        if args.skip_validation:
            config['output']['validate_data'] = False

        if args.run_analysis:
            config['output']['run_analysis'] = True

        return config