import os
import json
import random
import math
import carla


class SceneManager:
    """场景管理器 - 支持多种预定义场景"""

    # 场景预设配置
    SCENES = {
        'intersection_4way': {
            'description': '四路交叉口场景',
            'pedestrian_behavior': ['crossing', 'walking'],
            'vehicle_density': 0.7,
            'pedestrian_density': 0.8,
            'weather_variations': ['clear', 'cloudy']
        },
        'highway': {
            'description': '高速公路场景',
            'pedestrian_behavior': [],
            'vehicle_density': 0.9,
            'pedestrian_density': 0.0,
            'weather_variations': ['clear', 'rainy']
        },
        'urban_street': {
            'description': '城市街道场景',
            'pedestrian_behavior': ['walking', 'crossing', 'waiting'],
            'vehicle_density': 0.6,
            'pedestrian_density': 0.7,
            'weather_variations': ['clear', 'cloudy', 'rainy']
        },
        'night_scene': {
            'description': '夜间场景',
            'pedestrian_behavior': ['walking'],
            'vehicle_density': 0.5,
            'pedestrian_density': 0.4,
            'weather_variations': ['clear'],
            'time_of_day': 'night'
        },
        'rainy_intersection': {
            'description': '雨天交叉口',
            'pedestrian_behavior': ['crossing'],
            'vehicle_density': 0.6,
            'pedestrian_density': 0.5,
            'weather_variations': ['rainy']
        }
    }

    @staticmethod
    def setup_scene(world, config, scene_type='intersection_4way'):
        """设置特定场景"""
        if scene_type not in SceneManager.SCENES:
            print(f"未知场景类型: {scene_type}")
            return config

        scene_config = SceneManager.SCENES[scene_type]

        # 更新配置
        if 'time_of_day' in scene_config:
            config['scenario']['time_of_day'] = scene_config['time_of_day']

        if scene_config['weather_variations']:
            config['scenario']['weather'] = random.choice(scene_config['weather_variations'])

        # 调整交通密度
        base_vehicles = config['traffic']['background_vehicles']
        base_pedestrians = config['traffic']['pedestrians']

        config['traffic']['background_vehicles'] = int(base_vehicles * scene_config['vehicle_density'])
        config['traffic']['pedestrians'] = int(base_pedestrians * scene_config['pedestrian_density'])

        # 设置行人行为
        config['traffic']['pedestrian_behaviors'] = scene_config['pedestrian_behavior']

        # 应用场景特定设置
        SceneManager._apply_scene_specifics(world, scene_type)

        return config

    @staticmethod
    def _apply_scene_specifics(world, scene_type):
        """应用场景特定的设置"""
        try:
            if scene_type == 'highway':
                # 高速公路场景：设置较高车速
                for actor in world.get_actors():
                    if 'vehicle' in actor.type_id:
                        try:
                            actor.enable_constant_velocity(carla.Vector3D(25, 0, 0))
                        except:
                            pass

            elif scene_type == 'night_scene':
                # 夜间场景：调整灯光
                for actor in world.get_actors():
                    if 'vehicle' in actor.type_id:
                        try:
                            light_state = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
                            actor.set_light_state(light_state)
                        except:
                            pass

        except Exception as e:
            print(f"场景特定设置失败: {e}")

    @staticmethod
    def spawn_traffic_cones(world, center_location, num_cones=10):
        """生成交通锥桶"""
        blueprint_lib = world.get_blueprint_library()
        cone_bp = blueprint_lib.find('static.prop.roadcone')

        cones = []
        for i in range(num_cones):
            try:
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(3.0, 8.0)

                location = carla.Location(
                    x=center_location.x + distance * math.cos(angle),
                    y=center_location.y + distance * math.sin(angle),
                    z=center_location.z + 0.2
                )

                rotation = carla.Rotation(0, random.uniform(0, 360), 0)
                transform = carla.Transform(location, rotation)

                cone = world.spawn_actor(cone_bp, transform)
                cones.append(cone)

            except Exception as e:
                print(f"生成交通锥桶失败: {e}")

        return cones

    @staticmethod
    def spawn_construction_barriers(world, center_location, num_barriers=5):
        """生成施工障碍物"""
        blueprint_lib = world.get_blueprint_library()
        barrier_bp = blueprint_lib.find('static.prop.constructionbarrier')

        barriers = []
        for i in range(num_barriers):
            try:
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(5.0, 12.0)

                location = carla.Location(
                    x=center_location.x + distance * math.cos(angle),
                    y=center_location.y + distance * math.sin(angle),
                    z=center_location.z + 0.2
                )

                rotation = carla.Rotation(0, random.uniform(0, 360), 0)
                transform = carla.Transform(location, rotation)

                barrier = world.spawn_actor(barrier_bp, transform)
                barriers.append(barrier)

            except Exception as e:
                print(f"生成施工障碍物失败: {e}")

        return barriers

    @staticmethod
    def setup_traffic_accident(world, center_location, severity='minor'):
        """设置交通事故场景"""
        blueprint_lib = world.get_blueprint_library()

        # 生成事故车辆
        accident_vehicles = []
        for i in range(2):  # 两车事故
            try:
                vehicle_bp = random.choice(list(blueprint_lib.filter('vehicle.*')))

                # 稍微偏移位置，模拟碰撞
                offset_x = random.uniform(-3.0, 3.0) if i == 0 else random.uniform(2.0, 5.0)
                offset_y = random.uniform(-2.0, 2.0) if i == 0 else random.uniform(-5.0, -2.0)

                location = carla.Location(
                    x=center_location.x + offset_x,
                    y=center_location.y + offset_y,
                    z=center_location.z + 0.5
                )

                # 设置碰撞角度
                rotation = carla.Rotation(
                    0,
                    random.uniform(-30, 30) + (90 if i == 1 else 0),
                    0
                )

                transform = carla.Transform(location, rotation)
                vehicle = world.spawn_actor(vehicle_bp, transform)

                # 根据严重程度设置车辆损坏
                if severity == 'major':
                    vehicle.set_light_state(carla.VehicleLightState.Hazard)

                accident_vehicles.append(vehicle)

            except Exception as e:
                print(f"生成事故车辆失败: {e}")

        # 生成应急车辆（警车、救护车）
        emergency_vehicles = []
        try:
            police_bp = blueprint_lib.find('vehicle.dodge.charger_police')
            if police_bp:
                location = carla.Location(
                    x=center_location.x - 8.0,
                    y=center_location.y + 4.0,
                    z=center_location.z + 0.5
                )
                rotation = carla.Rotation(0, 45, 0)
                police = world.spawn_actor(police_bp, carla.Transform(location, rotation))
                police.set_light_state(carla.VehicleLightState.Special1)
                emergency_vehicles.append(police)
        except:
            pass

        return accident_vehicles, emergency_vehicles

    @staticmethod
    def save_scene_description(output_dir, scene_type, config, extra_info=None):
        """保存场景描述文件"""
        scene_info = {
            'scene_type': scene_type,
            'description': SceneManager.SCENES.get(scene_type, {}).get('description', '未知场景'),
            'config': config,
            'created': SceneManager._get_timestamp(),
            'extra_info': extra_info or {}
        }

        scene_file = os.path.join(output_dir, "metadata", "scene_description.json")
        with open(scene_file, 'w', encoding='utf-8') as f:
            json.dump(scene_info, f, indent=2, ensure_ascii=False)

        print(f"场景描述保存: {scene_file}")

    @staticmethod
    def _get_timestamp():
        from datetime import datetime
        return datetime.now().isoformat()