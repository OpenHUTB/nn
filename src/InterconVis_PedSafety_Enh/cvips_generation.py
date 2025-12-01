import carla
import argparse
import time
import random
import logging

# 配置日志
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--town', type=str, default='Town01', help='城镇地图')
    argparser.add_argument(
        '--num_vehicles', default=30, type=int, help='生成车辆数量')
    argparser.add_argument(
        '--num_pedestrians', default=50, type=int, help='生成行人数量')
    argparser.add_argument(
        '--seed', default=None, type=int, help='随机种子')
    argparser.add_argument(
        '--weather', default='clear', choices=['clear', 'rainy', 'cloudy', 'night'], help='天气预设')
    args = argparser.parse_args()

    if args.seed:
        random.seed(args.seed)

    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)

    # 全局列表，用于清理
    vehicles_list = []
    walkers_list = []
    all_id = []

    try:
        # 1. 加载地图
        print(f"正在加载地图: {args.town}")
        client.load_world(args.town)
        world = client.get_world()

        # 2. 设置同步模式 (保证物理计算准确，防止瞬移)
        settings = world.get_settings()
        settings.synchronous_mode = True  
        settings.fixed_delta_seconds = 0.05 # 20 FPS
        world.apply_settings(settings)

        # 3. 配置交通管理器
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_hybrid_physics_mode(True) # 远处车辆关闭物理计算
        traffic_manager.set_respawn_dormant_vehicles(True)
        
        if args.seed:
            traffic_manager.set_random_device_seed(args.seed)

        # 4. 设置环境
        configure_weather(world, args.weather)

        # ==============================
        # 生成车辆 (Vehicles)
        # ==============================
        blueprint_library = world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter('vehicle.*')
        vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]
        vehicle_bps = [x for x in vehicle_bps if not x.id.endswith('microlino')]

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.num_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.num_vehicles > number_of_spawn_points:
            logging.warning(f"请求车辆数 {args.num_vehicles} 超过生成点数量 {number_of_spawn_points}")
            args.num_vehicles = number_of_spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.num_vehicles:
                break
            bp = random.choice(vehicle_bps)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            if bp.has_attribute('driver_id'):
                bp.set_attribute('driver_id', random.choice(bp.get_attribute('driver_id').recommended_values))
            
            bp.set_attribute('role_name', 'autopilot')

            batch.append(SpawnActor(bp, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        print(f"✅ 已生成 {len(vehicles_list)} 辆车")

        for actor in world.get_actors(vehicles_list):
            traffic_manager.vehicle_percentage_speed_difference(actor, random.randint(-10, 10))
            traffic_manager.ignore_lights_percentage(actor, 0)
            traffic_manager.ignore_signs_percentage(actor, 0)
            traffic_manager.auto_lane_change(actor, True)

        # ==============================
        # 生成行人 (Pedestrians)
        # ==============================
        spawn_points = []
        for i in range(args.num_pedestrians):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        walker_bp_list = blueprint_library.filter('walker.pedestrian.*')
        walker_controller_bp = blueprint_library.find('controller.ai.walker')

        batch = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(walker_bp_list)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            batch.append(SpawnActor(walker_bp, spawn_point))

        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})

        batch = []
        walker_controller_list = []
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))

        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
                walker_controller_list.append(results[i].actor_id)

        all_id = vehicles_list + [x['id'] for x in walkers_list] + [x['con'] for x in walkers_list if 'con' in x]
        print(f"✅ 已生成 {len(walkers_list)} 个行人")

        world.tick()
        world.set_pedestrians_cross_factor(0.1) 

        for i, controller_id in enumerate(walker_controller_list):
            controller = world.get_actor(controller_id)
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            
            # ============ 修改点开始 ============
            # 设置更自然的速度: 0.5 ~ 1.3 m/s (约 1.8 ~ 4.7 km/h)
            # 之前的 1.0~1.8 太快了
            speed = random.uniform(0.5, 1.3)
            controller.set_max_speed(speed)
            # ============ 修改点结束 ============

        print("\n🚀 仿真已启动！按 Ctrl+C 停止...")
        
        while True:
            world.tick()

    except KeyboardInterrupt:
        print("\n🛑 正在停止仿真...")
    finally:
        print(f"🧹 清理 {len(all_id)} 个仿真对象...")
        for i, controller_id in enumerate(walker_controller_list):
            try:
                controller = world.get_actor(controller_id)
                controller.stop()
            except:
                pass
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            
        print("✅ 清理完成，程序退出。")

def configure_weather(world, weather_type):
    weather_presets = {
        'clear': carla.WeatherParameters.ClearNoon,
        'cloudy': carla.WeatherParameters.CloudyNoon,
        'rainy': carla.WeatherParameters.HardRainNoon,
        'night': carla.WeatherParameters.ClearNight
    }
    preset = weather_presets.get(weather_type, carla.WeatherParameters.ClearNoon)
    world.set_weather(preset)
    print(f"🌤️  天气设置为: {weather_type}")

if __name__ == '__main__':
    main()