import carla
import argparse
import time
import atexit
import random

# 全局变量
generated_actors = []
client = None
world = None

def main():
    global client, world

    parser = argparse.ArgumentParser(description='CARLA 0.9.14 车辆和行人生成工具 (优化观测版)')
    parser.add_argument('--town', type=str, default='Town01', help='城镇地图 (例如: Town01, Town04)')
    parser.add_argument('--num_vehicles', type=int, default=20, help='生成车辆数量')
    parser.add_argument('--num_pedestrians', type=int, default=100, help='生成行人数量')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--weather', type=str, default='clear', choices=['clear', 'rainy', 'cloudy'], help='天气类型')
    parser.add_argument('--time_of_day', type=str, default='noon', choices=['noon', 'sunset', 'night'], help='时段')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"✅ 已设置随机种子: {args.seed}")

    client, world = connect_carla_with_retry(args.town)
    if not world:
        return

    print("\n🌤️  正在配置环境...")
    configure_weather_and_time(world, args.weather, args.time_of_day)

    print("\n📌 开始生成场景...")
    generate_vehicles(world, args.num_vehicles)
    generate_pedestrians(world, args.num_pedestrians)
    print("\n✅ 场景生成完毕！")

    print("\n🚗 场景已启动！按 Ctrl+C 退出并清理资源...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 收到退出信号，开始清理资源...")

# ------------------------------ 辅助函数 ------------------------------
def connect_carla_with_retry(town_name, max_retries=3, retry_interval=5):
    global client
    for retry in range(max_retries):
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(15.0)
            client.load_world(town_name)
            world = client.get_world()
            print(f"✅ 成功连接 CARLA 并加载地图：{town_name}")
            return client, world
        except Exception as e:
            error_msg = str(e)
            if retry < max_retries - 1:
                print(f"❌ 连接失败（{retry + 1}/{max_retries}）：{error_msg}")
                print(f"⌛ {retry_interval}秒后重试...")
                time.sleep(retry_interval)
            else:
                print(f"❌ 连接失败（已达最大重试次数）：{error_msg}")
                print("💡 请检查：1. CARLA 服务器是否启动 2. 端口是否为 2000")
    return None, None

def configure_weather_and_time(world, weather_type, time_of_day):
    weather = carla.WeatherParameters()
    if weather_type == 'clear':
        weather.cloudiness = 0; weather.precipitation = 0
    elif weather_type == 'rainy':
        weather.cloudiness = 80; weather.precipitation = 50; weather.precipitation_deposits = 30; weather.wind_intensity = 10; weather.fog_density = 0.3
    elif weather_type == 'cloudy':
        weather.cloudiness = 70; weather.precipitation = 0; weather.fog_density = 0.2

    if time_of_day == 'noon':
        weather.sun_altitude_angle = 90; weather.ambient_light = 1.0
    elif time_of_day == 'sunset':
        weather.sun_altitude_angle = -15; weather.ambient_light = 0.3
    elif time_of_day == 'night':
        weather.sun_altitude_angle = -60; weather.ambient_light = 0.05; weather.moon_altitude_angle = 45; weather.moon_intensity = 0.8
    
    world.set_weather(weather)
    print(f"✅ 已配置环境：天气={weather_type}，时段={time_of_day}")

def generate_vehicles(world, num_vehicles):
    if num_vehicles <= 0:
        print("ℹ️  车辆数量为 0，不生成车辆。")
        return

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("⚠️  未找到车辆生成点。")
        return

    vehicle_bps = world.get_blueprint_library().filter('vehicle.*')
    tm = client.get_trafficmanager(8000)
    tm.set_global_distance_to_leading_vehicle(2.5)

    print(f"🚗 正在生成 {num_vehicles} 辆车辆...")
    for i in range(num_vehicles):
        spawn_point = random.choice(spawn_points)
        vehicle_bp = random.choice(vehicle_bps)
        try:
            max_speed_kmh = random.uniform(20, 30)
            vehicle_bp.set_attribute('speed', str(max_speed_kmh / 3.6))
        except Exception:
            pass
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            generated_actors.append(vehicle)
            vehicle.set_autopilot(True, tm.get_port())

    num_generated = len([a for a in generated_actors if a.type_id.startswith('vehicle')])
    print(f"✅ 成功生成 {num_generated} 辆车辆。")

def generate_pedestrians(world, num_pedestrians):
    """生成指定数量的行人并让他们随机行走 (优化观测版)"""
    if num_pedestrians <= 0:
        print("ℹ️  行人数量为 0，不生成行人。")
        return

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("⚠️  未找到任何生成点，无法生成行人。")
        return

    pedestrian_bps = world.get_blueprint_library().filter('*walker*')
    pedestrian_bps = [bp for bp in pedestrian_bps if bp.id.startswith('walker.pedestrian')]
    if not pedestrian_bps:
        print("❌ 未找到行人蓝图！请检查 CARLA 资产是否完整。")
        return

    controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    print(f"👤 正在生成 {num_pedestrians} 个行人... (优化分散度)")
    
    success_count = 0
    for i in range(num_pedestrians):
        retry_count = 5
        while retry_count > 0:
            spawn_point = random.choice(spawn_points)
            
            # --- 核心优化：增加随机偏移，让行人分布更分散 ---
            spawn_point.location.x += random.uniform(-8.0, 8.0)
            spawn_point.location.y += random.uniform(-8.0, 8.0)
            spawn_point.location.z += 0.5
            
            try:
                pedestrian_bp = random.choice(pedestrian_bps)
                pedestrian = world.spawn_actor(pedestrian_bp, spawn_point)
                if pedestrian:
                    generated_actors.append(pedestrian)
                    controller = world.spawn_actor(controller_bp, carla.Transform(), pedestrian)
                    generated_actors.append(controller)
                    
                    # --- 行为优化 ---
                    controller.start()
                    controller.go_to_location(world.get_random_location_from_navigation())
                    controller.set_max_speed(random.uniform(0.5, 1.5)) # 更自然的速度范围
                    
                    success_count += 1
                    break
            except RuntimeError:
                retry_count -= 1
                if retry_count == 0:
                    # 减少失败提示，避免刷屏
                    if i % 20 == 0:
                        print(f"⚠️  部分行人生成位置被占用，已跳过。")
        
        if (i + 1) % 25 == 0:
            print(f"👤 已生成 {success_count}/{i + 1} 个行人")

    print(f"✅ 成功生成 {success_count} 个行人。")

def clean_up_actors():
    global client
    if generated_actors:
        print(f"\n🧹 正在清理 {len(generated_actors)} 个仿真对象...")
        try:
            batch = [carla.command.DestroyActor(x) for x in generated_actors]
            if client:
                client.apply_batch(batch)
                time.sleep(1)
            generated_actors.clear()
            print("✅ 资源清理完成！")
        except Exception as e:
            print(f"⚠️  清理资源时发生错误: {e}")
    else:
        print("\n✅ 无需要清理的仿真对象。")

atexit.register(clean_up_actors)

if __name__ == "__main__":
    main()