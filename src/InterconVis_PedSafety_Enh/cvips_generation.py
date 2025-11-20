import carla
import argparse
import time
import atexit

# 全局变量：存储生成的CARLA Actor（车辆、传感器等），用于退出时清理
generated_actors = []

def main():
    # 1. 解析命令行参数（带输入校验）
    parser = argparse.ArgumentParser(description='CVIPS场景数据生成工具')
    parser.add_argument('--town', type=str, required=True, choices=['Town04', 'Town10HD', 'Town07'], 
                        help='CARLA城镇地图（支持：Town04/Town10HD/Town07）')
    parser.add_argument('--intersection', type=str, required=True, choices=['3way', '4way'], 
                        help='路口类型（3way=三叉路口，4way=四岔路口）')
    parser.add_argument('--weather', type=str, required=True, choices=['clear', 'rainy', 'cloudy'], 
                        help='天气条件（clear=晴天，rainy=雨天，cloudy=阴天）')
    parser.add_argument('--time_of_day', type=str, required=True, choices=['noon', 'sunset', 'night'], 
                        help='时段（noon=中午，sunset=日落，night=夜晚）')
    args = parser.parse_args()

    # 2. 连接CARLA服务器（带重试机制）
    client, world = connect_carla_with_retry(args.town)
    if not world:
        return  # 连接失败直接退出

    # 3. 配置天气和时段（让参数实际生效）
    configure_weather_and_time(world, args.weather, args.time_of_day)

    # 4. 基于路口类型生成场景（预留核心逻辑，可根据需求扩展）
    generate_intersection_scene(world, args.intersection)

    # 5. 保持场景运行（按 Ctrl+C 退出并清理资源）
    print("\n🚗 场景已启动！按 Ctrl+C 退出并清理资源...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 收到退出信号，开始清理资源...")

# ------------------------------ 辅助函数 ------------------------------
def connect_carla_with_retry(town_name, max_retries=3, retry_interval=5):
    """带重试机制的CARLA连接函数"""
    client = None
    world = None
    for retry in range(max_retries):
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(15.0)
            client.load_world(town_name)  # 加载指定地图
            world = client.get_world()
            print(f"✅ 成功连接CARLA并加载地图：{town_name}")
            return client, world
        except Exception as e:
            error_msg = str(e)
            if retry < max_retries - 1:
                print(f"❌ 连接失败（{retry+1}/{max_retries}）：{error_msg}")
                print(f"⌛ {retry_interval}秒后重试...")
                time.sleep(retry_interval)
            else:
                print(f"❌ 连接失败（已达最大重试次数）：{error_msg}")
                print("💡 请检查：1. CARLA服务器是否启动 2. 端口是否为2000 3. 地图名称是否正确")
    return None, None

def configure_weather_and_time(world, weather_type, time_of_day):
    """配置CARLA的天气和时段（让命令行参数生效）"""
    # CARLA预设天气映射（对应自定义参数）
    base_weather_mapping = {
        'clear': carla.WeatherParameters.ClearNoon,
        'rainy': carla.WeatherParameters.RainyNoon,
        'cloudy': carla.WeatherParameters.CloudyNoon
    }

    # 获取基础天气参数并根据时段调整
    weather = base_weather_mapping[weather_type]
    if time_of_day == 'sunset':
        weather.sun_altitude_angle = -15.0  # 日落：太阳高度角为负（地平线以下）
        weather.ambient_light = 0.3          # 降低环境光强度
        weather.directional_light_intensity = 0.5  # 降低直射光强度
        weather.fog_density = 0.1            # 日落添加轻微雾气
    elif time_of_day == 'night':
        weather.sun_altitude_angle = -60.0  # 夜晚：太阳高度角极低
        weather.ambient_light = 0.05         # 极低环境光
        weather.directional_light_intensity = 0.01  # 几乎无直射光
        weather.moon_altitude_angle = 45.0   # 月亮高度角
        weather.moon_intensity = 0.8         # 月亮亮度
        weather.stars_intensity = 0.5        # 星星亮度
    # noon（中午）：使用默认预设，无需额外调整

    # 应用天气设置
    world.set_weather(weather)
    print(f"✅ 已配置环境：天气={weather_type}，时段={time_of_day}")

def generate_intersection_scene(world, intersection_type):
    """基于路口类型生成场景（核心逻辑，可扩展）"""
    print(f"📌 开始生成{intersection_type}路口场景...")
    
    # ------------------------------
    # 此处为预留逻辑，可根据需求扩展：
    # 1. 定位地图中对应类型的路口坐标（需提前调研目标地图的路口位置）
    # 2. 在路口周围生成车辆、行人、红绿灯等Actor
    # 3. 设置车辆行驶路线、交通规则等
    # ------------------------------
    example_spawn_point = get_example_spawn_point(world)
    if example_spawn_point:
        # 示例：生成1辆测试车辆（添加到全局列表，方便退出时清理）
        vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        vehicle = world.spawn_actor(vehicle_bp, example_spawn_point)
        generated_actors.append(vehicle)
        print(f"✅ 已在路口附近生成测试车辆（ID：{vehicle.id}）")
    else:
        print("⚠️  未找到合适的车辆生成点，路口场景生成失败")

def get_example_spawn_point(world):
    """获取示例生成点（可根据实际地图路口位置修改）"""
    spawn_points = world.get_map().get_spawn_points()
    # 取前10个生成点中的第3个作为示例（可根据地图路口位置调整索引）
    return spawn_points[2] if len(spawn_points) > 2 else None

def clean_up_actors():
    """退出时清理所有生成的Actor（车辆、传感器等）"""
    if generated_actors:
        print(f"🧹 正在清理 {len(generated_actors)} 个仿真对象...")
        for actor in generated_actors:
            if actor.is_alive:
                actor.destroy()
        print("✅ 资源清理完成！")
    else:
        print("✅ 无需要清理的仿真对象")

# 注册退出回调：程序终止时自动清理资源
atexit.register(clean_up_actors)

if __name__ == "__main__":
    main()