#!/usr/bin/env python
"""
CARLA Basic Vehicle Spawn and Spectator Setup (sd_1/__main__.py)
完全适配 CARLA 0.9.15 版本（无任何天气预设依赖）

This script connects to a CARLA simulator instance, removes any
pre-existing vehicles with the role 'my_car', spawns a new
Tesla Model 3 at a default spawn point, and positions the
spectator camera behind the newly spawned vehicle.
新增功能：定时循环切换CARLA模拟器的天气（晴天、多云、雨天、雾天、日落等）

The script keeps the simulation running until interrupted (Ctrl+C),
but the vehicle does not move.
"""

# 导入CARLA模拟器的Python API
import carla
# 导入时间模块，用于延时和计时
import time


def remove_previous_vehicle(world: carla.World) -> None:
    """
    查找并销毁所有角色名为'my_car'的车辆Actor，避免重复生成导致冲突

    Args:
        world (carla.World): CARLA模拟器的世界对象，用于获取当前所有Actor
    """
    print("Searching for previous 'my_car' vehicles...")
    # 过滤出所有车辆类型的Actor（vehicle.* 匹配所有车辆蓝图）
    actors = world.get_actors().filter('vehicle.*')
    # 记录成功销毁的车辆数量
    removed_count = 0

    for actor in actors:
        # 检查Actor的角色名是否为'my_car'
        if actor.attributes.get('role_name') == 'my_car':
            print(f"  - Removing previous vehicle: {actor.type_id} (ID {actor.id})")
            # 销毁Actor，返回布尔值表示是否成功
            if actor.destroy():
                removed_count += 1
            else:
                print(f"  - Failed to remove vehicle {actor.id}")

    print(f"Removed {removed_count} previous vehicles.")


def set_spectator_behind_vehicle(world: carla.World, vehicle: carla.Vehicle) -> None:
    """
    将旁观者相机（spectator）定位到指定车辆的后上方，实现跟随视角

    Args:
        world (carla.World): CARLA模拟器的世界对象
        vehicle (carla.Vehicle): 目标车辆Actor，用于获取车辆的位置和姿态
    """
    # 获取旁观者相机Actor（CARLA中全局唯一的 spectator）
    spectator = world.get_spectator()
    # 获取车辆的当前位姿（位置+旋转）
    vehicle_transform = vehicle.get_transform()
    # 获取车辆的前向向量，用于计算相机的相对偏移（保证相机始终在车辆后方）
    forward_vector = vehicle_transform.get_forward_vector()

    # 计算相机偏移：向后15米，向上6米（基于车辆的前向向量，保证方向正确）
    camera_offset = carla.Location(
        x=-15 * forward_vector.x,
        y=-15 * forward_vector.y,
        z=6
    )
    # 构建旁观者相机的位姿：
    # 位置 = 车辆位置 + 偏移量
    # 旋转 = 俯仰角-20°（向下看），偏航角与车辆一致，滚转角为0
    spectator_transform = carla.Transform(
        vehicle_transform.location + camera_offset,
        carla.Rotation(
            pitch=-20,       # 俯仰角，负数表示向下看
            yaw=vehicle_transform.rotation.yaw,  # 偏航角与车辆一致
            roll=0           # 滚转角，保持水平
        )
    )

    # 尝试设置相机位姿，增加异常处理提高鲁棒性
    try:
        spectator.set_transform(spectator_transform)
        print("Spectator camera positioned behind the vehicle.")
    except Exception as e:
        print(f"Error setting spectator transform: {e}")


def get_weather_presets() -> list:
    """
    纯手动定义天气参数列表（完全不依赖CARLA预设，适配0.9.15）
    每个天气通过手动设置WeatherParameters的所有关键参数实现，确保兼容性

    Returns:
        list: 元组列表，每个元组包含（天气名称，carla.WeatherParameters对象）
    """
    # 1. 晴天中午（太阳高悬、无云、无雨、无雾）
    clear_noon = carla.WeatherParameters(
        sun_altitude_angle=75.0,    # 太阳高度角（75°=中午，天顶为90°）
        sun_azimuth_angle=90.0,     # 太阳方位角
        cloudiness=0.0,             # 云量（0=无云）
        precipitation=0.0,          # 降水量（0=无雨）
        precipitation_deposits=0.0, # 降水沉积（路面雨水）
        wind_intensity=5.0,         # 风力
        fog_density=0.0,            # 雾密度（0=无雾）
        fog_distance=0.0,           # 雾的可见距离
        fog_falloff=1.0,            # 雾的衰减率
        wetness=0.0,                # 路面湿度
        scattering_intensity=0.0,   # 光的散射强度
        mie_scattering_scale=0.0,   # 米氏散射比例
        rayleigh_scattering_scale=0.0 # 瑞利散射比例
    )

    # 2. 多云中午（高云量，阳光散射）
    cloudy_noon = carla.WeatherParameters(
        sun_altitude_angle=75.0,
        sun_azimuth_angle=90.0,
        cloudiness=80.0,            # 云量80%
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=10.0,
        fog_density=0.0,
        fog_distance=0.0,
        fog_falloff=1.0,
        wetness=0.0,
        scattering_intensity=0.1,
        mie_scattering_scale=0.1,
        rayleigh_scattering_scale=0.1
    )

    # 3. 小雨中午（少量降雨、路面微湿）
    light_rain_noon = carla.WeatherParameters(
        sun_altitude_angle=75.0,
        sun_azimuth_angle=90.0,
        cloudiness=90.0,            # 云量90%
        precipitation=20.0,         # 降水量20%（小雨）
        precipitation_deposits=5.0, # 路面雨水沉积5%
        wind_intensity=15.0,
        fog_density=5.0,            # 轻微雾霭
        fog_distance=50.0,
        fog_falloff=0.8,
        wetness=0.2,                # 路面湿度20%
        scattering_intensity=0.2,
        mie_scattering_scale=0.2,
        rayleigh_scattering_scale=0.2
    )

    # 4. 中雨中午（中等降雨、路面湿滑）
    mid_rain_noon = carla.WeatherParameters(
        sun_altitude_angle=75.0,
        sun_azimuth_angle=90.0,
        cloudiness=100.0,           # 满云
        precipitation=50.0,         # 降水量50%（中雨）
        precipitation_deposits=20.0, # 路面雨水沉积20%
        wind_intensity=20.0,
        fog_density=15.0,           # 雾密度15%
        fog_distance=30.0,
        fog_falloff=0.6,
        wetness=0.5,                # 路面湿度50%
        scattering_intensity=0.3,
        mie_scattering_scale=0.3,
        rayleigh_scattering_scale=0.3
    )

    # 5. 雾天中午（大雾、能见度低）
    mist_noon = carla.WeatherParameters(
        sun_altitude_angle=75.0,
        sun_azimuth_angle=90.0,
        cloudiness=50.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        fog_density=30.0,           # 雾密度30%（大雾）
        fog_distance=10.0,          # 雾的可见距离10米
        fog_falloff=0.5,
        wetness=0.1,
        scattering_intensity=0.4,
        mie_scattering_scale=0.4,
        rayleigh_scattering_scale=0.4
    )

    # 6. 晴天日落（太阳低垂、暖色调、无云）
    clear_sunset = carla.WeatherParameters(
        sun_altitude_angle=15.0,    # 太阳高度角15°（日落，地平线为0°）
        sun_azimuth_angle=180.0,    # 太阳方位角180°（西方）
        cloudiness=0.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        fog_density=0.0,
        fog_distance=0.0,
        fog_falloff=1.0,
        wetness=0.0,
        scattering_intensity=0.1,
        mie_scattering_scale=0.1,
        rayleigh_scattering_scale=0.1
    )

    # 7. 潮湿路面（无雨但路面湿滑、轻微雾）
    wet_road = carla.WeatherParameters(
        sun_altitude_angle=75.0,
        sun_azimuth_angle=90.0,
        cloudiness=30.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=10.0,
        fog_density=5.0,
        fog_distance=40.0,
        fog_falloff=0.9,
        wetness=0.8,                # 路面湿度80%（湿滑）
        scattering_intensity=0.1,
        mie_scattering_scale=0.1,
        rayleigh_scattering_scale=0.1
    )

    # 组合天气预设列表
    weather_presets = [
        ("Clear Noon", clear_noon),
        ("Cloudy Noon", cloudy_noon),
        ("Light Rain Noon", light_rain_noon),
        ("Mid Rain Noon", mid_rain_noon),
        ("Mist Noon", mist_noon),
        ("Clear Sunset", clear_sunset),
        ("Wet Road Noon", wet_road)
    ]

    return weather_presets


def switch_weather(world: carla.World, weather: carla.WeatherParameters, weather_name: str) -> None:
    """
    设置CARLA世界的天气，并打印切换信息

    Args:
        world (carla.World): CARLA模拟器的世界对象
        weather (carla.WeatherParameters): 目标天气参数对象
        weather_name (str): 天气名称，用于打印日志
    """
    try:
        # 设置世界天气
        world.set_weather(weather)
        print(f"\n=== Switched to weather: {weather_name} ===")
    except Exception as e:
        print(f"Error switching weather to {weather_name}: {e}")


def main() -> None:
    """
    主执行函数：
    1. 连接CARLA服务器
    2. 清理旧车辆
    3. 生成特斯拉Model3车辆
    4. 设置旁观者相机
    5. 定时循环切换天气
    6. 保持仿真运行直到用户中断
    """
    # 初始化变量，避免finally块中引用未定义的变量
    client: carla.Client = None
    world: carla.World = None
    vehicle: carla.Vehicle = None

    # 天气相关变量初始化
    weather_presets = get_weather_presets()  # 获取天气预设列表
    current_weather_index = 0  # 当前天气的索引
    weather_switch_interval = 10  # 天气切换间隔（秒）
    last_weather_switch_time = time.time()  # 上一次天气切换的时间戳

    try:
        # 连接到本地CARLA服务器（地址：localhost，端口：2000）
        client = carla.Client('localhost', 2000)
        # 设置连接超时时间（10秒），避免无限等待
        client.set_timeout(10.0)
        print("Connecting to CARLA server...")

        # 获取当前CARLA世界对象（包含地图、Actor、天气等信息）
        world = client.get_world()
        # 打印当前加载的地图名称
        print(f"Connected to world: {world.get_map().name}")

        # 清理之前运行残留的'my_car'车辆
        remove_previous_vehicle(world)

        # 获取地图的所有预设生成点（用于车辆/行人的生成）
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("Error: No spawn points found on the map!")
            return
        # 选择第一个生成点作为车辆生成位置
        spawn_point = spawn_points[0]

        # 获取蓝图库（包含所有可生成的Actor蓝图：车辆、行人、传感器等）
        vehicle_bp_library = world.get_blueprint_library()
        # 过滤出特斯拉Model3的蓝图（vehicle.tesla.model3 是CARLA中该车辆的唯一标识）
        vehicle_bp = vehicle_bp_library.filter('vehicle.tesla.model3')[0]
        # 设置车辆的角色名，方便后续清理
        vehicle_bp.set_attribute('role_name', 'my_car')

        # 尝试生成车辆（try_spawn_actor会检查生成点是否被占用，返回None表示失败）
        print("Attempting to spawn vehicle...")
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

        if vehicle is None:
            # 生成失败（可能生成点被占用）
            print(f"Error: Failed to spawn vehicle at {spawn_point.location}.")
            return

        # 打印生成成功的车辆信息
        print(f"Vehicle {vehicle.type_id} (ID {vehicle.id}) spawned successfully.")

        # 等待车辆稳定（等待一次仿真tick，再加0.5秒延时）
        world.wait_for_tick()
        time.sleep(0.5)

        # 设置旁观者相机到车辆后上方
        set_spectator_behind_vehicle(world, vehicle)

        # 初始化天气为第一个预设
        switch_weather(world, weather_presets[0][1], weather_presets[0][0])

        # 打印运行提示
        print(f"\nSimulation running. Vehicle is stationary.")
        print(f"Weather will switch every {weather_switch_interval} seconds.")
        print("Press Ctrl+C to stop.\n")

        # 主循环：保持仿真运行并定时切换天气
        while True:
            # 等待仿真tick（推进仿真时间）
            world.wait_for_tick()

            # 检查是否到达天气切换时间
            current_time = time.time()
            if current_time - last_weather_switch_time >= weather_switch_interval:
                # 切换到下一个天气（循环遍历预设列表）
                current_weather_index = (current_weather_index + 1) % len(weather_presets)
                weather_name, weather = weather_presets[current_weather_index]
                switch_weather(world, weather, weather_name)
                # 更新上一次切换时间
                last_weather_switch_time = current_time

            # 小延时，避免循环过于频繁（减少CPU占用）
            time.sleep(0.1)

    except KeyboardInterrupt:
        # 捕获用户Ctrl+C中断，友好退出
        print("\nScript stopped by user (Ctrl+C).")
    except Exception as e:
        # 捕获其他未预期的异常，打印错误信息和堆栈跟踪
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 资源清理：确保车辆被销毁，避免残留
        print("\nStarting resource cleanup...")
        if vehicle is not None and vehicle.is_alive:
            print(f"Destroying vehicle: {vehicle.type_id} (ID {vehicle.id})")
            if vehicle.destroy():
                print("Vehicle destroyed successfully.")
            else:
                print("Vehicle destroy() returned False.")
        else:
            print("Vehicle was None or not alive, no destruction needed.")

        print("Simulation finished.")


# 程序入口
if __name__ == '__main__':
    main()