import carla
import time
from spawn_car import create_vehicle
from cruise_control import get_vehicle_speed, speed_cruise_control
from traffic_manager import TrafficManager
from weather_manager import WeatherManager

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()

    vehicle = None
    traffic_manager = None
    weather_manager = None

    try:
        # 1. 生成主车辆
        vehicle = create_vehicle(world, carla_map)
        print("[OK] 主车辆已生成")

        # 2. 初始化天气系统
        print("\n=== 初始化天气系统 ===")
        weather_manager = WeatherManager(world)
        
        # 设置初始天气为晴天
        weather_manager.set_weather('sunny')

        # 3. 生成交通场景
        print("\n=== 生成交通场景 ===")
        traffic_manager = TrafficManager(world, carla_map, num_vehicles=15, num_walkers=8)

        # 4. 启动驾驶
        print("\n=== 启动巡航 ===")
        target_speed = 35  # km/h
        
        print("[OK] 天气系统演示模式已启动")
        print(f"目标速度: {target_speed} km/h")
        print("按 Ctrl+C 退出")
        print("\n天气将每30秒自动切换一次\n")

        weather_list = ['sunny', 'cloudy', 'rainy', 'stormy', 'soft_rain', 'night', 'rainy_night']
        weather_idx = 0

        # 主循环
        for i in range(1000):
            world.tick()

            # 获取速度
            speed = get_vehicle_speed(vehicle)

            # 速度控制
            throttle, brake = speed_cruise_control(speed, target_speed)
            ctrl = carla.VehicleControl(throttle=throttle, brake=brake)
            vehicle.apply_control(ctrl)

            # 每30秒切换天气
            if i % 600 == 0 and i > 0:
                weather_idx = (weather_idx + 1) % len(weather_list)
                weather_manager.set_weather(weather_list[weather_idx])

            # 每50帧输出状态
            if i % 50 == 0:
                nearby_vehicles = len(traffic_manager.get_nearby_vehicles(vehicle, radius=50))
                nearby_walkers = len(traffic_manager.get_nearby_walkers(vehicle, radius=30))
                
                print(f"\n--- 状态报告 (帧 {i}) ---")
                print(f"  当前速度: {speed:.1f} km/h")
                print(f"  目标速度: {target_speed} km/h")
                print(f"  当前天气: {weather_list[weather_idx]}")
                print(f"  附近车辆: {nearby_vehicles}")
                print(f"  附近行人: {nearby_walkers}")

            time.sleep(0.03)

        print("\n=== 运行结束 ===")

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        print("\n正在清理资源...")
        if traffic_manager:
            traffic_manager.cleanup()
        if vehicle:
            vehicle.destroy()
        print("[OK] 所有资源已清理")

if __name__ == "__main__":
    main()