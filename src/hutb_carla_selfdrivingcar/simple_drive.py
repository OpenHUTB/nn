import carla
import time
from spawn_car import create_vehicle
from cruise_control import get_vehicle_speed, speed_cruise_control
from traffic_manager import TrafficManager

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()

    vehicle = None
    traffic_manager = None

    try:
        # 生成主车辆
        vehicle = create_vehicle(world, carla_map)
        print("[OK] 主车辆已生成")

        # 生成大量交通车辆和行人
        print("\n=== 生成交通场景 ===")
        traffic_manager = TrafficManager(world, carla_map, num_vehicles=25, num_walkers=15)

        # 启动定速巡航（不检查交通灯）
        print("\n=== 启动定速巡航 ===")
        target_speed = 40  # km/h

        print("[OK] 简单驾驶模式启动")
        print(f"目标速度: {target_speed} km/h")

        # 主循环
        for i in range(800):
            world.tick()

            # 获取当前速度
            speed = get_vehicle_speed(vehicle)

            # 定速巡航控制
            throttle, brake = speed_cruise_control(speed, target_speed)
            ctrl = carla.VehicleControl(throttle=throttle, brake=brake)

            # 应用控制
            vehicle.apply_control(ctrl)

            # 每50帧输出一次状态
            if i % 50 == 0:
                nearby_vehicles = len(traffic_manager.get_nearby_vehicles(vehicle, radius=50))
                nearby_walkers = len(traffic_manager.get_nearby_walkers(vehicle, radius=30))
                print(f"\n--- 状态报告 (帧 {i}) ---")
                print(f"  当前速度: {speed:.1f} km/h")
                print(f"  目标速度: {target_speed} km/h")
                print(f"  附近车辆: {nearby_vehicles}")
                print(f"  附近行人: {nearby_walkers}")

            time.sleep(0.03)

        print("\n=== 运行结束 ===")

    finally:
        print("\n正在清理资源...")
        if traffic_manager:
            traffic_manager.cleanup()
        if vehicle:
            vehicle.destroy()
        print("[OK] 所有资源已清理")

if __name__ == "__main__":
    main()