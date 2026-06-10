import carla
import time
from spawn_car import create_vehicle
from cruise_control import get_vehicle_speed, speed_cruise_control
from traffic_manager import TrafficManager
from traffic_light_handler import TrafficLightHandler
from collision_detector import CollisionDetector

def main():
    # 连接到模拟器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()

    vehicle = None
    traffic_manager = None
    collision_detector = None

    try:
        # 1. 生成主车辆
        vehicle = create_vehicle(world, carla_map)
        print("[OK] 主车辆已生成")

        # 2. 初始化交通场景
        print("\n=== 初始化交通场景 ===")
        traffic_manager = TrafficManager(world, carla_map, num_vehicles=20, num_walkers=10)
        traffic_light_handler = TrafficLightHandler(world)
        collision_detector = CollisionDetector(world, vehicle)

        # 3. 启动定速巡航
        print("\n=== 启动定速巡航 ===")
        target_speed = 30  # km/h

        print("[OK] 作业：定速巡航 + 交通场景功能启动")
        print("功能包括：")
        print("  - 定速巡航控制")
        print("  - 交通灯识别和响应")
        print("  - 碰撞检测和避免")
        print("  - 其他车辆和行人交互")

        # 主循环
        for i in range(500):
            world.tick()

            # 获取当前速度
            speed = get_vehicle_speed(vehicle)

            # 默认定速巡航控制
            throttle, brake = speed_cruise_control(speed, target_speed)
            ctrl = carla.VehicleControl(throttle=throttle, brake=brake)

            # 交通灯检查
            should_stop, tl_distance, tl_state = traffic_light_handler.should_stop_for_traffic_light(
                vehicle, stop_distance=10
            )
            if should_stop:
                ctrl = traffic_light_handler.get_stop_control()
                state_name = str(tl_state).split('.')[-1]
                print(f"[TrafficLight] 前方 {tl_distance:.1f}m 有{state_name}灯，停车等待")

            # 碰撞检测
            collision_risk, obstacle_info = collision_detector.check_front_collision(
                traffic_manager, safe_distance=15
            )
            if collision_risk and obstacle_info:
                ctrl = collision_detector.get_avoidance_control(obstacle_info, speed)
                print(f"[Collision] 前方 {obstacle_info['distance']:.1f}m 有{obstacle_info['type']}，"
                      f"TTC={obstacle_info['ttc']:.1f}s，执行避撞")

            # 应用控制
            vehicle.apply_control(ctrl)

            # 每50帧输出一次状态
            if i % 50 == 0:
                collision_count = collision_detector.get_collision_count()
                nearby_vehicles = len(traffic_manager.get_nearby_vehicles(vehicle, radius=50))
                nearby_walkers = len(traffic_manager.get_nearby_walkers(vehicle, radius=30))
                print(f"\n--- 状态报告 (帧 {i}) ---")
                print(f"  当前速度: {speed:.1f} km/h")
                print(f"  目标速度: {target_speed} km/h")
                print(f"  附近车辆: {nearby_vehicles}")
                print(f"  附近行人: {nearby_walkers}")
                print(f"  碰撞次数: {collision_count}")

            time.sleep(0.05)

        # 最终报告
        print("\n=== 运行结束 ===")
        print(f"总碰撞次数: {collision_detector.get_collision_count()}")

    finally:
        # 清理资源
        print("\n正在清理资源...")
        if collision_detector:
            collision_detector.cleanup()
        if traffic_manager:
            traffic_manager.cleanup()
        if vehicle:
            vehicle.destroy()
        print("[OK] 所有资源已清理")

if __name__ == "__main__":
    main()