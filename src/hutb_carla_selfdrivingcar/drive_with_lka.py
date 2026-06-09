import carla
import time
from spawn_car import create_vehicle
from cruise_control import get_vehicle_speed, speed_cruise_control
from traffic_manager import TrafficManager
from lane_keep_assist import LaneKeepAssist

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    bp_lib = world.get_blueprint_library()

    vehicle = None
    traffic_manager = None
    lka = None

    try:
        # 1. 生成主车辆
        vehicle = create_vehicle(world, carla_map)
        print("[OK] 主车辆已生成")

        # 2. 初始化车道保持辅助
        print("\n=== 初始化车道保持辅助 ===")
        lka = LaneKeepAssist(vehicle, world, bp_lib)
        lka.setup_camera()

        # 3. 生成交通场景
        print("\n=== 生成交通场景 ===")
        traffic_manager = TrafficManager(world, carla_map, num_vehicles=20, num_walkers=10)

        # 4. 启动驾驶
        print("\n=== 启动车道保持巡航 ===")
        target_speed = 40  # km/h

        print("[OK] 车道保持模式已启动")
        print(f"目标速度: {target_speed} km/h")
        print("按 Ctrl+C 退出\n")

        # 主循环
        for i in range(800):
            world.tick()

            # 获取速度
            speed = get_vehicle_speed(vehicle)

            # 方向盘控制
            steer = lka.get_steering()

            # 速度控制
            throttle, brake = speed_cruise_control(speed, target_speed)

            # 应用控制
            ctrl = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
            vehicle.apply_control(ctrl)

            # 每50帧输出状态
            if i % 50 == 0:
                nearby_vehicles = len(traffic_manager.get_nearby_vehicles(vehicle, radius=50))
                nearby_walkers = len(traffic_manager.get_nearby_walkers(vehicle, radius=30))
                
                lka_status = "✓ 车道检测" if not lka.queue.empty() else "✗ 未检测"
                
                print(f"\n--- 状态报告 (帧 {i}) ---")
                print(f"  当前速度: {speed:.1f} km/h")
                print(f"  目标速度: {target_speed} km/h")
                print(f"  方向盘: {steer:.2f}")
                print(f"  车道检测: {lka_status}")
                print(f"  附近车辆: {nearby_vehicles}")
                print(f"  附近行人: {nearby_walkers}")

            time.sleep(0.03)

        print("\n=== 运行结束 ===")

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        print("\n正在清理资源...")
        if lka:
            lka.cleanup()
        if traffic_manager:
            traffic_manager.cleanup()
        if vehicle:
            vehicle.destroy()
        print("[OK] 所有资源已清理")

if __name__ == "__main__":
    main()