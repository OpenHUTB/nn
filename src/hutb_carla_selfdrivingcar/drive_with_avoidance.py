import carla
import time
import math
from spawn_car import create_vehicle
from cruise_control import get_vehicle_speed, speed_cruise_control
from traffic_manager import TrafficManager

def get_distance_between_vehicles(vehicle1, vehicle2):
    """计算两辆车之间的距离"""
    loc1 = vehicle1.get_location()
    loc2 = vehicle2.get_location()
    return loc1.distance(loc2)

def get_angle_to_vehicle(ego_vehicle, target_vehicle):
    """计算目标车辆相对于自车的角度"""
    ego_transform = ego_vehicle.get_transform()
    ego_loc = ego_transform.location
    target_loc = target_vehicle.get_location()
    
    # 计算方向向量
    dx = target_loc.x - ego_loc.x
    dy = target_loc.y - ego_loc.y
    
    # 计算角度（相对于车辆前进方向）
    forward_vector = ego_transform.get_forward_vector()
    angle = math.atan2(dy, dx) - math.atan2(forward_vector.y, forward_vector.x)
    return math.degrees(angle)

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
        traffic_manager = TrafficManager(world, carla_map, num_vehicles=30, num_walkers=20)

        # 启动定速巡航
        print("\n=== 启动智能巡航 ===")
        target_speed = 45  # km/h
        safe_distance = 20  # 安全距离（米）

        print("[OK] 智能巡航模式启动")
        print(f"目标速度: {target_speed} km/h")
        print(f"安全距离: {safe_distance} m")

        # 主循环
        for i in range(1000):
            world.tick()

            # 获取当前速度
            speed = get_vehicle_speed(vehicle)

            # 默认控制
            throttle, brake = speed_cruise_control(speed, target_speed)
            steering = 0.0
            ctrl = carla.VehicleControl(throttle=throttle, brake=brake, steer=steering)

            # 检查前方障碍物
            nearby_vehicles = traffic_manager.get_nearby_vehicles(vehicle, radius=50)
            obstacle_ahead = None
            min_distance = float('inf')

            for other_vehicle in nearby_vehicles:
                distance = get_distance_between_vehicles(vehicle, other_vehicle)
                angle = get_angle_to_vehicle(vehicle, other_vehicle)

                # 检查是否在前方 ±30 度范围内
                if -30 < angle < 30 and distance < safe_distance and distance < min_distance:
                    obstacle_ahead = other_vehicle
                    min_distance = distance

            # 如果前方有障碍物
            if obstacle_ahead:
                print(f"[避障] 前方 {min_distance:.1f}m 有车辆，减速")
                # 减速或停车
                ctrl.throttle = 0.0
                ctrl.brake = min(1.0, (safe_distance - min_distance) / safe_distance)

            # 应用控制
            vehicle.apply_control(ctrl)

            # 每50帧输出一次状态
            if i % 50 == 0:
                nearby_vehicles_count = len(traffic_manager.get_nearby_vehicles(vehicle, radius=50))
                nearby_walkers_count = len(traffic_manager.get_nearby_walkers(vehicle, radius=30))
                print(f"\n--- 状态报告 (帧 {i}) ---")
                print(f"  当前速度: {speed:.1f} km/h")
                print(f"  目标速度: {target_speed} km/h")
                print(f"  附近车辆: {nearby_vehicles_count}")
                print(f"  附近行人: {nearby_walkers_count}")

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