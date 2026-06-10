import carla
import math
import sys
sys.path.insert(0, '.')
from spawn_car import create_vehicle
from traffic_manager import TrafficManager
from adaptive_cruise_control import AdaptiveCruiseControl

def get_vehicle_speed(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

def speed_cruise_control(current_speed, target_speed):
    error = target_speed - current_speed
    throttle = min(max(0, error / 20), 1.0)
    brake = min(max(0, -error / 10), 1.0)
    return throttle, brake

def main():
    print("=== 自适应巡航控制演示 ===")
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    carla_map = world.get_map()
    vehicle = create_vehicle(world, carla_map)
    if not vehicle:
        print("[ERROR] 车辆生成失败")
        return
    
    print("\n=== 初始化自适应巡航控制 ===")
    acc = AdaptiveCruiseControl(vehicle, world, max_speed=40, safe_distance=20, follow_distance=40)
    print(f"[ACC] 安全距离: {acc.safe_distance}m, 跟车距离: {acc.follow_distance}m")
    print(f"[ACC] 最大速度: {acc.max_speed} km/h")
    
    print("\n=== 生成交通场景 ===")
    traffic_manager = TrafficManager(world, carla_map)
    traffic_manager.spawn_vehicles(15)
    traffic_manager.spawn_walkers(8)
    
    print("\n=== 启动自适应巡航 ===")
    print("[OK] ACC模式已启动")
    
    try:
        for frame in range(500):
            world.tick()
            current_speed = get_vehicle_speed(vehicle)
            target_speed = acc.get_target_speed()
            
            throttle, brake = speed_cruise_control(current_speed, target_speed)
            ctrl = carla.VehicleControl(throttle=throttle, brake=brake)
            vehicle.apply_control(ctrl)
            
            if frame % 50 == 0:
                front_distance = acc.detect_front_vehicle()
                mode = "巡航" if target_speed >= acc.max_speed * 0.9 else "跟车" if target_speed > 0 else "停车"
                print(f"\n--- 状态报告 (帧 {frame}) ---")
                print(f"  当前速度: {current_speed:.1f} km/h")
                print(f"  目标速度: {target_speed:.1f} km/h")
                print(f"  前车距离: {front_distance:.1f} m" if front_distance else "  前车距离: 无")
                print(f"  模式: {mode}")
                
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        print("\n=== 运行结束 ===")
        traffic_manager.cleanup()
        vehicle.destroy()
        print("[OK] 所有资源已清理")

if __name__ == "__main__":
    main()