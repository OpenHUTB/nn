#!/usr/bin/env python3
import carla
import time
from spawn_car import create_vehicle
from traffic_manager import TrafficManager
from auto_overtake import AutoOvertake
from cruise_control import speed_cruise_control

def main():
    print("=== 自动超车系统演示 ===\n")
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    
    print("[1] 生成主车辆...")
    vehicle = create_vehicle(world, carla_map)
    if vehicle is None:
        print("[ERROR] 主车辆生成失败！")
        return
    print("[OK] 主车辆已生成\n")
    
    print("[2] 初始化自动超车系统...")
    overtake = AutoOvertake(vehicle, world, min_distance=15.0, speed_diff=10.0, overtake_speed=50.0)
    print(f"[Overtake] 最小超车距离: {overtake.min_distance}m")
    print(f"[Overtake] 超车速度: {overtake.overtake_speed} km/h")
    print("[OK] 超车系统已就绪\n")
    
    print("[3] 生成交通场景...")
    traffic_manager = TrafficManager(world, carla_map)
    traffic_manager.spawn_vehicles(20)
    traffic_manager.spawn_walkers(8)
    print("[OK] 交通场景已生成\n")
    
    print("=== 启动巡航 ===\n")
    print(f"[OK] 巡航模式启动")
    print(f"目标速度: {overtake.overtake_speed} km/h\n")
    
    frame_count = 0
    
    try:
        for i in range(600):
            world.tick()
            frame_count += 1
            
            current_speed = overtake.get_vehicle_speed()
            steer, target_speed, state = overtake.update(current_speed)
            
            # 控制
            throttle, brake = speed_cruise_control(current_speed, target_speed)
            control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
            vehicle.apply_control(control)
            
            # 状态显示
            if frame_count % 50 == 0:
                print(f"--- 状态报告 (帧 {frame_count}) ---")
                print(f"  当前速度: {current_speed:.1f} km/h")
                print(f"  目标速度: {target_speed:.1f} km/h")
                print(f"  前车距离: {overtake.front_distance:.1f} m")
                print(f"  前车速度: {overtake.front_speed:.1f} km/h")
                print(f"  超车状态: {state}")
                print()
                
                if state == "detected":
                    print("[Overtake] 检测到慢车，准备超车")
                elif state == "changing_left":
                    print("[Overtake] 向左变道 ←")
                elif state == "overtaking":
                    print("[Overtake] 超车中 💨")
                elif state == "returning":
                    print("[Overtake] 返回车道 →")
                elif state == "completed":
                    print("[Overtake] 超车完成 ✓")
                print()
            
            time.sleep(0.1)
        
        print("=== 运行结束 ===")
        print(f"总运行帧数: {frame_count}")
        
    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断")
    finally:
        print("\n[INFO] 清理资源...")
        traffic_manager.cleanup()
        vehicle.destroy()
        print("[OK] 所有资源已清理")

if __name__ == "__main__":
    main()
