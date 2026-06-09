#!/usr/bin/env python3
import carla
import time
import math
import sys
from spawn_car import create_vehicle
from traffic_manager import TrafficManager
from aeb_system import AEBSystem
from cruise_control import speed_cruise_control

def main():
    print("=== 自动紧急刹车系统（AEB）演示 ===\n")
    
    # 连接模拟器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    blueprint_lib = world.get_blueprint_library()
    
    # 生成主车辆
    print("[1] 生成主车辆...")
    vehicle = create_vehicle(world, carla_map)
    if vehicle is None:
        print("[ERROR] 主车辆生成失败！")
        return
    print("[OK] 主车辆已生成\n")
    
    # 初始化AEB系统
    print("[2] 初始化AEB系统...")
    aeb_system = AEBSystem(vehicle, world, warning_distance=15.0, emergency_distance=5.0)
    print(f"[AEB] 警告距离: {aeb_system.warning_distance}m")
    print(f"[AEB] 紧急制动距离: {aeb_system.emergency_distance}m")
    print("[OK] AEB系统已就绪\n")
    
    # 生成交通场景
    print("[3] 生成交通场景...")
    traffic_manager = TrafficManager(world, carla_map)
    traffic_manager.spawn_vehicles(20)
    traffic_manager.spawn_walkers(8)
    print("[OK] 交通场景已生成\n")
    
    # 设置目标速度
    target_speed = 40  # km/h
    
    print("=== 启动巡航 ===\n")
    print("[OK] 巡航模式启动")
    print(f"目标速度: {target_speed} km/h\n")
    
    frame_count = 0
    try:
        for i in range(500):
            world.tick()
            frame_count += 1
            
            # 获取当前速度
            speed = aeb_system.get_vehicle_speed()
            
            # 更新AEB系统
            brake_intensity, distance, status = aeb_system.update()
            
            # 根据AEB状态控制车辆
            if brake_intensity > 0:
                # AEB正在工作，使用AEB的刹车控制
                pass
            else:
                # 正常巡航控制
                throttle, brake = speed_cruise_control(speed, target_speed)
                control = carla.VehicleControl(throttle=throttle, brake=brake)
                vehicle.apply_control(control)
            
            # 每50帧显示状态
            if frame_count % 50 == 0:
                # 获取附近车辆数量
                nearby_vehicles = len(traffic_manager.get_nearby_vehicles(vehicle, 30))
                nearby_walkers = len(traffic_manager.get_nearby_walkers(vehicle, 30))
                
                # AEB状态显示
                if status == "正常":
                    status_icon = "✓"
                elif status == "预制动":
                    status_icon = "⚠"
                else:
                    status_icon = "🚨"
                
                print(f"--- 状态报告 (帧 {frame_count}) ---")
                print(f"  当前速度: {speed:.1f} km/h")
                print(f"  目标速度: {target_speed} km/h")
                print(f"  前方距离: {distance:.1f} m")
                print(f"  AEB状态: {status} {status_icon}")
                print(f"  附近车辆: {nearby_vehicles}")
                print(f"  附近行人: {nearby_walkers}")
                print()
                
                # 如果触发紧急制动，显示警告
                if status == "紧急制动":
                    print("[WARNING] 🚨 检测到紧急情况！AEB已启动！")
                    print()
            
            time.sleep(0.1)
        
        print("=== 运行结束 ===")
        print(f"总运行帧数: {frame_count}")
        
    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断")
    finally:
        # 清理资源
        print("\n[INFO] 清理资源...")
        traffic_manager.cleanup()
        vehicle.destroy()
        print("[OK] 所有资源已清理")

if __name__ == "__main__":
    main()
