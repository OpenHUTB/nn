#!/usr/bin/env python3
import carla
import time
import sys
from spawn_car import create_vehicle
from traffic_manager import TrafficManager
from pedestrian_protection import PedestrianProtectionSystem
from cruise_control import speed_cruise_control

def main():
    print("=== 行人保护系统（PPS）演示 ===\n")
    
    # 连接模拟器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    
    # 生成主车辆
    print("[1] 生成主车辆...")
    vehicle = create_vehicle(world, carla_map)
    if vehicle is None:
        print("[ERROR] 主车辆生成失败！")
        return
    print("[OK] 主车辆已生成\n")
    
    # 初始化行人保护系统
    print("[2] 初始化行人保护系统...")
    pps = PedestrianProtectionSystem(
        vehicle, world, 
        safe_distance=25.0, 
        warning_distance=15.0, 
        danger_distance=8.0
    )
    print(f"[PPS] 安全距离: {pps.safe_distance}m")
    print(f"[PPS] 警告距离: {pps.warning_distance}m")
    print(f"[PPS] 危险距离: {pps.danger_distance}m")
    print("[OK] PPS系统已就绪\n")
    
    # 生成交通场景
    print("[3] 生成交通场景...")
    traffic_manager = TrafficManager(world, carla_map)
    traffic_manager.spawn_vehicles(15)
    traffic_manager.spawn_walkers(15)
    print("[OK] 交通场景已生成\n")
    
    # 设置目标速度
    target_speed = 35  # km/h
    
    print("=== 启动巡航 ===\n")
    print("[OK] 巡航模式启动")
    print(f"目标速度: {target_speed} km/h\n")
    
    frame_count = 0
    try:
        for i in range(500):
            world.tick()
            frame_count += 1
            
            # 获取当前速度
            speed = pps.get_vehicle_speed()
            
            # 更新PPS系统
            brake_intensity, distance, status, status_code = pps.update()
            
            # 根据PPS状态控制车辆
            if brake_intensity > 0:
                # PPS正在工作，使用PPS的刹车控制
                pass
            else:
                # 正常巡航控制
                throttle, brake = speed_cruise_control(speed, target_speed)
                control = carla.VehicleControl(throttle=throttle, brake=brake)
                vehicle.apply_control(control)
            
            # 每50帧显示状态
            if frame_count % 50 == 0:
                # 获取附近行人数量
                nearby_walkers = len(traffic_manager.get_nearby_walkers(vehicle, 30))
                
                # PPS状态显示
                if status_code == "safe":
                    status_icon = "✓"
                elif status_code == "caution":
                    status_icon = "⚠"
                elif status_code == "warning":
                    status_icon = "⚠️"
                else:
                    status_icon = "🚨"
                
                print(f"--- 状态报告 (帧 {frame_count}) ---")
                print(f"  当前速度: {speed:.1f} km/h")
                print(f"  目标速度: {target_speed} km/h")
                print(f"  前方行人距离: {distance:.1f} m")
                print(f"  PPS状态: {status} {status_icon}")
                print(f"  附近行人: {nearby_walkers}")
                print()
                
                # 如果检测到危险，显示警告
                if status_code == "danger":
                    print("[WARNING] 🚨 检测到行人！紧急制动！")
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
