#!/usr/bin/env python3
import carla
import time
from spawn_car import create_vehicle
from traffic_manager import TrafficManager
from traffic_sign_recognition import TrafficSignRecognition
from cruise_control import speed_cruise_control

def main():
    print("=== 交通标志识别系统（TSR）演示 ===\n")
    
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
    
    # 初始化交通标志识别系统
    print("[2] 初始化交通标志识别系统...")
    tsr = TrafficSignRecognition(vehicle, world)
    print("[TSR] 摄像头已设置")
    print("[OK] TSR系统已就绪\n")
    
    # 生成交通场景
    print("[3] 生成交通场景...")
    traffic_manager = TrafficManager(world, carla_map)
    traffic_manager.spawn_vehicles(15)
    traffic_manager.spawn_walkers(8)
    print("[OK] 交通场景已生成\n")
    
    # 设置目标速度
    default_speed = 50  # km/h
    
    print("=== 启动巡航 ===\n")
    print(f"[OK] 巡航模式启动")
    print(f"默认速度: {default_speed} km/h\n")
    
    frame_count = 0
    
    try:
        for i in range(500):
            world.tick()
            frame_count += 1
            
            # 获取当前速度
            v = vehicle.get_velocity()
            current_speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            
            # 获取交通标志信息
            sign_info = tsr.get_sign_info()
            detected_sign = sign_info['sign']
            target_speed = tsr.get_target_speed(default_speed)
            
            # 控制车辆
            throttle, brake = speed_cruise_control(current_speed, target_speed)
            control = carla.VehicleControl(throttle=throttle, brake=brake)
            vehicle.apply_control(control)
            
            # 每50帧显示状态
            if frame_count % 50 == 0:
                # 获取附近车辆数量
                nearby_vehicles = len(traffic_manager.get_nearby_vehicles(vehicle, 30))
                
                # 标志图标
                if detected_sign:
                    if '限速' in detected_sign:
                        sign_icon = '🚸'
                    elif '停车' in detected_sign:
                        sign_icon = '⛔'
                    elif '让行' in detected_sign:
                        sign_icon = '🚗'
                    else:
                        sign_icon = '📛'
                else:
                    sign_icon = ''
                
                print(f"--- 状态报告 (帧 {frame_count}) ---")
                print(f"  当前速度: {current_speed:.1f} km/h")
                print(f"  目标速度: {target_speed:.1f} km/h")
                print(f"  识别标志: {detected_sign or '无'} {sign_icon}")
                print(f"  标志距离: {sign_info['distance']:.1f} m")
                print(f"  附近车辆: {nearby_vehicles}")
                print()
                
                # 如果识别到标志，显示提示
                if detected_sign:
                    print(f"[TSR] 检测到交通标志: {detected_sign} {sign_icon}")
                    print(f"[TSR] 目标速度调整为: {target_speed} km/h")
                    print()
            
            time.sleep(0.1)
        
        print("=== 运行结束 ===")
        print(f"总运行帧数: {frame_count}")
        
    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断")
    finally:
        # 清理资源
        print("\n[INFO] 清理资源...")
        tsr.destroy()
        traffic_manager.cleanup()
        vehicle.destroy()
        print("[OK] 所有资源已清理")

if __name__ == "__main__":
    import math
    main()
