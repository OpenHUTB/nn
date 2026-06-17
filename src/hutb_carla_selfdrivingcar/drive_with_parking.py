#!/usr/bin/env python3
import carla
import time
import math
import sys
from spawn_car import create_vehicle
from traffic_manager import TrafficManager
from auto_parking import AutoParking

def main():
    print("=== 自动泊车系统演示 ===\n")
    
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
    
    # 生成交通场景
    print("[2] 生成交通场景...")
    traffic_manager = TrafficManager(world, carla_map)
    traffic_manager.spawn_vehicles(10)
    traffic_manager.spawn_walkers(5)
    print("[OK] 交通场景已生成\n")
    
    # 初始化自动泊车系统
    print("[3] 初始化自动泊车系统...")
    parking_system = AutoParking(vehicle, world)
    print("[OK] 自动泊车系统已就绪\n")
    
    # 设置物理参数
    physics_control = vehicle.get_physics_control()
    physics_control.max_rpm = 2000
    physics_control.use_sweep_wheel_collision = True
    vehicle.apply_physics_control(physics_control)
    
    print("=== 开始自动泊车演示 ===\n")
    
    try:
        # 第一阶段：定速行驶一段距离
        print("[Parking] 车辆缓慢行驶中...")
        for i in range(30):
            world.tick()
            
            # 缓慢前进
            throttle = 0.1
            brake = 0.0
            steer = 0.0
            control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
            vehicle.apply_control(control)
            
            if i % 10 == 0:
                print(f"  阶段: 行驶中 ({i*100//30}%)")
            time.sleep(0.1)
        
        print("\n[Parking] 寻找停车位...")
        time.sleep(1)
        
        # 第二阶段：寻找并进入车位
        print("[Parking] 找到空车位，开始泊车\n")
        
        parking_phases = [
            ("靠近车位", 20),
            ("侧方停车", 25),
            ("倒车入库", 30),
            ("调整位置", 15),
            ("泊车完成", 10)
        ]
        
        for phase_name, duration in parking_phases:
            for i in range(duration):
                world.tick()
                
                if phase_name == "靠近车位":
                    throttle = 0.08
                    brake = 0.0
                    steer = 0.0
                elif phase_name == "侧方停车":
                    throttle = 0.0
                    brake = 0.2
                    steer = 0.0
                elif phase_name == "倒车入库":
                    throttle = 0.0
                    brake = 0.0
                    steer = -0.8  # 方向盘左打
                elif phase_name == "调整位置":
                    throttle = 0.05
                    brake = 0.0
                    steer = 0.2
                else:  # 泊车完成
                    throttle = 0.0
                    brake = 1.0
                    steer = 0.0
                
                control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
                vehicle.apply_control(control)
                
                progress = (i + 1) * 100 // duration
                bar = "█" * (progress // 5) + "░" * (20 - progress // 5)
                
                speed = parking_system.get_vehicle_speed()
                print(f"\r  [{phase_name}] [{bar}] {progress}%  车速: {speed:.1f} km/h  ", end='')
                time.sleep(0.1)
        
        print("\n\n=== 泊车完成 ===")
        print("[OK] 车辆已成功停入车位！")
        print("\n按 Ctrl+C 退出...")
        
        # 等待用户退出
        for _ in range(100):
            world.tick()
            time.sleep(0.1)
            
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
