#!/usr/bin/env python3
import carla
import time
import sys
from spawn_car import create_vehicle
from traffic_manager import TrafficManager
from lane_change_assist import LaneChangeAssist
from cruise_control import speed_cruise_control

def main():
    print("=== 自动变道辅助系统（LCA）演示 ===\n")
    
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
    
    # 初始化自动变道辅助系统
    print("[2] 初始化自动变道辅助系统...")
    lca = LaneChangeAssist(vehicle, world, safe_distance=30.0)
    print(f"[LCA] 安全距离: {lca.safe_distance}m")
    print("[OK] LCA系统已就绪\n")
    
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
    change_cooldown = 0
    
    try:
        for i in range(600):
            world.tick()
            frame_count += 1
            
            # 获取当前速度
            speed = lca.get_vehicle_speed()
            
            # 检查车道安全性
            left_safe, right_safe = lca.check_all_lanes()
            
            # 更新变道系统
            steer_angle, state, progress = lca.update()
            
            # 变道冷却
            if change_cooldown > 0:
                change_cooldown -= 1
            else:
                # 自动尝试变道
                if state == "cruise" and frame_count > 100:
                    # 优先向左变道，如果左边不安全则尝试右边
                    if left_safe:
                        success = lca.start_lane_change('left')
                        if success:
                            print("[LCA] 开始向左变道 ←")
                            change_cooldown = 100
                    elif right_safe:
                        success = lca.start_lane_change('right')
                        if success:
                            print("[LCA] 开始向右变道 →")
                            change_cooldown = 100
            
            # 变道完成后重置状态
            if state == "completed":
                print("[LCA] 变道完成 ✓")
                lca.reset()
            
            # 控制车辆
            throttle, brake = speed_cruise_control(speed, target_speed)
            control = carla.VehicleControl(
                throttle=throttle, 
                brake=brake, 
                steer=steer_angle
            )
            vehicle.apply_control(control)
            
            # 每50帧显示状态
            if frame_count % 50 == 0:
                print(f"--- 状态报告 (帧 {frame_count}) ---")
                print(f"  当前速度: {speed:.1f} km/h")
                print(f"  目标速度: {target_speed} km/h")
                print(f"  左侧车道: {'安全 ✓' if left_safe else '有车 ✗'}")
                print(f"  右侧车道: {'安全 ✓' if right_safe else '有车 ✗'}")
                print(f"  变道状态: {state}")
                if state in ['left', 'right']:
                    print(f"  变道进度: {int(progress * 100)}%")
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
