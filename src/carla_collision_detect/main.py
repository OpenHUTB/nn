import carla
import time
import pygame
import math
import keyboard
import numpy as np
from vision_module import VisionSystem
from planner import LanePlanner

def main():
    print("===================================")
    print("🚗 欢迎使用 CARLA 碰撞与巡航测试系统")
    print("请选择本次生成的测试障碍物：")
    print("  [1] 测试车辆")
    print("  [2] 测试行人")
    print("===================================")
    choice = input("请输入选项 (1 或 2，默认按回车选 1): ").strip()
    target_type_name = "行人" if choice == '2' else "车辆"
    print(f"\n⏳ 正在连接 CARLA 服务器并准备生成 {target_type_name}...\n")

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    ego_vehicle = None
    dummy_target = None       
    collision_sensor = None   
    vision_system = None      

    pygame.init()
    screen = pygame.display.set_mode((400, 240)) 
    pygame.display.set_caption("CARLA 控制面板")
    
    pygame.font.init()
    font = pygame.font.SysFont("simhei", 24) 

    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2017')
        
        spawn_points = world.get_map().get_spawn_points()
        
        # 1. 寻找一条前方至少有 80 米没有路口的纯直道
        ego_spawn_point = None
        target_waypoint = None
        
        for sp in spawn_points:
            wp = world.get_map().get_waypoint(sp.location)
            # 必须当前不是路口
            if not wp.is_junction:
                # 探寻前方 60 米处的路点
                fwd_wps = wp.next(60.0)
                if fwd_wps and not fwd_wps[0].is_junction:
                    ego_spawn_point = sp
                    target_waypoint = fwd_wps[0]
                    break
                    
        if not ego_spawn_point:
            print("⚠️ 没找到完美的超长直道，将就用默认点。")
            ego_spawn_point = spawn_points[0]
            target_waypoint = world.get_map().get_waypoint(ego_spawn_point.location).next(40.0)[0]

        # 2. 生成主车
        ego_vehicle = world.try_spawn_actor(vehicle_bp, ego_spawn_point)
        
        if ego_vehicle:
            print("✅ 主车已生成！定速巡航模块已就绪。")
            vision_system = VisionSystem(ego_vehicle, world)
            lane_planner = LanePlanner(ego_vehicle, world) 

            # 3. 生成靶标 (使用地图路网获取的精确前方航点)
            target_transform = target_waypoint.transform
            target_transform.location.z += 0.5  # 稍微抬高防止卡地里
            
            if choice == '2':
                target_bp = bp_lib.filter('walker.pedestrian.*')[0]
            else:
                target_bp = bp_lib.find('vehicle.tesla.model3')
                
            dummy_target = world.try_spawn_actor(target_bp, target_transform)
            
            if dummy_target:
                print(f"🎯 前方固定坐标静态测试靶标 [{target_type_name}] 已生成！准备进行测试。")
            else:
                print(f"⚠️ {target_type_name} 生成失败，前方空间可能受限。")

            collision_bp = bp_lib.find('sensor.other.collision')
            collision_sensor = world.try_spawn_actor(collision_bp, carla.Transform(), attach_to=ego_vehicle)
            collision_flag = [False]

            def on_collision(event):
                if collision_flag[0]: return
                collision_flag[0] = True
                t = event.actor.get_transform()
                loc = t.location
                impulse = event.normal_impulse
                intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
                hit_type = "行人" if "walker" in event.other_actor.type_id else "车辆"
                
                print(f"\033[91m\n💥 [致命警告] 发生撞击! 撞击对象: {hit_type} ({event.other_actor.type_id}), "
                      f"碰撞冲量大小: {intensity:.0f}, "
                      f"坐标: (x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f})\033[0m")

            if collision_sensor:
                collision_sensor.listen(on_collision)
                print("✅ 碰撞传感器已挂载。")

            control = carla.VehicleControl()
            steer_cache, is_reverse, target_speed_kmh = 0.0, False, 0.0  
            Kp, Ki, error_sum = 0.15, 0.02, 0.0 

            prev_key_w = False
            prev_key_s = False
            prev_key_q = False

            vision_aeb_active = False 
            was_aeb_active = False 

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        
                if keyboard.is_pressed('esc'):
                    running = False

                curr_q = keyboard.is_pressed('q')
                if curr_q and not prev_key_q:
                    is_reverse = not is_reverse 
                prev_key_q = curr_q

                curr_w = keyboard.is_pressed('w')
                if curr_w and not prev_key_w:
                    target_speed_kmh += 5.0
                prev_key_w = curr_w

                curr_s = keyboard.is_pressed('s')
                if curr_s and not prev_key_s:
                    target_speed_kmh = max(0.0, target_speed_kmh - 5.0)
                prev_key_s = curr_s

                curr_space = keyboard.is_pressed('space')
                curr_a = keyboard.is_pressed('a')
                curr_d = keyboard.is_pressed('d')

                v = ego_vehicle.get_velocity() 
                speed_m_s = math.sqrt(v.x**2 + v.y**2 + v.z**2) 
                current_speed_kmh = speed_m_s * 3.6 
                error = target_speed_kmh - current_speed_kmh

                if target_speed_kmh > 0:
                    error_sum = max(min(error_sum + error, 40.0), -40.0) 
                else:
                    error_sum = 0.0

                if target_speed_kmh == 0.0:
                    control.throttle, control.brake = 0.0, 0.2 if current_speed_kmh > 0.5 else 1.0
                elif error > 0:
                    control.throttle, control.brake = min(max((error * Kp) + (error_sum * Ki), 0.0), 0.75), 0.0
                else:
                    control.throttle, control.brake = 0.0, min(max((-error * Kp) - (error_sum * Ki), 0.0), 0.5)

                if curr_space or collision_flag[0]:
                    control.hand_brake, control.throttle, control.brake, target_speed_kmh, error_sum = True, 0.0, 1.0, 0.0, 0.0             
                else:
                    control.hand_brake = False

                # ==========================================
                # 🌟 核心：感知、规划与控制
                # ==========================================
                vision_aeb_active = False
                
                if vision_system:
                    # 1. 视觉感知：获取最近障碍物距离
                    _, min_dist = vision_system.process_and_render()
                    
                    safe_dist = min_dist if min_dist != float('inf') else 100.0
                    planner_steer = lane_planner.get_lateral_control(safe_dist, current_speed_kmh)
                    
                    if not is_reverse and planner_steer is not None:
                        control.steer = planner_steer
                        
                    # 3. 纵向决策 (AES 减速 vs AEB 急刹)
                    if min_dist != float('inf'):
                        # 计算当前车速下的安全制动距离 (公式：v²/2a + 反应距离)
                        braking_dist = (speed_m_s ** 2) / (2 * 6.0) + 3.0 
                        
                        if lane_planner.is_changing_lane and not is_reverse:
                            # 【变道优先级高】：如果正在变道，我们选择“减速绕过”而不是“原地停死”
                            # 这样可以防止车子切到一半停在马路中间
                            target_speed_kmh = 15.0  
                            control.brake = 0.0     # 变道时禁制 AEB 介入
                            
                        elif min_dist <= braking_dist and current_speed_kmh > 2.0 and not is_reverse:
                            # 【制动保底】：如果没有在变道（比如旁边没路），则触发 AEB 紧急制动
                            vision_aeb_active = True
                            if not was_aeb_active:
                                print(f"\033[91m⚠️ 路径受阻且无法变道，触发 AEB 紧急刹车！\033[0m")
                            target_speed_kmh = 0.0   
                            control.throttle = 0.0
                            control.brake = 1.0      
                            control.hand_brake = True
                
                was_aeb_active = vision_aeb_active
                # ==========================================

                control.reverse = is_reverse
                ego_vehicle.apply_control(control)
                world.tick()

                spectator = world.get_spectator()
                transform = ego_vehicle.get_transform()
                spectator.set_transform(carla.Transform(
                    transform.location + carla.Location(z=5) - transform.get_forward_vector() * 10,
                    carla.Rotation(pitch=-20, yaw=transform.rotation.yaw)
                ))
                
                screen.fill((30, 30, 30)) 

                throttle_status = "开" if control.throttle > 0.01 else "关"
                brake_status = "开" if control.brake > 0.01 else "关"

                if vision_aeb_active:
                    info_text1 = font.render("⚠️ 视觉 AEB 介入制动中！", True, (255, 50, 50))
                else:
                    info_text1 = font.render("巡航系统已启动 (W/S 调速)", True, (255, 200, 0))
                    
                info_text2 = font.render(f"设定巡航: {target_speed_kmh:.1f} km/h", True, (255, 150, 200))
                info_text3 = font.render(f"当前车速: {current_speed_kmh:.1f} km/h", True, (0, 255, 255))
                info_text4 = font.render(f"底层输出 -> 油门:[{throttle_status}]  刹车:[{brake_status}]", True, (150, 150, 150))
                info_text5 = font.render(f"当前档位: {'[R] 倒车' if control.reverse else '[D] 前进'}", True, (255, 255, 255))
                
                screen.blit(info_text1, (20, 20))
                screen.blit(info_text2, (20, 60))
                screen.blit(info_text3, (20, 100))
                screen.blit(info_text4, (20, 140))
                screen.blit(info_text5, (20, 180))
                
                pygame.display.flip()
                
        else:
            print("❌ 生成失败，请尝试重启模拟器。")

    except KeyboardInterrupt:
        print("\n👋 停止程序")
    finally:
        keyboard.unhook_all()
        if vision_system:
            vision_system.destroy()
        if collision_sensor:
            collision_sensor.destroy()
        if dummy_target:
            dummy_target.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
            
        settings.synchronous_mode = False
        world.apply_settings(settings)
        pygame.quit() 
        print("🧹 环境已清理。")

if __name__ == '__main__':
    main()