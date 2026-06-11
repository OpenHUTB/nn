import carla
import time
import pygame
import math
import keyboard
import numpy as np
from vision_module import VisionSystem
from planner import LanePlanner
from acc_module import ACCController

def main():
    print("===================================")
    print("🚗 欢迎使用 CARLA 碰撞与巡航测试系统")
    print("请选择本次生成的测试障碍物：")
    print("  [1] 测试车辆")
    print("  [2] 测试行人")
    print("===================================")
    choice = input("请输入选项 (1 或 2，默认按回车选 1): ").strip()
    target_type_name = "行人" if choice == '2' else "车辆"
    print("  [1] 障碍物靠左生成")
    print("  [2] 障碍物靠右生成")
    side_choice = input("请选择障碍物位置 (1 或 2，默认按回车选 1): ").strip()
    spawn_side = "right" if side_choice == '2' else "left"
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
            if wp.is_junction:
                continue
                
            is_perfect_straight = True
            check_wp = wp
            start_yaw = wp.transform.rotation.yaw  # 💡 新增：记录起始路点的绝对航向角
            
            for _ in range(8):
                next_wps = check_wp.next(10.0)
                # 条件 1：不能是断头路或十字路口
                if not next_wps or next_wps[0].is_junction:
                    is_perfect_straight = False
                    break
                    
                # 💡 条件 2（核心修复）：弯道测谎仪！
                current_yaw = next_wps[0].transform.rotation.yaw
                yaw_diff = abs((current_yaw - start_yaw) % 360)
                if yaw_diff > 180: 
                    yaw_diff = 360 - yaw_diff
                    
                # 如果前方路点的朝向，和起点偏差超过 2 度，说明这是一条弯路，立刻抛弃！
                if yaw_diff > 2.0:
                    is_perfect_straight = False
                    break
                    
                check_wp = next_wps[0]

            # 2. 如果经过了重重考验，这 80 米是一条绝对纯正的直道
            if is_perfect_straight:
                target_wp = wp
                current_lane_id = target_wp.lane_id
                
                # 靠左/靠右生成的逻辑 (带防逆行双黄线保护)
                if spawn_side == "left":
                    while target_wp.get_left_lane() and target_wp.get_left_lane().lane_type == carla.LaneType.Driving and (target_wp.get_left_lane().lane_id * current_lane_id > 0):
                        target_wp = target_wp.get_left_lane()
                else:
                    while target_wp.get_right_lane() and target_wp.get_right_lane().lane_type == carla.LaneType.Driving and (target_wp.get_right_lane().lane_id * current_lane_id > 0):
                        target_wp = target_wp.get_right_lane()

                # 3. 锁定主车生成点
                ego_spawn_point = target_wp.transform
                ego_spawn_point.location.z += 0.5
                
                # 4. 在正前方 30 米处生成测试靶标
                target_waypoint = target_wp.next(15.0)[0]
                break
                    
        if not ego_spawn_point:
            print("⚠️ 没找到完美的超长直道，将就用默认点。")
            ego_spawn_point = spawn_points[0]
            target_waypoint = world.get_map().get_waypoint(ego_spawn_point.location).next(15.0)[0]

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
            saved_target_speed = None

            prev_key_w = False
            prev_key_s = False
            prev_key_q = False

            vision_aeb_active = False 
            was_aeb_active = False
            acc_sys = ACCController()

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                if choice != '2':
                    dummy_target.set_autopilot(True)
                    tm = client.get_trafficmanager()
                    tm.ignore_lights_percentage(dummy_target, 100)
                    tm.vehicle_percentage_speed_difference(dummy_target, 50.0)        
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

                # ==========================================
                # 1. 获取当前主车车速
                # ==========================================
                v = ego_vehicle.get_velocity() 
                speed_m_s = math.sqrt(v.x**2 + v.y**2 + v.z**2) 
                current_speed_kmh = speed_m_s * 3.6 

                # ==========================================
                # 2. 视觉感知与 ACC 状态判定
                # ==========================================
                vision_aeb_active = False
                min_dist = float('inf')
                obstacle_side = None
                is_following = False

                if vision_system:
                    _, min_dist, obstacle_side = vision_system.process_and_render()
                    
                    active_target_speed = target_speed_kmh
                    # 先让 ACC 计算速度，看看前车是死是活
                    if min_dist != float('inf') and not lane_planner.is_changing_lane:
                        active_target_speed = acc_sys.update_target_speed(min_dist, current_speed_kmh, target_speed_kmh)
                        # 🌟 核心：如果前车速度 > 5km/h，判定为活车，开启跟车模式！
                        if hasattr(acc_sys, 'lead_speed_kmh') and acc_sys.lead_speed_kmh > 5.0:
                            is_following = True

                # ==========================================
                # 3. 规划与横向控制 (决定是否变道)
                # ==========================================
                if vision_system:
                    safe_dist = min_dist if min_dist != float('inf') else 100.0
                    # 🌟 传入 is_following 标志位。如果是跟车，规划器会放弃变道，老老实实保持车道！
                    planner_steer = lane_planner.get_lateral_control(safe_dist, current_speed_kmh, obstacle_side, is_following)
                    
                    if not is_reverse and planner_steer is not None:
                        control.steer = planner_steer

                # ==========================================
                # 4. 纵向速度管理 (变道限速与恢复)
                # ==========================================
                if lane_planner.is_changing_lane and not is_reverse and not control.hand_brake:
                    if saved_target_speed is None:
                        saved_target_speed = target_speed_kmh 
                    active_target_speed = 15.0  # 变道慢行
                    control.brake = 0.0
                else:
                    if saved_target_speed is not None:
                        target_speed_kmh = saved_target_speed
                        saved_target_speed = None
                        active_target_speed = target_speed_kmh

                # ==========================================
                # 5. 底层 PI 速度控制器 (计算油门和刹车)
                # ==========================================
                error = active_target_speed - current_speed_kmh

                if active_target_speed > 0:
                    error_sum = max(min(error_sum + error, 40.0), -40.0) 
                else:
                    error_sum = 0.0

                if active_target_speed == 0.0:
                    control.throttle, control.brake = 0.0, 0.2 if current_speed_kmh > 0.5 else 1.0
                elif error > 0:
                    control.throttle, control.brake = min(max((error * Kp) + (error_sum * Ki), 0.0), 0.75), 0.0
                else:
                    control.throttle, control.brake = 0.0, min(max((-error * Kp) - (error_sum * Ki), 0.0), 0.5)

                # ==========================================
                # 6. 紧急制动 AEB 兜底 (只对静止障碍物生效)
                # ==========================================
                if min_dist != float('inf'):
                    braking_dist = (speed_m_s ** 2) / (2 * 6.0) + 3.0
                    # 如果距离极近，且没在变道，且不是在跟车(如果是跟车，ACC会柔和减速而不是直接抱死)，触发 AEB
                    if min_dist < braking_dist and not lane_planner.is_changing_lane and not is_following:
                        vision_aeb_active = True
                        control.throttle = 0.0
                        control.brake = 1.0

                if curr_space or collision_flag[0]:
                    control.hand_brake, control.throttle, control.brake = True, 0.0, 1.0
                    target_speed_kmh = 0.0
                    error_sum = 0.0             
                else:
                    control.hand_brake = False
                    
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