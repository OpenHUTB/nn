import carla
import time
import pygame
import math

def main():
    # ==========================================
    # 🌟 新增：启动时的交互选择菜单
    # ==========================================
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

    # 提前声明变量，防止后续报错
    ego_vehicle = None
    dummy_target = None       # 🌟 修改：统一称为 dummy_target
    collision_sensor = None   # 碰撞传感器变量

    pygame.init()
    screen = pygame.display.set_mode((400, 240)) 
    pygame.display.set_caption("CARLA 智能巡航系统 (PI控制)")
    
    pygame.font.init()
    font = pygame.font.SysFont("simhei", 24) 

    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2017')
        
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] 

        ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        
        if ego_vehicle:
            print("✅ 主车已生成！定速巡航模块已就绪。")
            target_transform = carla.Transform(
                carla.Location(x=-52.64, y=24.47, z=0.5),
                carla.Rotation(yaw=0.16)
            )
            
            # ==========================================
            # 🌟 新增模块 1：根据用户选择生成对应的蓝图
            # ==========================================
            if choice == '2':
                target_bp = bp_lib.filter('walker.pedestrian.*')[0]
            else:
                target_bp = bp_lib.find('vehicle.tesla.model3')
                
            dummy_target = world.try_spawn_actor(target_bp, target_transform)
            
            if dummy_target:
                print(f"🎯 前方固定坐标静态测试靶标 [{target_type_name}] 已生成！准备进行追尾测试。")
            else:
                print(f"⚠️ {target_type_name} 生成失败，前方空间可能受限。")

            # ==========================================
            # 🌟 新增模块 2：挂载并激活碰撞传感器
            # ==========================================
            collision_bp = bp_lib.find('sensor.other.collision')
            collision_sensor = world.try_spawn_actor(collision_bp, carla.Transform(), attach_to=ego_vehicle)

            collision_flag = [False]

            def on_collision(event):
                if collision_flag[0]:
                    return
                
                collision_flag[0] = True
                t = event.actor.get_transform()
                loc = t.location
                impulse = event.normal_impulse
                intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
                
                # 🌟 新增：智能判断撞击的是人还是车
                hit_type = "行人" if "walker" in event.other_actor.type_id else "车辆"
                
                print(f"\033[91m\n💥 [致命警告] 发生撞击! 撞击对象: {hit_type} ({event.other_actor.type_id}), "
                      f"碰撞冲量大小: {intensity:.0f}, "
                      f"坐标: (x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f})\033[0m")
                print(f"\033[93m⚠️ 自动刹车！\033[0m")

            if collision_sensor:
                collision_sensor.listen(on_collision)
                print("✅ 碰撞传感器已挂载。")
            # ==========================================

            control = carla.VehicleControl()
            steer_cache = 0.0  
            is_reverse = False 
            
            target_speed_kmh = 0.0  
            
            Kp = 0.15      
            Ki = 0.02      
            error_sum = 0.0 

            running = True
            while running:
                world.tick()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            is_reverse = not is_reverse 
                        elif event.key == pygame.K_w:
                            target_speed_kmh += 5.0
                        elif event.key == pygame.K_s:
                            target_speed_kmh = max(0.0, target_speed_kmh - 5.0)
                keys = pygame.key.get_pressed()
                if keys[pygame.K_ESCAPE]:
                    running = False

                v = ego_vehicle.get_velocity() 
                speed_m_s = math.sqrt(v.x**2 + v.y**2 + v.z**2) 
                current_speed_kmh = speed_m_s * 3.6 

                error = target_speed_kmh - current_speed_kmh

                if target_speed_kmh > 0:
                    error_sum += error
                    error_sum = max(min(error_sum, 40.0), -40.0) 
                else:
                    error_sum = 0.0

                if target_speed_kmh == 0.0:
                    control.throttle = 0.0
                    control.brake = 0.2 if current_speed_kmh > 0.5 else 1.0
                elif error > 0:
                    throttle_output = (error * Kp) + (error_sum * Ki)
                    control.throttle = min(max(throttle_output, 0.0), 0.75) 
                    control.brake = 0.0
                else:
                    brake_output = (-error * Kp) - (error_sum * Ki)
                    control.throttle = 0.0
                    control.brake = min(max(brake_output, 0.0), 0.5)

                if keys[pygame.K_SPACE] or collision_flag[0]:
                    control.hand_brake = True   
                    control.throttle = 0.0      
                    control.brake = 1.0         
                    target_speed_kmh = 0.0      
                    error_sum = 0.0             
                else:
                    control.hand_brake = False

                control.reverse = is_reverse

                steer_speed = 0.02 
                if keys[pygame.K_a]:
                    steer_cache = max(steer_cache - steer_speed, -1.0)
                elif keys[pygame.K_d]:
                    steer_cache = min(steer_cache + steer_speed, 1.0)
                else:
                    if steer_cache > 0:
                        steer_cache = max(steer_cache - steer_speed, 0.0)
                    elif steer_cache < 0:
                        steer_cache = min(steer_cache + steer_speed, 0.0)
                
                if abs(steer_cache) < 0.01:
                    steer_cache = 0.0
                control.steer = steer_cache

                ego_vehicle.apply_control(control)

                screen.fill((30, 30, 30)) 
                
                throttle_status = "开" if control.throttle > 0.01 else "关"
                brake_status = "开" if control.brake > 0.01 else "关"

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

                spectator = world.get_spectator()
                transform = ego_vehicle.get_transform()
                spectator.set_transform(carla.Transform(
                    transform.location + carla.Location(z=5) - transform.get_forward_vector() * 10,
                    carla.Rotation(pitch=-20, yaw=transform.rotation.yaw)
                ))
                
        else:
            print("❌ 生成失败，请尝试重启模拟器。")

    except KeyboardInterrupt:
        print("\n👋 停止程序")
    finally:
        # ==========================================
        # 🌟 修改：适配 dummy_target 的资源销毁
        # ==========================================
        if collision_sensor:
            collision_sensor.destroy()
        if dummy_target:
            dummy_target.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
            
        settings.synchronous_mode = False
        world.apply_settings(settings)
        pygame.quit() 
        print("🧹 环境已清理（主车、靶标、传感器均已销毁）")

if __name__ == '__main__':
    main()