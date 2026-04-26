import carla
import time
import pygame
import math

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    ego_vehicle = None

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
            
            control = carla.VehicleControl()
            steer_cache = 0.0  
            is_reverse = False 
            
            target_speed_kmh = 0.0  
            
            # ==========================================
            # 🌟 核心升级：PI 控制器参数
            # ==========================================
            Kp = 0.15      # 稍微调大基础油门反应
            Ki = 0.02      # 新增：积分系数（控制“记仇”的威力）
            error_sum = 0.0 # 新增：用于累积误差的容器

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

                # ==========================================
                # 🌟 核心算法：PI (比例-积分) 控制
                # ==========================================
                error = target_speed_kmh - current_speed_kmh

                if target_speed_kmh > 0:
                    # 如果设定了速度，就开始累积误差
                    error_sum += error
                    # 防止积分饱和 (Anti-Windup)：避免误差攒得太大导致一脚油门窜上天
                    error_sum = max(min(error_sum, 40.0), -40.0) 
                else:
                    # 如果目标速度是0，立刻清空记忆
                    error_sum = 0.0

                if target_speed_kmh == 0.0:
                    control.throttle = 0.0
                    control.brake = 0.2 if current_speed_kmh > 0.5 else 1.0
                elif error > 0:
                    # 动力不足时：现在的油门 = 基础油门 + 历史累积油门
                    throttle_output = (error * Kp) + (error_sum * Ki)
                    control.throttle = min(max(throttle_output, 0.0), 0.75) 
                    control.brake = 0.0
                else:
                    # 超速时也应用积分，平滑刹车
                    brake_output = (-error * Kp) - (error_sum * Ki)
                    control.throttle = 0.0
                    control.brake = min(max(brake_output, 0.0), 0.5)

                if keys[pygame.K_SPACE]:
                    control.hand_brake = True
                    control.throttle = 0.0
                    control.brake = 1.0
                    target_speed_kmh = 0.0 
                    error_sum = 0.0 
                else:
                    control.hand_brake = False

                control.reverse = is_reverse

                # 方向盘手动控制
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

                # UI 刷新
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
        if ego_vehicle:
            ego_vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        pygame.quit() 
        print("🧹 环境已清理")

if __name__ == '__main__':
    main()