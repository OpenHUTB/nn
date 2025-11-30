import carla
import pygame
import sys
import traceback
import math

# --- 全局变量 ---
actor_list = []
clock = pygame.time.Clock()

def main():
    # --- 1. 初始化Pygame ---
    pygame.init()
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Carla 直线行驶控制（修复版）")
    pygame.display.flip()
    print("✅ Pygame 窗口初始化完成")

    # --- 2. 连接到Carla并执行主要逻辑 ---
    try:
        # 连接 Carla 服务器
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        print("✅ 成功连接到 Carla 服务器")

        # 获取世界对象
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        print("✅ 成功获取世界对象")

        # 选择车辆蓝图（特斯拉Model3，设置为黄色便于识别）
        vehicle_bp = blueprint_library.filter('model3')[0]
        vehicle_bp.set_attribute('color', '255,255,0')  # 黄色车辆
        print(f"✅ 选择车辆蓝图: {vehicle_bp.id}（颜色：黄色）")

        # 遍历生成点，选择可用位置
        spawn_points = world.get_map().get_spawn_points()
        vehicle = None
        for i, spawn_point in enumerate(spawn_points[:20]):  # 遍历前20个生成点
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                actor_list.append(vehicle)
                print(f"✅ 车辆生成成功：位置=生成点 #{i}（{spawn_point.location.x:.1f}, {spawn_point.location.y:.1f}）")
                break
            except RuntimeError:
                print(f"⚠️  生成点 #{i} 被占用，尝试下一个...")
                continue

        if not vehicle:
            print("❌ 错误：所有生成点都被占用，无法生成车辆")
            return

        # 初始化车辆状态（关闭自动驾驶，松开手刹）
        vehicle.set_autopilot(False)
        vehicle.apply_control(carla.VehicleControl(
            throttle=0.0,
            brake=0.0,
            steer=0.0,
            hand_brake=False,
            reverse=False
        ))
        print("✅ 车辆状态初始化完成：自动驾驶关闭，手刹松开")

        # --- 关键修复：让 Carla 视角自动聚焦到车辆 ---
        spectator = world.get_spectator()
        vehicle_transform = vehicle.get_transform()
        # 镜头位置：车辆后方5米、上方2米，朝向与车辆一致
        spectator_transform = carla.Transform(
            location=vehicle_transform.location + carla.Location(x=-5.0, z=2.0),
            rotation=vehicle_transform.rotation
        )
        spectator.set_transform(spectator_transform)
        print("✅ Carla 视角已聚焦到车辆")

        # --- 3. 主循环 ---
        running = True
        throttle = 0.0
        brake = 0.0
        steer = 0.0
        speed_kmh = 0.0

        print("\n✅ 进入主循环，等待键盘操作...")
        print("操作说明：↑ 加速 | ↓ 刹车 | Q 退出")

        while running:
            # 事件处理（键盘操作）
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\n🔌 用户点击关闭窗口，准备退出")
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            print("\n🔌 用户按下 Q 键，准备退出")
                            running = False
                        elif event.key == pygame.K_UP:
                            throttle = 0.8  # 增大油门，确保车辆能启动
                            print(f"⚡ 油门开启：{throttle:.2f}")
                        elif event.key == pygame.K_DOWN:
                            brake = 1.0
                            print(f"🛑 刹车开启：{brake:.2f}")
                    elif event.type == pygame.KEYUP:
                        if event.key == pygame.K_UP:
                            throttle = 0.0
                            print("⚡ 油门关闭")
                        elif event.key == pygame.K_DOWN:
                            brake = 0.0
                            print("🛑 刹车关闭")
            except Exception as e:
                print(f"⚠️  事件处理时出错: {e}")
                continue

            # 车辆控制信号发送
            try:
                control = carla.VehicleControl(
                    throttle=throttle,
                    brake=brake,
                    steer=steer,
                    hand_brake=False,
                    reverse=False
                )
                vehicle.apply_control(control)
            except Exception as e:
                print(f"⚠️  车辆控制时出错: {e}")
                continue

            # 更新 Pygame 显示（速度、状态等）
            try:
                display.fill((0, 0, 0))  # 黑色背景

                # 计算车辆速度（兼容旧版本 Carla，无 length() 方法）
                velocity = vehicle.get_velocity()
                speed_mps = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                speed_kmh = speed_mps * 3.6  # 转换为 km/h

                # 显示状态文本
                status_text = [
                    f"当前状态：{'加速' if throttle > 0 else '刹车' if brake > 0 else '滑行'}",
                    f"油门：{throttle:.2f} | 刹车：{brake:.2f}",
                    f"当前速度：{speed_kmh:.2f} km/h",
                    "",
                    "操作说明：",
                    "↑ 键：加速（油门=0.8）",
                    "↓ 键：刹车（刹车=1.0）",
                    "Q 键：退出程序"
                ]

                # 渲染文本
                font = pygame.font.Font(None, 32)
                for i, text in enumerate(status_text):
                    text_color = (255, 255, 255) if i < 3 else (150, 150, 150)
                    text_surface = font.render(text, True, text_color)
                    display.blit(text_surface, (20, 20 + i * 40))

                pygame.display.flip()  # 刷新显示
            except Exception as e:
                print(f"⚠️  显示更新时出错: {e}")
                continue

            # 控制帧率（60 FPS）
            clock.tick(60)

    # --- 4. 异常处理 ---
    except Exception as e:
        print("\n" + "="*50)
        print("❌ 程序初始化阶段出错！")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        traceback.print_exc()
        print("="*50)

    # --- 5. 清理资源（确保车辆和传感器被销毁） ---
    finally:
        print("\n🧹 开始清理资源...")
        try:
            # 停止车辆并销毁
            if 'vehicle' in locals() and vehicle.is_alive:
                vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=True
                ))
                time.sleep(0.5)  # 等待车辆停止
                vehicle.destroy()
                print("✅ 车辆已销毁")

            # 销毁所有生成的 Actor
            for actor in actor_list:
                if actor.is_alive:
                    actor.destroy()
                    print(f"✅ 销毁 Actor：{actor.type_id}")

        except Exception as e:
            print(f"⚠️  清理资源时出错: {e}")

        # 关闭 Pygame
        pygame.quit()
        print("✅ Pygame 窗口已关闭")
        print("🧹 资源清理完成！")
        sys.exit()

if __name__ == '__main__':
    main()