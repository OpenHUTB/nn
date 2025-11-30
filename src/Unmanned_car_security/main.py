import carla
import pygame
import sys
import traceback

# --- 全局变量 ---
actor_list = []  # 用于跟踪创建的actor（车辆、传感器等）
clock = pygame.time.Clock()


def main():
    # --- 1. 初始化Pygame ---
    pygame.init()
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Carla 直线行驶控制")

    # --- 2. 连接到Carla并执行主要逻辑 ---
    try:
        # 连接到本地Carla服务器
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        print("✅ 成功连接到Carla服务器")

        # 获取世界对象
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        print("✅ 成功获取世界对象")

        # --- 3. 生成车辆 ---
        # 选择车辆蓝图（特斯拉Model3）
        vehicle_bp = blueprint_library.filter('model3')[0]
        print(f"✅ 选择车辆蓝图: {vehicle_bp.id}")

        # 获取所有生成点
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("❌ 错误: 地图中没有找到可用的生成点")
            return

        # --- 关键修改：自动寻找可用的生成点 ---
        vehicle = None
        for i, spawn_point in enumerate(spawn_points):
            try:
                # 尝试在当前生成点生成车辆
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                actor_list.append(vehicle)
                print(f"✅ 成功生成车辆: {vehicle.type_id}")
                print(f"📍 车辆生成位置: 生成点 #{i} ({spawn_point.location})")
                break  # 生成成功，跳出循环
            except RuntimeError as e:
                # 如果生成失败（碰撞），尝试下一个生成点
                print(f"⚠️  生成点 #{i} 有碰撞，尝试下一个...")
                continue

        # 如果所有生成点都尝试过仍失败
        if vehicle is None:
            print("❌ 错误: 所有生成点都被占用，无法生成车辆")
            return

        # 关闭自动驾驶，手动控制
        vehicle.set_autopilot(False)
        print("✅ 已关闭自动驾驶，切换为手动控制")

        # --- 4. 主循环 ---
        running = True
        throttle = 0.0
        brake = 0.0
        steer = 0.0  # 保持转向为0，即直线行驶

        while running:
            # --- 事件处理 ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        print("🔌 用户按下Q键，准备退出程序")
                        running = False
                    elif event.key == pygame.K_UP:
                        throttle = 0.5  # 按下上箭头，增加油门
                        print(f"⚡ 油门开启: {throttle}")
                    elif event.key == pygame.K_DOWN:
                        brake = 1.0  # 按下下箭头，踩下刹车
                        print(f"🛑 刹车开启: {brake}")
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP:
                        throttle = 0.0  # 松开上箭头，油门归零
                        print("⚡ 油门关闭")
                    elif event.key == pygame.K_DOWN:
                        brake = 0.0  # 松开下箭头，刹车归零
                        print("🛑 刹车关闭")

            # --- 车辆控制 ---
            vehicle.apply_control(carla.VehicleControl(
                throttle=throttle,
                brake=brake,
                steer=steer
            ))

            # --- 更新显示 ---
            display.fill((0, 0, 0))  # 黑色背景
            # 显示车辆状态
            status_text = [
                f"Throttle: {throttle:.2f}",
                f"Brake: {brake:.2f}",
                f"Steer: {steer:.2f}",
                "",
                "操作说明:",
                "↑ 加速",
                "↓ 刹车",
                "Q 退出"
            ]

            # 渲染文本
            font = pygame.font.Font(None, 30)
            for i, text in enumerate(status_text):
                text_surface = font.render(text, True, (255, 255, 255))
                display.blit(text_surface, (10, 10 + i * 35))

            pygame.display.flip()

            # --- 控制帧率 ---
            clock.tick(60)

    # --- 异常处理 ---
    except Exception as e:
        print("\n" + "=" * 50)
        print("❌ 程序运行出错！")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        print("\n详细错误堆栈:")
        traceback.print_exc()
        print("=" * 50)

    # --- 5. 清理资源 ---
    finally:
        print("\n🧹 开始清理资源...")
        # 停止车辆
        if 'vehicle' in locals() and vehicle and vehicle.is_alive:
            vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1))
            print("🛑 车辆已停止")

        # 销毁所有actor
        for actor in actor_list:
            if actor and actor.is_alive:
                actor.destroy()
                print(f"🗑️ 销毁actor: {actor.type_id}")

        # 关闭Pygame
        pygame.quit()
        print("🖥️ Pygame窗口已关闭")
        print("🧹 资源清理完成！")
        sys.exit()


if __name__ == '__main__':
    main()