import carla
import time

def main():
    vehicle = None
    camera_sensor = None
    spectator = None  # 控制模拟器视角，确保能看到车辆
    try:
        # 1. 连接Carla模拟器（延长超时，适配低配电脑）
        client = carla.Client("localhost", 2000)
        client.set_timeout(15.0)
        world = client.get_world()
        spectator = world.get_spectator()  # 获取视角控制器
        print("✅ 成功连接Carla模拟器！")
        print("📌 当前仿真地图：", world.get_map().name)

        # 2. 获取车辆蓝图（亮黄色车身，更易识别）
        vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '255,255,0')  # 亮黄色（RGB）
        print("🎨 已设置车辆颜色为亮黄色（易识别）")

        # 3. 选择合法生成点（优先选地图中心位置）
        spawn_points = world.get_map().get_spawn_points()
        if spawn_points:
            spawn_point = spawn_points[0]  # 可替换为spawn_points[10]等避免边缘位置
            # 生成车辆（增加重试，避免偶发碰撞失败）
            max_retry = 3
            for i in range(max_retry):
                try:
                    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                    break
                except:
                    if i == max_retry - 1:
                        raise Exception("车辆生成失败：生成点有碰撞")
                    time.sleep(0.5)

            print(f"🚗 成功生成特斯拉车辆，ID：{vehicle.id}")

            # 关键：将模拟器视角瞬移到车辆上方（确保能看到车）
            spectator_transform = carla.Transform(
                spawn_point.location + carla.Location(z=5),  # 车辆上方5米
                carla.Rotation(pitch=-15, yaw=spawn_point.rotation.yaw)  # 俯视视角
            )
            spectator.set_transform(spectator_transform)
            print("👀 模拟器视角已切换到车辆位置！")

            # 4. 简化摄像头（仅保留基础功能，不影响核心）
            camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            camera_sensor.listen(lambda img: print(f"📸 摄像头正常采集（帧号：{img.frame_number}）"))
            print("📹 摄像头已挂载，车辆开始行驶...")

            # 5. 车辆持续运行（简化控制逻辑，延长行驶时间）
            print("\n🚙 车辆开始持续行驶（10秒）...")
            # 持续直行（油门0.7，更明显的行驶效果）
            for _ in range(10):
                vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=0.0, brake=0.0))
                # 视角跟随车辆移动
                vehicle_loc = vehicle.get_location()
                spectator.set_transform(carla.Transform(
                    vehicle_loc + carla.Location(z=5),
                    carla.Rotation(pitch=-15, yaw=vehicle.get_transform().rotation.yaw)
                ))
                print(f"🔄 车辆当前位置：X={vehicle_loc.x:.2f}, Y={vehicle_loc.y:.2f}")
                time.sleep(1)

            # 停车
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            time.sleep(2)
            print("🛑 车辆已停车")

            # 6. 打印最终状态
            final_loc = vehicle.get_location()
            final_vel = vehicle.get_velocity()
            print(f"\n📊 车辆行驶完成：")
            print(f"   最终位置：X={final_loc.x:.2f}, Y={final_loc.y:.2f}")
            print(f"   最终速度：{((final_vel.x**2 + final_vel.y**2)**0.5):.2f} m/s")

        else:
            print("⚠️ 未找到合法的车辆生成点")

    except Exception as e:
        print(f"❌ 运行失败：{e}")
        print("\n🔍 排查建议：")
        print("1. 确认Carla模拟器是0.9.11版本")
        print("2. 模拟器窗口不要最小化")
        print("3. 尝试更换生成点：spawn_points[10]")

    # 7. 资源清理（延迟销毁，确保能看到车辆直到程序结束）
    finally:
        time.sleep(3)  # 程序结束前车辆多显示3秒
        if camera_sensor:
            camera_sensor.stop()
            camera_sensor.destroy()
        if vehicle:
            vehicle.destroy()
        print("\n✅ 车辆已销毁，程序结束")

if __name__ == "__main__":
    main()