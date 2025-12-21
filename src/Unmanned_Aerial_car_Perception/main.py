import carla
import time

def main():
    # 初始化变量，用于后续资源清理
    vehicle = None
    camera_sensor = None
    collision_sensor = None
    spectator = None  # 控制模拟器视角，确保能看到车辆
    try:
        # 1. 连接Carla模拟器（延长超时，适配低配电脑）
        client = carla.Client("localhost", 2000)
        client.set_timeout(15.0)
        world = client.get_world()
        spectator = world.get_spectator()  # 获取视角控制器
        print("✅ 成功连接Carla模拟器！")
        print("📌 当前仿真地图：", world.get_map().name)

        # 2. 获取车辆蓝图，设置红色车身
        vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '255,0,0')  # 红色车身
        print("🎨 已设置车辆颜色为红色")

        # 3. 选择合法生成点生成车辆（增加重试，避免碰撞失败）
        spawn_points = world.get_map().get_spawn_points()
        if spawn_points:
            spawn_point = spawn_points[10] if len(spawn_points) > 10 else spawn_points[0]
            # 生成车辆（重试3次，解决偶发碰撞问题）
            max_retry = 3
            for i in range(max_retry):
                try:
                    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                    break
                except:
                    if i == max_retry - 1:
                        raise Exception("车辆生成失败：生成点有碰撞，请更换spawn_points索引（如spawn_points[10]）")
                    time.sleep(0.5)

            print(f"🚗 成功生成特斯拉车辆，ID：{vehicle.id}")

            # 关键：将模拟器视角瞬移到车辆上方（确保能看到车）
            spectator_transform = carla.Transform(
                spawn_point.location + carla.Location(z=5),  # 车辆上方5米
                carla.Rotation(pitch=-15, yaw=spawn_point.rotation.yaw)  # 俯视视角
            )
            spectator.set_transform(spectator_transform)
            print("👀 模拟器视角已切换到车辆位置！")

            # 4. 添加RGB摄像头传感器（绑定到车辆）
            camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

            # 摄像头回调函数
            def camera_callback(image):
                print(f"📸 摄像头帧号：{image.frame_number} | 时间戳：{image.timestamp}")
            camera_sensor.listen(camera_callback)
            print("📹 已挂载RGB摄像头，开始采集画面！")

            # 5. 添加碰撞传感器（碰撞后紧急停车）
            collision_bp = world.get_blueprint_library().find('sensor.other.collision')
            collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
            def collision_callback(event):
                print("💥 检测到碰撞（建筑物/障碍物），紧急停车！")
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            collision_sensor.listen(collision_callback)
            print("🛡️ 已挂载碰撞传感器，开启碰撞保护！")

            # 6. 车辆行驶逻辑（手动规避，简化版）
            print("\n🚙 开始行驶（靠近建筑物会触发碰撞停车）...")
            # 阶段1：直行5秒
            vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=0.0, brake=0.0))
            time.sleep(5)
            # 阶段2：轻微转向，避开可能的建筑物
            vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=-0.5, brake=0.0))
            time.sleep(3)
            # 阶段3：停车
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            print("🛑 行驶结束，已停车！")

            # 7. 打印最终状态
            vehicle_location = vehicle.get_location()
            vehicle_velocity = vehicle.get_velocity()
            print(f"\n📊 车辆最终状态：")
            print(f"   位置：X={vehicle_location.x:.2f}, Y={vehicle_location.y:.2f}, Z={vehicle_location.z:.2f}")
            print(f"   速度：X={vehicle_velocity.x:.2f}, Y={vehicle_velocity.y:.2f}, Z={vehicle_velocity.z:.2f}")

        else:
            print("⚠️ 未找到合法的车辆生成点")

    except Exception as e:
        print(f"❌ 调用失败：{e}")
        print("\n🔍 排查建议：")
        print("1. 确认Carla模拟器是0.9.11版本")
        print("2. 更换生成点索引（如spawn_points[20]）")

    # 资源清理
    finally:
        time.sleep(3)
        if camera_sensor:
            camera_sensor.stop()
            camera_sensor.destroy()
            print("🗑️ 摄像头传感器已销毁")
        if collision_sensor:
            collision_sensor.stop()
            collision_sensor.destroy()
            print("🗑️ 碰撞传感器已销毁")
        if vehicle:
            vehicle.destroy()
            print("🗑️ 车辆已销毁")
        print("✅ 所有资源清理完成，程序正常退出")

if __name__ == "__main__":
    main()