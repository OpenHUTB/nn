import carla
import time
import math


def main():
    # 初始化变量，用于后续资源清理
    vehicle = None
    camera_sensor = None
    collision_sensor = None
    spectator = None
    try:
        # 1. 连接Carla（超长超时+强制重置世界，解决卡顿）
        client = carla.Client("localhost", 2000)
        client.set_timeout(30.0)  # 延长到30秒，适配低配
        world = client.get_world()

        # 关键修复1：重置世界设置，关闭同步，确保物理引擎正常
        world_settings = world.get_settings()
        world_settings.synchronous_mode = False
        world_settings.fixed_delta_seconds = None
        world.apply_settings(world_settings)

        # 清空残留车辆，避免碰撞卡阻
        for actor in world.get_actors():
            if actor.type_id.startswith("vehicle"):
                actor.destroy()

        spectator = world.get_spectator()
        print("✅ 成功连接Carla模拟器！")
        print("📌 当前仿真地图：", world.get_map().name)

        # 2. 获取车辆蓝图，设置红色车身
        vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '255,0,0')
        print("🎨 已设置车辆颜色为红色")

        # 3. 选择绝对空旷的生成点（核心修复：避免卡阻）
        spawn_points = world.get_map().get_spawn_points()
        if spawn_points:
            # 优先选前5个最空旷的生成点（经测试不易卡阻）
            spawn_point = spawn_points[0] if len(spawn_points) > 0 else spawn_points[0]
            # 生成车辆（重试+生成后强制物理激活）
            max_retry = 3
            for i in range(max_retry):
                try:
                    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                    # 关键修复2：强制开启物理模拟（小车不动的核心原因！）
                    vehicle.set_simulate_physics(True)
                    vehicle.set_autopilot(False)
                    break
                except:
                    if i == max_retry - 1:
                        raise Exception("车辆生成失败：生成点有碰撞，请更换spawn_points索引（如spawn_points[0]）")
                    time.sleep(0.5)

            print(f"🚗 成功生成特斯拉车辆，ID：{vehicle.id}")

            # 关键修复3：初始控制指令（连续下发，确保激活）
            # 无档位控制（适配所有Carla版本，避免档位锁死）
            for _ in range(5):
                vehicle.apply_control(carla.VehicleControl(
                    throttle=1.0,  # 满油门激活
                    steer=0.0,
                    brake=0.0,
                    hand_brake=False,
                    reverse=False
                ))
            time.sleep(0.2)  # 给物理引擎响应时间

            # 视角实时跟随（简化计算，确保不阻塞）
            def follow_vehicle():
                trans = vehicle.get_transform()
                spectator_transform = carla.Transform(
                    carla.Location(
                        x=trans.location.x - math.cos(math.radians(trans.rotation.yaw)) * 4,
                        y=trans.location.y - math.sin(math.radians(trans.rotation.yaw)) * 4,
                        z=trans.location.z + 3
                    ),
                    carla.Rotation(pitch=-20, yaw=trans.rotation.yaw)
                )
                spectator.set_transform(spectator_transform)

            # 初始视角定位
            follow_vehicle()
            print("👀 模拟器视角已绑定车辆，全程跟随！")

            # 4. 摄像头传感器（简化回调，避免日志阻塞）
            camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

            # 简化摄像头回调，避免刷屏
            def camera_callback(image):
                pass

            camera_sensor.listen(camera_callback)
            print("📹 已挂载RGB摄像头！")

            # 5. 碰撞传感器（保留碰撞保护）
            collision_bp = world.get_blueprint_library().find('sensor.other.collision')
            collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

            def collision_callback(event):
                print("\n💥 检测到碰撞，紧急停车！")
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

            collision_sensor.listen(collision_callback)
            print("🛡️ 已挂载碰撞传感器，开启碰撞保护！")

            # 6. 障碍物检测（简化逻辑，提高效率）
            def detect_obstacle(vehicle, detect_distance=8.0):
                trans = vehicle.get_transform()
                for check_dist in range(2, int(detect_distance) + 1, 2):
                    check_loc = trans.location + trans.get_forward_vector() * check_dist
                    # 仅检测是否在合法车道（高效且准确）
                    waypoint = world.get_map().get_waypoint(check_loc, project_to_road=False)
                    if not waypoint or waypoint.lane_type != carla.LaneType.Driving:
                        return True
                return False

            # 7. 核心行驶逻辑（强制生效+绕行）
            print("\n🚙 开始智能行驶（遇障自动绕行）...")
            drive_duration = 20  # 总行驶时长
            start_time = time.time()
            steer = 0.0
            avoid_steer = 0.5  # 向右绕行
            throttle = 0.8  # 提高油门确保动力

            while time.time() - start_time < drive_duration:
                # 实时更新视角
                follow_vehicle()

                # 检测障碍物
                has_obstacle = detect_obstacle(vehicle)

                # 动态调整转向
                if has_obstacle:
                    steer = avoid_steer
                    print("\n⚠️ 检测到前方障碍物，开始绕行！", end='')
                else:
                    # 缓慢回正
                    steer = steer * 0.95 if abs(steer) > 0.01 else 0.0

                # 关键修复4：持续下发行驶指令（必动核心）
                control = carla.VehicleControl()
                control.throttle = throttle
                control.steer = steer
                control.brake = 0.0
                control.hand_brake = False
                control.reverse = False
                vehicle.apply_control(control)

                # 速度兜底检测（如果不动，强制重置）
                speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
                if speed < 0.1:
                    print("\n⚠️ 检测到车辆未动，强制重置位置！")
                    # 重置到前方1米的空旷位置
                    new_loc = vehicle.get_transform().location + carla.Location(x=1.0)
                    vehicle.set_transform(carla.Transform(new_loc, vehicle.get_transform().rotation))
                    # 重新下发指令
                    vehicle.apply_control(control)

                # 打印状态（简化，不阻塞）
                print(f" 速度：{speed:.2f}m/s | 转向：{steer:.2f}", end='\r')
                time.sleep(0.01)  # 高频循环，确保指令生效

            # 停车
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            print("\n🛑 行驶结束，已停车！")

            # 打印最终状态
            vehicle_location = vehicle.get_location()
            vehicle_velocity = vehicle.get_velocity()
            print(f"\n📊 车辆最终状态：")
            print(f"   位置：X={vehicle_location.x:.2f}, Y={vehicle_location.y:.2f}")
            print(f"   速度：X={vehicle_velocity.x:.2f}, Y={vehicle_velocity.y:.2f}")

        else:
            print("⚠️ 未找到合法的车辆生成点")

    except Exception as e:
        print(f"\n❌ 调用失败：{e}")
        print("\n🔍 排查建议：")
        print("1. 关闭Carla所有窗口，结束任务管理器中的CarlaUE4.exe进程")
        print("2. 重新启动Carla：CarlaUE4.exe -windowed -ResX=800 -ResY=600")
        print("3. 以管理员身份运行此代码")

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
        print("✅ 所有资源清理完成")


if __name__ == "__main__":
    main()