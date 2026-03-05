import carla
import time
import math


def main():
    # 初始化变量，用于后续资源清理
    # 初始化变量
    vehicle = None
    camera_sensor = None
    collision_sensor = None
    spectator = None
    is_vehicle_alive = False  # 标记车辆是否真实存活

    # 核心配置（聚焦稳定性和运动性）
    CONFIG = {
        "init_control_times": 12,  # 初始激活指令次数（确保能动）
        "init_control_interval": 0.05,  # 每次激活指令间隔
        "init_total_delay": 0.8,  # 激活总延迟（适配物理引擎响应）
        "normal_throttle": 0.85,  # 正常行驶油门（保证动力）
        "avoid_throttle": 0.5,  # 绕障时油门
        "avoid_steer": 0.6,  # 绕障转向幅度
        "loop_interval": 0.008,  # 控制循环间隔（响应快）
        "detect_distance": 10.0,  # 障碍物检测距离
        "stuck_reset_dist": 2.0  # 卡停时重置距离
    }

    try:
        # 1. 连接Carla模拟器（超长超时+稳定性配置）
        client = carla.Client("localhost", 2000)
        client.set_timeout(60.0)  # 60秒超时，适配低配/卡顿场景
        world = client.get_world()
        print(f"✅ 成功连接Carla模拟器 | 地图：{world.get_map().name}")

        # 重置世界设置，关闭同步模式（物理引擎更稳定）
    spectator = None  # 控制模拟器视角，确保能看到车辆
    try:
        # 1. 连接Carla模拟器（延长超时，适配低配电脑）
        world_settings.fixed_delta_seconds = None
        world.apply_settings(world_settings)

        # 清理残留Actor（避免资源冲突）
        for actor in world.get_actors():
            if actor.type_id.startswith(("vehicle", "sensor")):
                actor.destroy()
        time.sleep(1)  # 等待清理完成
        print("🧹 已清理所有残留车辆/传感器")

        # 2. 选择安全生成点（避免卡阻）
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise Exception("❌ 未找到任何车辆生成点")

        # 优先选前5个生成点中最空旷的
        spawn_point = spawn_points[2] if len(spawn_points) >= 3 else spawn_points[0]
        print(f"📍 选定车辆生成点 | 位置：({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
        # 清空残留车辆，避免碰撞卡阻
        for actor in world.get_actors():
            if actor.type_id.startswith("vehicle"):
        # world = client.load_world("Town01")
        # print("🔄 已切换地图为：Town01")

        # 2. 获取车辆蓝图，设置红色车身
        # 3. 生成车辆（多次重试+存活校验）
        vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '255,0,0')  # 红色车身
        print("🎨 已设置车辆颜色为红色")

        # 3. 选择合法生成点生成车辆（增加重试，避免碰撞失败）
        spawn_points = world.get_map().get_spawn_points()
        if spawn_points:
            spawn_point = spawn_points[0]  # 可替换为spawn_points[10]避免边缘位置
            # 生成车辆（重试3次，解决偶发碰撞问题）
            max_retry = 3
            for i in range(max_retry):
                try:
                    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle_bp.set_attribute("color", "255,0,0")  # 红色车身

        # 5次重试生成，确保成功
        max_spawn_retry = 5
        for retry in range(max_spawn_retry):
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                # 校验车辆是否真的存活
                if vehicle and vehicle.is_alive:
                    vehicle.set_simulate_physics(True)  # 强制开启物理
                    vehicle.set_autopilot(False)
                    is_vehicle_alive = True
                    print(f"🚗 车辆生成成功 | ID：{vehicle.id} | 重试次数：{retry + 1}")
                    break
                except:
                    if i == max_retry - 1:
                        raise Exception("车辆生成失败：生成点有碰撞，请更换spawn_points索引（如spawn_points[10]）")
                    time.sleep(0.5)

            print(f"🚗 成功生成特斯拉车辆，ID：{vehicle.id}")
                else:
                    if vehicle:
                        vehicle.destroy()
            except Exception as e:
                if retry == max_spawn_retry - 1:
                    raise Exception(f"🚨 车辆生成失败（重试{max_spawn_retry}次）：{e}")
                time.sleep(0.8)

        # 4. 强制激活车辆（核心：确保小车能动）
        print("🔋 正在激活车辆物理状态...")
        # 连续下发激活指令，确保物理引擎响应
        for _ in range(CONFIG["init_control_times"]):
            vehicle.apply_control(carla.VehicleControl(
                throttle=1.0,  # 满油门激活
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False
            ))
            time.sleep(CONFIG["init_control_interval"])

        time.sleep(CONFIG["init_total_delay"])  # 给物理引擎足够响应时间

        # 校验激活状态：检查速度是否大于0
        init_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
        if init_speed < 0.1:
            print("⚠️ 车辆初始速度低，二次激活...")
            # 重置物理状态后再次激活
            vehicle.set_simulate_physics(False)
            time.sleep(0.2)
            vehicle.set_simulate_physics(True)
            vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
            time.sleep(0.3)

        # 5. 绑定视角（全程跟随，便于观察）
        spectator = world.get_spectator()

            # 关键：将模拟器视角瞬移到车辆上方（确保能看到车）
            spectator_transform = carla.Transform(
                spawn_point.location + carla.Location(z=5),  # 车辆上方5米
                carla.Rotation(pitch=-15, yaw=spawn_point.rotation.yaw)  # 俯视视角
        def follow_vehicle():
            trans = vehicle.get_transform()
            # 视角后移+升高，清晰观察车辆运动
            spectator_loc = carla.Location(
                x=trans.location.x - math.cos(math.radians(trans.rotation.yaw)) * 7,
                y=trans.location.y - math.sin(math.radians(trans.rotation.yaw)) * 7,
                z=trans.location.z + 4.5
            )
            spectator_rot = carla.Rotation(pitch=-30, yaw=trans.rotation.yaw)
            spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))

        follow_vehicle()
        print("👀 视角已绑定车辆，全程跟随")

        # 6. 简化传感器（非核心功能，失败不影响运动）
        # 碰撞传感器：碰撞后继续行驶，不停车
        try:
            collision_bp = world.get_blueprint_library().find("sensor.other.collision")
            collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

            def collision_cb(event):
                nonlocal steer
                print("\n💥 检测到碰撞，自动调整方向！")
                steer = -steer if abs(steer) > 0 else -CONFIG["avoid_steer"]
                vehicle.apply_control(carla.VehicleControl(
                    throttle=CONFIG["avoid_throttle"],
                    steer=steer,
                    brake=0.0
                ))

            collision_sensor.listen(collision_cb)
            print("🛡️ 碰撞传感器已挂载")
        except:
            print("⚠️ 碰撞传感器挂载失败（不影响车辆运动）")

        # 7. 障碍物检测（简化逻辑，确保行驶流畅）
        def detect_obstacle():
            trans = vehicle.get_transform()
            # 检测前方2-10米的障碍物
            for check_dist in range(2, int(CONFIG["detect_distance"]) + 1, 2):
                check_loc = trans.location + trans.get_forward_vector() * check_dist
                waypoint = world.get_map().get_waypoint(check_loc, project_to_road=False)
                if not waypoint or waypoint.lane_type != carla.LaneType.Driving:
                    return True
            return False

        # 8. 核心行驶逻辑（无限行驶，无时长限制）
        print("\n🚙 车辆开始行驶（无限时长）| 按 Ctrl+C 手动终止")
        print("------------------------------------------------")
        steer = 0.0
        run_time = 0  # 记录行驶时长（秒）

        # 无限循环行驶（替代固定时长，满足"行驶时长加长"需求）
        while True:
            # 实时校验车辆状态
            if not vehicle or not vehicle.is_alive:
                print("❌ 车辆异常消失，程序终止")
                break

            # 更新视角
            follow_vehicle()

            # 检测障碍物并调整转向
            has_obstacle = detect_obstacle()
            if has_obstacle:
                steer = CONFIG["avoid_steer"]  # 向右绕行
                throttle = CONFIG["avoid_throttle"]
                print(
                    f"\r⚠️ 前方有障碍 | 绕行中 | 行驶时长：{run_time:.0f}秒 | 速度：{math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y) * 3.6:.0f}km/h",
                    end="")
            else:
                # 平滑回正转向
                steer = steer * 0.9 if abs(steer) > 0.05 else 0.0
                throttle = CONFIG["normal_throttle"]
                print(
                    f"\r✅ 正常行驶 | 行驶时长：{run_time:.0f}秒 | 速度：{math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y) * 3.6:.0f}km/h | 转向：{steer:.2f}",
                    end="")

            # 持续下发行驶指令（核心：确保车辆一直运动）
            vehicle.apply_control(carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=0.0,
                hand_brake=False,
                reverse=False
            ))

            # 卡停处理：速度过低时重置位置
            current_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
            if current_speed < 0.1:
                print("\n⚠️ 车辆卡停，重置位置...")
                new_loc = vehicle.get_transform().location + carla.Location(x=CONFIG["stuck_reset_dist"])
                vehicle.set_transform(carla.Transform(new_loc, vehicle.get_transform().rotation))
                vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

            # 更新行驶时长
            run_time += CONFIG["loop_interval"]
            time.sleep(CONFIG["loop_interval"])

    # 手动终止处理（Ctrl+C）
    except KeyboardInterrupt:
        print(f"\n\n🛑 手动终止程序 | 车辆累计行驶时长：{run_time:.0f}秒")
    # 异常处理
    except Exception as e:
        print(f"\n❌ 程序异常：{str(e)}")
        print("\n🔧 快速修复建议：")
        print("1. 关闭Carla，在任务管理器结束CarlaUE4.exe")
        print("2. 以管理员身份重启Carla：CarlaUE4.exe -windowed -ResX=800 -ResY=600")
        print("3. 再次运行本代码")
    # 资源清理（仅在车辆存活时执行）
    finally:
        print("\n🧹 开始清理资源...")
        # 停车并销毁车辆
        if vehicle and is_vehicle_alive:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            time.sleep(1)
            vehicle.destroy()
            print("🗑️ 车辆已安全销毁")
        # 销毁传感器
        if collision_sensor:
            collision_sensor.stop()
            collision_sensor.destroy()
            print("🗑️ 碰撞传感器已销毁")
        if camera_sensor:
            camera_sensor.stop()
            camera_sensor.destroy()
            print("🗑️ 摄像头已销毁")
        print("✅ 所有资源清理完成！")
            spectator.set_transform(spectator_transform)
            print("👀 模拟器视角已切换到车辆位置！")

        # 2. 获取车辆蓝图，随机选择车辆颜色
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
            # 定义摄像头回调函数（保存图片/打印信息）
            def camera_callback(image):
                # 保存摄像头画面到本地（可选，取消注释即可）
                # image.save_to_disk(f'./camera_images/frame_{image.frame_number}.png')
                print(f"📸 摄像头帧号：{image.frame_number} | 时间戳：{image.timestamp}")

            # 绑定回调函数
            camera_sensor.listen(camera_callback)
            print("📹 已挂载RGB摄像头，开始采集画面！")

            # 5. 车辆多阶段控制（前进→右转→减速）
            print("\n🚙 开始车辆控制演示...")
            # 阶段1：直行3秒（油门0.7，行驶更明显）
            vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=0.0, brake=0.0))
            # 阶段1：直行3秒
            vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=0.0, brake=0.0))
            time.sleep(3)
            # 阶段2：右转2秒
            vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.5, brake=0.0))
            time.sleep(2)
            # 阶段3：减速停车
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            time.sleep(1)
            print("🛑 车辆已停车")

            # 6. 打印车辆最终状态
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
        print(f"❌ 调用失败：{e}")
        print("\n🔍 排查建议：")
        print("1. 确认Carla模拟器是0.9.11版本，与代码适配")
        print("2. 模拟器窗口不要最小化，保持前台显示")
        print("3. 尝试更换生成点索引：将spawn_points[0]改为spawn_points[10]/spawn_points[20]")

    # 7. 资源清理（延迟销毁，确保能看到车辆直到程序结束）
    finally:
        time.sleep(3)  # 程序结束前车辆多显示3秒

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