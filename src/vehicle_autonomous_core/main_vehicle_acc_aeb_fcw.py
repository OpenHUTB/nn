import carla
import math
import time

def main():
    # 连接 CARLA 服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    map = world.get_map()

    # 获取蓝图库
    blueprint_lib = world.get_blueprint_library()
    vehicle_bp = blueprint_lib.find('vehicle.tesla.model3')

    # 生成主车（自车）
    spawn_points = map.get_spawn_points()
    ego_spawn = spawn_points[0]
    ego_vehicle = world.spawn_actor(vehicle_bp, ego_spawn)
    print("✅ 主车生成成功，启动 ACC+AEB+FCW 主动安全演示")

    # 生成前方障碍车（可设置为移动或静止）
    obstacle_spawn = spawn_points[1]
    obstacle_spawn.location.x += 40.0  # 放在主车前方40米处
    obstacle_vehicle = world.spawn_actor(vehicle_bp, obstacle_spawn)
    obstacle_vehicle.set_autopilot(False)
    print("✅ 前方障碍车已生成")

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    try:
        # 巡航与安全参数
        target_speed = 30.0  # 目标巡航速度（km/h）
        time_gap = 1.5       # 安全时距（秒），动态安全距离计算用
        warning_distance = 20.0  # FCW预警距离
        brake_distance = 12.0    # 分级制动距离
        emergency_brake_distance = 6.0  # 紧急制动距离

        for _ in range(1200):
            world.tick()

            # 获取自车状态
            ego_transform = ego_vehicle.get_transform()
            ego_velocity = ego_vehicle.get_velocity()
            ego_speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
            ego_speed_ms = ego_speed / 3.6

            # 获取障碍车状态
            obstacle_transform = obstacle_vehicle.get_transform()
            obstacle_velocity = obstacle_vehicle.get_velocity()
            obstacle_speed = 3.6 * math.sqrt(obstacle_velocity.x**2 + obstacle_velocity.y**2)
            obstacle_speed_ms = obstacle_speed / 3.6

            # 计算两车直线距离与相对速度
            distance = ego_transform.location.distance(obstacle_transform.location)
            relative_speed = ego_speed_ms - obstacle_speed_ms

            # 动态安全距离（随车速变化）
            dynamic_safe_distance = max(emergency_brake_distance, ego_speed_ms * time_gap)

            # 车辆控制
            control = carla.VehicleControl()
            control.steer = 0.0  # 保持直线行驶

            # 分级安全控制逻辑（FCW -> 轻制动 -> 紧急制动 -> 巡航）
            if distance < emergency_brake_distance:
                # 1. 紧急制动（最高优先级）
                control.throttle = 0.0
                control.brake = 1.0
                print(f"🛑 【紧急制动】！距离：{distance:.1f}m | 自车速度：{ego_speed:.1f}km/h")

            elif distance < brake_distance:
                # 2. 分级制动（中等制动力）
                control.throttle = 0.0
                control.brake = 0.6
                print(f"⚠️  【分级制动】！距离：{distance:.1f}m | 准备减速")

            elif distance < warning_distance:
                # 3. 前方碰撞预警（FCW）+ 轻微减速
                control.throttle = 0.0
                control.brake = 0.2
                print(f"🔔 【碰撞预警】前方有风险 | 距离：{distance:.1f}m | 相对速度：{relative_speed:.1f}m/s")

            else:
                # 4. 正常自适应巡航（ACC）
                if ego_speed < target_speed:
                    control.throttle = 0.5
                else:
                    control.throttle = 0.0
                control.brake = 0.0
                print(f"🚗 【正常巡航】速度：{ego_speed:.1f}km/h | 距离障碍车：{distance:.1f}m | 动态安全距离：{dynamic_safe_distance:.1f}m")

            ego_vehicle.apply_control(control)
            time.sleep(0.05)

    finally:
        # 恢复设置并释放资源
        settings.synchronous_mode = False
        world.apply_settings(settings)
        ego_vehicle.destroy()
        obstacle_vehicle.destroy()
        print("\n✅ 所有车辆资源已安全释放")

if __name__ == '__main__':
    main()