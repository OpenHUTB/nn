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

    # 生成主车
    spawn_points = map.get_spawn_points()
    ego_spawn = spawn_points[0]
    ego_vehicle = world.spawn_actor(vehicle_bp, ego_spawn)
    print("✅ 主车生成成功，启动自动紧急制动（AEB）演示")

    # 生成前方静止障碍车（用于测试）
    obstacle_spawn = spawn_points[1]
    obstacle_spawn.location.x += 30.0  # 放在主车前方30米处
    obstacle_vehicle = world.spawn_actor(vehicle_bp, obstacle_spawn)
    obstacle_vehicle.set_autopilot(False)
    print("✅ 前方障碍车已生成")

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    try:
        # 基础巡航速度
        target_speed = 20.0  # km/h
        safe_distance = 10.0 # 安全距离阈值（米）

        for _ in range(1000):
            world.tick()

            # 获取主车状态
            ego_transform = ego_vehicle.get_transform()
            ego_velocity = ego_vehicle.get_velocity()
            ego_speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)

            # 获取障碍车状态
            obstacle_transform = obstacle_vehicle.get_transform()
            # 计算两车直线距离
            distance = ego_transform.location.distance(obstacle_transform.location)

            # 核心：自动紧急制动逻辑
            control = carla.VehicleControl()

            if distance < safe_distance:
                # 距离过近，紧急制动
                control.throttle = 0.0
                control.brake = 1.0
                print(f"🛑 前方距离过近！已触发紧急制动 | 距离：{distance:.1f}m")
            else:
                # 距离安全，保持巡航
                if ego_speed < target_speed:
                    control.throttle = 0.4
                else:
                    control.throttle = 0.0
                control.brake = 0.0
                print(f"🚗 正常巡航中 | 速度：{ego_speed:.1f}km/h | 距离障碍车：{distance:.1f}m")

            control.steer = 0.0  # 保持直线行驶
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