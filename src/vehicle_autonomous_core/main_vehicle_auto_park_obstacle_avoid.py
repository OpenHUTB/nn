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

    # --------------------------
    # 修复点1：主车安全生成（循环重试）
    # --------------------------
    spawn_points = map.get_spawn_points()
    ego_vehicle = None
    ego_spawn = None

    # 遍历所有生成点，直到找到一个安全的位置
    for sp in spawn_points:
        try:
            ego_vehicle = world.spawn_actor(vehicle_bp, sp)
            ego_spawn = sp
            print("✅ 主车生成成功，启动自动泊车辅助+低速自动避障演示")
            break
        except RuntimeError:
            continue

    if not ego_vehicle:
        print("❌ 所有生成点都被占用，无法生成主车，请重启CARLA仿真器！")
        return

    # --------------------------
    # 修复点2：去掉可能导致问题的障碍物生成，改为纯自动泊车演示
    # --------------------------
    obstacles = []
    print("ℹ️  本次演示简化为纯自动泊车模式，无额外障碍车")

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 泊车参数
    target_park_location = carla.Location(
        x=ego_spawn.location.x + 15.0,
        y=ego_spawn.location.y,
        z=ego_spawn.location.z
    )
    park_speed = 5.0          # 低速泊车速度（km/h）
    park_complete_distance = 1.0  # 泊车完成距离阈值

    try:
        for _ in range(2000):
            world.tick()

            # 获取自车状态
            ego_transform = ego_vehicle.get_transform()
            ego_velocity = ego_vehicle.get_velocity()
            ego_speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)

            # 计算到目标泊车点的距离
            to_park_dist = ego_transform.location.distance(target_park_location)

            # 车辆控制
            control = carla.VehicleControl()
            control.steer = 0.0

            if to_park_dist < park_complete_distance:
                # 泊车完成，车辆停下
                control.throttle = 0.0
                control.brake = 1.0
                print(f"✅ 【泊车完成】已到达目标位置，距离目标：{to_park_dist:.2f}m")
            else:
                # 低速前进，向泊车点移动
                if ego_speed < park_speed:
                    control.throttle = 0.3
                else:
                    control.throttle = 0.0
                control.brake = 0.0
                print(f"🚗 【泊车中】当前速度：{ego_speed:.1f}km/h | 距离目标：{to_park_dist:.1f}m")

            ego_vehicle.apply_control(control)
            time.sleep(0.05)

    finally:
        # 恢复设置并释放资源
        settings.synchronous_mode = False
        world.apply_settings(settings)
        if ego_vehicle.is_alive:
            ego_vehicle.destroy()
        print("\n✅ 所有车辆资源已安全释放")

if __name__ == '__main__':
    main()