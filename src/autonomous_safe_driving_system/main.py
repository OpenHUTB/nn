import carla
import math
import time

def main():
    # 连接CARLA服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    map = world.get_map()

    # 初始化仿真设置
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    ego_vehicle = None
    try:
        # 生成自车（直接写在这里，不依赖外部模块）
        blueprint_lib = world.get_blueprint_library()
        vehicle_bp = blueprint_lib.find('vehicle.tesla.model3')
        spawn_points = map.get_spawn_points()
        ego_vehicle = None
        for sp in spawn_points:
            try:
                ego_vehicle = world.spawn_actor(vehicle_bp, sp)
                break
            except RuntimeError:
                continue
        if not ego_vehicle:
            raise RuntimeError("无法生成自车，请重启CARLA")

        print("✅ 自车生成成功，启动安全驾驶系统")
        target_speed = 30.0

        # 主循环
        for _ in range(3000):
            world.tick()

            # 获取自车状态
            ego_transform = ego_vehicle.get_transform()
            ego_velocity = ego_vehicle.get_velocity()
            ego_speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)

            # 基础巡航控制
            control = carla.VehicleControl()
            if ego_speed < target_speed:
                control.throttle = 0.5
            else:
                control.throttle = 0.0
            control.brake = 0.0
            control.steer = 0.0

            ego_vehicle.apply_control(control)
            print(f"🚗 正常巡航 | 速度：{ego_speed:.1f} km/h")
            time.sleep(0.05)

    finally:
        # 清理仿真环境
        settings.synchronous_mode = False
        world.apply_settings(settings)
        if ego_vehicle and ego_vehicle.is_alive:
            ego_vehicle.destroy()
        print("\n✅ 仿真结束，资源已释放")

if __name__ == '__main__':
    main()