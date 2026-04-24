import carla
import time
import math

def main():
    vehicle = None
    world = None

    try:
        # 连接CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        print("[INFO] 成功连接CARLA服务器")

        # 同步模式设置
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # 生成车辆
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[2])

        # 开启自动驾驶
        vehicle.set_autopilot(True)
        print("[INFO] 车辆已启动，保持车道行驶中")

        # 视角跟随
        spectator = world.get_spectator()

        # 主循环
        for i in range(800):
            world.tick()
            trans = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                trans.location + carla.Location(x=-15, z=5),
                carla.Rotation(pitch=-25)
            ))

            # 稳定车速输出
            if i % 25 == 0:
                vel = vehicle.get_velocity()
                speed = 3.6 * math.hypot(vel.x, vel.y)
                print(f"[状态] 当前车速：{speed:.1f} km/h | 保持车道正常行驶")

        print("[INFO] 车道保持行驶完成")

    finally:
        # 安全销毁资源
        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        if vehicle and vehicle.is_alive:
            vehicle.destroy()
        print("[INFO] 资源已安全释放")

if __name__ == '__main__':
    main()