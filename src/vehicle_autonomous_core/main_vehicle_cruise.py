import carla
import time
import math

def main():
    client = None
    world = None
    vehicle = None

    try:
        # 连接CARLA服务器
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        print("[INFO] 成功连接CARLA服务器")

        # 设置同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # 生成车辆
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[1])

        # 开启自动驾驶（去掉不兼容的交通管理器设置）
        vehicle.set_autopilot(True)
        print("[INFO] 车辆已启动自动巡航模式")

        # 视角跟随
        spectator = world.get_spectator()

        # 主循环
        for i in range(800):
            world.tick()

            # 更新视角
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(x=-10, z=4.5),
                carla.Rotation(pitch=-20)
            ))

            # 定时输出状态
            if i % 20 == 0:
                velocity = vehicle.get_velocity()
                speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
                print(f"[状态] 巡航速度：{speed:.1f} km/h | 位置: ({transform.location.x:.1f}, {transform.location.y:.1f})")

        print("[INFO] 自动巡航完成")

    finally:
        # 恢复世界设置
        if world is not None:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)

        # 销毁车辆
        if vehicle and vehicle.is_alive:
            vehicle.destroy()

        print("[INFO] 资源已安全释放")

if __name__ == '__main__':
    main()