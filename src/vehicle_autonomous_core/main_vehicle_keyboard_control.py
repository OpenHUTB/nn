import carla
import time
import math

def main():
    vehicle = None
    world = None

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        print("[INFO] 成功连接CARLA")
        print("[INFO] 自动控制模式启动，车辆将按预设路径行驶")

        # 同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # 生成车辆
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

        # 固定视角，不漂移
        spectator = world.get_spectator()

        # 预设路径控制：前进→右转→左转→停止
        for i in range(1000):
            world.tick()

            control = carla.VehicleControl()
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 0.0

            # 阶段1：直线前进
            if i < 300:
                control.throttle = 0.4
            # 阶段2：轻微右转
            elif i < 500:
                control.throttle = 0.3
                control.steer = 0.3
            # 阶段3：轻微左转
            elif i < 700:
                control.throttle= 0.3
                control.steer=-0.3
            # 阶段4：减速停车
            else:
                control.brake=1.0

            vehicle.apply_control(control)

            # 视角锁定在车后面
            trans = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                trans.location + carla.Location(x=-12, z=4),
                carla.Rotation(pitch=-20)
            ))

            # 打印状态
            if i % 20 == 0:
                vel = vehicle.get_velocity()
                speed = 3.6 * math.hypot(vel.x, vel.y)
                print(f"[状态] 车速：{speed:.1f} km/h")

        print("[INFO] 行驶完成")

    finally:
        if vehicle and vehicle.is_alive:
            vehicle.destroy()
        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        print("[INFO] 资源已安全释放")

if __name__ == '__main__':
    main()