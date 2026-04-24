import carla
import time
import math

def main():
    vehicle = None
    world = None
    radar = None

    try:
        # 连接CARLA服务器
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
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[3])
        vehicle.set_autopilot(True)
        print("[INFO] 车辆已启动，开启障碍物检测功能")

        # 雷达传感器（障碍物检测专用，全新功能）
        radar_bp = bp_lib.find('sensor.other.radar')
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)

        # 雷达检测回调
        def radar_callback(radar_data):
            for detection in radar_data:
                dist = detection.depth
                if dist < 8:
                    print(f"[预警] 前方障碍物距离：{dist:.1f} 米")

        radar.listen(radar_callback)

        # 视角跟随
        spectator = world.get_spectator()

        # 主循环
        for i in range(800):
            world.tick()
            trans = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                trans.location + carla.Location(x=-14, z=5.5),
                carla.Rotation(pitch=-25)
            ))

            # 状态输出
            if i % 30 == 0:
                vel = vehicle.get_velocity()
                speed = 3.6 * math.hypot(vel.x, vel.y)
                print(f"[状态] 车速：{speed:.1f} km/h | 障碍物检测中")

    finally:
        # 资源安全释放
        if radar and radar.is_alive:
            radar.destroy()
        if vehicle and vehicle.is_alive:
            vehicle.destroy()
        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        print("[INFO] 所有资源已安全释放")

if __name__ == '__main__':
    main()