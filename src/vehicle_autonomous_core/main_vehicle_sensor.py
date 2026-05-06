import carla
import time
import math

def main():
    # 定义所有对象，确保退出时能安全销毁
    ego_vehicle = None
    camera = None
    lidar = None
    collision = None

    try:
        # 1. 连接 CARLA 服务器
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        print("[INFO] 成功连接 CARLA 服务器")

        # 2. 生成自动驾驶车辆
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_point = world.get_map().get_spawn_points()[0]
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        ego_vehicle.set_autopilot(True)
        print("[INFO] 车辆生成成功，已开启自动驾驶")

        # 3. 挂载 RGB 摄像头
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=2.5, z=1.3))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        print("[INFO] RGB 摄像头已挂载")

        # 4. 挂载激光雷达
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_transform = carla.Transform(carla.Location(z=1.8))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
        print("[INFO] 激光雷达已挂载")

        # 5. 挂载碰撞传感器
        collision_bp = bp_lib.find('sensor.other.collision')
        collision = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego_vehicle)
        print("[INFO] 碰撞传感器已挂载")

        # 6. 主循环：视角跟随 + 状态输出
        spectator = world.get_spectator()
        for i in range(500):
            world.tick()
            vehicle_transform = ego_vehicle.get_transform()
            # 设置跟随视角
            spectator.set_transform(carla.Transform(
                vehicle_transform.location + carla.Location(x=-10, z=4),
                carla.Rotation(pitch=-20)
            ))
            # 定时输出车速
            if i % 30 == 0:
                v = ego_vehicle.get_velocity()
                speed = 3.6 * math.hypot(v.x, v.y)
                print(f"[状态] 车速：{speed:.1f} km/h")

        print("[INFO] 程序运行完成")

    finally:
        # 安全销毁所有资源
        if camera and camera.is_alive:
            camera.destroy()
        if lidar and lidar.is_alive:
            lidar.destroy()
        if collision and collision.is_alive:
            collision.destroy()
        if ego_vehicle and ego_vehicle.is_alive:
            ego_vehicle.destroy()
        print("[INFO] 所有资源已安全释放")

if __name__ == '__main__':
    main()