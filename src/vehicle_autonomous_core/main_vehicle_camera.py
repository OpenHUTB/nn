import carla
import time
import math

def main():
    vehicle = None
    world = None
    camera = None

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
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[4])
        vehicle.set_autopilot(True)
        print("[INFO] 车辆已启动，开启车载摄像头监控")

        # 车载RGB摄像头
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '100')
        
        camera_transform = carla.Transform(carla.Location(x=2.5, z=1.5))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # 摄像头监听
        def camera_callback(image):
            print(f"[摄像头] 已采集画面 {image.frame} 帧")

        camera.listen(camera_callback)

        # 视角跟随
        spectator = world.get_spectator()

        # 主循环
        for i in range(800):
            world.tick()
            trans = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                trans.location + carla.Location(x=-16, z=6),
                carla.Rotation(pitch=-25)
            ))

            # 状态输出
            if i % 30 == 0:
                vel = vehicle.get_velocity()
                speed = 3.6 * math.hypot(vel.x, vel.y)
                print(f"[状态] 车速：{speed:.1f} km/h | 摄像头正常工作")

        print("[INFO] 车载摄像头监控完成")

    finally:
        # 安全销毁资源（修复is_alive调用问题）
        if camera and camera.is_alive:
            camera.destroy()
        if vehicle and vehicle.is_alive:
            vehicle.destroy()
        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        print("[INFO] 所有资源已安全释放")

if __name__ == '__main__':
    main()