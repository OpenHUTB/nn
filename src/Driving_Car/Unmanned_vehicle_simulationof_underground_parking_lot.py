import carla
import random
import time

def main():
    # 1. 连接Carla服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # 超时时间10秒

    try:
        # 2. 加载包含地下停车场的地图（Carla部分官方地图自带地下停车场，如Town07、Town10）
        # 若使用自定义地下停车场地图，替换为自定义地图名称
        world = client.load_world('Town07')  # Town07包含地下停车场区域
        blueprint_library = world.get_blueprint_library()

        # 3. 设置天气（地下停车场可设为阴天，增强真实感）
        weather = carla.WeatherParameters(
            cloudiness=90.0,
            precipitation=0.0,
            sun_altitude_angle=0.0  # 模拟地下无阳光
        )
        world.set_weather(weather)

        # 4. 获取地下停车场的生成点（可自定义坐标，或从地图预设点筛选）
        # 方式1：手动设置地下停车场坐标（需根据地图实际坐标调整）
        spawn_point = carla.Transform(
            carla.Location(x=-120.0, y=230.0, z=-1.0),  # z为负数表示地下
            carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)
        )
        # 方式2：从地图预设生成点中随机选（若预设点包含地下区域）
        # spawn_points = world.get_map().get_spawn_points()
        # spawn_point = random.choice(spawn_points)

        # 5. 生成无人车（选择特斯拉Model3为例）
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('color', 'black')  # 设置车辆颜色
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"无人车生成成功：{vehicle.id}")

        # 6. 为车辆添加传感器（摄像头+激光雷达，用于地下停车场感知）
        ## 6.1 前置RGB摄像头
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '90')
        # 摄像头安装位置（车辆前上方）
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.8))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        # 摄像头数据回调函数（保存图片）
        def camera_callback(image, data_dict):
            image.save_to_disk('./out/camera/%06d.png' % image.frame)
        camera.listen(lambda image: camera_callback(image, None))

        ## 6.2 激光雷达（地下停车场障碍物检测）
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '50')  # 地下停车场短距离感知足够
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('points_per_second', '100000')
        # 激光雷达安装位置（车辆顶部）
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        # 激光雷达数据回调函数（保存点云）
        def lidar_callback(point_cloud, data_dict):
            point_cloud.save_to_disk('./out/lidar/%06d.ply' % point_cloud.frame)
        lidar.listen(lambda point_cloud: lidar_callback(point_cloud, None))

        # 7. 车辆控制（简单的自动驾驶：匀速直线行驶，可替换为路径规划算法）
        vehicle.set_autopilot(True)  # 启用Carla内置自动驾驶
        # 若需自定义控制，可替换为以下代码：
        # control = carla.VehicleControl()
        # control.throttle = 0.3  # 油门
        # control.steer = 0.0     # 方向
        # vehicle.apply_control(control)

        # 8. 运行仿真（持续30秒，可根据需求调整）
        time.sleep(30)

    except Exception as e:
        print(f"仿真过程中出现错误：{e}")
    finally:
        # 9. 清理资源（销毁所有生成的actor）
        actors = [vehicle, camera, lidar] if 'vehicle' in locals() else []
        for actor in actors:
            if actor.is_alive:
                actor.destroy()
        print("仿真结束，资源已清理")

if __name__ == '__main__':
    main()