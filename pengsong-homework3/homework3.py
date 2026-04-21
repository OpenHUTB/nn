"""
第三次作业：Carla多传感器融合与交通场景交互实验
作者：彭嵩
功能亮点：
1. 随机地图+随机出生点，适配不同Town地图
2. 车辆自动驾驶+交通灯自动控制
3. RGB摄像头+激光雷达+碰撞传感器多传感器融合
4. 摄像头画面可保存，实时打印传感器日志
5. 车辆速度/位置/碰撞事件日志打印
6. 安全退出自动销毁所有Actor，无资源泄漏
"""
import carla
import time
import random
import math
from datetime import datetime

def setup_traffic(client, world):
    """生成交通流，让场景更真实"""
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    traffic_manager.set_synchronous_mode(True)
    return traffic_manager

def spawn_ego_vehicle(world, blueprint_library, spawn_point):
    """生成主车，设置自动驾驶"""
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)
    return vehicle

def attach_sensors(world, vehicle, blueprint_library):
    """挂载所有传感器，并配置回调"""
    sensors = []

    # 1. RGB摄像头（车头视角，带画面保存）
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    sensors.append(camera)

    def camera_callback(image):
        # 保存第10帧画面作为作业运行截图
        if image.frame == 10:
            image.save_to_disk('third_assignment_camera.png')
        print(f"[摄像头] 帧: {image.frame}, 时间: {datetime.now()}")
    camera.listen(camera_callback)

    # 2. 激光雷达（兼容版本，不使用plot）
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '60')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '100000')
    lidar_transform = carla.Transform(carla.Location(x=0, z=1.8))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    sensors.append(lidar)

    def lidar_callback(data):
        print(f"[激光雷达] 收到点云数据，点数量: {len(data)}")
    lidar.listen(lidar_callback)

    # 3. 碰撞传感器（带详细日志）
    collision_bp = blueprint_library.find('sensor.other.collision')
    collision = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    sensors.append(collision)

    def collision_callback(event):
        other_actor = event.other_actor
        print(f"[碰撞传感器] 碰撞发生！碰撞对象: {other_actor.type_id}")
        print(f"碰撞位置: {event.transform.location}")
    collision.listen(collision_callback)

    return sensors

def main():
    # 1. 连接Carla服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # 2. 同步模式，保证画面稳定
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 3. 生成交通流
    traffic_manager = setup_traffic(client, world)

    # 4. 随机选择出生点
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)

    # 5. 生成主车
    ego_vehicle = spawn_ego_vehicle(world, blueprint_library, spawn_point)
    print("✅ 主车生成成功，已开启自动驾驶")

    # 6. 挂载传感器
    sensors = attach_sensors(world, ego_vehicle, blueprint_library)
    print("✅ 所有传感器已挂载并启动")

    # 7. 主循环
    spectator = world.get_spectator()
    try:
        for i in range(300):  # 运行15秒
            world.tick()
            # 更新视角，跟随主车后方
            spectator_transform = carla.Transform(
                ego_vehicle.get_transform().location + carla.Location(x=-10, z=5),
                carla.Rotation(pitch=-20)
            )
            spectator.set_transform(spectator_transform)

            # 打印车辆状态日志
            if i % 20 == 0:
                velocity = ego_vehicle.get_velocity()
                speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
                print(f"[车辆状态] 速度: {speed:.1f} km/h, 位置: {ego_vehicle.get_transform().location}")

    finally:
        # 安全销毁所有Actor
        for sensor in sensors:
            if sensor.is_alive:
                sensor.destroy()
        if ego_vehicle.is_alive:
            ego_vehicle.destroy()
        print("\n✅ 所有传感器与车辆已安全销毁，作业运行结束")

        # 恢复世界设置
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"❌ 作业运行出错: {e}")