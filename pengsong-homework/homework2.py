"""
第二次作业：Carla 传感器综合实验
功能：车辆生成 + 自动驾驶 + RGB摄像头 + 激光雷达 + 碰撞传感器
环境：HUTB/Carla
"""
import carla
import time
import random

def run():
    # 1. 连接模拟器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    actors = []  # 用于存放所有生成的物体，最后统一销毁

    try:
        # 2. 随机选择出生点
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        # 3. 生成车辆（特斯拉）
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)  # 开启自动驾驶
        actors.append(vehicle)
        print("✅ 车辆生成成功，已开启自动驾驶")

        # 4. 生成 RGB 摄像头（车头视角）
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_transform = carla.Transform(carla.Location(x=2.5, z=1.2))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actors.append(camera)

        # 5. 生成激光雷达（LIDAR）
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_transform = carla.Transform(carla.Location(z=1.5))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        actors.append(lidar)
        print("✅ 激光雷达已启动")

        # 6. 生成碰撞检测传感器
        collision_bp = bp_lib.find('sensor.other.collision')
        collision = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        actors.append(collision)

        # 碰撞回调：发生碰撞就打印
        def on_collision(event):
            print(f"💥 碰撞触发！对象：{event.other_actor.type_id}")

        collision.listen(on_collision)
        print("✅ 碰撞传感器已启动")

        # 7. 视角跟随车辆
        spectator = world.get_spectator()
        for _ in range(30):
            spectator.set_transform(camera.get_transform())
            time.sleep(0.1)

        print("✅ 第二次作业运行中：车辆自动驾驶 + 双传感器 + 碰撞检测")
        time.sleep(20)

    finally:
        # 安全销毁所有演员
        for actor in actors:
            if actor.is_alive:
                actor.destroy()
        print("\n✅ 所有传感器/车辆已销毁，作业正常结束")

if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print("❌ 作业运行出错：", e)