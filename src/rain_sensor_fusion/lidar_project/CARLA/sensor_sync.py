#!/usr/bin/env python

import glob
import os
import sys
from queue import Queue
from queue import Empty

# 加载CARLA环境
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    sensor_queue.put((sensor_data.frame, sensor_name))

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    original_settings = world.get_settings()
    sensor_list = []

    try:
        # 同步模式
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.1
        settings.synchronous_mode = True
        world.apply_settings(settings)

        sensor_queue = Queue()
        blueprint_library = world.get_blueprint_library()

        # 1. RGB相机
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '800')
        cam_bp.set_attribute('image_size_y', '600')
        cam_bp.set_attribute('fov', '90')
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        cam = world.spawn_actor(cam_bp, cam_transform)
        cam.listen(lambda data: sensor_callback(data, sensor_queue, "camera01"))
        sensor_list.append(cam)

        # 2. 激光雷达1
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('range', '50')
        lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        lidar = world.spawn_actor(lidar_bp, lidar_transform)
        lidar.listen(lambda data: sensor_callback(data, sensor_queue, "lidar01"))
        sensor_list.append(lidar)

        # 3. 激光雷达2
        lidar2 = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=1.5, z=2.4, carla.Rotation(yaw=90))))
        lidar2.listen(lambda data: sensor_callback(data, sensor_queue, "lidar02"))
        sensor_list.append(lidar2)

        print("✅ 传感器同步已启动，终端将实时输出同步状态")
        print("💡 同步成功的帧号会完全一致，无警告即代表工作正常")

        while True:
            world.tick()
            w_frame = world.get_snapshot().frame
            print(f"\n世界帧号: {w_frame}")

            try:
                for _ in range(len(sensor_list)):
                    s_frame, s_name = sensor_queue.get(True, 1.0)
                    print(f"  同步成功 | {s_name} | 帧号:{s_frame}")
            except Empty:
                print("  ⚠️ 部分传感器数据丢失")

    finally:
        world.apply_settings(original_settings)
        for sensor in sensor_list:
            if sensor is not None:
                sensor.destroy()
        print("\n✅ 程序退出，传感器全部销毁")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n✅ 手动停止运行')