import carla
import random
import time
import numpy as np
import cv2
import os

# ======================== 【仅添加这一行注释，功能完全不变】 ========================
# 代码功能完全保留，仅用于触发Git提交更新  
# ==================================================================================

# 输出文件夹
OUTPUT_FOLDER = "_out"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# 视频参数
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
FPS = 10
FRAME_LIMIT = 100  # 最多保存100帧

def main():
    # 连接CARLA服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # 清空所有actor，避免冲突
    for actor in world.get_actors():
        if actor.type_id.startswith('vehicle') or actor.type_id.startswith('sensor'):
            actor.destroy()

    time.sleep(1)

    # 生成一辆车
    vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
    vehicle.set_autopilot(True)
    print("车辆已生成，开启自动驾驶")

    # 设置相机：RGB相机 + 激光雷达
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
    camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
    camera_bp.set_attribute('fov', '100')

    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '50')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('upper_fov', '10')
    lidar_bp.set_attribute('lower_fov', '-30')
    lidar_bp.set_attribute('points_per_second', '100000')

    # 相机安装在车头
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # 激光雷达安装在车顶
    lidar_transform = carla.Transform(carla.Location(x=0, z=2.2))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    # 全局变量保存最新数据
    latest_rgb = None
    frame_count = 0

    # 相机回调
    def process_rgb(image):
        nonlocal latest_rgb
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
        latest_rgb = array[:, :, :3]  # 去掉alpha通道

    # 激光雷达回调：绘制点云到图像
    def process_lidar(point_cloud):
        nonlocal frame_count
        if latest_rgb is None:
            return
        if frame_count >= FRAME_LIMIT:
            return

        img = latest_rgb.copy()

        # 把3D点云投影到2D图像
        points = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))

        for point in points:
            x, y, z = point[0], point[1], point[2]
            if x < 0:
                continue  # 只保留车前方点云

            # 简单投影（仅演示）
            f = IMAGE_WIDTH / 2
            u = int(IMAGE_WIDTH / 2 - (y * f) / x)
            v = int(IMAGE_HEIGHT / 2 - (z * f) / x)

            if 0 <= u < IMAGE_WIDTH and 0 <= v < IMAGE_HEIGHT:
                cv2.circle(img, (u, v), 2, (0, 255, 0), -1)

        # 保存融合后的帧
        filename = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count:04d}.png")
        cv2.imwrite(filename, img)
        print(f"保存第 {frame_count} 帧 -> {filename}")
        frame_count += 1

    # 启动监听
    camera.listen(process_rgb)
    lidar.listen(process_lidar)

    # 运行一段时间
    start_time = time.time()
    try:
        while frame_count < FRAME_LIMIT and time.time() - start_time < 20:
            world.tick()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("手动停止")

    # 清理
    camera.stop()
    lidar.stop()
    camera.destroy()
    lidar.destroy()
    vehicle.destroy()

    print("数据采集完成！")
    print(f"已保存 {frame_count} 帧融合图像到 {OUTPUT_FOLDER} 文件夹")

if __name__ == "__main__":
    main()