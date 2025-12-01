import sys
import os
import carla
import time
import numpy as np
import random

# -------------------------- 1. 配置 Carla 路径（方法 1：脚本内临时添加）--------------------------
carla_api_path = "D:/CARLA_0.9.10/WindowsNoEditor/PythonAPI"
if carla_api_path not in sys.path:
    sys.path.append(carla_api_path)

# -------------------------- 2. 全局变量 --------------------------
HOST = "localhost"  # Carla 服务器 IP（本地默认 localhost）
PORT = 2000  # Carla 服务器端口（默认 2000）
VEHICLE_MODEL = "model3"  # 车辆模型（可改为 "cybertruck"、"mustang" 等）
LIDAR_RANGE = 50  # 激光雷达探测范围（米）

# 全局对象（后续会初始化）
client = None
world = None
vehicle = None
lidar_sensor = None
camera_sensor = None


# -------------------------- 3. 核心功能函数 --------------------------
def connect_to_carla():
    """连接到 Carla 服务器"""
    global client, world
    try:
        # 创建客户端并连接
        client = carla.Client(HOST, PORT)
        client.set_timeout(10.0)  # 超时时间（10 秒）
        world = client.get_world()  # 获取 Carla 世界对象

        print(f"✅ 成功连接到 Carla！当前地图：{world.get_map().name}")
    except Exception as e:
        print(f"❌ 连接 Carla 失败：{e}")
        sys.exit(1)


def spawn_vehicle():
    """在 Carla 中生成车辆"""
    global vehicle
    try:
        # 获取车辆蓝图库
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(VEHICLE_MODEL)[0]  # 选择车辆模型

        # 获取地图中的生成点（选择第一个可用生成点）
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise Exception("❌ 地图中没有可用的车辆生成点")

        # 随机选择一个生成点
        spawn_point = random.choice(spawn_points)

        # 生成车辆
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)  # 开启自动驾驶，让车辆自动行驶

        print(f"✅ 成功生成车辆：{VEHICLE_MODEL}（位置：{spawn_point.location}）")

        # 等待一下，确保车辆完全生成
        time.sleep(1)

    except Exception as e:
        print(f"❌ 生成车辆失败：{e}")
        sys.exit(1)


def setup_lidar():
    """为车辆安装激光雷达传感器"""
    global lidar_sensor
    try:
        # 获取激光雷达蓝图
        blueprint_library = world.get_blueprint_library()
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")

        # 配置激光雷达参数
        lidar_bp.set_attribute("range", str(LIDAR_RANGE))  # 探测范围
        lidar_bp.set_attribute("points_per_second", "50000")  # 每秒点数
        lidar_bp.set_attribute("rotation_frequency", "10")  # 旋转频率（Hz）
        lidar_bp.set_attribute("channels", "32")  # 通道数

        # 激光雷达安装位置（车辆顶部，x 向前，z 向上）
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.4))

        # 生成激光雷达并挂载到车辆上
        lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

        # 注册激光雷达数据回调函数（每帧数据都会触发）
        lidar_sensor.listen(lambda data: lidar_callback(data))
        print(f"✅ 激光雷达已安装：探测范围 {LIDAR_RANGE} 米，回调函数已注册")
    except Exception as e:
        print(f"❌ 安装激光雷达失败：{e}")
        sys.exit(1)


def setup_camera():
    """为车辆安装摄像头传感器（用于观察）"""
    global camera_sensor
    try:
        # 获取摄像头蓝图
        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")

        # 配置摄像头参数
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "110")

        # 摄像头安装位置（车辆前方）
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        # 生成摄像头并挂载到车辆上
        camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # 注册摄像头数据回调函数
        camera_sensor.listen(lambda image: camera_callback(image))
        print("✅ 摄像头已安装")
    except Exception as e:
        print(f"❌ 安装摄像头失败：{e}")


def camera_callback(image):
    """摄像头数据回调函数"""
    # 这里可以处理图像数据，但为了性能我们只是简单确认摄像头在工作
    pass


def set_spectator_follow_vehicle():
    """设置观察者视角跟随车辆"""
    try:
        # 获取观察者对象
        spectator = world.get_spectator()

        # 设置观察者位置在车辆后方上方
        def update_spectator():
            if vehicle:
                transform = vehicle.get_transform()
                # 计算观察者位置（车辆后方10米，上方5米）
                location = transform.location
                rotation = transform.rotation

                # 计算后方位置
                x = location.x - 10 * np.cos(np.radians(rotation.yaw))
                y = location.y - 10 * np.sin(np.radians(rotation.yaw))
                z = location.z + 5

                spectator.set_transform(carla.Transform(
                    carla.Location(x=x, y=y, z=z),
                    carla.Rotation(pitch=-20, yaw=rotation.yaw)
                ))

        return update_spectator
    except Exception as e:
        print(f"❌ 设置观察者视角失败：{e}")
        return None


def lidar_callback(data):
    """激光雷达数据回调函数（处理每帧点云数据）"""
    try:
        # 使用 raw_data 属性并将其转换为点云
        point_cloud = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))

        # 打印基本信息（减少输出频率，避免控制台过于拥挤）
        if random.random() < 0.1:  # 只有10%的概率输出，减少控制台输出
            print(f"📡 激光雷达帧数据：共 {len(point_cloud)} 个点")

    except Exception as e:
        print(f"❌ 处理激光雷达数据时出错：{e}")


def main():
    try:
        # 1. 连接 Carla 服务器
        connect_to_carla()
        # 2. 生成车辆
        spawn_vehicle()
        # 3. 安装激光雷达
        setup_lidar()
        # 4. 安装摄像头（可选）
        setup_camera()
        # 5. 设置观察者视角跟随车辆
        update_spectator = set_spectator_follow_vehicle()

        # 6. 保持程序运行（持续接收激光雷达数据）
        print("\n⏳ 程序运行中，车辆将自动行驶，按 Ctrl+C 停止...")

        frame_count = 0
        while True:
            # 每帧更新观察者视角
            if update_spectator:
                update_spectator()

            frame_count += 1
            if frame_count % 100 == 0:  # 每100帧输出一次状态
                if vehicle:
                    location = vehicle.get_location()
                    velocity = vehicle.get_velocity()
                    speed = 3.6 * (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5  # 转换为km/h
                    print(
                        f"📍 车辆位置: x={location.x:.1f}, y={location.y:.1f}, z={location.z:.1f}, 速度: {speed:.1f} km/h")

            time.sleep(0.05)  # 控制循环频率

    except KeyboardInterrupt:
        print("\n\n🛑 程序被用户停止")
    finally:
        # 7. 清理资源（销毁车辆和传感器，避免 Carla 服务器残留）
        print("\n🧹 开始清理资源...")

        if camera_sensor:
            camera_sensor.destroy()
            print("✅ 摄像头已销毁")
        if lidar_sensor:
            lidar_sensor.stop()
            lidar_sensor.destroy()
            print("✅ 激光雷达已销毁")
        if vehicle:
            vehicle.destroy()
            print("✅ 车辆已销毁")
        print("🧹 资源清理完成！")


# -------------------------- 4. 运行主函数 --------------------------
if __name__ == "__main__":
    main()