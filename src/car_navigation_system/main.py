import carla
import time
import numpy as np
import cv2

# --------------------------
# 1. 初始化CARLA连接和环境
# --------------------------
client = carla.Client('localhost', 2000)  # 连接CARLA服务器
client.set_timeout(10.0)
world = client.load_world('Town01')  # 加载简单地图
settings = world.get_settings()
settings.synchronous_mode = True  # 同步模式，便于控制
world.apply_settings(settings)

# 获取地图 spawn 点
spawn_points = world.get_map().get_spawn_points()
if not spawn_points:
    raise Exception("No spawn points available")
spawn_point = spawn_points[0]  # 选择第一个 spawn 点

# --------------------------
# 2. 生成机器人载体（车辆）
# --------------------------
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')  # 选择特斯拉模型作为机器人载体
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(False)  # 关闭自动驾驶，手动控制

# --------------------------
# 3. 配置多模态传感器
# --------------------------
# 3.1 摄像头（视觉模态）
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')
camera_bp.set_attribute('fov', '90')
# 摄像头安装位置（车辆前方）
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# 3.2 激光雷达（LiDAR，距离感知模态）
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('channels', '32')  # 32线激光雷达
lidar_bp.set_attribute('range', '50')  # 最大探测距离50米
lidar_bp.set_attribute('points_per_second', '100000')
# LiDAR安装位置（车辆顶部）
lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

# --------------------------
# 4. 传感器数据处理回调
# --------------------------
# 存储传感器数据的变量
camera_image = None
lidar_data = None


# 摄像头数据回调（保存RGB图像）
def camera_callback(image):
    global camera_image
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # RGBA格式
    camera_image = array[:, :, :3]  # 转为RGB


# LiDAR数据回调（检测前方障碍物）
def lidar_callback(point_cloud):
    global lidar_data
    # 将点云数据转为numpy数组（x, y, z, 反射率）
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    lidar_data = data


# 绑定回调函数
camera.listen(lambda data: camera_callback(data))
lidar.listen(lambda data: lidar_callback(data))


# --------------------------
# 5. 多模态导航控制逻辑
# --------------------------
def avoid_obstacle():
    """基于LiDAR数据判断是否需要避障"""
    if lidar_data is None:
        return False, 0.0  # 无数据时不避障

    # 筛选车辆前方±30度范围内的点云（x正方向为前方）
    front_points = lidar_data[
        (lidar_data[:, 1] > -5) &  # y > -5（左侧边界）
        (lidar_data[:, 1] < 5) &  # y < 5（右侧边界）
        (lidar_data[:, 0] > 0)  # x > 0（前方）
        ]

    if len(front_points) == 0:
        return False, 0.0  # 无障碍物

    # 计算前方最近障碍物距离
    min_distance = np.min(front_points[:, 0])  # x坐标即距离
    return min_distance < 10.0, min_distance  # 距离小于10米时需要避障


# 控制参数
throttle = 0.4  # 油门
steer = 0.0  # 转向角

try:
    print("多模态导航系统启动，按Ctrl+C停止...")
    while True:
        # 刷新世界状态
        world.tick()

        # 检查障碍物
        need_avoid, distance = avoid_obstacle()

        # 控制逻辑
        if need_avoid:
            print(f"前方{distance:.2f}米检测到障碍物，开始避障...")
            steer = 0.5  # 向右转向避障
        else:
            steer = 0.0  # 直行

        # 控制车辆
        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=0.0
        ))

        # 显示摄像头画面（多模态可视化）
        if camera_image is not None:
            cv2.imshow('Camera View', camera_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("系统已停止")

finally:
    # 清理资源
    camera.stop()
    lidar.stop()
    vehicle.destroy()
    camera.destroy()
    lidar.destroy()
    world.apply_settings(settings)  # 恢复世界设置
    cv2.destroyAllWindows()
