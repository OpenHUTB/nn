import carla
import time
import numpy as np
import cv2
import math

# --------------------------
# 1. 初始化CARLA连接和环境
# --------------------------
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town01')
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 固定时间步长
world.apply_settings(settings)

# 设置天气
weather = carla.WeatherParameters(
    cloudiness=30.0,
    precipitation=0.0,
    sun_altitude_angle=70.0
)
world.set_weather(weather)

# 获取出生点
spawn_points = world.get_map().get_spawn_points()
if not spawn_points:
    raise Exception("No spawn points available")
spawn_point = spawn_points[0]

# --------------------------
# 2. 生成车辆和障碍物
# --------------------------
blueprint_library = world.get_blueprint_library()

# 主车辆（红色）
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
vehicle_bp.set_attribute('color', '255,0,0')  # 红色主车辆
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(False)

# 生成随机障碍物车辆
for i in range(6):
    if i >= len(spawn_points):
        break
    other_vehicles = blueprint_library.filter('vehicle.*')
    other_vehicle_bp = np.random.choice(other_vehicles)
    spawn_idx = (i + 8) % len(spawn_points)  # 分散的出生点
    other_vehicle = world.try_spawn_actor(other_vehicle_bp, spawn_points[spawn_idx])
    if other_vehicle:
        other_vehicle.set_autopilot(True)

# --------------------------
# 3. 配置传感器（含第三视角摄像头）
# --------------------------
# 3.1 前视摄像头（第一视角）
front_camera_bp = blueprint_library.find('sensor.camera.rgb')
front_camera_bp.set_attribute('image_size_x', '640')
front_camera_bp.set_attribute('image_size_y', '480')
front_camera_bp.set_attribute('fov', '90')
front_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
front_camera = world.spawn_actor(front_camera_bp, front_camera_transform, attach_to=vehicle)

# 3.2 第三视角摄像头（跟随车辆）
third_camera_bp = blueprint_library.find('sensor.camera.rgb')
third_camera_bp.set_attribute('image_size_x', '640')
third_camera_bp.set_attribute('image_size_y', '480')
third_camera_bp.set_attribute('fov', '110')  # 更宽的视角
# 安装在车辆后方上方，提供良好的第三视角
third_camera_transform = carla.Transform(
    carla.Location(x=-5.0, y=0.0, z=3.0),  # 位置：车后5米，高3米
    carla.Rotation(pitch=-15.0)  # 角度：向下倾斜15度
)
third_camera = world.spawn_actor(third_camera_bp, third_camera_transform, attach_to=vehicle)

# 3.3 激光雷达
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('channels', '32')
lidar_bp.set_attribute('range', '50')
lidar_bp.set_attribute('points_per_second', '100000')
lidar_bp.set_attribute('rotation_frequency', '10')
lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

# --------------------------
# 4. 传感器数据处理
# --------------------------
# 存储传感器数据
front_image = None
third_image = None
lidar_data = None
lidar_img = None


def front_camera_callback(image):
    global front_image
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    front_image = array[:, :, :3]


def third_camera_callback(image):
    global third_image
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    third_image = array[:, :, :3]


def lidar_callback(point_cloud):
    global lidar_data, lidar_img
    # 处理点云数据
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    lidar_data = data

    # 生成LiDAR可视化图像（鸟瞰图）
    lidar_img = np.zeros((400, 400, 3), dtype=np.uint8)
    center_x, center_y = 200, 200
    scale = 8  # 缩放因子

    for point in data:
        x, y = point[0], point[1]
        if 0 < x < 50 and -25 < y < 25:  # 扩大检测范围
            px = int(center_x - y * scale)
            py = int(center_y - x * scale)
            if 0 <= px < 400 and 0 <= py < 400:
                # 距离越近颜色越红
                color_intensity = min(1.0, x / 50.0)
                color = (
                    int(255 * (1 - color_intensity)),
                    int(255 * color_intensity),
                    0
                )
                cv2.circle(lidar_img, (px, py), 1, color, -1)

    # 绘制车辆位置和方向
    cv2.circle(lidar_img, (center_x, center_y), 5, (0, 0, 255), -1)
    cv2.line(lidar_img, (center_x, center_y), (center_x, center_y - 30), (0, 0, 255), 2)


# 绑定回调函数
front_camera.listen(lambda data: front_camera_callback(data))
third_camera.listen(lambda data: third_camera_callback(data))
lidar.listen(lambda data: lidar_callback(data))


# --------------------------
# 5. 优化的避障控制逻辑
# --------------------------
def detect_obstacles():
    """检测障碍物并分析左右安全区域"""
    if lidar_data is None:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0

    # 分区域检测：近距离(0-10m)、中距离(10-20m)、远距离(20-30m)
    # 近距离检测范围更窄，中远距离更宽，符合驾驶习惯
    near_points = lidar_data[
        (lidar_data[:, 0] > 0) & (lidar_data[:, 0] < 10) &  # 近距离
        (lidar_data[:, 1] > -3) & (lidar_data[:, 1] < 3)  # 窄范围
        ]

    mid_points = lidar_data[
        (lidar_data[:, 0] >= 10) & (lidar_data[:, 0] < 20) &  # 中距离
        (lidar_data[:, 1] > -6) & (lidar_data[:, 1] < 6)  # 中等范围
        ]

    far_points = lidar_data[
        (lidar_data[:, 0] >= 20) & (lidar_data[:, 0] < 30) &  # 远距离
        (lidar_data[:, 1] > -10) & (lidar_data[:, 1] < 10)  # 宽范围
        ]

    # 合并所有检测点
    front_points = np.vstack((near_points, mid_points, far_points)) if len(near_points) > 0 or len(
        mid_points) > 0 or len(far_points) > 0 else np.array([])

    if len(front_points) == 0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0

    # 计算最近障碍物距离
    min_distance = np.min(front_points[:, 0])

    # 区分左右障碍物
    left_points = front_points[front_points[:, 1] < 0]
    right_points = front_points[front_points[:, 1] > 0]

    # 计算左右最近距离
    left_min = np.min(left_points[:, 0]) if len(left_points) > 0 else float('inf')
    right_min = np.min(right_points[:, 0]) if len(right_points) > 0 else float('inf')

    # 计算左右安全区域大小（障碍物较少的区域）
    left_free = np.sum(left_points[:, 0] > 15) if len(left_points) > 0 else 1000
    right_free = np.sum(right_points[:, 0] > 15) if len(right_points) > 0 else 1000

    return min_distance < 20.0, min_distance, left_min, right_min, left_free, right_free


# 控制状态变量
throttle = 0.5
steer = 0.0
avoid_state = 0  # 0: 正常, 1: 右避障, 2: 左避障, 3: 回正
avoid_timer = 0
recovery_timer = 0  # 避障后回正计时器

try:
    print("多模态导航系统启动（带第三视角）")
    print("控制键: q-退出, w-加速, s-减速, a-左转向, d-右转向, r-重置方向")

    while True:
        world.tick()

        # 检测障碍物
        need_avoid, min_dist, left_min, right_min, left_free, right_free = detect_obstacles()

        # 智能避障逻辑
        if need_avoid:
            print(f"避障中 - 前方:{min_dist:.1f}m, 左:{left_min:.1f}m, 右:{right_min:.1f}m")

            if recovery_timer > 0:
                # 处于回正阶段，继续回正
                recovery_timer -= 1
                steer = steer * 0.8  # 逐渐回正
                if recovery_timer == 0:
                    avoid_state = 0

            elif avoid_timer <= 0:
                # 选择避障方向：综合考虑距离和安全区域
                # 右侧更安全的条件：右侧距离更远 或 右侧安全区域更大
                if (right_min > left_min + 2) or (right_free > left_free + 50):
                    avoid_state = 1
                    steer = 0.4  # 右转幅度
                    avoid_timer = 40  # 避障持续时间
                else:
                    avoid_state = 2
                    steer = -0.4  # 左转幅度
                    avoid_timer = 40
            else:
                avoid_timer -= 1
                # 避障后期开始准备回正
                if avoid_timer < 15:
                    steer *= 0.95

                # 避障结束后进入回正阶段
                if avoid_timer == 0:
                    recovery_timer = 20  # 回正持续时间

        else:
            # 无障碍物时保持直行或回正
            if abs(steer) > 0.1:
                steer *= 0.85  # 平滑回正
            else:
                steer = 0.0
            avoid_state = 0
            avoid_timer = 0
            recovery_timer = 0

        # 动态调整油门：根据障碍物距离
        if need_avoid:
            if min_dist < 8.0:
                throttle = 0.15  # 很近时减速
            elif min_dist < 15.0:
                throttle = 0.25  # 较近时减速
            else:
                throttle = 0.4  # 保持一定速度
        else:
            throttle = 0.4  # 正常速度

        # 应用控制
        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=0.0,
            hand_brake=False
        ))

        # 可视化显示
        if front_image is not None and third_image is not None and lidar_img is not None:
            # 前视摄像头添加信息
            front_display = front_image.copy()
            cv2.putText(front_display, f"距离: {min_dist:.1f}m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(front_display, f"转向: {steer:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if need_avoid:
                cv2.putText(front_display, "避障中", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 第三视角添加信息
            third_display = third_image.copy()
            cv2.putText(third_display, "第三视角", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 显示所有窗口
            cv2.imshow('前视摄像头', front_display)
            cv2.imshow('第三视角', third_display)
            cv2.imshow('LiDAR鸟瞰图', lidar_img)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                throttle = min(1.0, throttle + 0.1)
            elif key == ord('s'):
                throttle = max(0.0, throttle - 0.1)
            elif key == ord('a'):
                steer = max(-1.0, steer - 0.1)
            elif key == ord('d'):
                steer = min(1.0, steer + 0.1)
            elif key == ord('r'):
                steer = 0.0  # 重置转向

        time.sleep(0.01)

except KeyboardInterrupt:
    print("系统已停止")

finally:
    # 清理资源
    front_camera.stop()
    third_camera.stop()
    lidar.stop()

    # 销毁所有车辆
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()

    # 销毁所有传感器
    for actor in world.get_actors().filter('sensor.*'):
        actor.destroy()

    # 恢复世界设置
    settings.synchronous_mode = False
    world.apply_settings(settings)
    cv2.destroyAllWindows()