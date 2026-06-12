import carla
import random
import time
import numpy as np
from collections import deque

# -------------------------- 配置参数 --------------------------
# Carla服务器地址和端口
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
# 车辆行驶速度（m/s）
TARGET_SPEED = 5.0
# 避障相关阈值
OBSTACLE_DISTANCE_THRESHOLD = 8.0  # 障碍物距离阈值（m）
OBSTACLE_ANGLE_THRESHOLD = 30.0    # 障碍物检测角度范围（度），左右各30度
BRAKE_INTENSITY = 1.0              # 制动强度（0-1）
STEER_ANGLE = 0.2                  # 避障转向角度（-1到1，左负右正）
# 激光雷达参数
LIDAR_RANGE = 50.0                 # 激光雷达最大探测距离（m）
LIDAR_CHANNELS = 32                # 激光雷达通道数
LIDAR_POINTS_PER_SECOND = 100000   # 每秒点云数量

# -------------------------- 全局变量 --------------------------
# 存储激光雷达点云数据
lidar_data = deque(maxlen=1)
# 车辆控制对象
vehicle_control = carla.VehicleControl()

def main():
    client = None
    world = None
    vehicle = None
    lidar_sensor = None

    try:
        # 1. 连接Carla服务器
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # 2. 设置地图和天气
        # 加载Town03地图（可替换为Town01/Town02等）
        world = client.load_world('Town03')
        weather = carla.WeatherParameters(
            sun_altitude_angle=70.0,
            fog_density=0.0
        )
        world.set_weather(weather)

        # 3. 生成自动驾驶车辆
        vehicle_bp = random.choice(blueprint_library.filter('model3'))  # 选择特斯拉Model3
        vehicle_bp.set_attribute('color', '255,0,0')  # 设置车辆颜色为红色
        # 获取随机生成点（也可以自定义固定点）
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"车辆生成成功：{vehicle.id}")

        # 4. 挂载激光雷达传感器
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        # 配置激光雷达参数
        lidar_bp.set_attribute('range', str(LIDAR_RANGE))
        lidar_bp.set_attribute('channels', str(LIDAR_CHANNELS))
        lidar_bp.set_attribute('points_per_second', str(LIDAR_POINTS_PER_SECOND))
        lidar_bp.set_attribute('rotation_frequency', '10')  # 刷新率10Hz
        lidar_bp.set_attribute('upper_fov', '10.0')       # 上视场角
        lidar_bp.set_attribute('lower_fov', '-30.0')      # 下视场角
        # 激光雷达安装位置（车辆顶部，稍微靠前）
        lidar_transform = carla.Transform(carla.Location(x=0.8, z=1.8))
        lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        # 注册激光雷达数据回调函数
        lidar_sensor.listen(lambda data: lidar_callback(data))
        print("激光雷达挂载成功")

        # 5. 主循环：感知+避障控制
        print("开始避障控制...")
        while True:
            # 获取车辆当前状态
            vehicle_velocity = vehicle.get_velocity()
            current_speed = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2)

            # 检测前方障碍物
            obstacle_detected = detect_obstacle()

            if obstacle_detected:
                # 避障逻辑：紧急制动 + 小幅度转向
                vehicle_control.brake = BRAKE_INTENSITY
                vehicle_control.throttle = 0.0
                vehicle_control.steer = STEER_ANGLE  # 向右转向（可根据需要调整为左）
                print(f"检测到障碍物！当前速度：{current_speed:.2f}m/s，执行制动转向")
            else:
                # 无障碍物：保持匀速直行
                if current_speed < TARGET_SPEED:
                    vehicle_control.throttle = 0.5
                else:
                    vehicle_control.throttle = 0.0
                vehicle_control.brake = 0.0
                vehicle_control.steer = 0.0  # 直行

            # 应用控制指令到车辆
            vehicle.apply_control(vehicle_control)
            time.sleep(0.05)  # 控制频率20Hz

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序出错：{e}")
    finally:
        # 清理资源
        if lidar_sensor:
            lidar_sensor.destroy()
            print("激光雷达已销毁")
        if vehicle:
            vehicle.destroy()
            print("车辆已销毁")
        print("资源清理完成")

def lidar_callback(data):
    """
    激光雷达数据回调函数：将点云数据存储到全局队列
    """
    # 将点云数据转换为numpy数组（x, y, z, intensity）
    points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
    lidar_data.append(points)

def detect_obstacle():
    """
    检测车辆前方是否有障碍物
    逻辑：
    1. 过滤激光雷达点云中的有效点（x>0，即车辆前方）
    2. 计算点到车辆的距离和角度
    3. 判断是否有在阈值范围内的点（距离<阈值，角度在左右阈值内）
    """
    if not lidar_data:
        return False

    # 获取最新的点云数据
    points = lidar_data[0]
    # 过滤掉车辆后方的点（x<=0）和太远的点（x>阈值）
    front_points = points[(points[:, 0] > 0) & (points[:, 0] < OBSTACLE_DISTANCE_THRESHOLD)]
    if len(front_points) == 0:
        return False

    # 计算每个点的水平角（相对于车辆前进方向，单位：度）
    # 车辆前进方向为x轴正方向，y轴正方向为左，负方向为右
    angles = np.degrees(np.arctan2(front_points[:, 1], front_points[:, 0]))
    # 过滤角度在[-阈值, +阈值]范围内的点（前方左右各OBSTACLE_ANGLE_THRESHOLD度）
    valid_angles = np.abs(angles) <= OBSTACLE_ANGLE_THRESHOLD
    valid_points = front_points[valid_angles]

    # 如果有有效点，说明存在障碍物
    return len(valid_points) > 0

if __name__ == '__main__':
    main()