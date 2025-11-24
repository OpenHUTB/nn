# --------------------------
# 1. 初始化CARLA连接和环境
# --------------------------
# 导入必要的库
import carla
import time
import numpy as np
import cv2
import math
from collections import deque

# 连接到本地CARLA服务器，端口2000
client = carla.Client('localhost', 2000)
# 设置超时时间为15秒
client.set_timeout(15.0)
# 加载名为'Town01'的地图
world = client.load_world('Town01')

# 获取并设置世界的运行参数
settings = world.get_settings()
# 启用同步模式，这意味着仿真将等待客户端的每一个tick信号
settings.synchronous_mode = True
# 设置固定的时间步长，单位为秒
settings.fixed_delta_seconds = 0.1
# 启用子步进，用于更精确的物理模拟
settings.substepping = True
# 设置子步进的最大时间步长
settings.max_substep_delta_time = 0.01
# 设置最大子步进次数
settings.max_substeps = 10
# 应用这些设置
world.apply_settings(settings)

# 定义天气参数
weather = carla.WeatherParameters(
    cloudiness=30.0,      # 云量
    precipitation=0.0,    # 降雨量
    sun_altitude_angle=70.0 # 太阳高度角
)
# 应用天气设置
world.set_weather(weather)

# 获取地图对象
map = world.get_map()
# 获取地图中所有可用的出生点
spawn_points = map.get_spawn_points()
if not spawn_points:
    raise Exception("No spawn points available") # 如果没有找到出生点则抛出异常

# 选择一个更好的出生点（索引为10的点）
spawn_point = spawn_points[10]  # 选择更靠前的出生点

# --------------------------
# 2. 生成车辆和障碍物
# --------------------------
# 获取蓝图库，包含所有可生成的actor的蓝图
blueprint_library = world.get_blueprint_library()

# 主车辆（红色特斯拉Model3）
# 查找特斯拉Model3的车辆蓝图
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
# 设置车辆颜色为红色 (RGB)
vehicle_bp.set_attribute('color', '255,0,0')
# 在指定的出生点生成主车辆
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    raise Exception("无法生成主车辆")
# 禁用车辆的自动驾驶模式
vehicle.set_autopilot(False)
# 确保车辆的物理模拟是开启的
vehicle.set_simulate_physics(True)

print(f"车辆生成在位置: {spawn_point.location}")

# 生成障碍物车辆
obstacle_count = 3 # 计划生成3个障碍物
for i in range(obstacle_count):
    if i >= len(spawn_points):
        break
    # 过滤出所有车辆类型的蓝图
    other_vehicles = blueprint_library.filter('vehicle.*')
    # 随机选择一个车辆蓝图
    other_vehicle_bp = np.random.choice(other_vehicles)
    # 选择一个与主车辆不同的出生点
    spawn_idx = (i + 15) % len(spawn_points)
    # 尝试在指定位置生成障碍物车辆
    other_vehicle = world.try_spawn_actor(other_vehicle_bp, spawn_points[spawn_idx])
    if other_vehicle:
        # 为障碍物车辆启用自动驾驶
        other_vehicle.set_autopilot(True)

# --------------------------
# 3. 配置传感器
# --------------------------
# 配置第三视角RGB相机
# 查找RGB相机的传感器蓝图
third_camera_bp = blueprint_library.find('sensor.camera.rgb')
# 设置相机图像宽度
third_camera_bp.set_attribute('image_size_x', '640')
# 设置相机图像高度
third_camera_bp.set_attribute('image_size_y', '480')
# 设置相机视场角
third_camera_bp.set_attribute('fov', '110')
# 定义相机相对于车辆的变换（位置和旋转）
third_camera_transform = carla.Transform(
    carla.Location(x=-5.0, y=0.0, z=3.0), # 在车后5米，高3米处
    carla.Rotation(pitch=-15.0)           # 向下倾斜15度
)
# 生成相机传感器并将其附加到主车辆上
third_camera = world.spawn_actor(third_camera_bp, third_camera_transform, attach_to=vehicle)

# 配置激光雷达 (LiDAR)
# 查找激光雷达的传感器蓝图
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
# 设置激光雷达的通道数（线数）
lidar_bp.set_attribute('channels', '32')
# 设置激光雷达的最大探测范围
lidar_bp.set_attribute('range', '50')
# 设置激光雷达的每秒点数
lidar_bp.set_attribute('points_per_second', '100000')
# 设置激光雷达的旋转频率
lidar_bp.set_attribute('rotation_frequency', '10')
# 设置激光雷达的上视场角
lidar_bp.set_attribute('upper_fov', '15')
# 设置激光雷达的下视场角
lidar_bp.set_attribute('lower_fov', '-25')
# 定义激光雷达相对于车辆的变换
lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5)) # 在车顶中心，高2.5米处
# 生成激光雷达传感器并将其附加到主车辆上
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

# --------------------------
# 4. 传感器数据处理
# --------------------------
# 全局变量，用于存储来自传感器的最新数据
third_image = None # 存储相机图像
lidar_data = None  # 存储激光雷达点云

# 相机数据回调函数
def third_camera_callback(image):
    # 使用global关键字修改全局变量
    global third_image
    # 将原始图像数据（CARLA的Image对象）转换为numpy数组
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    # 重塑数组形状以匹配图像的尺寸 (高度, 宽度, 4通道RGBA)
    array = np.reshape(array, (image.height, image.width, 4))
    # 提取前3个通道（RGB）并存储
    third_image = array[:, :, :3]

# 激光雷达数据回调函数
def lidar_callback(point_cloud):
    # 使用global关键字修改全局变量
    global lidar_data
    # 将原始点云数据转换为numpy数组，每个点由4个32位浮点数表示 (x, y, z, intensity)
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    # 重塑数组形状，每行代表一个点 (x, y, z, intensity)
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    # 存储处理后的点云数据
    lidar_data = data

# 开始监听传感器数据，设置回调函数
third_camera.listen(third_camera_callback)
lidar.listen(lidar_callback)

# 等待2秒，确保传感器有足够的时间进行初始化并返回第一帧数据
time.sleep(2.0)  # 增加等待时间确保传感器初始化

# --------------------------
# 5. 路径规划与导航逻辑
# --------------------------
def get_next_waypoint(vehicle_location, distance=8.0):
    """
    获取车辆前方指定距离的路点
    :param vehicle_location: 车辆当前的位置
    :param distance: 希望获取的下一个路点距离当前车辆的距离
    :return: 下一个目标路点 (carla.Waypoint)
    """
    # 获取车辆当前位置所在的路点，并将位置投影到道路中心线上
    waypoint = map.get_waypoint(vehicle_location, project_to_road=True)

    # 尝试获取当前路点前方distance米处的路点
    next_waypoints = waypoint.next(distance)
    if next_waypoints:
        return next_waypoints[0] # 返回第一个找到的路点

    # 如果当前路点在路口(junction)，进行特殊处理
    if waypoint.is_junction:
        # 尝试在同一条道路上寻找下一个路点
        for wp in waypoint.next(distance):
            if wp.road_id == waypoint.road_id:
                return wp

        # 如果找不到同一条路的路点，则尝试变道
        # 检查是否可以向右变道
        if waypoint.lane_change & carla.LaneChange.Right:
            right_way = waypoint.get_right_lane()
            if right_way:
                return right_way.next(distance)[0]
        # 检查是否可以向左变道
        elif waypoint.lane_change & carla.LaneChange.Left:
            left_way = waypoint.get_left_lane()
            if left_way:
                return left_way.next(distance)[0]

    # 如果以上方法都失败，返回当前路点
    return waypoint

def calculate_steering_angle(vehicle_transform, target_waypoint):
    """
    计算到达目标路点所需的转向角
    :param vehicle_transform: 车辆当前的变换（位置和旋转）
    :param target_waypoint: 目标路点
    :return: 归一化的转向指令 (-1.0 到 1.0 之间)
    """
    # 获取车辆当前位置和目标路点位置
    vehicle_location = vehicle_transform.location
    target_location = target_waypoint.transform.location

    # 将车辆的偏航角（yaw）从度转换为弧度
    vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

    # 计算目标位置相对于车辆位置的向量 (dx, dy)
    dx = target_location.x - vehicle_location.x
    dy = target_location.y - vehicle_location.y

    # 将目标位置向量从世界坐标系转换到车辆局部坐标系
    # 这使得我们可以更容易地判断目标在车辆的左、右还是正前方
    local_x = dx * math.cos(vehicle_yaw) + dy * math.sin(vehicle_yaw)
    local_y = -dx * math.sin(vehicle_yaw) + dy * math.cos(vehicle_yaw)

    # 如果目标在车辆正后方（或非常接近），返回0转向角以避免除以零
    if abs(local_x) < 0.1:
        return 0.0

    # 计算目标方向与车辆当前朝向的夹角
    angle = math.atan2(local_y, local_x)

    # 定义最大转向角（例如60度），并将计算出的角度归一化到[-1, 1]范围
    max_angle = math.radians(60)
    steering = angle / max_angle

    # 使用np.clip确保转向值在有效范围内
    return np.clip(steering, -1.0, 1.0)

# --------------------------
# 6. 优化避障控制逻辑
# --------------------------
class ObstacleAvoidance:
    """
    障碍物检测与避障决策类
    封装了使用LiDAR数据进行障碍物检测和生成避障指令的逻辑
    """
    def __init__(self):
        """初始化避障控制器"""
        # 用于存储障碍物检测历史，防止频繁切换决策
        self.obstacle_history = deque(maxlen=10)
        # 紧急制动标志
        self.emergency_brake = False
        # 记录上一次的避障方向，用于决策的连续性
        self.last_avoid_direction = 0 # -1: 左, 0: 无, 1: 右

    def detect_obstacles(self, lidar_data, vehicle_speed):
        """
        修复版的障碍物检测算法
        处理LiDAR点云数据，检测前方障碍物并分析其位置
        :param lidar_data: 原始的LiDAR点云数据 (numpy array)
        :param vehicle_speed: 车辆当前速度 (m/s)
        :return: 包含障碍物检测结果的字典
        """
        # 如果没有LiDAR数据，返回默认结果（无障碍物）
        if lidar_data is None:
            return self._get_default_detection_result()

        # 如果点云为空，返回默认结果
        if len(lidar_data) == 0:
            return self._get_default_detection_result()

        try:
            # 1. 地面过滤：移除高度低于阈值的点，减少地面点的干扰
            ground_threshold = -0.5 # 地面阈值，单位为米
            valid_mask = lidar_data[:, 2] > ground_threshold # z坐标大于阈值的点被认为是有效的
            valid_points = lidar_data[valid_mask]

            if len(valid_points) == 0:
                return self._get_default_detection_result()

            # 2. 计算每个有效点相对于车辆的距离和角度
            # 计算x-y平面上的距离
            distances = np.sqrt(valid_points[:, 0] ** 2 + valid_points[:, 1] ** 2)
            # 计算与车辆x轴正方向的夹角（弧度）
            angles = np.arctan2(valid_points[:, 1], valid_points[:, 0])

            # 3. 定义感兴趣区域（ROI）：只关注车辆正前方的区域
            front_angle_range = np.radians(75) # 前方75度的范围
            # 过滤出在角度范围内且距离大于1米（避免过近的噪声点）的点
            front_mask = (np.abs(angles) <= front_angle_range) & (distances > 1.0)

            front_points = valid_points[front_mask]
            front_distances = distances[front_mask]
            front_angles = angles[front_mask]

            if len(front_points) == 0:
                return self._get_default_detection_result()

            # 4. 分区域检测：将前方区域分为近、中、远三个区域
            near_zone = front_distances < 8.0    # 近区：0-8米
            mid_zone = (front_distances >= 8.0) & (front_distances < 20.0) # 中区：8-20米
            far_zone = (front_distances >= 20.0) & (front_distances < 35.0) # 远区：20-35米

            # 5. 紧急制动检测：检测近区中非常接近的障碍物
            emergency_points = front_points[near_zone & (front_distances < 4.0)] # 4米内的点
            self.emergency_brake = len(emergency_points) > 10 # 如果超过10个点，则触发紧急制动

            # 6. 计算最近障碍物的距离和角度
            min_distance = np.min(front_distances)
            min_idx = np.argmin(front_distances)
            obstacle_angle = front_angles[min_idx] if len(front_angles) > min_idx else 0.0

            # 7. 分左右区域分析障碍物分布
            left_points_distances = front_distances[front_angles > 0] # 车辆左侧的点（角度为正）
            right_points_distances = front_distances[front_angles < 0] # 车辆右侧的点（角度为负）

            # 计算左右两侧最近障碍物的距离
            left_min = np.min(left_points_distances) if len(left_points_distances) > 0 else float('inf')
            right_min = np.min(right_points_distances) if len(right_points_distances) > 0 else float('inf')

            # 8. 计算左右两侧的自由空间（无障碍物的区域大小）
            safe_threshold = 15.0 # 认为15米外是安全的
            # 统计左右两侧距离大于安全阈值的点的数量，作为自由空间的度量
            left_free = np.sum(left_points_distances > safe_threshold) if len(left_points_distances) > 0 else 1000
            right_free = np.sum(right_points_distances > safe_threshold) if len(right_points_distances) > 0 else 1000

            # 9. 判断是否检测到障碍物的综合条件
            obstacle_detected = (np.sum(near_zone) > 5 or  # 近区有超过5个点
                                 np.sum(mid_zone) > 10 or # 中区有超过10个点
                                 min_distance < 12.0)     # 最近障碍物距离小于12米

            # 返回包含所有检测信息的字典
            return {
                'obstacle_detected': obstacle_detected,
                'min_distance': min_distance,
                'left_clearance': left_min,
                'right_clearance': right_min,
                'left_free_space': left_free,
                'right_free_space': right_free,
                'obstacle_angle': obstacle_angle,
                'emergency_brake': self.emergency_brake
            }

        except Exception as e:
            # 如果检测过程中发生错误，打印错误信息并返回默认结果
            print(f"障碍物检测错误: {e}")
            return self._get_default_detection_result()

    def _get_default_detection_result(self):
        """
        返回默认的检测结果（无障碍物）
        这是一个辅助函数，用于在没有数据或出错时提供一个安全的默认值
        :return: 默认的检测结果字典
        """
        return {
            'obstacle_detected': False,
            'min_distance': 30.0,
            'left_clearance': float('inf'),
            'right_clearance': float('inf'),
            'left_free_space': 1000,
            'right_free_space': 1000,
            'obstacle_angle': 0.0,
            'emergency_brake': False
        }

    def decide_avoidance_direction(self, detection_result, current_steer, vehicle_speed):
        """
        避障决策逻辑
        根据障碍物检测结果，决定车辆应采取的避障动作（转向和刹车）
        :param detection_result: 障碍物检测结果字典
        :param current_steer: 当前的转向值
        :param vehicle_speed: 当前车辆速度
        :return: (避障转向值, 避障刹车值, 是否紧急制动)
        """
        # 如果没有检测到障碍物，重置状态并返回无动作
        if not detection_result['obstacle_detected']:
            self.last_avoid_direction = 0
            return 0, 0, False

        # 从检测结果中提取关键信息
        min_dist = detection_result['min_distance']
        left_clear = detection_result['left_clearance']
        right_clear = detection_result['right_clearance']
        left_free = detection_result['left_free_space']
        right_free = detection_result['right_free_space']
        obstacle_angle = detection_result['obstacle_angle']

        # 如果需要紧急制动，返回最大刹车力度
        if detection_result['emergency_brake']:
            return 0, 1.0, True # (转向, 刹车, 紧急制动标志)

        # 1. 动态安全距离：速度越快，需要的安全距离越大
        safety_margin = max(3.0, vehicle_speed * 0.5)

        # 2. 计算左右两侧的安全得分，用于决策避障方向
        # 得分越高，表示该方向越安全
        left_score = (left_clear - safety_margin) + (left_free * 0.1)
        right_score = (right_clear - safety_margin) + (right_free * 0.1)

        # 3. 加入历史决策的权重，防止决策频繁跳动（ hysteresis 机制）
        if self.last_avoid_direction != 0:
            if self.last_avoid_direction == 1: # 上一次是向右避障
                right_score += 2.0 # 给右侧加分
            else: # 上一次是向左避障
                left_score += 2.0 # 给左侧加分

        # 4. 初始化避障控制量
        avoid_steer = 0.0 # 避障所需的转向
        avoid_brake = 0.0 # 避障所需的刹车

        # 5. 根据与障碍物的距离决定是否需要刹车
        if min_dist < safety_margin + 2.0:
            avoid_brake = 0.3 + (safety_margin - min_dist) * 0.1 # 距离越近，刹车力度越大

        # 6. 根据安全得分决定避障方向
        if right_score > left_score + 1.0: # 右侧明显更安全
            avoid_steer = -0.5 # 向右转向（CARLA中，负转向值为右转）
            self.last_avoid_direction = 1 # 记录这次是向右避障
        elif left_score > right_score + 1.0: # 左侧明显更安全
            avoid_steer = 0.5 # 向左转向（正转向值为左转）
            self.last_avoid_direction = -1 # 记录这次是向左避障
        else: # 两侧安全性相近，根据障碍物位置决定
            if obstacle_angle > 0: # 障碍物在左侧
                avoid_steer = -0.4 # 向右避让
                self.last_avoid_direction = 1
            else: # 障碍物在右侧或正前方
                avoid_steer = 0.4 # 向左避让
                self.last_avoid_direction = -1

        # 返回最终的避障决策
        return avoid_steer, avoid_brake, False

# --------------------------
# 7. 主控制循环
# --------------------------
# 初始化避障控制器实例
obstacle_avoidance = ObstacleAvoidance()

# 初始化控制状态变量
throttle = 1.0  # 油门值 (0.0 到 1.0)，初始设为最大
steer = 0.0     # 转向值 (-1.0 到 1.0)
brake = 0.0     # 刹车值 (0.0 到 1.0)
waypoint_distance = 8.0 # 目标路点距离

# 获取车辆初始位置的下一个路点
vehicle_location = vehicle.get_location()
waypoint = get_next_waypoint(vehicle_location, waypoint_distance)

# 使用滑动窗口对控制信号进行平滑滤波，减少控制抖动
steer_filter = deque(maxlen=3) # 存储最近3次的转向值
throttle_filter = deque(maxlen=2) # 存储最近2次的油门值

print("初始化车辆状态...")
# 再次确保车辆物理引擎开启
vehicle.set_simulate_physics(True)

# 应用一个初始的强力启动控制，让车辆动起来
print("应用强力启动控制...")
vehicle.apply_control(carla.VehicleControl(
    throttle=1.0,  # 最大油门
    steer=0.0,
    brake=0.0,
    hand_brake=False
))

try:
    print("自动驾驶系统启动（强力油门版本）")
    print("控制键: q-退出, w-加速, s-减速, a-左转向, d-右转向, r-重置方向, 空格-紧急制动")

    frame_count = 0 # 帧计数器
    stuck_count = 0 # 车辆卡住状态计数器
    last_position = vehicle.get_location() # 记录上一帧的位置，用于检测是否卡住

    # 主循环，持续运行直到用户退出
    while True:
        # 触发仿真世界进行一次步进
        world.tick()
        frame_count += 1

        # 获取车辆当前的状态信息
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle.get_location()
        vehicle_velocity = vehicle.get_velocity()
        # 计算车辆当前的速度 (m/s)
        vehicle_speed = math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2)

        # 每帧打印车辆状态信息
        print(
            f"帧 {frame_count}: 速度={vehicle_speed * 3.6:.1f}km/h, 位置=({vehicle_location.x:.1f}, {vehicle_location.y:.1f})")

        # 检测车辆是否卡住
        current_position = vehicle_location
        distance_moved = current_position.distance(last_position) # 计算与上一帧位置的距离
        if distance_moved < 0.1: # 如果移动距离小于0.1米，认为车辆卡住了
            stuck_count += 1
        else:
            stuck_count = 0 # 否则重置卡住计数器

        last_position = current_position # 更新上一帧位置

        # 如果车辆卡住超过10帧，尝试强力脱困
        if stuck_count > 10:
            print("车辆卡住，尝试强力脱困...")
            # 先短暂倒车
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=False,
                reverse=True # 启用倒车
            ))
            time.sleep(0.5) # 倒车0.5秒
            # 然后向前猛冲
            vehicle.apply_control(carla.VehicleControl(
                throttle=1.0,
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False # 关闭倒车
            ))
            stuck_count = 0 # 重置卡住计数器

        # 更新目标路点：如果车辆接近当前路点，则获取下一个
        current_distance = vehicle_location.distance(waypoint.transform.location)
        if current_distance < 4.0: # 如果距离目标路点小于4米
            waypoint = get_next_waypoint(vehicle_location, waypoint_distance) # 获取新的目标路点

        # 计算基础转向角：基于当前路点的导航转向
        base_steer = calculate_steering_angle(vehicle_transform, waypoint)

        # 使用LiDAR数据检测障碍物并生成避障指令
        detection_result = obstacle_avoidance.detect_obstacles(lidar_data, vehicle_speed)
        avoid_steer, avoid_brake, emergency_brake = obstacle_avoidance.decide_avoidance_direction(
            detection_result, steer, vehicle_speed
        )

        # 综合控制输出 - 简化逻辑，专注于让车动起来
        if emergency_brake:
            # 紧急制动状态
            throttle = 0.2
            brake = 0.1
            steer = base_steer * 0.1
            print("!!! 紧急制动 !!!")
        elif detection_result['obstacle_detected']:
            # 检测到障碍物，应用避障控制
            brake = avoid_brake
            # 动态调整油门：距离越近，油门越小，以获得更好的操控性
            obstacle_distance_factor = max(0.2, min(1.0, detection_result['min_distance'] / 15.0))
            throttle = 0.3 * obstacle_distance_factor  # 基础油门0.3，并根据距离动态调整
            steer = avoid_steer * 0.8 + base_steer * 0.2
            print(f"避障中 - 距离:{detection_result['min_distance']:.1f}m")
        else:
            # 正常行驶状态，无障碍物
            brake = 0.0
            steer = base_steer * 0.5

            # 强力油门策略：根据当前速度动态调整油门大小
            if vehicle_speed < 2.0:  # 低速时，使用较大油门加速
                throttle = 0.4
            elif vehicle_speed < 5.0: # 中低速时，适当减小油门
                throttle = 0.3
            elif vehicle_speed < 7.0: # 中高速时，进一步减小油门
                throttle = 0.2
            else: # 高速时，使用最小油门维持速度
                throttle = 0.1

            steer = base_steer # 使用基础转向角

        # 应用控制信号平滑滤波
        steer_filter.append(steer)
        throttle_filter.append(throttle)

        # 计算滤波后的控制信号
        smoothed_steer = np.mean(steer_filter)
        smoothed_throttle = np.mean(throttle_filter)

        # 构造最终的车辆控制命令
        control = carla.VehicleControl(
            throttle=smoothed_throttle,
            steer=smoothed_steer,
            brake=brake,
            hand_brake=False,
            reverse=False
        )

        # 打印即将应用的控制命令
        print(f"控制输出: 油门={control.throttle:.2f}, 刹车={control.brake:.2f}, 转向={control.steer:.2f}")
        # 将控制命令应用到车辆上
        vehicle.apply_control(control)

        # 可视化显示：如果相机图像可用
        if third_image is not None:
            # 创建图像副本，避免修改原始数据
            display_image = third_image.copy()
            # 在图像上绘制车辆速度信息
            cv2.putText(display_image, f"Speed: {vehicle_speed * 3.6:.1f} km/h", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # 在图像上绘制油门值
            cv2.putText(display_image, f"Throttle: {throttle:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # 在图像上绘制转向值
            cv2.putText(display_image, f"Steer: {steer:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 根据障碍物检测结果，在图像上绘制不同的状态文本
            if detection_result['obstacle_detected']:
                status_text = f"OBSTACLE: {detection_result['min_distance']:.1f}m"
                cv2.putText(display_image, status_text, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # 红色文本表示有障碍物
            else:
                cv2.putText(display_image, "CLEAR", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # 绿色文本表示无障碍物

            # 显示图像窗口
            cv2.imshow('第三视角 - 强力油门系统', display_image)

            # 监听键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break # 按下'q'键退出程序
            elif key == ord('w'):
                throttle = min(1.0, throttle + 0.1) # 按下'w'键增加油门
            elif key == ord('s'):
                throttle = max(0.0, throttle - 0.1) # 按下's'键减少油门
            elif key == ord('a'):
                steer = max(-1.0, steer - 0.1) # 按下'a'键向左转
            elif key == ord('d'):
                steer = min(1.0, steer + 0.1) # 按下'd'键向右转
            elif key == ord('r'):
                steer = 0.0 # 按下'r'键重置转向
            elif key == ord(' '):
                brake = 1.0 # 按下空格键紧急制动
                throttle = 0.0

        # 短暂休眠，降低CPU占用率（在同步模式下，这个sleep的影响不大，但仍是个好习惯）
        time.sleep(0.01)

except KeyboardInterrupt:
    # 如果用户按下Ctrl+C，捕获中断信号并打印信息
    print("系统已停止")
except Exception as e:
    # 如果程序运行中发生其他错误，打印错误信息和堆栈跟踪
    print(f"系统错误: {e}")
    import traceback
    traceback.print_exc()

finally:
    # 程序退出前的清理工作
    print("正在清理资源...")
    # 停止传感器数据监听
    third_camera.stop()
    lidar.stop()

    # 销毁所有生成的车辆和传感器actor
    for actor in world.get_actors():
        if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('sensor.'):
            actor.destroy()

    # 恢复世界设置为异步模式，方便下次运行
    settings.synchronous_mode = False
    world.apply_settings(settings)
    # 关闭所有OpenCV创建的窗口
    cv2.destroyAllWindows()
    print("资源清理完成")