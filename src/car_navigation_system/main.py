# --------------------------
# 1. 初始化CARLA连接和环境
# --------------------------
import carla
import time
import numpy as np
import cv2
import math
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os


# 修复1: 简化的神经网络架构（增加障碍物检测通道）
class SimpleDrivingNetwork(nn.Module):
    """
    简化的驾驶网络 - 增加障碍物感知
    """

    def __init__(self):
        super(SimpleDrivingNetwork, self).__init__()

        # 图像处理分支 (简化)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # 状态信息维度: 速度 + 转向历史 + 障碍物信息
        state_dim = 7  # 增加障碍物相关维度

        # 融合层
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 4 * 4 + state_dim, 128),  # 增加网络容量
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # [throttle, brake, steer]
        )

    def forward(self, image, state):
        # 处理图像
        visual_features = self.conv_layers(image)
        visual_features = visual_features.view(visual_features.size(0), -1)

        # 融合特征
        combined = torch.cat([visual_features, state], dim=1)

        # 输出控制
        control = self.fc_layers(combined)
        throttle_brake = torch.sigmoid(control[:, :2])
        steer = torch.tanh(control[:, 2:])

        return torch.cat([throttle_brake, steer], dim=1)


# 新增：障碍物检测器类
class ObstacleDetector:
    def __init__(self, world, vehicle, max_distance=50.0):
        self.world = world
        self.vehicle = vehicle
        self.max_distance = max_distance
        self.blueprint_library = world.get_blueprint_library()
        self.last_obstacle_info = {
            'has_obstacle': False,
            'distance': float('inf'),
            'relative_angle': 0.0,
            'obstacle_type': None
        }

    def get_obstacle_info(self):
        """检测前方障碍物信息"""
        try:
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation

            # 获取车辆前方的向量
            forward_vector = vehicle_transform.get_forward_vector()

            # 获取世界中的所有车辆（排除自身）
            all_vehicles = self.world.get_actors().filter('vehicle.*')

            min_distance = float('inf')
            closest_obstacle = None
            relative_angle = 0.0

            for other_vehicle in all_vehicles:
                if other_vehicle.id == self.vehicle.id:
                    continue

                other_location = other_vehicle.get_location()

                # 计算距离
                distance = vehicle_location.distance(other_location)

                if distance > self.max_distance:
                    continue

                # 计算相对位置向量
                relative_vector = carla.Location(
                    other_location.x - vehicle_location.x,
                    other_location.y - vehicle_location.y,
                    0
                )

                # 计算角度（车辆前方与障碍物方向的夹角）
                forward_2d = carla.Vector3D(forward_vector.x, forward_vector.y, 0)
                relative_2d = carla.Vector3D(relative_vector.x, relative_vector.y, 0)

                # 归一化向量
                forward_2d_norm = math.sqrt(forward_2d.x ** 2 + forward_2d.y ** 2)
                relative_2d_norm = math.sqrt(relative_2d.x ** 2 + relative_2d.y ** 2)

                if forward_2d_norm > 0 and relative_2d_norm > 0:
                    dot_product = forward_2d.x * relative_2d.x + forward_2d.y * relative_2d.y
                    cos_angle = dot_product / (forward_2d_norm * relative_2d_norm)
                    cos_angle = max(-1.0, min(1.0, cos_angle))  # 限制范围
                    angle = math.acos(cos_angle)

                    # 转换为角度
                    angle_deg = math.degrees(angle)

                    # 只考虑前方±60度范围内的障碍物
                    if angle_deg <= 60 and distance < min_distance:
                        min_distance = distance
                        closest_obstacle = other_vehicle
                        relative_angle = angle_deg if relative_2d.y >= 0 else -angle_deg

            if closest_obstacle is not None and min_distance < self.max_distance:
                self.last_obstacle_info = {
                    'has_obstacle': True,
                    'distance': min_distance,
                    'relative_angle': relative_angle,
                    'obstacle_type': closest_obstacle.type_id,
                    'obstacle_speed': self.get_vehicle_speed(closest_obstacle)
                }
            else:
                self.last_obstacle_info = {
                    'has_obstacle': False,
                    'distance': float('inf'),
                    'relative_angle': 0.0,
                    'obstacle_type': None,
                    'obstacle_speed': 0.0
                }

            return self.last_obstacle_info

        except Exception as e:
            print(f"障碍物检测错误: {e}")
            return self.last_obstacle_info

    def get_vehicle_speed(self, vehicle):
        """获取车辆速度"""
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        return speed * 3.6  # 转换为km/h

    def visualize_obstacles(self, image, vehicle_transform):
        """在图像上可视化障碍物检测结果"""
        if not self.last_obstacle_info['has_obstacle']:
            return image

        height, width = image.shape[:2]

        # 计算障碍物在图像中的位置（简化投影）
        distance = self.last_obstacle_info['distance']
        angle = self.last_obstacle_info['relative_angle']

        # 归一化角度到图像坐标
        x_pos = int(width / 2 + (angle / 60) * (width / 2))

        # 根据距离计算大小和颜色
        if distance < 10:
            color = (0, 0, 255)  # 红色，很近
            radius = 15
        elif distance < 20:
            color = (0, 165, 255)  # 橙色
            radius = 10
        else:
            color = (0, 255, 255)  # 黄色
            radius = 5

        # 绘制障碍物指示器
        cv2.circle(image, (x_pos, int(height * 0.8)), radius, color, -1)
        cv2.putText(image, f"{distance:.1f}m", (x_pos - 20, int(height * 0.8) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image


# 修复2: 改进的神经网络控制器（整合障碍物检测）
class ImprovedNeuralController:
    def __init__(self, obstacle_detector=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 使用简化网络
        self.model = SimpleDrivingNetwork().to(self.device)
        self.model.eval()

        # 控制历史，用于平滑
        self.control_history = deque(maxlen=5)

        # 障碍物检测器
        self.obstacle_detector = obstacle_detector

        # 修复3: 更保守的初始控制
        self.last_throttle = 0.3
        self.last_brake = 0.0
        self.last_steer = 0.0

        # 避障参数
        self.emergency_brake_distance = 5.0  # 紧急刹车距离
        self.safe_following_distance = 8.0  # 安全跟车距离
        self.obstacle_avoidance_steer = 0.0

    def preprocess_image(self, image):
        """修复图像预处理"""
        if image is None:
            # 返回黑色图像
            return torch.zeros((1, 3, 120, 160), device=self.device)

        try:
            # 调整图像尺寸，减少计算量
            small_img = cv2.resize(image, (160, 120))
            img_tensor = torch.from_numpy(small_img).float().to(self.device)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
            return img_tensor
        except Exception as e:
            print(f"图像预处理错误: {e}")
            return torch.zeros((1, 3, 120, 160), device=self.device)

    def preprocess_state(self, speed, steer_history, obstacle_info):
        """修复状态预处理，加入障碍物信息"""
        has_obstacle = 1.0 if obstacle_info['has_obstacle'] else 0.0
        normalized_distance = min(obstacle_info['distance'] / 50.0, 1.0) if obstacle_info['has_obstacle'] else 1.0
        normalized_angle = obstacle_info['relative_angle'] / 60.0 if obstacle_info['has_obstacle'] else 0.0

        state_data = [
            speed / 20.0,  # 归一化速度
            steer_history[-1] if steer_history else 0.0,  # 最近转向
            steer_history[-2] if len(steer_history) > 1 else 0.0,  # 前一次转向
            np.mean(steer_history) if steer_history else 0.0,  # 平均转向
            has_obstacle,  # 是否有障碍物
            normalized_distance,  # 归一化距离
            normalized_angle  # 归一化角度
        ]
        return torch.tensor(state_data, device=self.device).unsqueeze(0)

    def apply_obstacle_avoidance(self, throttle, brake, steer, obstacle_info, speed):
        """应用避障逻辑"""
        if not obstacle_info['has_obstacle']:
            return throttle, brake, steer

        distance = obstacle_info['distance']
        angle = obstacle_info['relative_angle']

        # 紧急情况：前方有近距离障碍物
        if distance < self.emergency_brake_distance:
            print(f"紧急刹车！距离障碍物: {distance:.1f}m")
            return 0.0, 1.0, steer  # 全力刹车

        # 中距离障碍物：减速并准备转向
        elif distance < self.safe_following_distance:
            # 计算安全速度比例
            safe_speed_ratio = (distance - 2.0) / (self.safe_following_distance - 2.0)
            safe_speed_ratio = max(0.1, min(1.0, safe_speed_ratio))

            # 如果当前速度过高，减速
            if speed > 5.0 * safe_speed_ratio:
                throttle = 0.0
                brake = 0.3

            # 如果障碍物在正前方，尝试轻微转向避开
            if abs(angle) < 10:  # 正前方±10度内
                # 根据障碍物距离决定转向幅度
                avoid_steer = 0.5 if angle >= 0 else -0.5
                # 平滑转向
                steer = 0.7 * steer + 0.3 * avoid_steer

        # 远距离障碍物：轻微调整
        elif distance < 20.0:
            # 轻微减速
            if speed > 10.0:
                throttle *= 0.8

            # 如果障碍物在正前方，轻微转向
            if abs(angle) < 15:
                avoid_steer = 0.2 if angle >= 0 else -0.2
                steer = 0.8 * steer + 0.2 * avoid_steer

        return throttle, brake, steer

    def get_control(self, image, speed, steer_history, obstacle_info):
        """修复控制生成逻辑，加入避障"""
        try:
            with torch.no_grad():
                # 预处理
                img_tensor = self.preprocess_image(image)
                state_tensor = self.preprocess_state(speed, steer_history, obstacle_info)

                # 神经网络推理
                control_output = self.model(img_tensor, state_tensor)

                # 提取控制指令
                throttle = control_output[0, 0].item()
                brake = control_output[0, 1].item()
                steer = control_output[0, 2].item()

                # 修复4: 添加安全限制
                throttle = max(0.0, min(0.8, throttle))  # 限制最大油门
                brake = max(0.0, min(0.5, brake))  # 限制最大刹车
                steer = max(-0.5, min(0.5, steer))  # 限制转向幅度

                # 应用避障逻辑
                throttle, brake, steer = self.apply_obstacle_avoidance(
                    throttle, brake, steer, obstacle_info, speed
                )

                return throttle, brake, steer

        except Exception as e:
            print(f"神经网络控制错误: {e}")
            # 返回安全默认值
            return 0.3, 0.0, 0.0


# 修复5: 传统控制器作为备份（整合障碍物检测）
class TraditionalController:
    """可靠的传统控制逻辑"""

    def __init__(self, world, obstacle_detector=None):
        self.world = world
        self.map = world.get_map()
        self.waypoint_distance = 10.0
        self.last_waypoint = None
        self.obstacle_detector = obstacle_detector
        self.emergency_brake_distance = 6.0
        self.safe_following_distance = 10.0

    def apply_obstacle_avoidance(self, throttle, brake, steer, vehicle, obstacle_info):
        """传统控制器的避障逻辑"""
        if not obstacle_info['has_obstacle']:
            return throttle, brake, steer

        distance = obstacle_info['distance']
        angle = obstacle_info['relative_angle']
        vehicle_speed = math.sqrt(vehicle.get_velocity().x ** 2 +
                                  vehicle.get_velocity().y ** 2 +
                                  vehicle.get_velocity().z ** 2) * 3.6  # km/h

        # 紧急刹车
        if distance < self.emergency_brake_distance:
            print(f"传统控制：紧急刹车！距离: {distance:.1f}m")
            return 0.0, 1.0, 0.0

        # 减速跟随
        elif distance < self.safe_following_distance:
            # 计算所需的安全距离（基于速度）
            required_distance = max(5.0, vehicle_speed * 0.3)  # 0.3秒车距

            if distance < required_distance:
                # 距离太近，减速
                if vehicle_speed > 10:
                    throttle = 0.0
                    brake = 0.4
                else:
                    throttle = 0.1
                    brake = 0.0

                # 如果障碍物在正前方，尝试变道
                if abs(angle) < 15:
                    # 获取当前车道和相邻车道
                    location = vehicle.get_location()
                    waypoint = self.map.get_waypoint(location)

                    # 尝试获取左侧车道
                    left_lane = waypoint.get_left_lane()
                    right_lane = waypoint.get_right_lane()

                    if left_lane and left_lane.lane_type == carla.LaneType.Driving:
                        steer = -0.3  # 向左变道
                    elif right_lane and right_lane.lane_type == carla.LaneType.Driving:
                        steer = 0.3  # 向右变道
                    else:
                        # 没有可用的相邻车道，轻微转向避开
                        steer = 0.2 if angle >= 0 else -0.2

        return throttle, brake, steer

    def get_control(self, vehicle):
        """基于路点的传统控制，整合避障"""
        # 获取车辆状态
        transform = vehicle.get_transform()
        location = vehicle.get_location()
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6  # km/h

        # 获取障碍物信息
        obstacle_info = None
        if self.obstacle_detector:
            obstacle_info = self.obstacle_detector.get_obstacle_info()

        # 获取路点
        waypoint = self.map.get_waypoint(location, project_to_road=True)
        next_waypoints = waypoint.next(self.waypoint_distance)

        if not next_waypoints:
            # 如果没有找到路点，尝试获取当前路点
            next_waypoints = [waypoint]

        target_waypoint = next_waypoints[0]
        self.last_waypoint = target_waypoint

        # 计算转向
        vehicle_yaw = math.radians(transform.rotation.yaw)
        target_loc = target_waypoint.transform.location

        dx = target_loc.x - location.x
        dy = target_loc.y - location.y

        local_x = dx * math.cos(vehicle_yaw) + dy * math.sin(vehicle_yaw)
        local_y = -dx * math.sin(vehicle_yaw) + dy * math.cos(vehicle_yaw)

        if abs(local_x) < 0.1:
            steer = 0.0
        else:
            angle = math.atan2(local_y, local_x)
            steer = np.clip(angle / math.radians(45), -1.0, 1.0)

        # 速度控制（基于障碍物距离调整）
        throttle = 0.0
        brake = 0.0

        if obstacle_info and obstacle_info['has_obstacle']:
            distance = obstacle_info['distance']

            # 根据障碍物距离调整速度
            if distance < 15:
                if speed > 20:
                    throttle = 0.0
                    brake = 0.3
                elif speed > 10:
                    throttle = 0.1
                    brake = 0.0
                else:
                    throttle = 0.3
                    brake = 0.0
            elif distance < 30:
                if speed > 30:
                    throttle = 0.0
                    brake = 0.1
                else:
                    throttle = 0.4
                    brake = 0.0
            else:
                # 没有近距离障碍物，正常行驶
                if speed < 20:
                    throttle = 0.6
                    brake = 0.0
                elif speed < 40:
                    throttle = 0.4
                    brake = 0.0
                else:
                    throttle = 0.2
                    brake = 0.1
        else:
            # 没有障碍物，正常行驶
            if speed < 20:
                throttle = 0.6
                brake = 0.0
            elif speed < 40:
                throttle = 0.4
                brake = 0.0
            else:
                throttle = 0.2
                brake = 0.1

        # 应用避障逻辑
        if obstacle_info:
            throttle, brake, steer = self.apply_obstacle_avoidance(
                throttle, brake, steer, vehicle, obstacle_info
            )

        return throttle, brake, steer


# CARLA初始化部分...
# 连接到本地CARLA服务器，端口2000
client = carla.Client('localhost', 2000)
client.set_timeout(15.0)
world = client.load_world('Town01')

# 获取并设置世界的运行参数
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.1
world.apply_settings(settings)

# 定义天气参数
weather = carla.WeatherParameters(
    cloudiness=30.0,
    precipitation=0.0,
    sun_altitude_angle=70.0
)
world.set_weather(weather)

# 获取地图和出生点
map = world.get_map()
spawn_points = map.get_spawn_points()
if not spawn_points:
    raise Exception("No spawn points available")

# 选择更合适的出生点
spawn_point = spawn_points[10]

# 生成车辆
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
vehicle_bp.set_attribute('color', '255,0,0')
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

if not vehicle:
    raise Exception("无法生成主车辆")

vehicle.set_autopilot(False)
vehicle.set_simulate_physics(True)

print(f"车辆生成在位置: {spawn_point.location}")

# 生成障碍物车辆
obstacle_count = 3
for i in range(obstacle_count):
    if i >= len(spawn_points):
        break
    other_vehicles = blueprint_library.filter('vehicle.*')
    other_vehicle_bp = np.random.choice(other_vehicles)
    spawn_idx = (i + 15) % len(spawn_points)
    other_vehicle = world.try_spawn_actor(other_vehicle_bp, spawn_points[spawn_idx])
    if other_vehicle:
        other_vehicle.set_autopilot(True)

# 配置传感器（简化配置）
third_camera_bp = blueprint_library.find('sensor.camera.rgb')
third_camera_bp.set_attribute('image_size_x', '640')
third_camera_bp.set_attribute('image_size_y', '480')
third_camera_bp.set_attribute('fov', '110')
third_camera_transform = carla.Transform(
    carla.Location(x=-5.0, y=0.0, z=3.0),
    carla.Rotation(pitch=-15.0)
)
third_camera = world.spawn_actor(third_camera_bp, third_camera_transform, attach_to=vehicle)

front_camera_bp = blueprint_library.find('sensor.camera.rgb')
front_camera_bp.set_attribute('image_size_x', '640')
front_camera_bp.set_attribute('image_size_y', '480')
front_camera_bp.set_attribute('fov', '90')
front_camera_transform = carla.Transform(
    carla.Location(x=2.0, y=0.0, z=1.5),
    carla.Rotation(pitch=0.0)
)
front_camera = world.spawn_actor(front_camera_bp, front_camera_transform, attach_to=vehicle)

# 传感器数据存储
third_image = None
front_image = None


def third_camera_callback(image):
    global third_image
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    third_image = array[:, :, :3]


def front_camera_callback(image):
    global front_image
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    front_image = array[:, :, :3]


third_camera.listen(third_camera_callback)
front_camera.listen(front_camera_callback)

time.sleep(2.0)

# 新增：初始化障碍物检测器
print("初始化障碍物检测器...")
obstacle_detector = ObstacleDetector(world, vehicle, max_distance=50.0)

# 修复6: 初始化控制器（传入障碍物检测器）
nn_controller = ImprovedNeuralController(obstacle_detector)
traditional_controller = TraditionalController(world, obstacle_detector)

# 控制变量
throttle = 0.3  # 更保守的初始油门
steer = 0.0
brake = 0.0
NEURAL_NETWORK_MODE = False  # 默认使用传统控制，更稳定

# 转向历史，用于平滑
steer_history = deque(maxlen=10)

# 障碍物信息历史
obstacle_history = deque(maxlen=5)

print("初始化车辆状态...")
vehicle.set_simulate_physics(True)

# 修复7: 更温和的启动控制
print("应用启动控制...")
vehicle.apply_control(carla.VehicleControl(
    throttle=0.5,  # 降低初始油门
    steer=0.0,
    brake=0.0,
    hand_brake=False
))

try:
    print("自动驾驶系统启动 - 初始模式: 传统控制")
    print("控制键: q-退出, m-切换控制模式, r-重置车辆, t-传统模式, n-神经网络模式")

    frame_count = 0
    stuck_count = 0
    last_position = vehicle.get_location()
    success_count = 0  # 成功运行计数器
    collision_count = 0  # 碰撞计数器
    last_collision_time = 0  # 上次碰撞时间

    # 主循环
    while True:
        world.tick()
        frame_count += 1

        # 获取车辆状态
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle.get_location()
        vehicle_velocity = vehicle.get_velocity()
        vehicle_speed = math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2)

        # 检测障碍物
        obstacle_info = obstacle_detector.get_obstacle_info()
        obstacle_history.append(obstacle_info)

        # 显示障碍物信息
        if obstacle_info['has_obstacle']:
            print(f"障碍物检测: 距离={obstacle_info['distance']:.1f}m, "
                  f"角度={obstacle_info['relative_angle']:.1f}°, "
                  f"类型={obstacle_info['obstacle_type']}")

        print(f"帧 {frame_count}: 速度={vehicle_speed * 3.6:.1f}km/h, "
              f"模式={'神经网络' if NEURAL_NETWORK_MODE else '传统'}, "
              f"障碍物={'有' if obstacle_info['has_obstacle'] else '无'}")

        # 修复8: 改进的卡住检测
        current_position = vehicle_location
        distance_moved = current_position.distance(last_position)

        # 更精确的卡住检测
        is_moving = distance_moved > 0.2 or vehicle_speed > 1.0
        if not is_moving:
            stuck_count += 1
        else:
            stuck_count = 0
            success_count += 1  # 成功运行一帧

        last_position = current_position

        # 修复9: 更智能的卡住恢复
        if stuck_count > 15:  # 1.5秒后认为卡住
            print("检测到车辆卡住，执行恢复程序...")

            # 先完全停止
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=1.0, hand_brake=True
            ))
            time.sleep(0.5)

            # 检查是否有前方障碍物
            if obstacle_info['has_obstacle'] and obstacle_info['distance'] < 10:
                print("前方有障碍物，尝试倒车...")
                # 倒车
                vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=0.0, reverse=True
                ))
                time.sleep(1.0)
            else:
                # 然后尝试不同方向的脱困
                recovery_steer = random.choice([-0.5, 0.5])  # 随机选择方向
                vehicle.apply_control(carla.VehicleControl(
                    throttle=0.8, steer=recovery_steer, brake=0.0, hand_brake=False
                ))
                time.sleep(1.0)

            stuck_count = 0
            success_count = 0

        # 每成功运行100帧显示一次状态
        if success_count % 100 == 0:
            print(f"已成功运行 {success_count} 帧")

        # 控制逻辑
        if NEURAL_NETWORK_MODE:
            # 神经网络控制（传入障碍物信息）
            nn_throttle, nn_brake, nn_steer = nn_controller.get_control(
                front_image, vehicle_speed, steer_history, obstacle_info
            )

            # 修复10: 更激进的控制平滑
            throttle = 0.3 * throttle + 0.7 * nn_throttle
            brake = 0.3 * brake + 0.7 * nn_brake
            steer = 0.2 * steer + 0.8 * nn_steer

            # 记录转向历史
            steer_history.append(steer)

        else:
            # 传统控制 - 更稳定
            throttle, brake, steer = traditional_controller.get_control(vehicle)
            steer_history.append(steer)

        # 应用控制
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False
        )

        vehicle.apply_control(control)

        # 显示和输入处理
        if third_image is not None:
            display_image = third_image.copy()

            # 可视化障碍物检测结果
            display_image = obstacle_detector.visualize_obstacles(display_image, vehicle_transform)

            # 显示信息
            cv2.putText(display_image, f"Speed: {vehicle_speed * 3.6:.1f} km/h", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, f"Mode: {'Neural' if NEURAL_NETWORK_MODE else 'Traditional'}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, f"Throttle: {throttle:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, f"Steer: {steer:.2f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, f"Brake: {brake:.2f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 显示障碍物信息
            if obstacle_info['has_obstacle']:
                cv2.putText(display_image, f"Obstacle: {obstacle_info['distance']:.1f}m", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_image, "Obstacle: None", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 卡住警告
            if stuck_count > 5:
                cv2.putText(display_image, "STUCK DETECTED!", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 安全区域标记
            cv2.rectangle(display_image, (240, 360), (400, 480), (0, 255, 0), 2)  # 前方安全区域

            cv2.imshow('自动驾驶系统 - 带障碍物检测', display_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                NEURAL_NETWORK_MODE = not NEURAL_NETWORK_MODE
                print(f"切换到{'神经网络' if NEURAL_NETWORK_MODE else '传统'}控制模式")
            elif key == ord('t'):
                NEURAL_NETWORK_MODE = False
                print("切换到传统控制模式")
            elif key == ord('n'):
                NEURAL_NETWORK_MODE = True
                print("切换到神经网络控制模式")
            elif key == ord('r'):
                # 重置车辆
                vehicle.set_transform(spawn_point)
                throttle = 0.3
                steer = 0.0
                brake = 1.0
                stuck_count = 0
                success_count = 0
                collision_count = 0
                steer_history.clear()
                obstacle_history.clear()
                print("车辆已重置")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("系统已停止")
except Exception as e:
    print(f"系统错误: {e}")
    import traceback

    traceback.print_exc()

finally:
    print("正在清理资源...")
    third_camera.stop()
    front_camera.stop()

    # 销毁actor
    for actor in world.get_actors():
        if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('sensor.'):
            actor.destroy()

    # 恢复设置
    settings.synchronous_mode = False
    world.apply_settings(settings)
    cv2.destroyAllWindows()
    print("资源清理完成")