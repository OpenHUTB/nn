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


# 修复1: 简化的神经网络架构
class SimpleDrivingNetwork(nn.Module):
    """
    简化的驾驶网络 - 更适合实时控制
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

        # 状态信息维度: 速度 + 转向历史
        state_dim = 4

        # 融合层
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 4 * 4 + state_dim, 64),
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


# 修复2: 改进的神经网络控制器
class ImprovedNeuralController:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 使用简化网络
        self.model = SimpleDrivingNetwork().to(self.device)
        self.model.eval()

        # 控制历史，用于平滑
        self.control_history = deque(maxlen=5)

        # 修复3: 更保守的初始控制
        self.last_throttle = 0.3
        self.last_brake = 0.0
        self.last_steer = 0.0

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

    def preprocess_state(self, speed, steer_history):
        """修复状态预处理"""
        state_data = [
            speed / 20.0,  # 归一化速度
            steer_history[-1] if steer_history else 0.0,  # 最近转向
            steer_history[-2] if len(steer_history) > 1 else 0.0,  # 前一次转向
            np.mean(steer_history) if steer_history else 0.0  # 平均转向
        ]
        return torch.tensor(state_data, device=self.device).unsqueeze(0)

    def get_control(self, image, speed, steer_history):
        """修复控制生成逻辑"""
        try:
            with torch.no_grad():
                # 预处理
                img_tensor = self.preprocess_image(image)
                state_tensor = self.preprocess_state(speed, steer_history)

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

                return throttle, brake, steer

        except Exception as e:
            print(f"神经网络控制错误: {e}")
            # 返回安全默认值
            return 0.3, 0.0, 0.0


# 修复5: 传统控制器作为备份
class TraditionalController:
    """可靠的传统控制逻辑"""

    def __init__(self, world):
        self.world = world
        self.map = world.get_map()
        self.waypoint_distance = 10.0
        self.last_waypoint = None

    def get_control(self, vehicle):
        """基于路点的传统控制"""
        # 获取车辆状态
        transform = vehicle.get_transform()
        location = vehicle.get_location()
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

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

        # 速度控制
        if speed < 5.0:  # 18 km/h
            throttle = 0.6
            brake = 0.0
        elif speed < 10.0:  # 36 km/h
            throttle = 0.3
            brake = 0.0
        else:
            throttle = 0.1
            brake = 0.1

        return throttle, brake, steer


# CARLA初始化部分保持不变...
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

# 修复6: 初始化控制器
nn_controller = ImprovedNeuralController()
traditional_controller = TraditionalController(world)

# 控制变量
throttle = 0.3  # 更保守的初始油门
steer = 0.0
brake = 0.0
NEURAL_NETWORK_MODE = False  # 默认使用传统控制，更稳定

# 转向历史，用于平滑
steer_history = deque(maxlen=10)

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

    # 主循环
    while True:
        world.tick()
        frame_count += 1

        # 获取车辆状态
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle.get_location()
        vehicle_velocity = vehicle.get_velocity()
        vehicle_speed = math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2)

        print(
            f"帧 {frame_count}: 速度={vehicle_speed * 3.6:.1f}km/h, 模式={'神经网络' if NEURAL_NETWORK_MODE else '传统'}")

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
            # 神经网络控制
            nn_throttle, nn_brake, nn_steer = nn_controller.get_control(
                front_image, vehicle_speed, steer_history
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

            # 卡住警告
            if stuck_count > 5:
                cv2.putText(display_image, "STUCK DETECTED!", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('自动驾驶系统 - 修复版', display_image)

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
                steer_history.clear()

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