import carla
import time
import numpy as np
import cv2
import math
from collections import deque

# --------------------------
# 1. 初始化CARLA连接和环境
# --------------------------
client = carla.Client('localhost', 2000)
client.set_timeout(15.0)
world = client.load_world('Town01')

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.1
settings.substepping = True
settings.max_substep_delta_time = 0.01
settings.max_substeps = 10
world.apply_settings(settings)

weather = carla.WeatherParameters(
    cloudiness=30.0,
    precipitation=0.0,
    sun_altitude_angle=70.0
)
world.set_weather(weather)

map = world.get_map()
spawn_points = map.get_spawn_points()
if not spawn_points:
    raise Exception("No spawn points available")

# 选择一个更好的出生点
spawn_point = spawn_points[10]  # 选择更靠前的出生点

# --------------------------
# 2. 生成车辆和障碍物
# --------------------------
blueprint_library = world.get_blueprint_library()

# 主车辆（红色）
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
vehicle_bp.set_attribute('color', '255,0,0')
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    raise Exception("无法生成主车辆")
vehicle.set_autopilot(False)
vehicle.set_simulate_physics(True)

print(f"车辆生成在位置: {spawn_point.location}")

# 生成障碍物
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

# --------------------------
# 3. 配置传感器
# --------------------------
third_camera_bp = blueprint_library.find('sensor.camera.rgb')
third_camera_bp.set_attribute('image_size_x', '640')
third_camera_bp.set_attribute('image_size_y', '480')
third_camera_bp.set_attribute('fov', '110')
third_camera_transform = carla.Transform(
    carla.Location(x=-5.0, y=0.0, z=3.0),
    carla.Rotation(pitch=-15.0)
)
third_camera = world.spawn_actor(third_camera_bp, third_camera_transform, attach_to=vehicle)

# 激光雷达配置
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('channels', '32')
lidar_bp.set_attribute('range', '50')
lidar_bp.set_attribute('points_per_second', '100000')
lidar_bp.set_attribute('rotation_frequency', '10')
lidar_bp.set_attribute('upper_fov', '15')
lidar_bp.set_attribute('lower_fov', '-25')
lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

# --------------------------
# 4. 传感器数据处理
# --------------------------
third_image = None
lidar_data = None


def third_camera_callback(image):
    global third_image
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    third_image = array[:, :, :3]


def lidar_callback(point_cloud):
    global lidar_data
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    lidar_data = data


third_camera.listen(third_camera_callback)
lidar.listen(lidar_callback)

time.sleep(2.0)  # 增加等待时间确保传感器初始化


# --------------------------
# 5. 路径规划与导航逻辑
# --------------------------
def get_next_waypoint(vehicle_location, distance=8.0):
    """获取车辆前方指定距离的路点"""
    waypoint = map.get_waypoint(vehicle_location, project_to_road=True)

    next_waypoints = waypoint.next(distance)
    if next_waypoints:
        return next_waypoints[0]

    if waypoint.is_junction:
        for wp in waypoint.next(distance):
            if wp.road_id == waypoint.road_id:
                return wp

        if waypoint.lane_change & carla.LaneChange.Right:
            right_way = waypoint.get_right_lane()
            if right_way:
                return right_way.next(distance)[0]
        elif waypoint.lane_change & carla.LaneChange.Left:
            left_way = waypoint.get_left_lane()
            if left_way:
                return left_way.next(distance)[0]

    return waypoint


def calculate_steering_angle(vehicle_transform, target_waypoint):
    """计算到达目标路点所需的转向角"""
    vehicle_location = vehicle_transform.location
    target_location = target_waypoint.transform.location

    vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
    dx = target_location.x - vehicle_location.x
    dy = target_location.y - vehicle_location.y

    local_x = dx * math.cos(vehicle_yaw) + dy * math.sin(vehicle_yaw)
    local_y = -dx * math.sin(vehicle_yaw) + dy * math.cos(vehicle_yaw)

    if abs(local_x) < 0.1:
        return 0.0

    angle = math.atan2(local_y, local_x)
    max_angle = math.radians(60)
    steering = angle / max_angle

    return np.clip(steering, -1.0, 1.0)


# --------------------------
# 6. 优化避障控制逻辑
# --------------------------
class ObstacleAvoidance:
    def __init__(self):
        self.obstacle_history = deque(maxlen=10)
        self.emergency_brake = False
        self.last_avoid_direction = 0

    def detect_obstacles(self, lidar_data, vehicle_speed):
        """修复版的障碍物检测算法"""
        if lidar_data is None:
            return self._get_default_detection_result()

        if len(lidar_data) == 0:
            return self._get_default_detection_result()

        try:
            # 地面过滤
            ground_threshold = -0.5
            valid_mask = lidar_data[:, 2] > ground_threshold
            valid_points = lidar_data[valid_mask]

            if len(valid_points) == 0:
                return self._get_default_detection_result()

            # 计算距离和角度
            distances = np.sqrt(valid_points[:, 0] ** 2 + valid_points[:, 1] ** 2)
            angles = np.arctan2(valid_points[:, 1], valid_points[:, 0])

            # 定义检测区域
            front_angle_range = np.radians(75)
            front_mask = (np.abs(angles) <= front_angle_range) & (distances > 1.0)

            front_points = valid_points[front_mask]
            front_distances = distances[front_mask]
            front_angles = angles[front_mask]

            if len(front_points) == 0:
                return self._get_default_detection_result()

            # 分区域检测
            near_zone = front_distances < 8.0
            mid_zone = (front_distances >= 8.0) & (front_distances < 20.0)
            far_zone = (front_distances >= 20.0) & (front_distances < 35.0)

            # 紧急制动检测
            emergency_points = front_points[near_zone & (front_distances < 4.0)]
            self.emergency_brake = len(emergency_points) > 10

            # 计算最小距离和障碍物角度
            min_distance = np.min(front_distances)
            min_idx = np.argmin(front_distances)
            obstacle_angle = front_angles[min_idx] if len(front_angles) > min_idx else 0.0

            # 分左右区域分析
            left_points_distances = front_distances[front_angles > 0]
            right_points_distances = front_distances[front_angles < 0]

            # 计算左右侧最小距离
            left_min = np.min(left_points_distances) if len(left_points_distances) > 0 else float('inf')
            right_min = np.min(right_points_distances) if len(right_points_distances) > 0 else float('inf')

            # 计算自由空间
            safe_threshold = 15.0
            left_free = np.sum(left_points_distances > safe_threshold) if len(left_points_distances) > 0 else 1000
            right_free = np.sum(right_points_distances > safe_threshold) if len(right_points_distances) > 0 else 1000

            # 障碍物检测条件
            obstacle_detected = (np.sum(near_zone) > 5 or
                                 np.sum(mid_zone) > 10 or
                                 min_distance < 12.0)

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
            print(f"障碍物检测错误: {e}")
            return self._get_default_detection_result()

    def _get_default_detection_result(self):
        """返回默认的检测结果"""
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
        """避障决策逻辑"""
        if not detection_result['obstacle_detected']:
            self.last_avoid_direction = 0
            return 0, 0, False

        min_dist = detection_result['min_distance']
        left_clear = detection_result['left_clearance']
        right_clear = detection_result['right_clearance']
        left_free = detection_result['left_free_space']
        right_free = detection_result['right_free_space']
        obstacle_angle = detection_result['obstacle_angle']

        # 紧急制动情况
        if detection_result['emergency_brake']:
            return 0, 1.0, True

        # 基于安全距离的避障决策
        safety_margin = max(3.0, vehicle_speed * 0.5)

        # 计算左右侧的安全得分
        left_score = (left_clear - safety_margin) + (left_free * 0.1)
        right_score = (right_clear - safety_margin) + (right_free * 0.1)

        # 考虑当前转向的连续性
        if self.last_avoid_direction != 0:
            if self.last_avoid_direction == 1:
                right_score += 2.0
            else:
                left_score += 2.0

        # 决策避障方向
        avoid_steer = 0.0
        avoid_brake = 0.0

        if min_dist < safety_margin + 2.0:
            avoid_brake = 0.3 + (safety_margin - min_dist) * 0.1

        if right_score > left_score + 1.0:
            avoid_steer = -0.5
            self.last_avoid_direction = -1
        elif left_score > right_score + 1.0:
            avoid_steer = 0.5
            self.last_avoid_direction = 1
        else:
            # 两侧条件相似，基于障碍物角度决策
            if obstacle_angle > 0:
                avoid_steer = -0.4
                self.last_avoid_direction = 1
            else:
                avoid_steer = 0.4
                self.last_avoid_direction = -1

        return avoid_steer, avoid_brake, False


# --------------------------
# 7. 主控制循环
# --------------------------
# 初始化避障控制器
obstacle_avoidance = ObstacleAvoidance()

# 控制状态变量
throttle = 1.0  # 直接使用最大油门
steer = 0.0
brake = 0.0
waypoint_distance = 8.0

# 获取初始路点
vehicle_location = vehicle.get_location()
waypoint = get_next_waypoint(vehicle_location, waypoint_distance)

# 控制平滑滤波器
steer_filter = deque(maxlen=3)
throttle_filter = deque(maxlen=2)

print("初始化车辆状态...")
# 确保车辆物理引擎开启
vehicle.set_simulate_physics(True)

# 直接应用强力控制
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

    frame_count = 0
    stuck_count = 0
    last_position = vehicle.get_location()

    while True:
        world.tick()
        frame_count += 1

        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle.get_location()
        vehicle_velocity = vehicle.get_velocity()
        vehicle_speed = math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2)

        # 每帧都打印状态信息
        print(
            f"帧 {frame_count}: 速度={vehicle_speed * 3.6:.1f}km/h, 位置=({vehicle_location.x:.1f}, {vehicle_location.y:.1f})")

        # 检测是否卡住
        current_position = vehicle_location
        distance_moved = current_position.distance(last_position)
        if distance_moved < 0.1:  # 几乎没移动
            stuck_count += 1
        else:
            stuck_count = 0

        last_position = current_position

        # 如果卡住超过10帧，尝试强力脱困
        if stuck_count > 10:
            print("车辆卡住，尝试强力脱困...")
            # 先倒车再前进
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=False,
                reverse=True
            ))
            time.sleep(0.5)
            vehicle.apply_control(carla.VehicleControl(
                throttle=1.0,
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False
            ))
            stuck_count = 0

        # 更新目标路点
        current_distance = vehicle_location.distance(waypoint.transform.location)
        if current_distance < 4.0:
            waypoint = get_next_waypoint(vehicle_location, waypoint_distance)

        # 计算基础转向角
        base_steer = calculate_steering_angle(vehicle_transform, waypoint)

        # 检测障碍物并决策避障
        detection_result = obstacle_avoidance.detect_obstacles(lidar_data, vehicle_speed)
        avoid_steer, avoid_brake, emergency_brake = obstacle_avoidance.decide_avoidance_direction(
            detection_result, steer, vehicle_speed
        )

        # 综合控制输出 - 简化逻辑，专注于让车动起来
        if emergency_brake:
            throttle = 0.2
            brake = 0.1
            steer = base_steer * 0.1
            print("!!! 紧急制动 !!!")
        elif detection_result['obstacle_detected']:
            brake = avoid_brake
            throttle = 0.4  # 避障时也保持高油门
            steer = avoid_steer * 0.1 + base_steer * 0.1

            throttle = 0.0
            brake = 1.0
            steer = base_steer * 0.3
            print("!!! 紧急制动 !!!")
        elif detection_result['obstacle_detected']:
            brake = avoid_brake
            throttle = 0.8  # 避障时也保持高油门
            steer = avoid_steer * 0.8 + base_steer * 0.2
            print(f"避障中 - 距离:{detection_result['min_distance']:.1f}m")
        else:
            # 正常行驶 - 使用强力油门
            brake = 0.0
            steer = base_steer * 0.
            steer = base_steer*0.1

            # 强力油门策略
            if vehicle_speed < 2.0:  # 低速时最大油门
                throttle = 0.4
            elif vehicle_speed < 5.0:
                throttle = 0.3
            elif vehicle_speed < 7.0:
                throttle = 0.2
            else:
                throttle = 0.1

            steer = base_steer

            # 强力油门策略
            if vehicle_speed < 10.0:  # 低速时最大油门
                throttle = 1.0
            elif vehicle_speed < 20.0:
                throttle = 0.8
            elif vehicle_speed < 30.0:
                throttle = 0.6
            else:
                throttle = 0.4

        # 应用平滑滤波
        steer_filter.append(steer)
        throttle_filter.append(throttle)

        smoothed_steer = np.mean(steer_filter)
        smoothed_throttle = np.mean(throttle_filter)

        # 应用强力控制
        control = carla.VehicleControl(
            throttle=smoothed_throttle,
            steer=smoothed_steer,
            brake=brake,
            hand_brake=False,
            reverse=False
        )

        print(f"控制输出: 油门={control.throttle:.2f}, 刹车={control.brake:.2f}, 转向={control.steer:.2f}")
        vehicle.apply_control(control)

        # 可视化显示
        if third_image is not None:
            display_image = third_image.copy()
            cv2.putText(display_image, f"Speed: {vehicle_speed * 3.6:.1f} km/h", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, f"Throttle: {throttle:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, f"Steer: {steer:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if detection_result['obstacle_detected']:
                status_text = f"OBSTACLE: {detection_result['min_distance']:.1f}m"
                cv2.putText(display_image, status_text, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_image, "CLEAR", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('第三视角 - 强力油门系统', display_image)

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
                steer = 0.0
            elif key == ord(' '):
                brake = 1.0
                throttle = 0.0

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
    lidar.stop()

    for actor in world.get_actors():
        if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('sensor.'):
            actor.destroy()

    settings.synchronous_mode = False
    world.apply_settings(settings)
    cv2.destroyAllWindows()
    print("资源清理完成")