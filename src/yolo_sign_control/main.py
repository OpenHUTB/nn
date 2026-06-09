import sys
import glob
import os
import threading
import copy
import cv2
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 控制台仪表盘（不依赖Pygame显示）
from console_dashboard import render_dashboard, enable_ansi_support
# 数据记录器
from data_logger import DataLogger

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    # 使用 insert(0, ...) 强制优先加载官方路径，防止虚拟环境里的旧包冲突
    # 注意：请将此处改为你的实际CARLA安装路径
    # 你的CARLA路径：E:/CARLA/WindowsNoEditor
    carla_paths = [
        'E:/CARLA/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg',
        'E:/CARLA/WindowsNoEditor/PythonAPI/carla/dist/carla-*-%s.egg',
        'E:/CARLA/WindowsNoEditor/PythonAPI/carla/dist/*.egg',
        '../CARLA/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg',
        '../../CARLA/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg',
        'C:/CARLA/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg',
        'D:/CARLA/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg'
    ]

    carla_path = None
    for path_pattern in carla_paths:
        try:
            if '%d' in path_pattern:
                path_matches = glob.glob(path_pattern % (
                    sys.version_info.major,
                    sys.version_info.minor,
                    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))
            else:
                path_matches = glob.glob(path_pattern)

            if path_matches:
                carla_path = path_matches[0]
                print(f"Found CARLA module: {carla_path}")
                break
        except Exception:
            continue

    if carla_path:
        sys.path.insert(0, carla_path)
        print(f"Using official CARLA module from: {carla_path}")
    else:
        print("Warning: Could not find official CARLA egg file in common paths.")
        print("Please ensure CARLA is installed and the path is correctly set.")
        print("Your CARLA: E:\\CARLA\\WindowsNoEditor")
except IndexError:
    print("Warning: Could not find official CARLA egg file.")

import carla
import random
import time
import pygame
import numpy as np
import math
from ultralytics import YOLO
import torch

# Initialize Pygame for display
def init_pygame(width, height):
    pygame.init()
    pygame.font.init()  # Initialize font system
    display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Driver's View")
    return display

def process_image(image):
    """
    简化版本的图像解析（仅用于兼容）
    """
    return parse_image_to_surface_and_array(image)

# Load YOLOv8 pretrained model for traffic sign detection
model = YOLO("yolov8n.pt")  # Use yolov8n.pt for fast inference

# Run detection on RGB numpy image from CARLA camera
def detect_traffic_signs(image_np):
    results = model.predict(source=image_np, imgsz=640, conf=0.5, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False)
    detections = results[0].boxes.data.cpu().numpy()
    names = results[0].names

    signs_detected = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = names[int(cls)]
        signs_detected.append((label, conf, (int(x1), int(y1), int(x2), int(y2))))
    return signs_detected

# ==============================================================================
# -- 颜色配置（不同检测类别使用不同颜色）---------------------------------------
# ==============================================================================
DETECTION_COLORS = {
    'person':        (255, 0,   0),    # 红色 - 行人
    'car':           (0,   0,   255),  # 蓝色 - 车辆
    'truck':         (0,   100, 200),  # 深蓝 - 卡车
    'bus':           (100, 100, 255),  # 浅蓝 - 公交车
    'bicycle':       (255, 165, 0),    # 橙色 - 自行车
    'motorcycle':    (255, 100, 0),    # 橙红 - 摩托车
    'traffic light': (255, 255, 0),    # 黄色 - 交通灯
    'stop sign':     (255, 0,   255),  # 紫色 - 停车标志
    'default':       (0,   255, 0),    # 绿色 - 其他
}

# 需要碰撞预警的重要类别
COLLISION_RELEVANT_CLASSES = ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']


def get_color_for_label(label):
    """根据检测标签获取对应颜色"""
    for key, color in DETECTION_COLORS.items():
        if key in label.lower():
            return color
    return DETECTION_COLORS['default']


# ==============================================================================
# -- 在Pygame画面上绘制检测边界框 ---------------------------------------------
# ==============================================================================

def draw_detections_on_surface(surface, signs, image_width, image_height):
    """
    在Pygame表面绘制YOLO检测结果的边界框、标签和置信度
    """
    if not signs:
        return surface

    # 获取Pygame的draw模块引用
    pg_draw = pygame.draw

    for label, conf, bbox in signs:
        x1, y1, x2, y2 = bbox
        color = get_color_for_label(label)

        # 根据置信度调整边界框线条粗细
        thickness = max(2, int(conf * 4))

        # 绘制边界框
        pg_draw.rect(surface, color, (x1, y1, x2 - x1, y2 - y1), thickness)

        # 绘制标签背景（在边界框上方）
        try:
            font = pygame.font.Font(None, 20)
            text = f"{label} {conf:.2f}"
            text_surface = font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect()

            # 标签放在边界框左上角
            text_bg = pygame.Surface((text_rect.width + 4, text_rect.height + 2))
            text_bg.set_alpha(180)
            text_bg.fill(color)
            surface.blit(text_bg, (x1, max(0, y1 - text_rect.height - 2)))
            surface.blit(text_surface, (x1 + 2, max(0, y1 - text_rect.height)))
        except:
            pass

    return surface


def estimate_distance(label, bbox, image_height):
    """
    基于边界框大小和物体类型估算距离（米）
    使用已知物体典型尺寸进行单目距离估计
    """
    x1, y1, x2, y2 = bbox
    bbox_height = y2 - y1
    bbox_width = x2 - x1

    if bbox_height < 5:
        return None  # 物体太小，无法可靠估计

    # 典型物体高度（米）
    typical_heights = {
        'person': 1.7,
        'car': 1.5,
        'truck': 3.0,
        'bus': 3.2,
        'bicycle': 1.0,
        'motorcycle': 1.2,
        'stop sign': 2.0,
        'traffic light': 1.0,
    }

    height = typical_heights.get('default', 1.5)
    for key, h in typical_heights.items():
        if key in label.lower():
            height = h
            break

    # 简单距离估计：距离 = (已知高度 * 焦距) / 图像中高度
    # 焦距用图像高度近似（假设FOV=90度）
    focal_length = image_height  # 近似值
    distance = (height * focal_length) / bbox_height
    return distance


def get_nearest_obstacle(signs, image_height):
    """
    找出最近的障碍物（车辆、行人等），返回（标签, 距离, 边界框）
    """
    nearest_dist = float('inf')
    nearest_sign = None

    for label, conf, bbox in signs:
        if any(cls in label.lower() for cls in COLLISION_RELEVANT_CLASSES):
            dist = estimate_distance(label, bbox, image_height)
            if dist is not None and dist < nearest_dist:
                nearest_dist = dist
                nearest_sign = (label, conf, bbox, dist)

    return nearest_sign

# ==============================================================================
# -- 速度平滑控制器 --------------------------------------------------------------
# ==============================================================================

class SpeedController:
    """速度平滑控制器 - 渐进式加减速，模拟真实驾驶"""

    def __init__(self, ramp_rate=0.04):
        self.current_throttle = 0.0
        self.current_brake = 0.0
        self.ramp_rate = ramp_rate  # 每帧最大变化量

    def update(self, target_throttle, target_brake):
        """
        平滑过渡到目标值
        返回 (平滑后的油门, 平滑后的刹车)
        """
        # 油门平滑（渐进式增加/减少）
        if self.current_throttle < target_throttle:
            self.current_throttle = min(
                target_throttle,
                self.current_throttle + self.ramp_rate
            )
        elif self.current_throttle > target_throttle:
            self.current_throttle = max(
                target_throttle,
                self.current_throttle - self.ramp_rate * 1.5  # 收油比加油快一点
            )

        # 刹车平滑（渐进式踩/松）
        if self.current_brake < target_brake:
            self.current_brake = min(
                target_brake,
                self.current_brake + self.ramp_rate * 0.8  # 刹车建立稍慢
            )
        elif self.current_brake > target_brake:
            self.current_brake = max(
                target_brake,
                self.current_brake - self.ramp_rate * 1.2  # 松刹车稍快
            )

        return self.current_throttle, self.current_brake

    def reset(self):
        """重置控制器状态"""
        self.current_throttle = 0.0
        self.current_brake = 0.0

def parse_image_to_surface_and_array(image):
    """
    修复CARLA 0.9.14的图像显示问题，正确转换色彩空间
    """
    try:
        # CARLA 0.9.14 不支持 convert 方法的 ColorConverter.Raw
        # 直接处理原始数据

        # 原始数据是BGRA格式 (4通道)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)

        # 检查数据大小是否正确
        expected_size = image.width * image.height * 4
        if len(array) != expected_size:
            print(f"Warning: Data size mismatch. Expected {expected_size}, got {len(array)}")
            return None, None

        # 重塑为(height, width, 4)的BGRA数组
        array = array.reshape((image.height, image.width, 4))

        # 方法1: 使用OpenCV进行色彩转换（推荐）
        # BGRA -> BGR -> RGB
        bgra = array[:, :, :4]
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)

        # 转置为(width, height, 3)以符合pygame.surfarray的要求
        rgb_transposed = np.transpose(rgb, (1, 0, 2))

        # 创建Pygame表面
        surface = pygame.surfarray.make_surface(rgb_transposed)

        # 返回surface和用于YOLO的数组（高度, 宽度, 3）
        return surface, rgb
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Calculate the steering angle between vehicle and waypoint
def get_steering_angle(vehicle_transform, waypoint_transform):
    v_loc = vehicle_transform.location
    v_forward = vehicle_transform.get_forward_vector()
    wp_loc = waypoint_transform.location
    direction = wp_loc - v_loc
    direction = carla.Vector3D(direction.x, direction.y, 0.0)

    v_forward = carla.Vector3D(v_forward.x, v_forward.y, 0.0)
    norm_dir = math.sqrt(direction.x ** 2 + direction.y ** 2)
    norm_fwd = math.sqrt(v_forward.x ** 2 + v_forward.y ** 2)
    dot = v_forward.x * direction.x + v_forward.y * direction.y
    angle = math.acos(dot / (norm_dir * norm_fwd + 1e-5))
    cross = v_forward.x * direction.y - v_forward.y * direction.x
    if cross < 0:
        angle *= -1
    return angle

# Control based on traffic signs with collision avoidance and smooth control
def control_vehicle_based_on_sign(vehicle, detected_signs, simulation_time, image_height=600, speed_controller=None):
    velocity = vehicle.get_velocity()
    current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6

    # 初始化返回值
    collision_warning = None
    is_emergency = False
    braking_force = 0.0
    throttle_value = 0.6  # 巡航油门

    # Traffic light control
    try:
        traffic_light_state = vehicle.get_traffic_light_state()
        if traffic_light_state == carla.TrafficLightState.Red:
            if speed_controller:
                smooth_t, smooth_b = speed_controller.update(0.0, 1.0)
                vehicle.apply_control(carla.VehicleControl(throttle=smooth_t, brake=smooth_b))
            else:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            return True, "RED LIGHT"
    except Exception:
        pass

    # --- 碰撞预警：检测前方障碍物（车辆/行人） ---
    nearest = get_nearest_obstacle(detected_signs, image_height)
    if nearest is not None:
        label, conf, bbox, distance = nearest

        # 根据距离分级响应
        if distance < 8.0:
            # 紧急制动 - 距离太近（直接刹死，安全第一）
            braking_force = 1.0
            throttle_value = 0.0
            is_emergency = True
            collision_warning = f"EMERGENCY BRAKE! {label} at {distance:.1f}m"
        elif distance < 15.0:
            # 强减速
            braking_force = min(0.8, 1.0 - distance / 15.0)
            throttle_value = 0.0
            collision_warning = f"WARNING! {label} at {distance:.1f}m"
        elif distance < 25.0:
            # 轻微减速
            braking_force = 0.2
            throttle_value = 0.1
            collision_warning = f"Caution: {label} at {distance:.1f}m"
        elif distance < 40.0:
            # 提前预警，收油滑行
            throttle_value = max(0.1, 0.4 - (40.0 - distance) / 100.0)
            collision_warning = f"{label} ahead at {distance:.1f}m"

    # --- 交通标志检测控制 ---
    for sign, conf, _ in detected_signs:
        if "stop" in sign.lower() and conf > 0.5:
            # 停车标志 - 完全制动
            throttle_value = 0.0
            braking_force = 1.0
            is_emergency = True
            collision_warning = f"STOP SIGN! ({conf:.0%})"
            if speed_controller:
                smooth_t, smooth_b = speed_controller.update(0.0, 1.0)
                vehicle.apply_control(carla.VehicleControl(throttle=smooth_t, brake=smooth_b, steer=vehicle.get_control().steer))
            else:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            return True, collision_warning

        elif "speed limit" in sign.lower():
            digits = [int(s) for s in sign.split() if s.isdigit()]
            if digits:
                speed_limit = digits[0]
                if current_speed > speed_limit:
                    # 超速了，松油门+轻刹车
                    braking_force = max(braking_force, 0.15)
                    throttle_value = min(throttle_value, 0.05)
                    collision_warning = f"Speed limit: {speed_limit} km/h"
                elif current_speed < speed_limit - 5:
                    # 低于限速，缓慢加速
                    throttle_value = max(throttle_value, 0.4)

    # 通过平滑控制器应用非紧急控制
    if speed_controller:
        if is_emergency and distance < 8.0:
            # 紧急情况直接刹死
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.0, brake=1.0, steer=vehicle.get_control().steer
            ))
        else:
            smooth_throttle, smooth_brake = speed_controller.update(throttle_value, braking_force)
            vehicle.apply_control(carla.VehicleControl(
                throttle=smooth_throttle,
                brake=smooth_brake,
                steer=vehicle.get_control().steer
            ))
    elif braking_force > 0 or is_emergency:
        # 没有控制器时直接应用（旧逻辑）
        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle_value,
            brake=braking_force,
            steer=vehicle.get_control().steer
        ))

    return is_emergency, collision_warning

# Spawn traffic signs
def spawn_dynamic_elements(world, blueprint_library):
    spawn_points = world.get_map().get_spawn_points()
    signs = []
    speed_values = [20, 40, 60, 60, 40, 60, 40, 20]
    sign_bp = [bp for bp in blueprint_library if 'static.prop.speedlimit' in bp.id or 'static.prop.stop' in bp.id]

    for i, speed in enumerate(speed_values):
        for bp in sign_bp:
            if f"speedlimit.{speed}" in bp.id:
                transform = spawn_points[i % len(spawn_points)]
                transform.location.z = 0
                actor = world.try_spawn_actor(bp, transform)
                if actor:
                    signs.append(actor)
                break

    stop_signs = [bp for bp in blueprint_library if 'static.prop.stop' in bp.id]
    if stop_signs:
        transform = spawn_points[-1]
        transform.location.z = 0
        actor = world.try_spawn_actor(stop_signs[0], transform)
        if actor:
            signs.append(actor)
    return signs

# Main function
def main():
    actor_list = []
    max_speed = 0

    # 启用控制台 ANSI 支持（Windows 终端）
    enable_ansi_support()

    # 初始化数据记录器
    logger = DataLogger()

    # 初始化速度平滑控制器（渐进式加减速）
    speed_ctrl = SpeedController(ramp_rate=0.04)

    try:
        # 确保pygame已初始化
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        elements = spawn_dynamic_elements(world, blueprint_library)
        actor_list.extend(elements)

        # Spawn ego vehicle
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_point = random.choice(map.get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        # Spawn NPC traffic
        for _ in range(10):
            traffic_bp = random.choice(blueprint_library.filter('vehicle.*'))
            traffic_spawn = random.choice(map.get_spawn_points())
            npc = world.try_spawn_actor(traffic_bp, traffic_spawn)
            if npc:
                npc.set_autopilot(True)
                actor_list.append(npc)

        # Camera
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")
        # 修复画面裂纹：移除可能导致问题的后处理效果
        camera_bp.set_attribute("enable_postprocess_effects", "False")  # 关闭后处理
        camera_bp.set_attribute("motion_blur_intensity", "0.0")
        camera_bp.set_attribute("sensor_tick", "0.0")  # 改为0，让相机以最大帧率运行
        camera_bp.set_attribute("gamma", "1.0")  # 降低gamma值
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        # 初始化Pygame显示
        display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Driver's View - CARLA Traffic Sign Detection")
        display.fill((0, 0, 0))
        pygame.display.flip()

        # 存储图像数据
        camera_surface = [None]
        camera_array = [None]
        camera_lock = threading.Lock()  # 添加线程锁保护图像数据

        def image_callback(image):
            # 优化图像处理逻辑
            surface, array = parse_image_to_surface_and_array(image)
            if surface is not None:
                # 使用锁保护数据写入,防止数据竞争
                with camera_lock:
                    # 为surface创建副本以避免显示撕裂
                    camera_surface[0] = surface.copy()
                    camera_array[0] = array.copy() if array is not None else None

        camera.listen(image_callback)

        # Top-down view
        spectator = world.get_spectator()
        def update_spectator():
            t = vehicle.get_transform()
            spectator.set_transform(carla.Transform(t.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

        clock = pygame.time.Clock()
        start_time = time.time()

        while True:
            update_spectator()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            # Steering control
            trans = vehicle.get_transform()
            waypoint = map.get_waypoint(trans.location)
            next_wps = waypoint.next(2.0)
            if not next_wps:
                # 没有后续路径点时尝试更远距离，或跳过本帧
                next_wps = waypoint.next(5.0)
                if not next_wps:
                    continue
            next_wp = next_wps[0]
            angle = get_steering_angle(trans, next_wp.transform)
            steer = max(-1.0, min(1.0, angle * 2.0))

            # 默认控制：巡航状态
            control = carla.VehicleControl()
            control.steer = steer

            # 通过速度控制器平滑输出油门/刹车
            target_throttle = 0.6  # 巡航目标油门
            target_brake = 0.0
            smooth_t, smooth_b = speed_ctrl.update(target_throttle, target_brake)
            control.throttle = smooth_t
            control.brake = smooth_b
            vehicle.apply_control(control)

            # Speed & time
            elapsed = time.time() - start_time
            m = int(elapsed // 60)
            s = int(elapsed % 60)
            vel = vehicle.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6
            if speed > max_speed:
                max_speed = round(speed, 1)

            # Display and detection
            if camera_surface[0] is not None and camera_array[0] is not None:
                try:
                    # 使用锁保护数据读取,确保读取完整帧
                    with camera_lock:
                        surface_copy = camera_surface[0].copy()
                        array_copy = camera_array[0].copy()

                    # YOLO 检测（使用副本）
                    signs = detect_traffic_signs(array_copy)

                    # 碰撞预警 + 车辆控制（使用速度平滑控制器）
                    is_emergency, warning_text = control_vehicle_based_on_sign(
                        vehicle, signs, elapsed,
                        image_height=600,
                        speed_controller=speed_ctrl
                    )
                    if is_emergency:
                        # 紧急情况下跳过默认控制
                        pass

                    # 在画面上绘制检测边界框
                    surface_copy = draw_detections_on_surface(
                        surface_copy, signs, 800, 600
                    )

                    # 找出最近障碍物用于显示
                    nearest_obs = get_nearest_obstacle(signs, 600)

                    # 记录数据到 CSV
                    try:
                        logger.record(
                            elapsed=elapsed,
                            speed=speed,
                            throttle=speed_ctrl.current_throttle,
                            brake=speed_ctrl.current_brake,
                            steer=control.steer,
                            signs=signs,
                            warning_text=warning_text if 'warning_text' in dir() else None,
                            is_emergency=is_emergency if 'is_emergency' in dir() else False,
                            traffic_light_state=vehicle.get_traffic_light_state(),
                            nearest_obs=nearest_obs,
                        )
                    except Exception:
                        pass

                    # 双缓冲显示以减少画面撕裂
                    display.fill((0, 0, 0))  # 先清除屏幕
                    display.blit(surface_copy, (0, 0))

                    # ---- 增强HUD显示 ----
                    try:
                        # 使用更安全的字体加载方式
                        font_large = pygame.font.Font(None, 26)
                        font_small = pygame.font.Font(None, 20)
                        font_warning = pygame.font.Font(None, 36)
                    except:
                        font_large = None
                        font_small = None
                        font_warning = None

                    if font_large:
                        # 基本信息（左上角）
                        display.blit(font_large.render(f"Time: {m:02d}:{s:02d}", True, (0,255,0)), (10,10))
                        display.blit(font_large.render(f"Speed: {speed:.1f} km/h", True, (255,255,0)), (10,40))
                        display.blit(font_large.render(f"Max: {max_speed} km/h", True, (255,0,0)), (10,70))

                    if font_small:
                        # 检测统计信息（右上角）
                        total_detected = len(signs)
                        display.blit(font_small.render(f"Detected: {total_detected} objects", True, (200,200,200)), (600, 10))

                        if total_detected > 0:
                            # 按类别统计
                            class_counts = {}
                            for label, _, _ in signs:
                                class_counts[label] = class_counts.get(label, 0) + 1
                            y_offset = 30
                            for cls_name, count in list(class_counts.items())[:5]:
                                color = get_color_for_label(cls_name)
                                display.blit(font_small.render(f"{cls_name}: {count}", True, color), (600, y_offset))
                                y_offset += 20

                        # 碰撞预警信息（屏幕中央底部）
                        if warning_text:
                            # 闪烁效果 (每0.5秒切换)
                            blink = int(time.time() * 2) % 2 == 0
                            if blink:
                                if is_emergency or "EMERGENCY" in warning_text or "WARNING" in warning_text:
                                    warn_render = font_warning.render(warning_text, True, (255, 0, 0))
                                else:
                                    warn_render = font_large.render(warning_text, True, (255, 200, 0))
                                warn_rect = warn_render.get_rect(center=(400, 550))
                                display.blit(warn_render, warn_rect)

                        # 最近障碍物距离（右下角）
                        if nearest_obs:
                            label, conf, bbox, dist = nearest_obs
                            color = get_color_for_label(label)
                            dist_text = f"{label}: {dist:.1f}m"
                            display.blit(font_small.render(dist_text, True, color), (600, 570))

                        # 碰撞预警计时器状态（左下角）
                        if is_emergency:
                            display.blit(font_large.render("!!! EMERGENCY !!!", True, (255, 0, 0)), (10, 570))

                    pygame.display.flip()  # 更新整个屏幕

                    # 控制台实时状态面板（不依赖Pygame）
                    render_dashboard(
                        elapsed=elapsed,
                        speed=speed,
                        max_speed=max_speed,
                        signs=signs,
                        warning_text=warning_text if 'warning_text' in dir() else None,
                        is_emergency=is_emergency if 'is_emergency' in dir() else False,
                        throttle=speed_ctrl.current_throttle,
                        brake=speed_ctrl.current_brake,
                        steer=control.steer,
                        traffic_light_state=vehicle.get_traffic_light_state(),
                        nearest_obs=nearest_obs if 'nearest_obs' in dir() else None,
                    )
                except Exception as e:
                    print(f"Display error: {e}")
                    import traceback
                    traceback.print_exc()

            # 保持稳定的帧率
            clock.tick(30)  # 固定帧率到30 FPS
            if elapsed > 120:
                break

    finally:
        # 安全清理所有actor，防止连接断开后destroy再次抛异常
        for actor in actor_list:
            try:
                if actor.is_alive():
                    actor.destroy()
            except Exception:
                pass
        try:
            pygame.quit()
        except Exception:
            pass

        # 输出数据记录报告
        try:
            logger.close_and_report()
        except Exception:
            pass

if __name__ == "__main__":
    main()
