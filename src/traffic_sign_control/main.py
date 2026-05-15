import carla
import random
import time
import pygame
import numpy as np
import math
import os
import threading
import argparse
from ultralytics import YOLO
import torch

# 初始化Pygame用于显示
def init_pygame(width, height):
    pygame.init()
    display = pygame.display.set_mode((width, height))
    pygame.display.set_caption("驾驶员视角")
    return display

# 将CARLA图像转换为numpy数组（RGB）
def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]  # 丢弃alpha通道
    return array

# 在图像上绘制检测结果
def draw_detections(image_np, detected_signs):
    # 创建Pygame表面
    surface = pygame.Surface((image_np.shape[1], image_np.shape[0]))
    surface.blit(pygame.image.frombuffer(image_np.tobytes(), (image_np.shape[1], image_np.shape[0]), "RGB"), (0, 0))
    
    # 绘制检测框和标签
    font = pygame.font.Font(None, 24)
    for sign, conf, bbox in detected_signs:
        x1, y1, x2, y2 = bbox
        # 绘制矩形框
        pygame.draw.rect(surface, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)
        # 绘制标签
        label_text = f"{sign}: {conf:.2f}"
        text_surface = font.render(label_text, True, (0, 255, 0))
        surface.blit(text_surface, (x1, y1 - 25))
    
    return surface

# 绘制圆角矩形
def draw_rounded_rect(surface, color, rect, radius, width=0):
    pygame.draw.rect(surface, color, rect, width, border_radius=radius)

# 绘制车辆状态监控面板
def draw_vehicle_status(surface, vehicle, detected_signs, traffic_light_state=None):
    # 使用默认字体
    font = pygame.font.Font(None, 16)

    status_width = 220
    status_height = 220  # 增加高度以容纳红绿灯状态
    padding = 15
    margin = 10

    # 计算面板位置
    panel_x = surface.get_width() - status_width - margin
    panel_y = margin

    # 创建状态面板背景（半透明圆角面板）
    draw_rounded_rect(surface, (0, 0, 0, 180), (panel_x, panel_y, status_width, status_height), 10)

    # 添加轻微的阴影效果
    shadow_offset = 3
    draw_rounded_rect(surface, (0, 0, 0, 100), (panel_x + shadow_offset, panel_y + shadow_offset, status_width, status_height), 10)

    # 绘制面板边框
    draw_rounded_rect(surface, (200, 200, 200), (panel_x, panel_y, status_width, status_height), 10, 1)

    # 获取车辆状态
    velocity = vehicle.get_velocity()
    current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # m/s转换为km/h
    transform = vehicle.get_transform()
    steering_angle = transform.rotation.yaw
    throttle = vehicle.get_control().throttle
    brake = vehicle.get_control().brake

    # 确定颜色编码
    speed_color = (255, 255, 255)
    throttle_color = (0, 255, 0) if throttle > 0 else (255, 255, 255)
    brake_color = (255, 0, 0) if brake > 0 else (255, 255, 255)

    # 检查是否检测到交通标志
    detected_speed_limit = None
    for sign, conf, bbox in detected_signs:
        if "speed limit" in sign.lower():
            digits = [int(s) for s in sign.split() if s.isdigit()]
            if digits:
                detected_speed_limit = digits[0]
                break

    # 速度颜色特殊处理（如果检测到限速标志）
    if detected_speed_limit:
        if current_speed > detected_speed_limit:
            speed_color = (255, 0, 0)  # 超速 - 红色
        elif current_speed < detected_speed_limit * 0.9:
            speed_color = (0, 255, 0)  # 速度过低 - 绿色
        else:
            speed_color = (255, 255, 0)  # 速度合适 - 黄色

    # 绘制标题
    title_font = pygame.font.Font(None, 14)
    title_surface = title_font.render("VEHICLE STATUS", True, (200, 200, 200))
    surface.blit(title_surface, (panel_x + (status_width - title_surface.get_width()) // 2, panel_y + 5))

    # 绘制分隔线
    pygame.draw.line(surface, (100, 100, 100), (panel_x + padding, panel_y + 25), (panel_x + status_width - padding, panel_y + 25), 1)

    # 绘制状态信息 - 动力系统组
    y_offset = 40
    power_font = pygame.font.Font(None, 12)
    power_surface = power_font.render("POWER SYSTEM", True, (150, 150, 150))
    surface.blit(power_surface, (panel_x + padding, panel_y + y_offset))
    y_offset += 20

    # 绘制速度
    speed_text = f"SPEED: {current_speed:6.2f} km/h"
    speed_surface = font.render(speed_text, True, speed_color)
    surface.blit(speed_surface, (panel_x + padding, panel_y + y_offset))
    y_offset += 20

    # 绘制油门
    throttle_text = f"THROTTLE: {throttle:5.2f}"
    throttle_surface = font.render(throttle_text, True, throttle_color)
    surface.blit(throttle_surface, (panel_x + padding, panel_y + y_offset))
    y_offset += 20

    # 绘制刹车
    brake_text = f"BRAKE: {brake:5.2f}"
    brake_surface = font.render(brake_text, True, brake_color)
    surface.blit(brake_surface, (panel_x + padding, panel_y + y_offset))
    y_offset += 25

    # 绘制分隔线
    pygame.draw.line(surface, (100, 100, 100), (panel_x + padding, panel_y + y_offset - 5), (panel_x + status_width - padding, panel_y + y_offset - 5), 1)

    # 绘制状态信息 - 车辆状态组
    status_font = pygame.font.Font(None, 12)
    status_surface = status_font.render("VEHICLE STATE", True, (150, 150, 150))
    surface.blit(status_surface, (panel_x + padding, panel_y + y_offset))
    y_offset += 20

    # 绘制转向角
    steer_text = f"STEER: {steering_angle:6.2f}°"
    steer_surface = font.render(steer_text, True, (255, 255, 255))
    surface.blit(steer_surface, (panel_x + padding, panel_y + y_offset))
    y_offset += 20

    # 绘制位置
    pos_text = f"POS: ({transform.location.x:5.1f}, {transform.location.y:5.1f})"
    pos_surface = font.render(pos_text, True, (255, 255, 255))
    surface.blit(pos_surface, (panel_x + padding, panel_y + y_offset))
    y_offset += 25

    # 绘制分隔线
    pygame.draw.line(surface, (100, 100, 100), (panel_x + padding, panel_y + y_offset - 5), (panel_x + status_width - padding, panel_y + y_offset - 5), 1)

    # 绘制红绿灯状态 - 清晰可视化
    light_font = pygame.font.Font(None, 12)
    light_surface = light_font.render("TRAFFIC LIGHT", True, (150, 150, 150))
    surface.blit(light_surface, (panel_x + padding, panel_y + y_offset))
    y_offset += 20

    # 绘制红绿灯状态指示器
    if traffic_light_state is not None:
        # 根据状态设置颜色和文本
        if traffic_light_state == carla.TrafficLightState.Red:
            light_color = (255, 0, 0)
            light_text = "RED - STOP"
        elif traffic_light_state == carla.TrafficLightState.Yellow:
            light_color = (255, 255, 0)
            light_text = "YELLOW - SLOW"
        elif traffic_light_state == carla.TrafficLightState.Green:
            light_color = (0, 255, 0)
            light_text = "GREEN - GO"
        else:
            light_color = (128, 128, 128)
            light_text = "UNKNOWN"

        # 绘制圆形指示灯
        pygame.draw.circle(surface, light_color, (panel_x + padding + 10, panel_y + y_offset + 8), 8)
        pygame.draw.circle(surface, (255, 255, 255), (panel_x + padding + 10, panel_y + y_offset + 8), 8, 2)

        # 绘制状态文本
        light_text_surface = font.render(light_text, True, light_color)
        surface.blit(light_text_surface, (panel_x + padding + 25, panel_y + y_offset))
    else:
        # 无红绿灯状态
        no_light_text = font.render("NO LIGHT DETECTED", True, (128, 128, 128))
        surface.blit(no_light_text, (panel_x + padding, panel_y + y_offset))

# 查找最新训练权重
def find_latest_weights():
    """查找最新的训练权重文件"""
    runs_dir = 'runs/detect'
    if not os.path.exists(runs_dir):
        return None

    best_weights = None
    latest_time = 0

    for root, dirs, files in os.walk(runs_dir):
        if 'best.pt' in files:
            weight_path = os.path.join(root, 'best.pt')
            mtime = os.path.getmtime(weight_path)
            if mtime > latest_time:
                latest_time = mtime
                best_weights = weight_path

    return best_weights

# 加载YOLOv8预训练模型用于交通标志检测
def parse_args():
    # 查找最新训练权重作为默认
    default_model = find_latest_weights() or 'yolov8n.pt'

    parser = argparse.ArgumentParser(description="CARLA交通标志检测与数据采集")
    parser.add_argument('--model', type=str, default=default_model,
                       help=f'YOLO模型路径 (默认: 自动选择最新训练权重)')
    parser.add_argument('--save-interval', type=int, default=30,
                       help='自动保存帧间隔 (默认: 30帧保存1张)')
    parser.add_argument('--output-dir', type=str, default='dataset',
                       help='数据集输出目录 (默认: dataset)')
    args = parser.parse_args()

    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        print(f"提示: 使用默认模型请运行: python main.py")
        exit(1)

    return args

# 初始化模型（延迟加载）
model = None

# COCO数据集中交通相关类别: stop sign(11), traffic light(9)
TRAFFIC_RELEVANT_CLASSES = {9, 11}

# CARLA语义分割中的交通标志类别ID
CARLA_TRAFFIC_SIGN_LABEL = 12
CARLA_TRAFFIC_LIGHT_LABEL = 18

# 数据集采集管理器
class DatasetCollector:
    def __init__(self, output_dir='dataset'):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'images')
        self.semantic_dir = os.path.join(output_dir, 'semantic')
        self.labels_dir = os.path.join(output_dir, 'labels')

        # 创建目录结构
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.semantic_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        self.frame_count = 0
        print(f"数据集采集初始化: {output_dir}")

    def save_frame(self, rgb_image, seg_image, detections):
        """保存一帧数据（RGB、语义分割、YOLO标注）"""
        if rgb_image is None:
            return

        self.frame_count += 1
        frame_name = f"frame_{self.frame_count:06d}"

        # 保存RGB图像
        rgb_path = os.path.join(self.images_dir, f"{frame_name}.jpg")
        rgb_surface = pygame.surfarray.make_surface(rgb_image.swapaxes(0, 1))
        pygame.image.save(rgb_surface, rgb_path)

        # 保存语义分割图像（如果有）
        if seg_image is not None:
            seg_path = os.path.join(self.semantic_dir, f"{frame_name}.png")
            seg_surface = pygame.surfarray.make_surface(seg_image.swapaxes(0, 1))
            pygame.image.save(seg_surface, seg_path)

        # 保存YOLO格式标注
        label_path = os.path.join(self.labels_dir, f"{frame_name}.txt")
        self._save_yolo_labels(label_path, detections, rgb_image.shape)

        print(f"保存帧 {frame_name}: RGB + {'语义分割 + ' if seg_image is not None else ''}标注")

    def _save_yolo_labels(self, label_path, detections, image_shape):
        """保存YOLO格式标注文件"""
        height, width = image_shape[:2]
        with open(label_path, 'w') as f:
            for sign, conf, bbox in detections:
                x1, y1, x2, y2 = bbox
                # 转换为YOLO格式 (x_center, y_center, width, height) - 归一化
                x_center = (x1 + x2) / 2.0 / width
                y_center = (y1 + y2) / 2.0 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height

                # 类别ID映射：stop sign=0, traffic light=1
                if "stop" in sign.lower():
                    class_id = 0
                else:
                    class_id = 1

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

# 异步检测器：在独立线程中运行YOLO推理，避免阻塞主循环
class AsyncDetector:
    def __init__(self, detect_interval=3, is_custom_model=False):
        self._lock = threading.Lock()
        self._latest_signs = []
        self._image_to_detect = None
        self._running = True
        self._detect_interval = detect_interval  # 每N帧检测一次
        self._frame_count = 0
        self._is_custom_model = is_custom_model  # 是否使用自定义模型
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()

    def _detect_loop(self):
        while self._running:
            img = None
            with self._lock:
                if self._image_to_detect is not None:
                    img = self._image_to_detect.copy()
                    self._image_to_detect = None

            if img is not None:
                results = model.predict(
                    source=img, imgsz=640, conf=0.25,
                    iou=0.45,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    verbose=False,
                    half=torch.cuda.is_available()
                )
                detections = results[0].boxes
                names = results[0].names

                signs = []
                for i in range(len(detections)):
                    cls_id = int(detections.cls[i])
                    conf = float(detections.conf[i])

                    # 自定义模型：所有类别都保留（只有stop_sign和traffic_light）
                    # COCO预训练模型：只保留交通相关类别
                    if not self._is_custom_model and cls_id not in TRAFFIC_RELEVANT_CLASSES:
                        continue

                    x1, y1, x2, y2 = detections.xyxy[i].cpu().numpy().astype(int)
                    label = names[cls_id]
                    signs.append((label, conf, (int(x1), int(y1), int(x2), int(y2))))

                with self._lock:
                    self._latest_signs = signs
            else:
                time.sleep(0.001)

    def should_detect(self):
        """判断当前帧是否应该进行检测（帧跳过逻辑）"""
        self._frame_count += 1
        return self._frame_count % self._detect_interval == 0

    def submit_image(self, image_np):
        with self._lock:
            self._image_to_detect = image_np

    def get_latest_signs(self):
        with self._lock:
            return list(self._latest_signs)

    def stop(self):
        self._running = False

# 计算车辆与目标航点之间的转向角度
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

# 车辆控制器类 - 基于简单的阈值控制和角度计算
class SimpleVehicleController:
    def __init__(self):
        self.target_speed = 30.0  # 目标速度 km/h
        self.prev_steer = 0.0  # 上一次的转向角，用于平滑
    
    def update_control(self, vehicle, waypoint):
        """
        基于waypoint更新车辆控制
        参考EgoVehicleController的简单有效控制方法
        """
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # km/h
        
        # 简单的速度控制：基于阈值
        if speed < self.target_speed:
            throttle = 0.5
            brake = 0.0
        else:
            throttle = 0.0
            brake = 0.1
        
        # 简单的车道保持控制
        if waypoint:
            # 获取下一个航点
            next_waypoint = waypoint.next(5.0)[0]
            if next_waypoint:
                # 计算到下一个航点的角度
                next_location = next_waypoint.transform.location
                angle = math.atan2(next_location.y - transform.location.y,
                                next_location.x - transform.location.x)
                angle = math.degrees(angle) - transform.rotation.yaw
                angle = (angle + 180) % 360 - 180  # 归一化到[-180, 180]
                
                # 基于角度计算转向角，限制在±0.5范围内
                steer = max(-0.5, min(0.5, angle / 90.0))
                
                # 平滑转向角变化
                steer = 0.7 * self.prev_steer + 0.3 * steer
                self.prev_steer = steer
            else:
                steer = self.prev_steer
        else:
            steer = self.prev_steer
        
        return throttle, brake, steer
    
    def set_target_speed(self, speed):
        self.target_speed = speed
    
    def get_target_speed(self):
        return self.target_speed

# 全局车辆控制器
simple_controller = SimpleVehicleController()

# 根据检测到的标志执行操作
def control_vehicle_based_on_sign(vehicle, detected_signs, lights, simulation_time, controller):
    velocity = vehicle.get_velocity()
    current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # m/s转换为km/h
    print(f"当前车辆速度: {current_speed:.2f} km/h")

    traffic_light_state = vehicle.get_traffic_light_state()
    if traffic_light_state == carla.TrafficLightState.Red:
        print("交通灯: 红色 - 停车等待")
        controller.set_target_speed(0)
        return
    elif traffic_light_state == carla.TrafficLightState.Yellow:
        print("交通灯: 黄色 - 减速慢行")
        controller.set_target_speed(15)
        return
    elif traffic_light_state == carla.TrafficLightState.Green:
        print("交通灯: 绿色 - 正常通行")
        controller.set_target_speed(30)

    # 检查检测到的标志，设置目标速度
    for sign, conf, bbox in detected_signs:
        print(f"检测到交通标志: {sign}，置信度 {conf:.2f}")
        if "stop" in sign.lower() and conf > 0.5:
            print("操作: 检测到停止标志！应用完全制动。")
            controller.set_target_speed(0)
            return
        elif "speed limit" in sign.lower():
            digits = [int(s) for s in sign.split() if s.isdigit()]
            if digits:
                speed_limit = digits[0]
                print(f"操作: 将速度调整为 {speed_limit} km/h")
                controller.set_target_speed(speed_limit)
                return
    
    # 如果没有检测到特殊标志，恢复默认速度
    if controller.get_target_speed() == 0:
        controller.set_target_speed(30.0)

# 生成带有红色计时和动态速度限制的交通灯
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
                    print(f"在索引 {i} 处生成了限速 {speed} 标志")
                break

    stop_signs = [bp for bp in blueprint_library if 'static.prop.stop' in bp.id]
    if stop_signs:
        transform = spawn_points[-1]
        transform.location.z = 0
        actor = world.try_spawn_actor(stop_signs[0], transform)
        if actor:
            signs.append(actor)
            print("在末尾生成了停止标志")

    return signs

# 主函数
def main():
    # 解析命令行参数（已包含模型存在性检查）
    args = parse_args()

    # 加载模型
    global model
    print(f"\n{'='*50}")
    print(f"加载模型: {args.model}")
    if 'runs/detect' in args.model:
        print(">>> 使用自定义训练模型 <<<")
    else:
        print(">>> 使用预训练模型 <<<")
    print(f"{'='*50}\n")
    model = YOLO(args.model)

    actor_list = []
    # 判断是否使用自定义模型（非默认yolov8n.pt）
    is_custom = os.path.basename(args.model) != 'yolov8n.pt'
    detector = AsyncDetector(is_custom_model=is_custom)
    collector = DatasetCollector(args.output_dir)
    frame_counter = 0
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        
        # 读取地图配置（由 switch_map.py 保存）
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_config.txt")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                target_map = f.read().strip()
            print(f"正在加载地图: {target_map} ...")
            world = client.load_world(target_map)
        else:
            print("未找到地图配置，使用默认地图 Town03 ...")
            world = client.load_world('Town03')
        
        world.set_weather(carla.WeatherParameters.ClearNoon)
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        # 生成交通元素
        elements = spawn_dynamic_elements(world, blueprint_library)
        actor_list.extend(elements)

        # 生成车辆
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_point = random.choice(map.get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print(f"车辆生成位置: {spawn_point.location}")

        # 生成随机交通
        for _ in range(10):
            traffic_bp = random.choice(blueprint_library.filter('vehicle.*'))
            traffic_spawn = random.choice(map.get_spawn_points())
            traffic_vehicle = world.try_spawn_actor(traffic_bp, traffic_spawn)
            if traffic_vehicle:
                traffic_vehicle.set_autopilot(True)
                actor_list.append(traffic_vehicle)

        # RGB相机设置
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        # 语义分割相机设置
        seg_camera_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        seg_camera_bp.set_attribute("image_size_x", "800")
        seg_camera_bp.set_attribute("image_size_y", "600")
        seg_camera_bp.set_attribute("fov", "90")
        seg_camera = world.spawn_actor(seg_camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(seg_camera)

        # 设置Pygame显示
        display = init_pygame(800, 600)

        image_surface = [None]
        seg_surface = [None]

        def image_callback(image):
            image_surface[0] = process_image(image)

        def seg_callback(image):
            # 语义分割图像转换（CARLA输出的是类别ID，需要转换为彩色）
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            # 提取语义标签（R通道）
            semantic_labels = array[:, :, 2]  # CARLA语义分割存储在BGR的R通道

            # 简单的颜色映射（可以根据需要扩展）
            color_map = {
                0: (0, 0, 0),        # Unlabeled - 黑色
                1: (0, 0, 255),      # Building - 蓝色
                2: (0, 255, 0),      # Fence - 绿色
                3: (255, 255, 255),  # Other - 白色
                4: (255, 0, 0),      # Pedestrian - 红色
                5: (255, 255, 0),    # Pole - 黄色
                6: (0, 255, 255),    # RoadLine - 青色
                7: (128, 64, 128),   # Road - 灰紫色
                8: (64, 0, 128),     # SideWalk - 紫色
                9: (64, 64, 0),      # Vegetation - 橄榄色
                10: (0, 128, 192),   # Vehicles - 深青色
                11: (0, 0, 128),     # Wall - 深红色
                12: (192, 128, 128), # TrafficSign - 浅蓝色
                13: (0, 128, 128),   # Sky - 灰青色
                14: (128, 128, 128), # Ground - 灰色
                15: (64, 0, 64),     # Bridge - 深紫色
                16: (64, 64, 64),    # RailTrack - 深灰色
                17: (128, 0, 128),   # GuardRail - 紫红色
                18: (192, 192, 128), # TrafficLight - 浅黄色
                19: (0, 0, 192),     # Static - 深蓝色
                20: (128, 128, 0),   # Dynamic - 橄榄色
            }

            # 创建彩色图像
            colored = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            for label, color in color_map.items():
                mask = semantic_labels == label
                colored[mask] = color

            seg_surface[0] = colored

        camera.listen(image_callback)
        seg_camera.listen(seg_callback)

        spectator = world.get_spectator()
        def update_spectator():
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(z=50),
                carla.Rotation(pitch=-90)
            ))

        clock = pygame.time.Clock()
        start_time = time.time()
        fps_font = pygame.font.Font(None, 20)

        while True:
            # 推进模拟世界一步，触发传感器回调（相机图像）
            world.tick()

            update_spectator()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
                # 手动保存当前帧（按S键）
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    if image_surface[0] is not None:
                        detected_signs = detector.get_latest_signs()
                        collector.save_frame(image_surface[0], seg_surface[0], detected_signs)

            transform = vehicle.get_transform()
            waypoint = map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)

            # 使用简单控制器计算控制
            throttle, brake, steer = simple_controller.update_control(vehicle, waypoint)

            # 应用控制
            control = carla.VehicleControl()
            control.throttle = throttle
            control.steer = steer
            control.brake = brake
            vehicle.apply_control(control)

            if image_surface[0] is not None:
                # 帧跳过逻辑：只在需要时提交图像给检测器
                if detector.should_detect():
                    detector.submit_image(image_surface[0])

                # 获取最新的检测结果（非阻塞，立即返回）
                detected_signs = detector.get_latest_signs()

                # 帧间隔自动保存
                frame_counter += 1
                if frame_counter % args.save_interval == 0:
                    collector.save_frame(image_surface[0], seg_surface[0], detected_signs)

                # 获取红绿灯状态
                traffic_light_state = vehicle.get_traffic_light_state()

                # 基于检测结果控制车辆
                simulation_time = time.time() - start_time
                control_vehicle_based_on_sign(vehicle, detected_signs, world.get_actors().filter("traffic.traffic_light"), simulation_time, simple_controller)

                # 绘制检测结果
                surface = draw_detections(image_surface[0], detected_signs)
                # 绘制车辆状态面板（包含红绿灯状态）
                draw_vehicle_status(surface, vehicle, detected_signs, traffic_light_state)
                # 绘制FPS
                fps_text = f"FPS: {clock.get_fps():.1f}"
                fps_surface = fps_font.render(fps_text, True, (255, 255, 0))
                surface.blit(fps_surface, (10, 10))
                # 绘制数据采集状态
                status_text = f"Frames: {collector.frame_count} | Press 'S' to save"
                status_surface = fps_font.render(status_text, True, (255, 255, 255))
                surface.blit(status_surface, (10, 30))
                display.blit(surface, (0, 0))
            else:
                # 等待相机图像时显示提示
                display.fill((0, 0, 0))
                font = pygame.font.Font(None, 36)
                text = font.render("等待相机图像...", True, (255, 255, 255))
                display.blit(text, (display.get_width()//2 - 100, display.get_height()//2))

            pygame.display.flip()
            clock.tick(30)

            if time.time() - start_time > 120:
                print("2分钟已过，停止模拟。")
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                break

    finally:
        detector.stop()
        print("清理actors...")
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print("完成。")

if __name__ == "__main__":
    main()