import carla
import random
import time
import pygame
import numpy as np
import math
import os
import threading
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

# 加载YOLOv8预训练模型用于交通标志检测
model = YOLO("yolov8n.pt")

# COCO数据集中交通相关类别: stop sign(11), traffic light(9)
TRAFFIC_RELEVANT_CLASSES = {9, 11}

# 异步检测器：在独立线程中运行YOLO推理，避免阻塞主循环
class AsyncDetector:
    def __init__(self, detect_interval=3):
        self._lock = threading.Lock()
        self._latest_signs = []
        self._image_to_detect = None
        self._running = True
        self._detect_interval = detect_interval  # 每N帧检测一次
        self._frame_count = 0
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
                    if cls_id not in TRAFFIC_RELEVANT_CLASSES:
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
    actor_list = []
    detector = AsyncDetector()
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

        # 设置Pygame显示
        display = init_pygame(800, 600)

        image_surface = [None]
        def image_callback(image):
            image_surface[0] = process_image(image)
        camera.listen(image_callback)

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