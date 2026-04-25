import carla
import random
import time
import pygame
import numpy as np
import math
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
def draw_vehicle_status(surface, vehicle, detected_signs):
    # 使用默认字体
    font = pygame.font.Font(None, 16)
    
    status_width = 220
    status_height = 180
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

# 加载YOLOv8预训练模型用于交通标志检测
model = YOLO("yolov8n.pt")  # 使用yolov8n.pt进行快速推理

# 在来自CARLA相机的RGB numpy图像上运行检测
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

# 根据检测到的标志执行操作
def control_vehicle_based_on_sign(vehicle, detected_signs, lights, simulation_time):
    velocity = vehicle.get_velocity()
    current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # m/s转换为km/h
    print(f"当前车辆速度: {current_speed:.2f} km/h")

    traffic_light_state = vehicle.get_traffic_light_state()
    if traffic_light_state == carla.TrafficLightState.Red:
        print("交通灯: 红色 - 应用制动")
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        return

    for sign, conf, bbox in detected_signs:
        print(f"检测到交通标志: {sign}，置信度 {conf:.2f}")
        if "stop" in sign.lower() and conf > 0.5:
            print("操作: 检测到停止标志！应用完全制动。")
            control = carla.VehicleControl()
            control.brake = 1.0
            vehicle.apply_control(control)
            time.sleep(2)
        elif "speed limit" in sign.lower():
            digits = [int(s) for s in sign.split() if s.isdigit()]
            if digits:
                speed_limit = digits[0]
                print(f"操作: 将速度调整为 {speed_limit} km/h")
                desired_speed = speed_limit * 1000 / 3600
                if current_speed < speed_limit:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0))
                else:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.5))

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
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
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

        while True:
            update_spectator()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            transform = vehicle.get_transform()
            waypoint = map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            next_waypoint = waypoint.next(2.0)[0]
            angle = get_steering_angle(transform, next_waypoint.transform)
            steer = max(-1.0, min(1.0, angle * 2.0))

            control = carla.VehicleControl()
            control.throttle = 0.5
            control.steer = steer
            control.brake = 0.0
            vehicle.apply_control(control)

            if image_surface[0] is not None:
                detected_signs = detect_traffic_signs(image_surface[0])
                simulation_time = time.time() - start_time
                control_vehicle_based_on_sign(vehicle, detected_signs, world.get_actors().filter("traffic.traffic_light"), simulation_time)

                # 绘制检测结果
                surface = draw_detections(image_surface[0], detected_signs)
                # 绘制车辆状态面板
                draw_vehicle_status(surface, vehicle, detected_signs)
                display.blit(surface, (0, 0))
                pygame.display.flip()

            clock.tick(30)

            if time.time() - start_time > 120:
                print("2分钟已过，停止模拟。")
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                break

    finally:
        print("清理actors...")
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print("完成。")

if __name__ == "__main__":
    main()