import carla
import random
import time
import pygame
import numpy as np
import math
from ultralytics import YOLO
import torch

# 初始化Pygame显示窗口
def init_pygame(width, height):
    pygame.init()
    display = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Driver's View")
    return display

# 转换CARLA图像为RGB numpy数组
def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]  # 丢弃alpha通道
    return array

# 加载YOLOv8模型，检测交通标志
model = YOLO("yolov8n.pt")  # 轻量级模型，速度快
# 检测来自CARLA摄像头的RGB数码图像
def detect_traffic_signs(image_np):
    # 模型检测：自动适配GPU/CPU，置信度0.5过滤
    results = model.predict(source=image_np, imgsz=640, conf=0.5, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False)
    detections = results[0].boxes.data.cpu().numpy()
    names = results[0].names

    # 解析检测结果：(标志类别, 置信度, 检测框坐标)
    signs_detected = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = names[int(cls)]
        signs_detected.append((label, conf, (int(x1), int(y1), int(x2), int(y2))))
    return signs_detected

# 计算车辆到目标路点的转向角度
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
    angle = math.acos(dot / (norm_dir * norm_fwd + 1e-5))# 避免除零错误
    cross = v_forward.x * direction.y - v_forward.y * direction.x
    if cross < 0:
        angle *= -1
    return angle

# 根据检测到的标志/红绿灯控制车辆
def control_vehicle_based_on_sign(vehicle, detected_signs, lights, simulation_time):
    # 计算当前车速（m/s → km/h）
    velocity = vehicle.get_velocity()
    current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # m/s to km/h
    print(f"当前车辆速度: {current_speed:.2f} km/h")

    traffic_light_state = vehicle.get_traffic_light_state()
    if traffic_light_state == carla.TrafficLightState.Red:
        print("红绿灯：红色 - 刹车中")
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        return

    # 处理STOP标志
    for sign, conf, bbox in detected_signs:
        print(f"检测到交通标志: {sign} ,置信度： {conf:.2f}")
        if "stop" in sign.lower() and conf > 0.5:
            print("行动：发现停车标志，踩满刹车")
            control = carla.VehicleControl()
            control.brake = 1.0
            vehicle.apply_control(control)
            time.sleep(2)# 停留2秒
        # 处理限速标志
        elif "speed limit" in sign.lower():
            digits = [int(s) for s in sign.split() if s.isdigit()]
            if digits:
                speed_limit = digits[0]
                print(f"动作：调整速度至{speed_limit} km/h")
                desired_speed = speed_limit * 1000 / 3600
                if current_speed < speed_limit:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0))
                else:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.5))

# 生成动态交通标志
def spawn_dynamic_elements(world, blueprint_library):
    spawn_points = world.get_map().get_spawn_points()
    signs = []
    speed_values = [20, 40, 60, 60, 40, 60, 40, 20]

    # 筛选限速/STOP标志蓝图
    sign_bp = [bp for bp in blueprint_library if 'static.prop.speedlimit' in bp.id or 'static.prop.stop' in bp.id]

    # 生成限速标志
    for i, speed in enumerate(speed_values):
        for bp in sign_bp:
            if f"speedlimit.{speed}" in bp.id:
                transform = spawn_points[i % len(spawn_points)]
                transform.location.z = 0
                actor = world.try_spawn_actor(bp, transform)
                if actor:
                    signs.append(actor)
                    print(f"索引{i}处生成的限速{speed}标志")
                break

    # 生成STOP标志
    stop_signs = [bp for bp in blueprint_library if 'static.prop.stop' in bp.id]
    if stop_signs:
        transform = spawn_points[-1]
        transform.location.z = 0
        actor = world.try_spawn_actor(stop_signs[0], transform)
        if actor:
            signs.append(actor)
            print("生成了停止标志")

    return signs

# 主函数
def main():
    actor_list = []
    try:
        # 连接CARLA模拟器
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        print("连接CARLA模拟器成功")

        # 生成交通标志
        elements = spawn_dynamic_elements(world, blueprint_library)
        actor_list.extend(elements)

        # 生成主车辆 特斯拉Model3
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_point = random.choice(map.get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print(f"车辆生成于: {spawn_point.location}")

        # 生成10辆随机交通车并开启自动行驶
        for _ in range(10):
            traffic_bp = random.choice(blueprint_library.filter('vehicle.*'))
            traffic_spawn = random.choice(map.get_spawn_points())
            traffic_vehicle = world.try_spawn_actor(traffic_bp, traffic_spawn)
            if traffic_vehicle:
                traffic_vehicle.set_autopilot(True)
                actor_list.append(traffic_vehicle)

        # 挂载RGB摄像头到主车辆
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        # 初始化Pygame窗口
        display = init_pygame(800, 600)

        # 摄像头回调：接收并转换图像
        image_surface = [None]
        def image_callback(image):
            image_surface[0] = process_image(image)
        camera.listen(image_callback)
        print("摄像头挂载完成")

        clock = pygame.time.Clock()
        start_time = time.time()

        # 实时显示
        while True:
            # 窗口退出逻辑
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            # 车辆自动转向控制
            transform = vehicle.get_transform()
            waypoint = map.get_waypoint(transform.location, project_to_road=True,lane_type=carla.LaneType.Driving)
            next_waypoint = waypoint.next(2.0)[0]  # 下一个2米的路点
            angle = get_steering_angle(transform, next_waypoint.transform)
            steer = max(-1.0, min(1.0, angle * 2.0))  # 限制转向范围

            # 应用车辆控制：默认油门0.3，自动转向
            control = carla.VehicleControl()
            control.throttle = 0.3
            control.steer = steer
            control.brake = 0.0
            vehicle.apply_control(control)

            # 检测交通标志并控车
            if image_surface[0] is not None:
                detected_signs = detect_traffic_signs(image_surface[0])
                simulation_time = time.time() - start_time
                control_vehicle_based_on_sign(vehicle, detected_signs,world.get_actors().filter("traffic.traffic_light"), simulation_time)
                # 渲染摄像头画面到Pygame窗口
                surface = pygame.image.frombuffer(image_surface[0].tobytes(), (800, 600), "RGB")
                display.blit(surface, (0, 0))
                pygame.display.flip()

            # 限制帧率30FPS
            clock.tick(30)

    finally:
        # 清理Actor
        print("清理actors")
        for actor in actor_list:
            actor.destroy()
        print("完成.")

if __name__ == "__main__":
    main()