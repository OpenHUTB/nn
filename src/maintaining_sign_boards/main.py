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
    array = array.reshape((image.height, image.width, 4))  # CARLA原始格式：BGRA
    array = array[:, :, :3]  # 丢弃Alpha通道，得到BGR
    array = array[:, :, ::-1]  # BGR转RGB
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


# 检测前方障碍物（车辆）并返回距离
def detect_front_obstacle(vehicle, world, max_distance=50.0, angle_threshold=35.0):
    """
    检测车辆前方是否有障碍物
    :param vehicle: 主车
    :param world: CARLA世界
    :param max_distance: 最大检测距离（米）
    :param angle_threshold: 检测角度范围（左右各多少度）
    :return: 最近障碍物的距离（如果没有则返回None）
    """
    # 获取主车的位置和朝向
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_forward = vehicle_transform.get_forward_vector()

    # 获取世界中所有其他车辆
    all_vehicles = world.get_actors().filter('vehicle.*')

    min_distance = None

    for target_vehicle in all_vehicles:
        # 跳过自己
        if target_vehicle.id == vehicle.id:
            continue

        # 计算目标车辆相对于主车的位置
        target_location = target_vehicle.get_transform().location
        distance = vehicle_location.distance(target_location)

        # 如果距离超过最大检测范围，跳过
        if distance > max_distance:
            continue

        # 计算目标车辆相对于主车的方向向量
        direction_vector = target_location - vehicle_location
        direction_vector = carla.Vector3D(direction_vector.x, direction_vector.y, 0.0)  # 忽略Z轴
        direction_vector_norm = math.sqrt(direction_vector.x ** 2 + direction_vector.y ** 2)

        if direction_vector_norm < 1e-5:
            continue

        # 计算方向向量与主车前进方向的夹角
        dot = vehicle_forward.x * direction_vector.x + vehicle_forward.y * direction_vector.y
        angle_rad = math.acos(dot / (direction_vector_norm + 1e-5))
        angle_deg = math.degrees(angle_rad)

        # 如果目标在前方角度范围内
        if angle_deg < angle_threshold:
            if min_distance is None or distance < min_distance:
                min_distance = distance

    return min_distance

# 根据检测到的标志/红绿灯控制车辆
def control_vehicle_based_on_sign(vehicle, detected_signs, lights, simulation_time,base_control,world):
    # 计算当前车速（m/s → km/h）
    velocity = vehicle.get_velocity()
    current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # m/s to km/h
    print(f"当前车辆速度: {current_speed:.2f} km/h")

    # 障碍物检测
    obstacle_distance = detect_front_obstacle(vehicle, world)
    if obstacle_distance is not None:
        print(f"前方障碍物距离: {obstacle_distance:.2f} 米")
        # 重新计算安全距离：车速越快，距离越长，完全避免追尾
        safe_distance = current_speed * 0.6 + 10.0  # 基础10米+车速补偿

        # 分级刹车
        if obstacle_distance < safe_distance:
            print("行动：前方有障碍物，分级减速")
            # 立刻收油门，切断动力
            base_control.throttle = 0.0

            if obstacle_distance < safe_distance * 0.3:
                # 极近：满刹
                base_control.brake = 1.0
            elif obstacle_distance < safe_distance * 0.6:
                # 较近：重刹
                base_control.brake = 0.7
            else:
                # 较远：轻刹减速
                base_control.brake = 0.3
            return base_control

    traffic_light_state = vehicle.get_traffic_light_state()
    if traffic_light_state == carla.TrafficLightState.Red:
        print("红绿灯：红色 - 刹车中")
        if current_speed > 1.0:  # 如果速度大于1km/h，轻踩刹车
            base_control.throttle = 0.0
            base_control.brake = min(0.8, current_speed / 50)  # 速度越快，刹车越重（最大0.8）
        else:  # 速度很慢时，踩死刹车停稳
            base_control.throttle = 0.0
            base_control.brake = 1.0
        return base_control

    # 处理STOP标志
    for sign, conf, bbox in detected_signs:
        print(f"检测到交通标志: {sign} ,置信度： {conf:.2f}")
        if "stop" in sign.lower() and conf > 0.5:
            if current_speed > 1.0:
                print("行动：发现停车标志，平滑减速")
                base_control.throttle = 0.0
                base_control.brake = min(0.8, current_speed / 50)  # 平滑减速
            else:
                print("行动：车辆停稳，等待2秒")
                base_control.throttle = 0.0
                base_control.brake = 1.0
                time.sleep(2)
            return base_control
        # 处理限速标志
        elif "speed limit" in sign.lower():
            digits = [int(s) for s in sign.split() if s.isdigit()]
            if digits:
                speed_limit = digits[0]
                print(f"动作：调整速度至{speed_limit} km/h")
                speed_error = current_speed - speed_limit
                if speed_error > 5:  # 超速较多，轻踩刹车
                    base_control.throttle = 0.0
                    base_control.brake = min(0.5, speed_error / 50)
                elif speed_error < -5:  # 太慢了，轻踩油门
                    base_control.throttle = min(0.5, -speed_error / 50)
                    base_control.brake = 0.0
                else:  # 接近限速，保持
                    base_control.throttle = 0.3
                    base_control.brake = 0.0
                return base_control

    # 如果没有任何标志/红灯，返回基础控制
    return base_control


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
        actors_to_destroy = []
        vehicle_actors = world.get_actors().filter('vehicle.*')
        for actor in vehicle_actors:
            actors_to_destroy.append(actor)
        walker_actors = world.get_actors().filter('walker.pedestrian.*')
        for actor in walker_actors:
            actors_to_destroy.append(actor)
        for actor in actors_to_destroy:
            actor.destroy()

        # 生成交通标志
        elements = spawn_dynamic_elements(world, blueprint_library)
        actor_list.extend(elements)

        # 生成主车辆 特斯拉Model3
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

        fixed_location = carla.Location(x=105.906349, y=67.419144, z=0.5)
        fixed_waypoint = map.get_waypoint(fixed_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if fixed_waypoint:
            fixed_rotation = fixed_waypoint.transform.rotation
            spawn_point = carla.Transform(fixed_location, fixed_rotation)
            print("使用固定生成点（已对齐车道方向）")
        else:
            all_spawn_points = map.get_spawn_points()
            spawn_point = random.choice(all_spawn_points)
            print("无法获取固定点的道路方向，使用随机生成点")

        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if not vehicle:
            all_spawn_points = map.get_spawn_points()
            spawn_point = random.choice(all_spawn_points)
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            print("固定点生成失败，使用随机生成点重试")

        actor_list.append(vehicle)
        print(f"车辆最终生成于: {spawn_point.location}")

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

        pygame.font.init()
        hud_font = pygame.font.SysFont("Arial", 20)  # 使用Arial字体，大小20

        # 摄像头回调：接收并转换图像
        image_surface = [None]
        def image_callback(image):
            image_surface[0] = process_image(image)
        camera.listen(image_callback)
        print("摄像头挂载完成")

        clock = pygame.time.Clock()
        start_time = time.time()
        last_steer = 0.0

        # 实时显示
        while True:
            # 窗口退出逻辑
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            # 车辆自动转向控制
            transform = vehicle.get_transform()
            waypoint = map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            next_waypoint = waypoint.next(8.0)[0]
            angle = get_steering_angle(transform, next_waypoint.transform)

            # 减少打方向幅度
            raw_steer = angle * 0.7

            # 忽略微小抖动
            if abs(raw_steer) < 0.08:
                raw_steer = 0.0

            # 平滑滤波
            smooth_steer = 0.2 * raw_steer + 0.8 * last_steer

            # 每帧转向最大变化量限制（防止摆动累积放大）
            steer_delta = smooth_steer - last_steer
            steer_delta = np.clip(steer_delta, -0.07, 0.07)
            steer = last_steer + steer_delta

            steer = np.clip(steer, -0.7, 0.7)  # 方向不打满，更稳定

            # 保存上一帧角度
            last_steer = steer

            # 应用车辆控制：默认油门0.3，自动转向
            final_control = carla.VehicleControl()
            final_control.throttle = 0.3
            final_control.steer = steer
            final_control.brake = 0.0

            # 检测交通标志并控车
            if image_surface[0] is not None:
                detected_signs = detect_traffic_signs(image_surface[0])
                simulation_time = time.time() - start_time
                # 传入基础控制，得到最终控制
                final_control = control_vehicle_based_on_sign(
                    vehicle,
                    detected_signs,
                    world.get_actors().filter("traffic.traffic_light"),
                    simulation_time,
                    final_control,
                    world
                )
                # 渲染摄像头画面到Pygame窗口
                surface = pygame.image.frombuffer(image_surface[0].tobytes(), (800, 600), "RGB")
                display.blit(surface, (0, 0))

                # 1. 显示当前车速
                velocity = vehicle.get_velocity()
                current_speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6
                speed_text = hud_font.render(f"Speed: {current_speed:.1f} km/h", True, (255, 255, 255))
                display.blit(speed_text, (10, 10))

                # 2. 显示前方障碍物距离
                obstacle_distance = detect_front_obstacle(vehicle, world)
                if obstacle_distance:
                    obstacle_text = hud_font.render(f"Obstacle: {obstacle_distance:.1f} m", True, (255, 0, 0))
                else:
                    obstacle_text = hud_font.render("Obstacle: None", True, (0, 255, 0))
                display.blit(obstacle_text, (10, 40))

                # 3. 显示检测到的交通标志
                if detected_signs:
                    sign_label = detected_signs[0][0]  # 取第一个检测到的标志
                    sign_text = hud_font.render(f"Sign: {sign_label}", True, (255, 255, 0))
                else:
                    sign_text = hud_font.render("Sign: None", True, (200, 200, 200))
                display.blit(sign_text, (10, 70))

                pygame.display.flip()

            # 统一执行最终控制指令
            vehicle.apply_control(final_control)

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