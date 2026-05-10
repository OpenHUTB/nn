import carla
import random
import time
import pygame
import numpy as np
import math

# 核心配置（重点优化红绿灯相关参数）
CONFIG = {
    "CARLA_HOST": "localhost",
    "CARLA_PORT": 2000,
    "CAMERA_WIDTH": 800,
    "CAMERA_HEIGHT": 600,
    "SAFE_STOP_DISTANCE": 15,
    "MIN_STOP_DISTANCE": 3,
    "DETECTION_CONF": 0.65,
    "DEFAULT_CRUISE_SPEED": 40,
    "INTERSECTION_SPEED": 25,
    "SPEED_ADJUST_SMOOTH": 0.3,
    # ========== 新增：高速防失控核心参数 ==========
    "STEER_SMOOTH_FACTOR": 0.8,  # 转向平滑系数，越大越稳
    "MAX_STEER_CHANGE": 0.08,  # 每帧最大转向变化量，防止猛打方向
    "BASE_PREVIEW_DISTANCE": 3.0,  # 基础预瞄距离
    "MAX_PREVIEW_DISTANCE": 10.0,  # 最大预瞄距离（高速用）
    "STEER_DEAD_ZONE": 0.03,  # 转向死区，小误差不修正，防止高频抖动
    "MAX_THROTTLE": 0.4,  # 最大油门限制，防止急加速打滑
    "MIN_TIRE_FRICTION": 2.5,  # 轮胎摩擦系数，提升抓地力防打滑
    "CAMERA_SMOOTH_FACTOR": 0.15  # 视角平滑系数
}

# ================== 全局变量（新增防失控相关缓存） ==================
need_vehicle_reset = False
current_speed_limit = CONFIG["DEFAULT_CRUISE_SPEED"]
current_steer = 0.0
smooth_camera_pos = None
# 新增：油门平滑缓存
current_throttle = 0.0


# ================== 基础初始化函数（完全保留） ==================
def init_pygame(width, height):
    pygame.init()
    display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA V6.0 防失控稳定版")
    return display


def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3].copy()
    return array


# ================== YOLO检测函数（完全保留） ==================
model = YOLO("yolov8n.pt")
TRAFFIC_DETECT_CLASSES = {
    9: "stop sign",
    8: "traffic light",
    1: "bicycle",
    0: "person",
    2: "car"
}


def detect_traffic_elements(image_np):
    global current_speed_limit
    results = model.predict(
        source=image_np,
        imgsz=640,
        conf=CONFIG["DETECTION_CONF"],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=False,
        classes=list(TRAFFIC_DETECT_CLASSES.keys())
    )
    detections = results[0].boxes.data.cpu().numpy()
    names = results[0].names

    detected_list = []
    traffic_light_state = None
    detected_speed_limit = None

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = names[int(cls)]
        if label == "traffic light":
            roi = image_np[int(y1):int(y2), int(x1):int(x2)]
            if roi.size != 0:
                red_mean = np.mean(roi[:, :, 0])
                green_mean = np.mean(roi[:, :, 1])
                if red_mean > green_mean + 15:
                    traffic_light_state = "Red"
                elif green_mean > red_mean + 15:
                    traffic_light_state = "Green"
            detected_list.append((label, traffic_light_state, conf, (int(x1), int(y1), int(x2), int(y2))))
        elif "speed limit" in label.lower():
            digits = [int(s) for s in label.split() if s.isdigit()]
            if digits:
                detected_speed_limit = digits[0]
                current_speed_limit = detected_speed_limit
                print(f"【V6.0 限速管控】检测到限速标志：{detected_speed_limit} km/h")
            detected_list.append((label, detected_speed_limit, conf, (int(x1), int(y1), int(x2), int(y2))))
        else:
            detected_list.append((label, None, conf, (int(x1), int(y1), int(x2), int(y2))))

    if detected_speed_limit is None:
        current_speed_limit = CONFIG["DEFAULT_CRUISE_SPEED"]

    return detected_list, traffic_light_state


# ================== 工具函数 ==================
def get_speed(vehicle):
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6


# ========== 修复：速度自适应转向角计算，核心解决高速转向失控 ==========
def get_steer(vehicle_transform, waypoint_transform, current_speed):
    v_loc = vehicle_transform.location
    v_forward = vehicle_transform.get_forward_vector()
    wp_loc = waypoint_transform.location

    direction = carla.Vector3D(wp_loc.x - v_loc.x, wp_loc.y - v_loc.y, 0.0)
    v_forward = carla.Vector3D(v_forward.x, v_forward.y, 0.0)

    dir_norm = math.hypot(direction.x, direction.y)
    fwd_norm = math.hypot(v_forward.x, v_forward.y)
    if dir_norm < 1e-5 or fwd_norm < 1e-5:
        return 0.0

    dot = (v_forward.x * direction.x + v_forward.y * direction.y) / (dir_norm * fwd_norm)
    dot = max(-1.0, min(1.0, dot))
    angle = math.acos(dot)
    cross = v_forward.x * direction.y - v_forward.y * direction.x
    if cross < 0:
        angle *= -1

    # ========== 核心防失控：速度越快，转向灵敏度越低 ==========
    # 速度超过20km/h后，转向增益线性衰减，60km/h时转向灵敏度只剩20%
    speed_gain = max(0.2, 1.0 - (current_speed / 60) * 0.8)
    final_steer = angle * 1.0 * speed_gain

    # 限制最大转向角度，速度越快，能打的方向越小
    max_steer_angle = max(0.1, 0.8 - (current_speed / 100) * 0.7)
    return max(-max_steer_angle, min(max_steer_angle, final_steer))


def get_distance_to_intersection(vehicle, map):
    vehicle_loc = vehicle.get_transform().location
    waypoint = map.get_waypoint(vehicle_loc, project_to_road=True)
    check_distance = 0
    current_wp = waypoint
    for _ in range(50):
        next_wps = current_wp.next(2.0)
        if not next_wps:
            break
        current_wp = next_wps[0]
        check_distance += 2.0
        if current_wp.is_junction or len(current_wp.next(2.0)) > 1:
            return check_distance
    return 999


# ================== 碰撞回调函数（完全保留） ==================
def on_collision(event):
    global need_vehicle_reset, current_steer, current_throttle
    need_vehicle_reset = True
    collision_force = event.normal_impulse.length()
    print(f"【V4.0 碰撞保护】检测到碰撞！强度：{collision_force:.1f}，准备重置车辆")
    # 碰撞后重置控制缓存
    current_steer = 0.0
    current_throttle = 0.0


# ========== 新增：车辆物理参数优化，底层解决加速打滑 ==========
def optimize_vehicle_physics(vehicle):
    physics_control = vehicle.get_physics_control()
    # 提升轮胎摩擦系数，防止打滑
    for wheel in physics_control.wheels:
        wheel.tire_friction = CONFIG["MIN_TIRE_FRICTION"]
    # 优化转向曲线，速度越快转向越轻
    physics_control.steering_curve = [
        carla.Vector2D(x=0, y=1.0),
        carla.Vector2D(x=50, y=0.5),
        carla.Vector2D(x=100, y=0.2)
    ]
    # 优化电车扭矩曲线，防止起步扭矩爆炸
    physics_control.torque_curve = [
        carla.Vector2D(x=0, y=300),
        carla.Vector2D(x=1000, y=400),
        carla.Vector2D(x=3000, y=200)
    ]
    # 降低换挡时间，适配电车特性
    physics_control.gear_switch_time = 0.01
    # 增加车身重量，提升高速稳定性
    physics_control.mass = 1800
    # 应用优化后的物理参数
    vehicle.apply_physics_control(physics_control)
    print("【防失控优化】车辆物理参数已优化，抓地力与稳定性提升")


# ================== 主函数 ==================
def main():
    global need_vehicle_reset, current_speed_limit, current_steer, smooth_camera_pos, current_throttle
    actor_list = []
    try:
        # 连接CARLA
        client = carla.Client(CONFIG["CARLA_HOST"], CONFIG["CARLA_PORT"])
        client.set_timeout(10.0)
        world = client.get_world()
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        # 生成主车
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_point = random.choice(map.get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print("主车生成成功")

        # ========== 新增：优化车辆物理，底层解决加速打滑 ==========
        optimize_vehicle_physics(vehicle)

        # 挂载碰撞传感器
        collision_bp = blueprint_library.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        collision_sensor.listen(on_collision)
        actor_list.append(collision_sensor)

        # 生成背景车辆
        traffic_count = random.randint(10, 15)
        spawned_traffic = 0
        for _ in range(traffic_count):
            traffic_bp = random.choice(blueprint_library.filter('vehicle.*'))
            traffic_spawn = random.choice(map.get_spawn_points())
            traffic_vehicle = world.try_spawn_actor(traffic_bp, traffic_spawn)
            if traffic_vehicle:
                traffic_vehicle.set_autopilot(True)
                actor_list.append(traffic_vehicle)
                spawned_traffic += 1
        print(f"生成背景车辆：{spawned_traffic}辆")

        # 生成限速标志
        speed_signs = []
        speed_values = [20, 30, 40, 50, 60]
        sign_bp_list = [bp for bp in blueprint_library if 'static.prop.speedlimit' in bp.id]
        for i, speed in enumerate(speed_values):
            target_bp = next((bp for bp in sign_bp_list if f"speedlimit.{speed}" in bp.id), None)
            if target_bp:
                spawn_point = map.get_spawn_points()[i * 3 % len(map.get_spawn_points())]
                spawn_point.location.z = 1.5
                sign_actor = world.try_spawn_actor(target_bp, spawn_point)
                if sign_actor:
                    speed_signs.append(sign_actor)
                    actor_list.append(sign_actor)
                    print(f"【V6.0 限速管控】生成{speed}km/h限速标志")

        # 生成车载摄像头
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(CONFIG["CAMERA_WIDTH"]))
        camera_bp.set_attribute("image_size_y", str(CONFIG["CAMERA_HEIGHT"]))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        # 摄像头回调
        image_surface = [None]

        def image_callback(image):
            image_surface[0] = process_image(image)

        camera.listen(image_callback)

        # 初始化显示与字体
        display = init_pygame(CONFIG["CAMERA_WIDTH"], CONFIG["CAMERA_HEIGHT"])
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 22, bold=True)

        # 平滑第三人称视角
        spectator = world.get_spectator()

        def update_spectator():
            global smooth_camera_pos
            transform = vehicle.get_transform()
            target_pos = transform.location + transform.get_forward_vector() * -10 + carla.Location(z=8)
            target_rot = carla.Rotation(pitch=-15, yaw=transform.rotation.yaw, roll=0)

            if smooth_camera_pos is None:
                smooth_camera_pos = target_pos
            else:
                smooth_camera_pos.x = smooth_camera_pos.x * (1 - CONFIG["CAMERA_SMOOTH_FACTOR"]) + target_pos.x * \
                                      CONFIG["CAMERA_SMOOTH_FACTOR"]
                smooth_camera_pos.y = smooth_camera_pos.y * (1 - CONFIG["CAMERA_SMOOTH_FACTOR"]) + target_pos.y * \
                                      CONFIG["CAMERA_SMOOTH_FACTOR"]
                smooth_camera_pos.z = smooth_camera_pos.z * (1 - CONFIG["CAMERA_SMOOTH_FACTOR"]) + target_pos.z * \
                                      CONFIG["CAMERA_SMOOTH_FACTOR"]

            spectator.set_transform(carla.Transform(smooth_camera_pos, target_rot))

        # 主循环
        while True:
            # 仅保留退出逻辑
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            update_spectator()
            control = carla.VehicleControl()
            current_speed = get_speed(vehicle)
            vehicle_transform = vehicle.get_transform()

            # 碰撞重置逻辑（完全保留）
            if need_vehicle_reset:
                control.throttle = 0.0
                control.brake = 1.0
                control.steer = 0.0
                vehicle.apply_control(control)
                time.sleep(1)

                new_spawn_point = random.choice(map.get_spawn_points())
                vehicle.set_transform(new_spawn_point)
                vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))

                need_vehicle_reset = False
                current_speed_limit = CONFIG["DEFAULT_CRUISE_SPEED"]
                smooth_camera_pos = None
                current_steer = 0.0
                current_throttle = 0.0
                print(f"【V4.0 碰撞保护】车辆已重置到新位置：{new_spawn_point.location}")
                continue

            # 图像检测
            traffic_light_state = None
            detected_list = []
            if image_surface[0] is not None:
                detected_list, traffic_light_state = detect_traffic_elements(image_surface[0])

            # 红绿灯逻辑（完全保留）
            native_light_state = vehicle.get_traffic_light_state().name
            final_light_state = traffic_light_state if traffic_light_state else native_light_state
            distance_to_intersection = get_distance_to_intersection(vehicle, map)
            should_stop = False

            if final_light_state == "Red":
                dynamic_stop_distance = CONFIG["SAFE_STOP_DISTANCE"] + (current_speed / 10)
                if distance_to_intersection < dynamic_stop_distance:
                    should_stop = True

            # 停车逻辑
            if should_stop:
                if distance_to_intersection < CONFIG["MIN_STOP_DISTANCE"] or current_speed < 5:
                    control.throttle = 0.0
                    control.brake = 1.0
                    control.steer = 0.0
                else:
                    brake_strength = 0.5 + (CONFIG["SAFE_STOP_DISTANCE"] - distance_to_intersection) / CONFIG[
                        "SAFE_STOP_DISTANCE"] * 0.5
                    control.throttle = 0.0
                    control.brake = min(brake_strength, 1.0)
                    control.steer = 0.0
                # 停车时重置控制缓存
                current_steer = 0.0
                current_throttle = 0.0
            else:
                # ========== 核心防失控：动态预瞄距离，速度越快看的越远 ==========
                preview_distance = min(CONFIG["MAX_PREVIEW_DISTANCE"],
                                       CONFIG["BASE_PREVIEW_DISTANCE"] + current_speed / 10)
                waypoint = map.get_waypoint(vehicle_transform.location, project_to_road=True,
                                            lane_type=carla.LaneType.Driving)
                next_waypoints = waypoint.next(preview_distance)
                if next_waypoints:
                    next_waypoint = next_waypoints[0]
                    # 计算目标转向角（传入当前速度做自适应衰减）
                    target_steer = get_steer(vehicle_transform, next_waypoint.transform, current_speed)

                    # 转向死区，小误差不修正，防止高频抖动
                    if abs(target_steer - current_steer) < CONFIG["STEER_DEAD_ZONE"]:
                        target_steer = current_steer

                    # 转向平滑滤波，防止突变
                    target_steer = current_steer * CONFIG["STEER_SMOOTH_FACTOR"] + target_steer * (
                                1 - CONFIG["STEER_SMOOTH_FACTOR"])
                    # 限制每帧最大转向变化量，防止猛打方向
                    steer_change = target_steer - current_steer
                    steer_change = max(-CONFIG["MAX_STEER_CHANGE"], min(CONFIG["MAX_STEER_CHANGE"], steer_change))
                    current_steer = max(-1.0, min(1.0, current_steer + steer_change))
                    # 赋值最终转向角
                    control.steer = current_steer

                # ========== 核心防失控：平滑油门控制，防止急加速打滑 ==========
                target_speed = current_speed_limit
                if distance_to_intersection < 30:
                    target_speed = min(target_speed, CONFIG["INTERSECTION_SPEED"])

                speed_error = target_speed - current_speed
                if speed_error > 1:
                    # 目标油门，限制最大值防止打滑
                    target_throttle = min(CONFIG["MAX_THROTTLE"], CONFIG["SPEED_ADJUST_SMOOTH"] * speed_error)
                    # 油门平滑过渡，防止瞬间踩满
                    current_throttle = current_throttle * 0.8 + target_throttle * 0.2
                    control.throttle = current_throttle
                    control.brake = 0.0
                elif speed_error < -1:
                    # 超速平滑刹车
                    target_brake = min(0.6, abs(CONFIG["SPEED_ADJUST_SMOOTH"] * speed_error))
                    control.brake = target_brake
                    control.throttle = 0.0
                    current_throttle = 0.0
                else:
                    # 匀速巡航
                    control.throttle = 0.15
                    control.brake = 0.0

            # 应用控制指令
            vehicle.apply_control(control)

            # 固定清晰的速度/状态显示
            if image_surface[0] is not None:
                surface = pygame.image.frombuffer(image_surface[0].tobytes(),
                                                  (CONFIG["CAMERA_WIDTH"], CONFIG["CAMERA_HEIGHT"]), "RGB")
                display.blit(surface, (0, 0))

                # 黑色背景防遮挡
                pygame.draw.rect(display, (0, 0, 0), (10, 10, 300, 100), border_radius=5)
                # 状态文字
                speed_text = font.render(f"当前车速: {current_speed:.1f} km/h", True, (0, 255, 0))
                limit_text = font.render(f"限速: {current_speed_limit} km/h", True, (255, 255, 0))
                light_text = font.render(f"红绿灯: {final_light_state}", True,
                                         (255, 0, 0) if final_light_state == "Red" else (0, 255, 0))
                throttle_text = font.render(f"油门开度: {control.throttle:.2f}", True, (255, 255, 255))
                display.blit(speed_text, (20, 20))
                display.blit(limit_text, (20, 50))
                display.blit(light_text, (20, 80))

                # 绘制检测框
                for label, _, conf, bbox in detected_list:
                    x1, y1, x2, y2 = bbox
                    pygame.draw.rect(display, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)
                    label_text = font.render(f"{label} {conf:.2f}", True, (255, 255, 255), (0, 0, 0))
                    display.blit(label_text, (x1, y1 - 25))

                pygame.display.flip()

            # 固定30帧，稳定画面
            clock.tick(30)

    # 完全保留原有的资源清理逻辑
    except Exception as e:
        print(f"发生严重错误: {e}")
    finally:
        print("正在安全清理资源...")
        for actor in actor_list:
            if actor and 'sensor' in actor.type_id:
                try:
                    actor.stop()
                except:
                    pass
        time.sleep(0.5)
        for actor in actor_list:
            if actor:
                try:
                    actor.destroy()
                except:
                    pass
        pygame.quit()
        print("程序结束")


if __name__ == "__main__":
    main()