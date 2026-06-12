import carla
import random
import time
import pygame
import numpy as np
import math
from ultralytics import YOLO
import torch

CONFIG = {
    "CARLA_HOST": "localhost",
    "CARLA_PORT": 2000,
    "CAMERA_WIDTH": 640,
    "CAMERA_HEIGHT": 480,
    "SAFE_STOP_DISTANCE": 15,
    "MIN_STOP_DISTANCE": 3,
    "DETECTION_CONF": 0.7,
    "DEFAULT_CRUISE_SPEED": 40,
    "INTERSECTION_SPEED": 25,
    "SPEED_ADJUST_SMOOTH": 0.3,
    "STEER_SMOOTH_FACTOR": 0.8,
    "MAX_STEER_CHANGE": 0.08,
    "BASE_PREVIEW_DISTANCE": 3.0,
    "MAX_PREVIEW_DISTANCE": 10.0,
    "STEER_DEAD_ZONE": 0.03,
    "MAX_THROTTLE": 0.4,
    "MIN_TIRE_FRICTION": 2.5,
    "CAMERA_SMOOTH_FACTOR": 0.15,
    "SAFE_FOLLOW_DISTANCE": 12,
    "MIN_FOLLOW_DISTANCE": 4,
    "FOLLOW_SPEED_GAIN": 0.6,
    "STOP_FOLLOW_DISTANCE": 3,
    "FRONT_VEHICLE_COUNT": 2,
    "FRONT_VEHICLE_DISTANCE": 25,
    "TRAFFIC_LIGHT_DETECT_AREA": 0.6,
    "TRAFFIC_LIGHT_MIN_HEIGHT": 30,
    "MAX_TRAFFIC_LIGHT_DISTANCE": 50,
    "PYGAME_FPS": 25,
    "YOLO_INFERENCE_INTERVAL": 2,
    "LANE_WIDTH": 3.5,
    "MAX_FOLLOW_DISTANCE": 80,
    "EMERGENCY_BRAKE_DISTANCE": 8,
    "CRITICAL_BRAKE_DISTANCE": 4
}

# 全局状态
need_vehicle_reset = False
current_speed_limit = CONFIG["DEFAULT_CRUISE_SPEED"]
current_steer = 0.0
smooth_camera_pos = None
current_throttle = 0.0
front_vehicle_distance = 999
front_vehicle_exist = False
acc_active = False
frame_count = 0
smooth_front_distance = 999
smooth_filter_alpha = 0.3
# ========== 新增：天气全局状态 ==========
current_weather = "晴天"
weather_presets = {
    pygame.K_1: ("晴天", carla.WeatherParameters.ClearNoon),
    pygame.K_2: ("阴天", carla.WeatherParameters.CloudyNoon),
    pygame.K_3: ("小雨", carla.WeatherParameters.SoftRainNoon),
    pygame.K_4: ("大雨", carla.WeatherParameters.HardRainNoon),
    pygame.K_5: ("黑夜", carla.WeatherParameters.ClearNight),
    pygame.K_0: ("晴天", carla.WeatherParameters.ClearNoon)
}


def init_pygame(width, height):
    pygame.init()
    pygame.display.set_caption("CARLA 天气快捷切换")
    return pygame.display.set_mode((width, height))


def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    return array.reshape((image.height, image.width, 4))[:, :, :3].copy()


# YOLO模型
model = YOLO("yolov8n.pt")
TRAFFIC_CLASSES = {9: "stop sign", 8: "traffic light", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CAR_HEIGHT = 1.5
CAMERA_FOCAL = 1000
TL_HEIGHT = 0.8


def detect_traffic(image_np, vehicle_transform):
    global current_speed_limit, front_vehicle_distance, front_vehicle_exist, smooth_front_distance
    results = model.predict(
        source=image_np, imgsz=640, conf=CONFIG["DETECTION_CONF"],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=False, classes=list(TRAFFIC_CLASSES.keys())
    )
    detections = results[0].boxes.data.cpu().numpy()
    names = results[0].names

    detected = []
    tl_state = None
    detected_speed = None
    front_vehicle_distance = 999
    front_vehicle_exist = False
    min_dist = 999

    h, w = image_np.shape[:2]
    tl_x1 = w * 0.2
    tl_x2 = w * 0.8
    tl_y2 = h * 0.6
    img_center_x = w / 2

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = names[int(cls)]
        bbox_h = y2 - y1
        bbox_cx = (x1 + x2) / 2

        if label == "traffic light":
            if tl_x1 < bbox_cx < tl_x2 and y1 < tl_y2 and bbox_h > 30:
                roi = image_np[int(y1):int(y2), int(x1):int(x2), :]
                if roi.size != 0:
                    roi_top = roi[:int(roi.shape[0] / 2), :]
                    r = np.mean(roi_top[:, :, 0])
                    g = np.mean(roi_top[:, :, 1])
                    if r > g + 20:
                        tl_state = "Red"
                    elif g > r + 20:
                        tl_state = "Green"
                dist = (TL_HEIGHT * CAMERA_FOCAL) / bbox_h
                detected.append((label, f"{tl_state} {dist:.1f}m", conf, (int(x1), int(y1), int(x2), int(y2))))

        elif "speed limit" in label.lower():
            digits = [int(s) for s in label.split() if s.isdigit()]
            if digits:
                detected_speed = digits[0]
                current_speed_limit = detected_speed
            detected.append((label, detected_speed, conf, (int(x1), int(y1), int(x2), int(y2))))

        elif label in ["car", "truck", "bus", "motorcycle"]:
            dist = 999
            is_front = False
            if abs(bbox_cx - img_center_x) < w * 0.2 and bbox_h > 20:
                if bbox_h > 0:
                    dist = (CAR_HEIGHT * CAMERA_FOCAL) / bbox_h
                    if dist < 80:
                        offset_pixel = bbox_cx - img_center_x
                        offset_meter = (offset_pixel * dist) / CAMERA_FOCAL
                        if abs(offset_meter) < 1.75:
                            is_front = True

            if is_front and dist < min_dist:
                min_dist = dist
                front_vehicle_distance = dist
                front_vehicle_exist = True

            detected.append(
                (label, f"{dist:.1f}m{'(本车道)' if is_front else ''}", conf, (int(x1), int(y1), int(x2), int(y2))))

        else:
            detected.append((label, None, conf, (int(x1), int(y1), int(x2), int(y2))))

    # 平滑距离
    if front_vehicle_exist:
        smooth_front_distance = smooth_front_distance * 0.7 + front_vehicle_distance * 0.3
        front_vehicle_distance = smooth_front_distance
    else:
        smooth_front_distance = 999

    if detected_speed is None:
        current_speed_limit = CONFIG["DEFAULT_CRUISE_SPEED"]

    return detected, tl_state


def get_speed(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) * 3.6


def get_steer(v_transform, wp_transform, speed):
    v_loc = v_transform.location
    v_forward = v_transform.get_forward_vector()
    wp_loc = wp_transform.location

    dir_vec = carla.Vector3D(wp_loc.x - v_loc.x, wp_loc.y - v_loc.y, 0)
    v_forward = carla.Vector3D(v_forward.x, v_forward.y, 0)

    dir_norm = math.hypot(dir_vec.x, dir_vec.y)
    fwd_norm = math.hypot(v_forward.x, v_forward.y)
    if dir_norm < 1e-5 or fwd_norm < 1e-5:
        return 0.0

    dot = (v_forward.x * dir_vec.x + v_forward.y * dir_vec.y) / (dir_norm * fwd_norm)
    dot = max(-1.0, min(1.0, dot))
    angle = math.acos(dot)
    cross = v_forward.x * dir_vec.y - v_forward.y * dir_vec.x
    if cross < 0: angle *= -1

    # 速度越高转向越轻
    speed_gain = max(0.2, 1.0 - (speed / 60) * 0.8)
    final_steer = angle * 1.0 * speed_gain

    max_steer = max(0.1, 0.8 - (speed / 100) * 0.7)
    return max(-max_steer, min(max_steer, final_steer))


def get_intersection_dist(vehicle, map):
    loc = vehicle.get_transform().location
    wp = map.get_waypoint(loc, project_to_road=True)
    dist = 0
    current_wp = wp
    for _ in range(50):
        next_wps = current_wp.next(2.0)
        if not next_wps: break
        current_wp = next_wps[0]
        dist += 2.0
        if current_wp.is_junction:
            return dist
    return 999


def on_collision(event):
    global need_vehicle_reset, current_steer, current_throttle, acc_active, smooth_front_distance
    need_vehicle_reset = True
    print(f"碰撞！强度：{event.normal_impulse.length():.1f}，准备重置")
    current_steer = 0.0
    current_throttle = 0.0
    acc_active = False
    smooth_front_distance = 999


def optimize_physics(vehicle):
    physics = vehicle.get_physics_control()
    for wheel in physics.wheels:
        wheel.tire_friction = 2.5
    physics.steering_curve = [
        carla.Vector2D(0, 1.0),
        carla.Vector2D(50, 0.5),
        carla.Vector2D(100, 0.2)
    ]
    physics.torque_curve = [
        carla.Vector2D(0, 300),
        carla.Vector2D(1000, 400),
        carla.Vector2D(3000, 200)
    ]
    physics.mass = 1800
    vehicle.apply_physics_control(physics)
    print("车辆物理参数优化完成")


def spawn_front_cars(vehicle, world, map, bp_lib, actor_list):
    v_transform = vehicle.get_transform()
    current_wp = map.get_waypoint(v_transform.location, project_to_road=True)
    count = 0
    for i in range(CONFIG["FRONT_VEHICLE_COUNT"]):
        next_wps = current_wp.next(CONFIG["FRONT_VEHICLE_DISTANCE"] * (i + 1))
        if not next_wps: break
        spawn_wp = next_wps[0]
        car_bp = random.choice(bp_lib.filter('vehicle.*'))
        car = world.try_spawn_actor(car_bp, spawn_wp.transform)
        if car:
            car.set_autopilot(True)
            actor_list.append(car)
            count += 1
    print(f"前方生成测试车辆：{count}辆")
    return count


def main():
    global need_vehicle_reset, current_speed_limit, current_steer, smooth_camera_pos, current_throttle
    global front_vehicle_distance, front_vehicle_exist, acc_active, frame_count, smooth_front_distance
    global current_weather  # 新增
    actor_list = []
    try:
        # 连接CARLA
        client = carla.Client(CONFIG["CARLA_HOST"], CONFIG["CARLA_PORT"])
        client.set_timeout(10.0)
        world = client.get_world()
        map = world.get_map()
        bp_lib = world.get_blueprint_library()

        # 生成主车
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_point = random.choice(map.get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print("主车生成成功")
        optimize_physics(vehicle)

        # 碰撞传感器
        collision_bp = bp_lib.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        collision_sensor.listen(on_collision)
        actor_list.append(collision_sensor)

        # 生成交通
        front_cars = spawn_front_cars(vehicle, world, map, bp_lib, actor_list)
        traffic_count = random.randint(6, 10)
        for _ in range(traffic_count):
            car_bp = random.choice(bp_lib.filter('vehicle.*'))
            car = world.try_spawn_actor(car_bp, random.choice(map.get_spawn_points()))
            if car:
                car.set_autopilot(True)
                actor_list.append(car)
        print(f"背景车辆：{traffic_count}辆")

        # 生成限速标志
        speed_signs = []
        speeds = [30, 40, 50, 60]
        sign_bps = [bp for bp in bp_lib if 'static.prop.speedlimit' in bp.id]
        for i, speed in enumerate(speeds):
            target_bp = next((bp for bp in sign_bps if f"speedlimit.{speed}" in bp.id), None)
            if target_bp:
                spawn_point = map.get_spawn_points()[i * 5 % len(map.get_spawn_points())]
                spawn_point.location.z = 1.5
                sign = world.try_spawn_actor(target_bp, spawn_point)
                if sign:
                    speed_signs.append(sign)
                    actor_list.append(sign)

        # 车载摄像头
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(CONFIG["CAMERA_WIDTH"]))
        camera_bp.set_attribute("image_size_y", str(CONFIG["CAMERA_HEIGHT"]))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        image_surface = [None]

        def image_callback(image):
            image_surface[0] = process_image(image)

        camera.listen(image_callback)

        # 初始化显示
        display = init_pygame(CONFIG["CAMERA_WIDTH"], CONFIG["CAMERA_HEIGHT"])
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 20, bold=True)

        # 平滑视角
        spectator = world.get_spectator()

        def update_spectator():
            global smooth_camera_pos
            transform = vehicle.get_transform()
            target_pos = transform.location + transform.get_forward_vector() * -10 + carla.Location(z=8)
            target_rot = carla.Rotation(pitch=-15, yaw=transform.rotation.yaw, roll=0)

            if smooth_camera_pos is None:
                smooth_camera_pos = target_pos
            else:
                smooth_camera_pos.x = smooth_camera_pos.x * 0.85 + target_pos.x * 0.15
                smooth_camera_pos.y = smooth_camera_pos.y * 0.85 + target_pos.y * 0.15
                smooth_camera_pos.z = smooth_camera_pos.z * 0.85 + target_pos.z * 0.15

            spectator.set_transform(carla.Transform(smooth_camera_pos, target_rot))

        # 主循环
        while True:
            frame_count += 1
            # ========== 新增：天气快捷切换事件监听 ==========
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
                # 数字键切换天气
                if event.type == pygame.KEYDOWN and event.key in weather_presets:
                    weather_name, weather_params = weather_presets[event.key]
                    world.set_weather(weather_params)
                    current_weather = weather_name
                    print(f"已切换天气：{weather_name}")

            update_spectator()
            control = carla.VehicleControl()
            current_speed = get_speed(vehicle)
            v_transform = vehicle.get_transform()

            # 碰撞重置
            if need_vehicle_reset:
                control.throttle = 0.0
                control.brake = 1.0
                control.steer = 0.0
                vehicle.apply_control(control)
                time.sleep(1)

                new_spawn = random.choice(map.get_spawn_points())
                vehicle.set_transform(new_spawn)
                vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))

                need_vehicle_reset = False
                current_speed_limit = CONFIG["DEFAULT_CRUISE_SPEED"]
                smooth_camera_pos = None
                current_steer = 0.0
                current_throttle = 0.0
                acc_active = False
                smooth_front_distance = 999
                print(f"车辆已重置到：{new_spawn.location}")
                continue

            # 目标检测
            detected_list = []
            tl_state = None
            if image_surface[0] is not None and frame_count % 2 == 0:
                detected_list, tl_state = detect_traffic(image_surface[0], v_transform)

            # 紧急制动（最高优先级）
            emergency_brake = False
            if front_vehicle_exist and front_vehicle_distance < CONFIG["EMERGENCY_BRAKE_DISTANCE"]:
                if front_vehicle_distance < CONFIG["CRITICAL_BRAKE_DISTANCE"]:
                    control.brake = 1.0
                else:
                    control.brake = 0.8
                control.throttle = 0.0
                control.steer = 0.0
                vehicle.apply_control(control)
                acc_active = False
                emergency_brake = True

            if not emergency_brake:
                # ACC跟车（次高优先级）
                target_speed = current_speed_limit
                acc_active = False
                if front_vehicle_exist:
                    acc_active = True
                    safe_dist = CONFIG["SAFE_FOLLOW_DISTANCE"] + (current_speed / 10) * 2

                    if front_vehicle_distance < CONFIG["STOP_FOLLOW_DISTANCE"]:
                        target_speed = 0
                    elif front_vehicle_distance < safe_dist:
                        target_speed = current_speed_limit * (front_vehicle_distance / safe_dist)
                        target_speed = max(0, target_speed)

                # 红绿灯停车（只有当前方没有车时才执行）
                else:
                    intersection_dist = get_intersection_dist(vehicle, map)
                    native_tl = vehicle.get_traffic_light_state().name
                    final_light_state = native_tl if native_tl != "Unknown" else tl_state

                    if intersection_dist < 50 and (final_light_state == "Red" or final_light_state == "Yellow"):
                        stop_dist = CONFIG["SAFE_STOP_DISTANCE"] + (current_speed / 10)
                        if intersection_dist < stop_dist:
                            if intersection_dist < 3 or current_speed < 5:
                                target_speed = 0
                            else:
                                target_speed = current_speed * (intersection_dist / stop_dist) * 0.5
                                target_speed = max(0, target_speed)

                # 路口预减速
                intersection_dist = get_intersection_dist(vehicle, map)
                if intersection_dist < 30:
                    target_speed = min(target_speed, CONFIG["INTERSECTION_SPEED"])

                # 横向转向控制
                preview_dist = min(10, 3 + current_speed / 10)
                wp = map.get_waypoint(v_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
                next_wps = wp.next(preview_dist)
                if next_wps:
                    next_wp = next_wps[0]
                    target_steer = get_steer(v_transform, next_wp.transform, current_speed)

                    if abs(target_steer - current_steer) < 0.03:
                        target_steer = current_steer

                    target_steer = current_steer * 0.8 + target_steer * 0.2
                    steer_change = target_steer - current_steer
                    steer_change = max(-0.08, min(0.08, steer_change))
                    current_steer = max(-1.0, min(1.0, current_steer + steer_change))
                    control.steer = current_steer

                # 纵向速度控制
                speed_error = target_speed - current_speed
                if speed_error > 1:
                    target_throttle = min(0.4, 0.3 * speed_error)
                    current_throttle = current_throttle * 0.8 + target_throttle * 0.2
                    control.throttle = current_throttle
                    control.brake = 0.0
                elif speed_error < -1:
                    target_brake = min(0.6, abs(0.3 * speed_error))
                    control.brake = target_brake
                    control.throttle = 0.0
                    current_throttle = 0.0
                else:
                    control.throttle = 0.15
                    control.brake = 0.0

                vehicle.apply_control(control)

            # 画面渲染
            if image_surface[0] is not None:
                surface = pygame.image.frombuffer(image_surface[0].tobytes(),
                                                  (CONFIG["CAMERA_WIDTH"], CONFIG["CAMERA_HEIGHT"]), "RGB")
                display.blit(surface, (0, 0))

                # 绘制检测框
                for label, info, conf, bbox in detected_list:
                    x1, y1, x2, y2 = bbox
                    color = (255, 0, 0) if "(本车道)" in str(info) else (0, 255, 0)
                    pygame.draw.rect(display, color, (x1, y1, x2 - x1, y2 - y1), 2)
                    label_text = font.render(f"{label} {info}", True, (255, 255, 255), (0, 0, 0))
                    display.blit(label_text, (x1, y1 - 20))

                # ========== 新增：显示当前天气状态 ==========
                pygame.draw.rect(display, (0, 0, 0), (10, 10, 200, 140), border_radius=5)
                speed_text = font.render(f" {current_speed:.1f} km/h", True, (0, 255, 0))
                limit_text = font.render(f" {current_speed_limit} km/h", True, (255, 255, 0))
                acc_text = font.render(f"ACC: {'A' if acc_active else 'B'}", True,
                                       (255, 165, 0) if acc_active else (200, 200, 200))
                distance_text = font.render(f"{front_vehicle_distance:.1f}m", True,
                                            (255, 0, 0) if front_vehicle_distance < 10 else (0, 255, 0))
                weather_text = font.render(f"天气: {current_weather}", True, (0, 255, 255))

                display.blit(speed_text, (20, 20))
                display.blit(limit_text, (20, 45))
                display.blit(acc_text, (20, 70))
                display.blit(distance_text, (20, 95))
                display.blit(weather_text, (20, 120))

                pygame.display.flip()

            clock.tick(CONFIG["PYGAME_FPS"])

    except Exception as e:
        print(f"错误: {e}")
    finally:
        print("清理资源...")
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