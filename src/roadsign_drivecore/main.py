import carla
import random
import time
import pygame
import numpy as np
import math
from ultralytics import YOLO
import torch
import datetime

CONFIG = {
    "CARLA_HOST": "localhost",
    "CARLA_PORT": 2000,
    "CAMERA_WIDTH": 640,
    "CAMERA_HEIGHT": 480,
    "SAFE_STOP_DISTANCE": 20,
    "MIN_STOP_DISTANCE": 6,
    "DETECTION_CONF": 0.65,
    "DEFAULT_CRUISE_SPEED": 40,
    "INTERSECTION_SPEED": 20,
    "SPEED_ADJUST_SMOOTH": 0.3,
    "STEER_SMOOTH_FACTOR": 0.8,
    "MAX_STEER_CHANGE": 0.08,
    "BASE_PREVIEW_DISTANCE": 3.0,
    "MAX_PREVIEW_DISTANCE": 10.0,
    "STEER_DEAD_ZONE": 0.03,
    "MAX_THROTTLE": 0.35,
    "MIN_TIRE_FRICTION": 2.5,
    "CAMERA_SMOOTH_FACTOR": 0.15,
    "SAFE_FOLLOW_DISTANCE": 15,
    "MIN_FOLLOW_DISTANCE": 6,
    "FOLLOW_SPEED_GAIN": 0.4,
    "STOP_FOLLOW_DISTANCE": 6,
    "FRONT_VEHICLE_COUNT": 0,
    "FRONT_VEHICLE_DISTANCE": 30,
    "TRAFFIC_LIGHT_DETECT_AREA": 0.7,
    "TRAFFIC_LIGHT_MIN_HEIGHT": 20,
    "MAX_TRAFFIC_LIGHT_DISTANCE": 70,
    "PYGAME_FPS": 25,
    "YOLO_INFERENCE_INTERVAL": 1,
    "LANE_WIDTH": 3.5,
    "MAX_FOLLOW_DISTANCE": 100,
    "EMERGENCY_BRAKE_DISTANCE": 15,
    "CRITICAL_BRAKE_DISTANCE": 8,
    "STUCK_THRESHOLD": 10,
    "AUTO_WEATHER_INTERVAL": 45,
    "PRE_BRAKE_DISTANCE": 25,
    "STATIC_VEHICLE_BONUS": 3
}

# Global state
current_speed_limit = CONFIG["DEFAULT_CRUISE_SPEED"]
current_steer = 0.0
smooth_camera_pos = None
current_throttle = 0.0
front_vehicle_distance = 999
front_vehicle_exist = False
front_vehicle_speed = 0.0
acc_active = False
frame_count = 0
smooth_front_distance = 999
smooth_filter_alpha = 0.3
current_weather = "Clear"
last_movement_time = 0.0
last_weather_switch_time = 0.0
current_weather_index = 0
log_file = None

weather_presets = [
    ("Clear", carla.WeatherParameters.ClearNoon),
    ("Cloudy", carla.WeatherParameters.CloudyNoon),
    ("LightRain", carla.WeatherParameters.SoftRainNoon),
    ("HeavyRain", carla.WeatherParameters.HardRainNoon),
    ("Night", carla.WeatherParameters.ClearNight)
]


def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    if log_file:
        log_file.write(full_message + "\n")
        log_file.flush()


def init_pygame(width, height):
    pygame.init()
    pygame.display.set_caption("CARLA V5.3.9")
    return pygame.display.set_mode((width, height))


def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    return array.reshape((image.height, image.width, 4))[:, :, :3].copy()


# YOLO model
model = YOLO("yolov8n.pt")
TRAFFIC_CLASSES = {9: "stop sign", 8: "traffic light", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CAR_HEIGHT = 1.5
CAMERA_FOCAL = 1000
TL_HEIGHT = 0.8


def detect_traffic(image_np, vehicle_transform):
    global current_speed_limit, front_vehicle_distance, front_vehicle_exist, smooth_front_distance, front_vehicle_speed
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
    front_vehicle_speed = 0.0
    min_dist = 999

    h, w = image_np.shape[:2]
    tl_x1 = w * 0.15
    tl_x2 = w * 0.85
    tl_y2 = h * 0.7
    img_center_x = w / 2

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = names[int(cls)]
        bbox_h = y2 - y1
        bbox_cx = (x1 + x2) / 2

        if label == "traffic light":
            if tl_x1 < bbox_cx < tl_x2 and y1 < tl_y2 and bbox_h > 20:
                roi = image_np[int(y1):int(y2), int(x1):int(x2), :]
                if roi.size != 0:
                    roi_top = roi[:int(roi.shape[0] / 3), :]
                    roi_mid = roi[int(roi.shape[0] / 3):int(roi.shape[0] * 2 / 3), :]
                    r = np.mean(roi_top[:, :, 0])
                    g = np.mean(roi_mid[:, :, 1])
                    if r > g + 30:
                        tl_state = "Red"
                    elif g > r + 30:
                        tl_state = "Green"
                    else:
                        tl_state = "Yellow"
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
            if abs(bbox_cx - img_center_x) < w * 0.25 and bbox_h > 15:
                if bbox_h > 0:
                    dist = (CAR_HEIGHT * CAMERA_FOCAL) / bbox_h
                    if dist < 120:
                        offset_pixel = bbox_cx - img_center_x
                        offset_meter = (offset_pixel * dist) / CAMERA_FOCAL
                        if abs(offset_meter) < 2.0:
                            is_front = True

            if is_front and dist < min_dist:
                min_dist = dist
                front_vehicle_distance = dist
                front_vehicle_exist = True
                if frame_count > 1:
                    front_vehicle_speed = (smooth_front_distance - dist) * CONFIG["PYGAME_FPS"] * 3.6

            detected.append(
                (label, f"{dist:.1f}m{'(F)' if is_front else ''}", conf, (int(x1), int(y1), int(x2), int(y2))))

        else:
            detected.append((label, None, conf, (int(x1), int(y1), int(x2), int(y2))))

    if front_vehicle_exist:
        smooth_front_distance = smooth_front_distance * 0.6 + front_vehicle_distance * 0.4
        front_vehicle_distance = smooth_front_distance
    else:
        smooth_front_distance = 999
        front_vehicle_speed = 0.0

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

    speed_gain = max(0.2, 1.0 - (speed / 60) * 0.8)
    final_steer = angle * 1.0 * speed_gain

    max_steer = max(0.1, 0.8 - (speed / 100) * 0.7)
    return max(-max_steer, min(max_steer, final_steer))


def get_intersection_dist(vehicle, map):
    loc = vehicle.get_transform().location
    wp = map.get_waypoint(loc, project_to_road=True)
    dist = 0
    current_wp = wp
    for _ in range(70):
        next_wps = current_wp.next(2.0)
        if not next_wps: break
        current_wp = next_wps[0]
        dist += 2.0
        if current_wp.is_junction:
            return dist
    return 999


def on_collision(event):
    global current_steer, current_throttle, acc_active, smooth_front_distance
    log(f"!!! COLLISION !!! Force: {event.normal_impulse.length():.1f}")
    current_steer = 0.0
    current_throttle = 0.0
    acc_active = False
    smooth_front_distance = 999


def optimize_physics(vehicle):
    physics = vehicle.get_physics_control()
    for wheel in physics.wheels:
        wheel.tire_friction = 3.0
    physics.steering_curve = [
        carla.Vector2D(0, 1.0),
        carla.Vector2D(50, 0.5),
        carla.Vector2D(100, 0.2)
    ]
    physics.torque_curve = [
        carla.Vector2D(0, 300),
        carla.Vector2D(1000, 350),
        carla.Vector2D(3000, 150)
    ]
    physics.mass = 1800
    vehicle.apply_physics_control(physics)
    log("Physics optimized")


def get_valid_spawn_point(map):
    for _ in range(100):
        try:
            x = random.uniform(-200, 200)
            y = random.uniform(-200, 200)
            location = carla.Location(x=x, y=y, z=0.6)

            waypoint = map.get_waypoint(location, project_to_road=True)
            if waypoint and waypoint.lane_type == carla.LaneType.Driving:
                transform = waypoint.transform
                transform.location.z = 0.6
                return transform
        except:
            continue

    spawn_points = map.get_spawn_points()
    return random.choice(spawn_points)


def spawn_initial_traffic(world, map, bp_lib, actor_list, ego_vehicle):
    traffic_count = random.randint(3, 5)
    count = 0
    for _ in range(traffic_count):
        try:
            car_bp = random.choice(bp_lib.filter('vehicle.*'))
            spawn_point = get_valid_spawn_point(map)
            if spawn_point.location.distance(ego_vehicle.get_transform().location) > 100:
                car = world.try_spawn_actor(car_bp, spawn_point)
                if car:
                    car.set_autopilot(True)
                    actor_list.append(car)
                    count += 1
                    time.sleep(0.1)
        except Exception as e:
            log(f"Error spawning vehicle: {e}")
    log(f"Initial traffic spawned: {count} cars")
    return count


def main():
    global current_speed_limit, current_steer, smooth_camera_pos, current_throttle
    global front_vehicle_distance, front_vehicle_exist, front_vehicle_speed, acc_active, frame_count, smooth_front_distance
    global current_weather, last_movement_time, last_weather_switch_time, current_weather_index, log_file
    actor_list = []
    try:
        log_filename = f"carla_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        log_file = open(log_filename, "w", encoding="utf-8")
        log(f"Program started. Log file: {log_filename}")

        client = carla.Client(CONFIG["CARLA_HOST"], CONFIG["CARLA_PORT"])
        client.set_timeout(15.0)
        world = client.get_world()
        map = world.get_map()
        bp_lib = world.get_blueprint_library()

        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_point = get_valid_spawn_point(map)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        log(f"Ego vehicle spawned at: {spawn_point.location}")
        optimize_physics(vehicle)

        collision_bp = bp_lib.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        collision_sensor.listen(on_collision)
        actor_list.append(collision_sensor)

        spawn_initial_traffic(world, map, bp_lib, actor_list, vehicle)

        speed_signs = []
        speeds = [30, 40, 50, 60]
        sign_bps = [bp for bp in bp_lib if 'static.prop.speedlimit' in bp.id]
        for i, speed in enumerate(speeds):
            target_bp = next((bp for bp in sign_bps if f"speedlimit.{speed}" in bp.id), None)
            if target_bp:
                spawn_point = get_valid_spawn_point(map)
                spawn_point.location.z = 1.5
                sign = world.try_spawn_actor(target_bp, spawn_point)
                if sign:
                    speed_signs.append(sign)
                    actor_list.append(sign)

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

        display = init_pygame(CONFIG["CAMERA_WIDTH"], CONFIG["CAMERA_HEIGHT"])
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 20, bold=True)

        spectator = world.get_spectator()

        def update_spectator():
            global smooth_camera_pos
            transform = vehicle.get_transform()
            target_pos = transform.location + transform.get_forward_vector() * -12 + carla.Location(z=10)
            target_rot = carla.Rotation(pitch=-20, yaw=transform.rotation.yaw, roll=0)

            if smooth_camera_pos is None:
                smooth_camera_pos = target_pos
            else:
                smooth_camera_pos.x = smooth_camera_pos.x * 0.85 + target_pos.x * 0.15
                smooth_camera_pos.y = smooth_camera_pos.y * 0.85 + target_pos.y * 0.15
                smooth_camera_pos.z = smooth_camera_pos.z * 0.85 + target_pos.z * 0.15

            spectator.set_transform(carla.Transform(smooth_camera_pos, target_rot))

        last_movement_time = time.time()
        last_weather_switch_time = time.time()
        weather_name, weather_params = weather_presets[current_weather_index]
        world.set_weather(weather_params)
        current_weather = weather_name
        log(f"Initial weather: {weather_name}")

        time.sleep(1.0)

        # Main loop
        while True:
            frame_count += 1
            current_time = time.time()

            # 天气自动切换
            if current_time - last_weather_switch_time >= CONFIG["AUTO_WEATHER_INTERVAL"]:
                current_weather_index = (current_weather_index + 1) % len(weather_presets)
                weather_name, weather_params = weather_presets[current_weather_index]
                world.set_weather(weather_params)
                current_weather = weather_name
                last_weather_switch_time = current_time
                log(f"Auto weather changed: {weather_name}")

            update_spectator()
            control = carla.VehicleControl()
            current_speed = get_speed(vehicle)
            v_transform = vehicle.get_transform()

            # 自动防卡住逻辑
            if current_speed > 1.0:
                last_movement_time = current_time
            elif current_time - last_movement_time > CONFIG["STUCK_THRESHOLD"]:
                log("Vehicle stuck detected, auto resetting...")
                control.throttle = 0.0
                control.brake = 1.0
                control.steer = 0.0
                vehicle.apply_control(control)
                time.sleep(0.5)

                new_spawn = get_valid_spawn_point(map)
                vehicle.set_transform(new_spawn)
                vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))

                current_speed_limit = CONFIG["DEFAULT_CRUISE_SPEED"]
                smooth_camera_pos = None
                current_steer = 0.0
                current_throttle = 0.0
                acc_active = False
                smooth_front_distance = 999
                last_movement_time = current_time
                log(f"Vehicle reset to: {new_spawn.location}")
                time.sleep(1.0)
                continue

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    log("Program exit requested")
                    return

            # Object detection
            detected_list = []
            tl_state = None
            if image_surface[0] is not None:
                detected_list, tl_state = detect_traffic(image_surface[0], v_transform)

            # 紧急制动（最高优先级）
            emergency_brake = False
            if front_vehicle_exist and front_vehicle_distance < CONFIG["EMERGENCY_BRAKE_DISTANCE"]:
                if front_vehicle_distance < CONFIG["CRITICAL_BRAKE_DISTANCE"]:
                    control.brake = 1.0
                else:
                    control.brake = 0.9
                control.throttle = 0.0
                control.steer = 0.0
                vehicle.apply_control(control)
                acc_active = False
                emergency_brake = True
                log(f"EMERGENCY BRAKE! Distance: {front_vehicle_distance:.1f}m, Speed: {current_speed:.1f}")

            if not emergency_brake:
                target_speed = current_speed_limit
                acc_active = False

                # 红绿灯逻辑（优先使用CARLA原生状态）
                intersection_dist = get_intersection_dist(vehicle, map)
                native_tl = vehicle.get_traffic_light_state().name
                final_light_state = native_tl if native_tl != "Unknown" else tl_state

                if intersection_dist < 70 and (final_light_state == "Red" or final_light_state == "Yellow"):
                    stop_dist = CONFIG["SAFE_STOP_DISTANCE"] + (current_speed / 10) * 2.5

                    if intersection_dist < CONFIG["PRE_BRAKE_DISTANCE"]:
                        target_speed = min(target_speed, 25)

                    if intersection_dist < stop_dist:
                        if intersection_dist < 6 or current_speed < 5:
                            target_speed = 0
                        else:
                            target_speed = current_speed * (intersection_dist / stop_dist) * 0.4
                            target_speed = max(0, target_speed)

                # ACC跟车逻辑
                if front_vehicle_exist:
                    acc_active = True
                    static_bonus = CONFIG["STATIC_VEHICLE_BONUS"] if front_vehicle_speed < 5 else 0
                    safe_dist = CONFIG["SAFE_FOLLOW_DISTANCE"] + (current_speed / 10) * 3 + static_bonus

                    if front_vehicle_distance < CONFIG["STOP_FOLLOW_DISTANCE"]:
                        target_speed = 0
                    elif front_vehicle_distance < safe_dist:
                        speed_factor = max(0.6, front_vehicle_speed / current_speed) if current_speed > 0 else 0.6
                        target_speed = current_speed_limit * (front_vehicle_distance / safe_dist) * speed_factor * 0.8
                        target_speed = max(0, target_speed)

                # 路口预减速
                if intersection_dist < 40:
                    target_speed = min(target_speed, CONFIG["INTERSECTION_SPEED"])

                # 路径规划
                try:
                    preview_dist = min(10, 3 + current_speed / 10)
                    wp = map.get_waypoint(v_transform.location, project_to_road=True)
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
                    else:
                        log("No waypoint found, resetting...")
                        last_movement_time = 0
                        continue
                except Exception as e:
                    log(f"Waypoint error: {e}, resetting...")
                    last_movement_time = 0
                    continue

                # 起步增强逻辑
                speed_error = target_speed - current_speed
                if current_speed < 2 and target_speed > 5:
                    target_throttle = min(CONFIG["MAX_THROTTLE"], 0.4)
                    current_throttle = current_throttle * 0.7 + target_throttle * 0.3
                    control.throttle = current_throttle
                    control.brake = 0.0
                elif speed_error > 1:
                    target_throttle = min(CONFIG["MAX_THROTTLE"], 0.25 * speed_error)
                    current_throttle = current_throttle * 0.9 + target_throttle * 0.1
                    control.throttle = current_throttle
                    control.brake = 0.0
                elif speed_error < -1:
                    target_brake = min(0.6, abs(0.4 * speed_error))
                    control.brake = target_brake
                    control.throttle = 0.0
                    current_throttle = 0.0
                else:
                    control.throttle = 0.12
                    control.brake = 0.0


                vehicle.apply_control(control)
                # ===== 全自动前照灯控制（程序自主完成，无需人工操作）=====
                sun_alt = world.get_weather().sun_altitude_angle
                if sun_alt < 15:
                    vehicle.set_light_state(carla.VehicleLightState.LowBeam)
                elif sun_alt > 20:
                    vehicle.set_light_state(carla.VehicleLightState.NONE)

            # Render
            if image_surface[0] is not None:
                surface = pygame.image.frombuffer(image_surface[0].tobytes(),
                                                  (CONFIG["CAMERA_WIDTH"], CONFIG["CAMERA_HEIGHT"]), "RGB")
                display.blit(surface, (0, 0))

                # Draw detection boxes
                for label, info, conf, bbox in detected_list:
                    x1, y1, x2, y2 = bbox
                    color = (255, 0, 0) if "(F)" in str(info) else (0, 255, 0)
                    pygame.draw.rect(display, color, (x1, y1, x2 - x1, y2 - y1), 2)
                    label_text = font.render(f"{label} {info}", True, (255, 255, 255), (0, 0, 0))
                    display.blit(label_text, (x1, y1 - 20))

                # ========== 最终界面：删除限速，添加天气倒计时 ==========
                pygame.draw.rect(display, (0, 0, 0), (10, 10, 180, 120), border_radius=5)
                speed_text = font.render(f"Speed: {current_speed:.1f} km/h", True, (0, 255, 0))
                weather_text = font.render(f"Weather: {current_weather}", True, (0, 255, 255))
                # 计算天气倒计时
                weather_time_left = int(CONFIG["AUTO_WEATHER_INTERVAL"] - (current_time - last_weather_switch_time))
                countdown_text = font.render(f"Next: {weather_time_left}s", True, (255, 165, 0))
                tl_text = font.render(f"Light: {final_light_state}", True,
                                      (255, 0, 0) if final_light_state == "Red" else (0, 255, 0))

                display.blit(speed_text, (20, 20))
                display.blit(weather_text, (20, 45))
                display.blit(countdown_text, (20, 70))
                display.blit(tl_text, (20, 95))

                # FPS显示
                fps = clock.get_fps()
                fps_text = font.render(f"FPS: {fps:.0f}", True, (0, 255, 255))
                display.blit(fps_text, (CONFIG["CAMERA_WIDTH"] - 100, 20))

                pygame.display.flip()

            clock.tick(CONFIG["PYGAME_FPS"])

    except Exception as e:
        log(f"Error: {e}")
    finally:
        log("Cleaning up...")
        if log_file:
            log_file.close()
        for actor in actor_list:
            if actor and 'sensor' in actor.type_id:
                try:
                    actor.stop()
                except:
                    pass
        time.sleep(1.5)
        for actor in actor_list:
            if actor:
                try:
                    actor.destroy()
                except:
                    pass
        pygame.quit()
        log("Program ended")


if __name__ == "__main__":
    main()