import carla
import random
import time
import pygame
import numpy as np
import math
from ultralytics import YOLO
import torch

# ================== 核心配置（完全保留V7.0） ==================
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
    "STEER_SMOOTH_FACTOR": 0.8,
    "MAX_STEER_CHANGE": 0.08,
    "BASE_PREVIEW_DISTANCE": 3.0,
    "MAX_PREVIEW_DISTANCE": 10.0,
    "STEER_DEAD_ZONE": 0.03,
    "MAX_THROTTLE": 0.4,
    "MIN_TIRE_FRICTION": 2.5,
    "CAMERA_SMOOTH_FACTOR": 0.15,
    "SAFE_FOLLOW_DISTANCE": 10,
    "MIN_FOLLOW_DISTANCE": 3,
    "FOLLOW_SPEED_GAIN": 0.8,
    "STOP_FOLLOW_DISTANCE": 2.5
}

# ================== 全局变量（完全保留V7.0） ==================
need_vehicle_reset = False
current_speed_limit = CONFIG["DEFAULT_CRUISE_SPEED"]
current_steer = 0.0
smooth_camera_pos = None
current_throttle = 0.0
front_vehicle_distance = 999
front_vehicle_exist = False
acc_active = False


# ================== 基础初始化函数（完全保留） ==================
def init_pygame(width, height):
    pygame.init()
    display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA V7.0 ACC自适应")
    return display


def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3].copy()
    return array


# ================== 修复：YOLO检测函数，解决distance变量未定义崩溃 ==================
model = YOLO("yolov8n.pt")
TRAFFIC_DETECT_CLASSES = {
    9: "stop sign",
    8: "traffic light",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}
CAR_REAL_HEIGHT = 1.5
CAMERA_FOCAL_LENGTH = 1000


def detect_traffic_elements(image_np):
    global current_speed_limit, front_vehicle_distance, front_vehicle_exist
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
    front_vehicle_distance = 999
    front_vehicle_exist = False
    min_distance = 999

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = names[int(cls)]
        bbox_height = y2 - y1
        bbox_center_x = (x1 + x2) / 2
        image_center_x = image_np.shape[1] / 2

        # 红绿灯状态识别（原有逻辑完全保留）
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
        # 限速标志识别（原有逻辑完全保留）
        elif "speed limit" in label.lower():
            digits = [int(s) for s in label.split() if s.isdigit()]
            if digits:
                detected_speed_limit = digits[0]
                current_speed_limit = detected_speed_limit
                print(f"【V6.0 限速管控】检测到限速标志：{detected_speed_limit} km/h")
            detected_list.append((label, detected_speed_limit, conf, (int(x1), int(y1), int(x2), int(y2))))
        # ========== 修复：车辆检测分支，确保distance变量一定被初始化 ==========
        elif label in ["car", "truck", "bus", "motorcycle"]:
            distance = 999  # 修复：先初始化distance变量
            # 只识别画面中心区域的前车
            if abs(bbox_center_x - image_center_x) < image_np.shape[1] * 0.25 and bbox_height > 20:
                if bbox_height > 0:
                    distance = (CAR_REAL_HEIGHT * CAMERA_FOCAL_LENGTH) / bbox_height
                    if distance < min_distance:
                        min_distance = distance
                        front_vehicle_distance = distance
                        front_vehicle_exist = True
            detected_list.append((label, f"{distance:.1f}m", conf, (int(x1), int(y1), int(x2), int(y2))))
        else:
            detected_list.append((label, None, conf, (int(x1), int(y1), int(x2), int(y2))))

    if detected_speed_limit is None:
        current_speed_limit = CONFIG["DEFAULT_CRUISE_SPEED"]

    return detected_list, traffic_light_state


# ================== 工具函数（完全保留V7.0） ==================
def get_speed(vehicle):
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6


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

    speed_gain = max(0.2, 1.0 - (current_speed / 60) * 0.8)
    final_steer = angle * 1.0 * speed_gain

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
    global need_vehicle_reset, current_steer, current_throttle, acc_active
    need_vehicle_reset = True
    collision_force = event.normal_impulse.length()
    print(f"【V4.0 碰撞保护】检测到碰撞！强度：{collision_force:.1f}，准备重置车辆")
    current_steer = 0.0
    current_throttle = 0.0
    acc_active = False


# ================== 车辆物理优化函数（完全保留） ==================
def optimize_vehicle_physics(vehicle):
    physics_control = vehicle.get_physics_control()
    for wheel in physics_control.wheels:
        wheel.tire_friction = CONFIG["MIN_TIRE_FRICTION"]
    physics_control.steering_curve = [
        carla.Vector2D(x=0, y=1.0),
        carla.Vector2D(x=50, y=0.5),
        carla.Vector2D(x=100, y=0.2)
    ]
    physics_control.torque_curve = [
        carla.Vector2D(x=0, y=300),
        carla.Vector2D(x=1000, y=400),
        carla.Vector2D(x=3000, y=200)
    ]
    physics_control.gear_switch_time = 0.01
    physics_control.mass = 1800
    vehicle.apply_physics_control(physics_control)
    print("【防失控优化】车辆物理参数已优化，抓地力与稳定性提升")


# ================== 主函数（完全保留V7.0，仅依赖修复后的检测函数） ==================
def main():
    global need_vehicle_reset, current_speed_limit, current_steer, smooth_camera_pos, current_throttle
    global front_vehicle_distance, front_vehicle_exist, acc_active
    actor_list = []
    try:
        client = carla.Client(CONFIG["CARLA_HOST"], CONFIG["CARLA_PORT"])
        client.set_timeout(10.0)
        world = client.get_world()
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_point = random.choice(map.get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print("主车生成成功")
        optimize_vehicle_physics(vehicle)

        collision_bp = blueprint_library.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        collision_sensor.listen(on_collision)
        actor_list.append(collision_sensor)

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

        camera_bp = blueprint_library.find("sensor.camera.rgb")
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
        font = pygame.font.SysFont("Arial", 22, bold=True)

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

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            update_spectator()
            control = carla.VehicleControl()
            current_speed = get_speed(vehicle)
            vehicle_transform = vehicle.get_transform()

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
                acc_active = False
                print(f"【V4.0 碰撞保护】车辆已重置到新位置：{new_spawn_point.location}")
                continue

            traffic_light_state = None
            detected_list = []
            if image_surface[0] is not None:
                detected_list, traffic_light_state = detect_traffic_elements(image_surface[0])

            native_light_state = vehicle.get_traffic_light_state().name
            final_light_state = traffic_light_state if traffic_light_state else native_light_state
            distance_to_intersection = get_distance_to_intersection(vehicle, map)
            should_stop = False

            if final_light_state == "Red":
                dynamic_stop_distance = CONFIG["SAFE_STOP_DISTANCE"] + (current_speed / 10)
                if distance_to_intersection < dynamic_stop_distance:
                    should_stop = True

            if should_stop:
                acc_active = False
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
                current_steer = 0.0
                current_throttle = 0.0
            else:
                dynamic_safe_distance = CONFIG["SAFE_FOLLOW_DISTANCE"] + (current_speed / 10) * 2
                acc_active = False
                target_speed = current_speed_limit

                if front_vehicle_exist and front_vehicle_distance < dynamic_safe_distance + 10:
                    acc_active = True
                    if front_vehicle_distance < CONFIG["STOP_FOLLOW_DISTANCE"]:
                        target_speed = 0
                    elif front_vehicle_distance < dynamic_safe_distance:
                        target_speed = current_speed * (front_vehicle_distance / dynamic_safe_distance) * CONFIG[
                            "FOLLOW_SPEED_GAIN"]
                        target_speed = max(0, min(target_speed, current_speed_limit))
                    print(
                        f"【V7.0 ACC激活】前车距离：{front_vehicle_distance:.1f}m | 安全车距：{dynamic_safe_distance:.1f}m | 跟车目标速度：{target_speed:.1f}km/h")

                if distance_to_intersection < 30:
                    target_speed = min(target_speed, CONFIG["INTERSECTION_SPEED"])

                preview_distance = min(CONFIG["MAX_PREVIEW_DISTANCE"],
                                       CONFIG["BASE_PREVIEW_DISTANCE"] + current_speed / 10)
                waypoint = map.get_waypoint(vehicle_transform.location, project_to_road=True,
                                            lane_type=carla.LaneType.Driving)
                next_waypoints = waypoint.next(preview_distance)
                if next_waypoints:
                    next_waypoint = next_waypoints[0]
                    target_steer = get_steer(vehicle_transform, next_waypoint.transform, current_speed)

                    if abs(target_steer - current_steer) < CONFIG["STEER_DEAD_ZONE"]:
                        target_steer = current_steer

                    target_steer = current_steer * CONFIG["STEER_SMOOTH_FACTOR"] + target_steer * (
                                1 - CONFIG["STEER_SMOOTH_FACTOR"])
                    steer_change = target_steer - current_steer
                    steer_change = max(-CONFIG["MAX_STEER_CHANGE"], min(CONFIG["MAX_STEER_CHANGE"], steer_change))
                    current_steer = max(-1.0, min(1.0, current_steer + steer_change))
                    control.steer = current_steer

                speed_error = target_speed - current_speed
                if speed_error > 1:
                    target_throttle = min(CONFIG["MAX_THROTTLE"], CONFIG["SPEED_ADJUST_SMOOTH"] * speed_error)
                    current_throttle = current_throttle * 0.8 + target_throttle * 0.2
                    control.throttle = current_throttle
                    control.brake = 0.0
                elif speed_error < -1:
                    target_brake = min(0.6, abs(CONFIG["SPEED_ADJUST_SMOOTH"] * speed_error))
                    control.brake = target_brake
                    control.throttle = 0.0
                    current_throttle = 0.0
                else:
                    control.throttle = 0.15
                    control.brake = 0.0

            vehicle.apply_control(control)

            if image_surface[0] is not None:
                surface = pygame.image.frombuffer(image_surface[0].tobytes(),
                                                  (CONFIG["CAMERA_WIDTH"], CONFIG["CAMERA_HEIGHT"]), "RGB")
                display.blit(surface, (0, 0))

                pygame.draw.rect(display, (0, 0, 0), (10, 10, 350, 140), border_radius=5)
                speed_text = font.render(f" {current_speed:.1f} km/h", True, (0, 255, 0))
                limit_text = font.render(f" {current_speed_limit} km/h", True, (255, 255, 0))
                light_text = font.render(f" {final_light_state}", True,
                                         (255, 0, 0) if final_light_state == "Red" else (0, 255, 0))
                acc_text = font.render(f"ACC状态: {'激活' if acc_active else '待机'}", True,
                                       (255, 165, 0) if acc_active else (200, 200, 200))
                distance_text = font.render(f" {front_vehicle_distance:.1f}m", True,
                                            (255, 0, 0) if front_vehicle_distance < 10 else (0, 255, 0))

                display.blit(speed_text, (20, 20))
                display.blit(limit_text, (20, 45))
                display.blit(light_text, (20, 70))
                display.blit(acc_text, (20, 95))
                display.blit(distance_text, (20, 120))

                for label, info, conf, bbox in detected_list:
                    x1, y1, x2, y2 = bbox
                    pygame.draw.rect(display, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)
                    label_text = font.render(f"{label} {info} {conf:.2f}", True, (255, 255, 255), (0, 0, 0))
                    display.blit(label_text, (x1, y1 - 25))

                pygame.display.flip()

            clock.tick(30)

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