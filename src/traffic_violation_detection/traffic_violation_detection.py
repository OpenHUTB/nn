import sys
import glob
import random
import numpy as np
import queue
import cv2
import os
import xml.etree.ElementTree as ET
import time
import pygame

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((400, 300))

import carla

# Connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(60.0)

# 全局违章变量
violation_info = {
    "speeding": False,
    "red_light": False,
    "ignore_sign": False,
    "current_speed": 0.0
}
# 新增：违章次数统计
violation_count = {
    "speeding_cnt": 0,
    "red_light_cnt": 0
}
# 记录上一帧违章状态，避免同一违章重复计数
last_violation = {
    "speeding": False,
    "red_light": False
}

SPEED_LIMIT = 50.0  # 固定限速


# Load a different map
def load_map(map_name):
    return client.load_world(map_name)


# Function to spawn vehicles
def spawn_vehicles(num_vehicles, world, spawn_points):
    vehicle_bp_lib = world.get_blueprint_library().filter('vehicle.*')
    spawned_vehicles = []

    for _ in range(num_vehicles):
        vehicle_bp = random.choice(vehicle_bp_lib)
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            spawned_vehicles.append(vehicle)
        else:
            print("Failed to spawn vehicle")

    return spawned_vehicles


# 获取车辆当前速度（km/h）
def get_vehicle_speed(vehicle):
    vel = vehicle.get_velocity()
    speed = 3.6 * np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    return round(speed, 2)


# ------------------------------
# 夜间/弱光 红绿灯识别优化
# ------------------------------
def detect_traffic_light_vision(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)

    if brightness < 50:
        img_rgb = cv2.convertScaleAbs(img_rgb, alpha=1.8, beta=30)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    lower_red1 = np.array([0, 120, 120])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 120, 120])
    upper_red2 = np.array([179, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    kernel = np.ones((2, 2), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    red_pixels = cv2.countNonZero(mask_red)
    threshold = 40 if brightness < 50 else 80
    return red_pixels > threshold


# 违章判断 + 次数统计
def detect_violations(vehicle, img_rgb):
    speed = get_vehicle_speed(vehicle)
    violation_info["current_speed"] = speed
    violation_info["speeding"] = speed > SPEED_LIMIT
    violation_info["red_light"] = detect_traffic_light_vision(img_rgb)
    violation_info["ignore_sign"] = False

    # 违章次数累加：只在从无违章变有违章时计数一次
    if violation_info["speeding"] and not last_violation["speeding"]:
        violation_count["speeding_cnt"] += 1
    if violation_info["red_light"] and not last_violation["red_light"]:
        violation_count["red_light_cnt"] += 1

    # 更新上一帧状态
    last_violation["speeding"] = violation_info["speeding"]
    last_violation["red_light"] = violation_info["red_light"]


# 绘制违章信息 + 实时违章次数
def draw_violation_info(img):
    speed = violation_info["current_speed"]
    cv2.putText(img, f"Speed: {speed} km/h", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # 显示违章统计次数
    cv2.putText(img, f"Speeding Count: {violation_count['speeding_cnt']}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
    cv2.putText(img, f"RedLight Count: {violation_count['red_light_cnt']}", (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    if violation_info["speeding"]:
        cv2.putText(img, "VIOLATION: SPEEDING!", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    if violation_info["red_light"]:
        cv2.putText(img, "VIOLATION: RED LIGHT!", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)


# Define the map you want to load
world = client.load_world('Town05')

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Initialize Traffic Manager
traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)

# Get map spawn points
spawn_points = world.get_map().get_spawn_points()

# Spawn vehicles and walkers
num_vehicles = 10
vehicles = spawn_vehicles(num_vehicles, world, spawn_points)

# Get the blueprint library
bp_lib = world.get_blueprint_library().filter('*')

# Spawn vehicle
vehicle_bp = bp_lib.find('vehicle.audi.a2')
try:
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle")
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)

# Disable Autopilot for manual control
vehicle.set_autopilot(True, traffic_manager.get_port())
print("✅ 自动驾驶已启用")
traffic_manager.ignore_lights_percentage(vehicle, 0.0)
traffic_manager.vehicle_percentage_speed_difference(vehicle, -50)

# Spawn camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1024')
camera_bp.set_attribute('image_size_y', '1024')
camera_bp.set_attribute('fov', '70')

camera_init_trans = carla.Transform(carla.Location(x=1, z=2), carla.Rotation(pitch=-3))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue(maxsize=50)


# Camera listener
def image_callback(image):
    if not image_queue.full():
        image_queue.put(image)


camera.listen(image_callback)

# 保存路径
current_dirc = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dirc, '..', '..', '..'))
data_path = os.path.join(project_root, "OutPut", "data01")
output_dir = data_path
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Function to get current weather parameters
def get_weather_params(world):
    weather = world.get_weather()
    return {
        "cloudiness": weather.cloudiness,
        "precipitation": weather.precipitation,
        "precipitation_deposits": weather.precipitation_deposits,
        "wind_intensity": weather.wind_intensity,
        "sun_azimuth_angle": weather.sun_azimuth_angle,
        "sun_altitude_angle": weather.sun_altitude_angle,
        "fog_density": weather.fog_density,
        "wetness": weather.wetness
    }


# Function to build the projection matrix
def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]


# Get the attributes from the camera
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()
K = build_projection_matrix(image_w, image_h, fov)

DISTANCE_THRESHOLD = 50.0
captured_sign_locations = set()
last_capture_time = 0
capture_cooldown = 5


def dot_product(v1, v2):
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def get_signs_bounding_boxes(vehicle_transform, camera_transform, K, world_2_camera):
    global captured_sign_locations, last_capture_time
    bounding_boxes = []
    camera_location = camera_transform.location
    vehicle_location = vehicle_transform.location

    vehicle_right_vector = vehicle_transform.get_right_vector()

    for obj in world.get_level_bbs(carla.CityObjectLabel.TrafficSigns):
        distance = obj.location.distance(vehicle_location)
        vector_to_object = obj.location - vehicle_location

        if distance < DISTANCE_THRESHOLD:
            right_side_dot_product = dot_product(vehicle_right_vector, vector_to_object)
            if right_side_dot_product > 0:
                vector_to_camera = obj.location - camera_location
                camera_dot_product = dot_product(camera_transform.get_forward_vector(), vector_to_camera)

                sign_location_tuple = (round(obj.location.x, 2), round(obj.location.y, 2), round(obj.location.z, 2))
                if camera_dot_product > 0 and sign_location_tuple not in captured_sign_locations:
                    verts = [v for v in obj.get_world_vertices(carla.Transform())]
                    x_coords = [get_image_point(v, K, world_2_camera)[0] for v in verts]
                    y_coords = [get_image_point(v, K, world_2_camera)[1] for v in verts]
                    xmin, xmax = int(min(x_coords)), int(max(x_coords))
                    ymin, ymax = int(min(y_coords)), int(max(y_coords))

                    area = (xmax - xmin) * (ymax - ymin)
                    min_area_threshold = 10

                    if xmin >= 0 and ymin >= 0 and xmax < image_w and ymax < image_h:
                        aspect_ratio = (xmax - xmin) / float(ymax - ymin) if (ymax - ymin) != 0 else 0
                        if area > min_area_threshold and 0.5 < aspect_ratio < 2.0:
                            bounding_boxes.append(
                                {'label': 'TrafficSign', 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

                            current_time = time.time()
                            if current_time - last_capture_time > capture_cooldown:
                                captured_sign_locations.add(sign_location_tuple)
                                last_capture_time = current_time

    return bounding_boxes


# 保存XML
def create_xml_file(image_name, bboxes, width, height, weather_params):
    annotation = ET.Element("annotation")
    filename = ET.SubElement(annotation, "filename")
    filename.text = image_name

    size = ET.SubElement(annotation, "size")
    width_elem = ET.SubElement(size, "width")
    height_elem = ET.SubElement(size, "height")
    depth_elem = ET.SubElement(size, "depth")
    width_elem.text = str(width)
    height_elem.text = str(height)
    depth_elem.text = "3"

    weather_category = get_weather_category(weather_params)
    weather = ET.SubElement(annotation, "weather")
    condition = ET.SubElement(weather, "condition")
    condition.text = str(weather_category)

    violation_node = ET.SubElement(annotation, "violation")
    ET.SubElement(violation_node, "speeding").text = str(violation_info["speeding"])
    ET.SubElement(violation_node, "red_light").text = str(violation_info["red_light"])

    for bbox in bboxes:
        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = bbox['label']

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(bbox['xmin'])
        ET.SubElement(bndbox, "ymin").text = str(bbox['ymin'])
        ET.SubElement(bndbox, "xmax").text = str(bbox['xmax'])
        ET.SubElement(bndbox, "ymax").text = str(bbox['ymax'])

    tree = ET.ElementTree(annotation)
    xml_file = os.path.join(output_dir, image_name.replace(".png", ".xml"))
    tree.write(xml_file)


# 天气切换
weather_conditions = [
    'rainy',
    'sunny',
    'night',
    'foggy'
]


def update_weather(world, condition):
    if condition == 'rainy':
        weather = carla.WeatherParameters(
            cloudiness=80.0,
            precipitation=80.0,
            precipitation_deposits=80.0,
            wind_intensity=10.0,
            sun_azimuth_angle=270.0,
            sun_altitude_angle=10.0,
            fog_density=10.0,
            wetness=70.0
        )
    elif condition == 'sunny':
        weather = carla.WeatherParameters(
            cloudiness=20.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=5.0,
            sun_azimuth_angle=180.0,
            sun_altitude_angle=60.0,
            fog_density=0.0,
            wetness=0.0
        )
    elif condition == 'night':
        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=3.0,
            sun_azimuth_angle=0.0,
            sun_altitude_angle=-5.0,
            fog_density=0.0,
            wetness=0.0
        )
    elif condition == 'foggy':
        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=3.0,
            sun_azimuth_angle=0.0,
            sun_altitude_angle=0.0,
            fog_density=60.0,
            wetness=0.0
        )
    else:
        raise ValueError("Unknown weather condition")
    world.set_weather(weather)


def get_weather_category(w):
    if w['cloudiness'] > 70 or w['precipitation'] > 50:
        return 0
    elif w['sun_altitude_angle'] > 30:
        return 1
    elif w['fog_density'] > 50:
        return 2
    else:
        return 3


# 手动控制
def handle_input(vehicle):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    control = vehicle.get_control()
    if keys[pygame.K_w]:
        control.throttle = 1.0
    if keys[pygame.K_s]:
        control.brake = 1.0
    if keys[pygame.K_a]:
        control.steer = -1.0
    if keys[pygame.K_d]:
        control.steer = 1.0
    if keys[pygame.K_r]:
        control.throttle = 1.0
        control.reverse = True
    if keys[pygame.K_s] and control.reverse:
        control.brake = 1.0

    vehicle.apply_control(control)


# NMS
def compute_iou(box1, box2):
    x1 = max(box1['xmin'], box2['xmin'])
    y1 = max(box1['ymin'], box2['ymin'])
    x2 = min(box1['xmax'], box2['xmax'])
    y2 = min(box1['ymax'], box2['ymax'])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    box2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])

    if box1_area == 0 or box2_area == 0:
        return 0.0
    return inter_area / float(box1_area + box2_area - inter_area)


def non_maximum_suppression(bboxes, iou_threshold=0.2):
    if len(bboxes) == 0:
        return []

    bboxes = sorted(bboxes, key=lambda x: (x['xmax'] - x['xmin']) * (x['ymax'] - x['ymin']), reverse=True)
    final_bboxes = []

    while bboxes:
        current_box = bboxes.pop(0)
        final_bboxes.append(current_box)
        bboxes = [box for box in bboxes if compute_iou(current_box, box) < iou_threshold]
    return final_bboxes


# 主循环
weather_transition_interval = 10
last_weather_change_time = time.time()
current_condition_index = 0

try:
    while True:
        world.tick()
        time.sleep(0.033)
        pygame.event.pump()
        handle_input(vehicle)

        image = image_queue.get()
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        img_rgb = img[:, :, :3].astype(np.uint8)

        # 违章检测
        detect_violations(vehicle, img_rgb)

        current_time = time.time()
        if current_time - last_weather_change_time > weather_transition_interval:
            current_condition = weather_conditions[current_condition_index]
            update_weather(world, current_condition)
            current_condition_index = (current_condition_index + 1) % len(weather_conditions)
            last_weather_change_time = current_time

        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        bboxes = get_signs_bounding_boxes(vehicle.get_transform(), camera.get_transform(), K, world_2_camera)
        bboxes = non_maximum_suppression(bboxes)

        if bboxes:
            img_rgb_with_bboxes = img_rgb.copy()
            for box in bboxes:
                cv2.rectangle(img_rgb_with_bboxes, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), (0, 0, 255),
                              2)
            draw_violation_info(img_rgb_with_bboxes)

            image_name = f"image_{int(time.time())}.png"
            cv2.imwrite(os.path.join(output_dir, image_name), img_rgb)
            create_xml_file(image_name, bboxes, image_w, image_h, get_weather_params(world))
            cv2.imshow('CARLA - Violation Detection', img_rgb_with_bboxes)
        else:
            draw_violation_info(img_rgb)
            cv2.imshow('CARLA - Violation Detection', img_rgb)

        if cv2.waitKey(10) & 0xFF == ord('x'):
            print("X key pressed")
            break

finally:
    cv2.destroyAllWindows()
    vehicle.destroy()
    camera.destroy()
    pygame.quit()
    for v in vehicles:
        v.destroy()