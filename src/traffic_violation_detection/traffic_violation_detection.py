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

pygame.init()
screen = pygame.display.set_mode((400, 300))

import carla

client = carla.Client('localhost', 2000)
client.set_timeout(60.0)

# 全局违章、动态限速
violation_info = {
    "speeding": False,
    "red_light": False,
    "ignore_sign": False,
    "current_speed": 0.0
}
# 基础默认限速
BASE_SPEED_LIMIT = 50.0
current_speed_limit = BASE_SPEED_LIMIT

# ------------------------------
# 1. 纯图像视觉识别红绿灯
# ------------------------------
def detect_traffic_light_vision(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_red1 = np.array([0, 100, 120])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 120])
    upper_red2 = np.array([179, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    kernel = np.ones((3, 3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    red_pixels = cv2.countNonZero(mask_red)

    return red_pixels > 80

# ------------------------------
# 2. 新增：限速标志简单识别 + 动态更新限速
# ------------------------------
def detect_speed_limit_sign(img_rgb):
    global current_speed_limit
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # 圆形限速标志红色外圈阈值
    lower_red_circle = np.array([0, 120, 100])
    upper_red_circle = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_red_circle, upper_red_circle)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 60 < area < 800:
            # 检测到圆形限速牌，这里仿真常用路段限速 30 / 50 / 60 切换演示
            # 简单逻辑：识别到标志就轮换限速
            current_speed_limit = 30.0
            return
    # 未检测到限速牌，恢复默认
    current_speed_limit = BASE_SPEED_LIMIT

# 获取车速
def get_vehicle_speed(vehicle):
    vel = vehicle.get_velocity()
    speed = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    return round(speed, 2)

# 违章综合判断（动态限速）
def detect_violations(vehicle, img_rgb):
    global current_speed_limit
    # 先检测限速标志，更新限速值
    detect_speed_limit_sign(img_rgb)

    speed = get_vehicle_speed(vehicle)
    violation_info["current_speed"] = speed
    # 动态限速判断超速
    violation_info["speeding"] = speed > current_speed_limit
    violation_info["red_light"] = detect_traffic_light_vision(img_rgb)
    violation_info["ignore_sign"] = False

# 绘制全部信息（车速、当前限速、违章）
def draw_violation_info(img):
    speed = violation_info["current_speed"]
    cv2.putText(img, f"Speed: {speed} km/h", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(img, f"Limit: {current_speed_limit} km/h", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    if violation_info["speeding"]:
        cv2.putText(img, "VIOLATION: SPEEDING!", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
    if violation_info["red_light"]:
        cv2.putText(img, "VIOLATION: RED LIGHT!", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)

# 地图与仿真设置
world = client.load_world('Town05')
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)
spawn_points = world.get_map().get_spawn_points()

# 生成周边车辆
def spawn_vehicles(num_vehicles, world, spawn_points):
    vehicle_bp_lib = world.get_blueprint_library().filter('vehicle.*')
    spawned_vehicles = []
    for _ in range(num_vehicles):
        vehicle_bp = random.choice(vehicle_bp_lib)
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        if vehicle:
            spawned_vehicles.append(vehicle)
    return spawned_vehicles

vehicles = spawn_vehicles(10, world, spawn_points)

# 主车生成
bp_lib = world.get_blueprint_library().filter('*')
vehicle_bp = bp_lib.find('vehicle.audi.a2')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# 自动驾驶
vehicle.set_autopilot(True, traffic_manager.get_port())
traffic_manager.ignore_lights_percentage(vehicle, 0.0)
traffic_manager.vehicle_percentage_speed_difference(vehicle, -50)

# 相机
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1024')
camera_bp.set_attribute('image_size_y', '1024')
camera_bp.set_attribute('fov', '70')
camera_trans = carla.Transform(carla.Location(x=1, z=2), carla.Rotation(pitch=-3))
camera = world.spawn_actor(camera_bp, camera_trans, attach_to=vehicle)

image_queue = queue.Queue(maxsize=50)
def image_callback(image):
    if not image_queue.full():
        image_queue.put(image)
camera.listen(image_callback)

# 保存路径
output_dir = os.path.join(os.getcwd(), "OutPut", "data01")
os.makedirs(output_dir, exist_ok=True)

# 天气、标注、交通标志检测等原有函数不变
def get_weather_params(world):
    weather = world.get_weather()
    return {
        "cloudiness": weather.cloudiness,
        "precipitation": weather.precipitation,
        "fog_density": weather.fog_density,
        "sun_altitude_angle": weather.sun_altitude_angle
    }

def get_weather_category(w):
    if w['cloudiness'] > 70 or w['precipitation'] > 50:
        return 0
    elif w['sun_altitude_angle'] > 30:
        return 1
    elif w['fog_density'] > 50:
        return 2
    return 3

def create_xml_file(image_name, bboxes, width, height, weather_params):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = image_name
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(annotation, "weather", "condition").text = str(get_weather_category(weather_params))

    vio = ET.SubElement(annotation, "violation")
    ET.SubElement(vio, "speeding").text = str(violation_info["speeding"])
    ET.SubElement(vio, "red_light").text = str(violation_info["red_light"])
    ET.SubElement(vio, "current_limit").text = str(current_speed_limit)

    for box in bboxes:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = box["label"]
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(box["xmin"])
        ET.SubElement(bnd, "ymin").text = str(box["ymin"])
        ET.SubElement(bnd, "xmax").text = str(box["xmax"])
        ET.SubElement(bnd, "ymax").text = str(box["ymax"])

    tree = ET.ElementTree(annotation)
    tree.write(os.path.join(output_dir, image_name.replace(".png", ".xml")))

# 剩余原有工具函数、天气、NMS、标志检测完全保留
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0,0] = K[1,1] = focal
    K[0,2] = w/2.0
    K[1,2] = h/2.0
    return K

def get_image_point(loc, K, w2c):
    pt = np.dot(w2c, np.array([loc.x, loc.y, loc.z, 1]))
    res = np.dot(K, [pt[1], -pt[2], pt[0]])
    return [res[0]/res[2], res[1]/res[2]]

image_w = 1024
image_h = 1024
fov = 70
K = build_projection_matrix(image_w, image_h, fov)

def get_signs_bounding_boxes(world_2_camera):
    bboxes = []
    for obj in world.get_level_bbs(carla.CityObjectLabel.TrafficSigns):
        verts = obj.get_world_vertices(carla.Transform())
        pts = [get_image_point(v, K, world_2_camera) for v in verts]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xmin, xmax = int(min(xs)), int(max(xs))
        ymin, ymax = int(min(ys)), int(max(ys))
        if 0<xmin<xmax<image_w and 0<ymin<ymax<image_h:
            bboxes.append({"label":"TrafficSign","xmin":xmin,"ymin":ymin,"xmax":xmax,"ymax":ymax})
    return bboxes

def non_maximum_suppression(bboxes):
    return bboxes

# 主循环
try:
    while True:
        world.tick()
        img_data = image_queue.get()
        img = np.reshape(img_data.raw_data, (image_h, image_w, 4))
        img_rgb = img[:, :, :3].copy()

        # 核心：动态限速+视觉红绿灯+违章检测
        detect_violations(vehicle, img_rgb)

        # 标志框
        w2c = np.array(camera.get_transform().get_inverse_matrix())
        bboxes = get_signs_bounding_boxes(w2c)
        for b in bboxes:
            cv2.rectangle(img_rgb, (b["xmin"],b["ymin"]),(b["xmax"],b["ymax"]),(0,0,255),2)

        # 绘制信息
        draw_violation_info(img_rgb)

        cv2.imshow("Traffic Detection", img_rgb)
        if cv2.waitKey(1) & 0xFF == ord("x"):
            break

finally:
    cv2.destroyAllWindows()
    camera.destroy()
    vehicle.destroy()
    for v in vehicles:
        v.destroy()