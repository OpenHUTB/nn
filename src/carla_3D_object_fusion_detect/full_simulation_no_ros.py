import carla
import time
import random
import os
import cv2
import numpy as np
import sys
import math
import traceback
import csv
from datetime import datetime

# ====================== 路径 ======================
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
p1 = os.path.dirname(current_dir)
p2 = os.path.dirname(p1)
p3 = os.path.dirname(p2)

image_folder = os.path.join(p3, "images")
lidar_folder = os.path.join(p3, "lidar")
collision_folder = os.path.join(p3, "collision")
semantic_folder = os.path.join(p3, "semantic")
trajectory_folder = os.path.join(p3, "trajectory")
os.makedirs(image_folder, exist_ok=True)
os.makedirs(lidar_folder, exist_ok=True)
os.makedirs(collision_folder, exist_ok=True)
os.makedirs(semantic_folder, exist_ok=True)
os.makedirs(trajectory_folder, exist_ok=True)

# ====================== 配置 ======================
SAVE_INTERVAL = 5 * 60
COLLISION_COOLDOWN_SEC = 3.0

# 动态交通配置（基础值，会随速度自适应）
BASE_SPAWN_INTERVAL = 5.0
BASE_MAX_VEHICLES = 4
BASE_MAX_PEDESTRIANS = 5
MAX_SPAWN_ATTEMPTS = 1
LOW_SPEED_THRESHOLD = 20   # km/h
HIGH_SPEED_THRESHOLD = 60  # km/h

# 实际使用的动态值（会在主循环中更新）
current_spawn_interval = BASE_SPAWN_INTERVAL
current_max_vehicles = BASE_MAX_VEHICLES
current_max_pedestrians = BASE_MAX_PEDESTRIANS
current_remove_distance = 80  # 随速度自适应

SPAWN_RADIUS = 40
REMOVE_DISTANCE_BASE = 80
REMOVE_DISTANCE_HIGH = 120

# 障碍物警告配置
OBSTACLE_WARNING_DISTANCE = 10.0
OBSTACLE_DANGER_DISTANCE = 5.0
OBSTACLE_FOV_ANGLE = 60.0
OBSTACLE_MAX_HEIGHT = 2.0

# 轨迹记录文件
trajectory_csv_path = os.path.join(trajectory_folder, f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
trajectory_file = None
trajectory_writer = None

# 天气预设
weather_presets = {
    "sunny": carla.WeatherParameters(
        cloudiness=0.0, precipitation=0.0, precipitation_deposits=0.0,
        wind_intensity=0.0, sun_azimuth_angle=0.0, sun_altitude_angle=70.0,
        fog_density=0.0, fog_distance=0.0, wetness=0.0
    ),
    "rainy": carla.WeatherParameters(
        cloudiness=90.0, precipitation=90.0, precipitation_deposits=90.0,
        wind_intensity=20.0, sun_azimuth_angle=0.0, sun_altitude_angle=30.0,
        fog_density=10.0, fog_distance=100.0, wetness=90.0
    ),
    "foggy": carla.WeatherParameters(
        cloudiness=50.0, precipitation=0.0, precipitation_deposits=0.0,
        wind_intensity=5.0, sun_azimuth_angle=0.0, sun_altitude_angle=20.0,
        fog_density=80.0, fog_distance=30.0, wetness=20.0
    ),
    "night": carla.WeatherParameters(
        cloudiness=20.0, precipitation=0.0, precipitation_deposits=0.0,
        wind_intensity=5.0, sun_azimuth_angle=0.0, sun_altitude_angle=-30.0,
        fog_density=0.0, fog_distance=0.0, wetness=0.0
    )
}
weather_names = ["sunny", "rainy", "foggy", "night"]
current_weather_idx = 1

# 全局变量
last_save_time = time.time()
latest_camera = None
latest_follow = None
latest_lidar = None
latest_semantic = None
display_mode = "rgb"

collision_cooldown = False
collision_cooldown_time = 0
error_count = 0
frame_count = 0
fps = 0
last_fps_time = time.time()

closest_obstacle_distance = float('inf')
obstacle_warning_active = False

spawned_vehicles = []
spawned_pedestrians = []
all_spawned_actors = []
last_spawn_time = time.time()
vehicle_blueprints = []
pedestrian_blueprints = []

latest_gnss = None
latest_imu = None
weather_change_time = 0

# 相机投影矩阵（用于车道线投影，需在传感器就绪后计算）
camera_intrinsic = None
camera_follow_transform = None

# ====================== 连接 CARLA ======================
def connect_carla(retries=3):
    for i in range(retries):
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(10.0)
            client.get_server_version()
            print(f"✅ 连接成功，版本: {client.get_server_version()}")
            return client
        except Exception as e:
            print(f"连接失败 ({i+1}/{retries}): {e}")
            time.sleep(2)
    print("❌ 无法连接 CARLA，请确保模拟器已启动")
    sys.exit(1)

client = connect_carla()
world = client.get_world()

world.set_weather(weather_presets["rainy"])
print("✅ 初始天气: 雨天 (按 W 切换)")

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_points = world.get_map().get_spawn_points()
if not spawn_points:
    raise RuntimeError("地图无生成点")
spawn_point = random.choice(spawn_points)
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
if vehicle is None:
    raise RuntimeError("自车生成失败")
vehicle.set_autopilot(True)
print("✅ 自车已生成")

def prepare_blueprints():
    global vehicle_blueprints, pedestrian_blueprints
    try:
        all_vehicles = blueprint_library.filter('vehicle.*')
        vehicle_blueprints = [v for v in all_vehicles 
                              if int(v.get_attribute('number_of_wheels')) == 4 
                              and 'tesla' not in v.id]
        pedestrian_blueprints = list(blueprint_library.filter('walker.pedestrian.*'))
        print(f"✅ 车辆蓝图: {len(vehicle_blueprints)}, 行人蓝图: {len(pedestrian_blueprints)}")
    except Exception as e:
        print(f"准备蓝图失败: {e}")
        vehicle_blueprints = []
        pedestrian_blueprints = []

prepare_blueprints()

def is_location_occupied(location, radius=2.5):
    try:
        actors = world.get_actors()
        for actor in actors:
            if actor.id == vehicle.id:
                continue
            if actor.get_location().distance(location) < radius:
                return True
        return False
    except:
        return True

def spawn_random_vehicle_near(ego_location):
    if len(spawned_vehicles) >= current_max_vehicles or not vehicle_blueprints:
        return None
    available = []
    for sp in spawn_points:
        if sp.location.distance(ego_location) < SPAWN_RADIUS:
            if not is_location_occupied(sp.location, radius=3.0):
                available.append(sp)
    if not available:
        return None
    chosen = random.choice(available)
    blueprint = random.choice(vehicle_blueprints)
    try:
        new_vehicle = world.spawn_actor(blueprint, chosen)
        if new_vehicle:
            new_vehicle.set_autopilot(True)
            spawned_vehicles.append(new_vehicle)
            all_spawned_actors.append(new_vehicle)
        return new_vehicle
    except Exception as e:
        print(f"生成车辆异常: {e}")
    return None

def spawn_random_pedestrian_near(ego_location):
    if len(spawned_pedestrians) >= current_max_pedestrians or not pedestrian_blueprints:
        return None
    angle = random.uniform(0, 2*math.pi)
    radius = random.uniform(12, SPAWN_RADIUS)
    x = ego_location.x + radius * math.cos(angle)
    y = ego_location.y + radius * math.sin(angle)
    z = ego_location.z + 0.5
    spawn_loc = carla.Location(x=x, y=y, z=z)
    if is_location_occupied(spawn_loc, radius=1.5):
        return None
    blueprint = random.choice(pedestrian_blueprints)
    try:
        new_walker = world.spawn_actor(blueprint, carla.Transform(spawn_loc))
        if new_walker:
            controller_bp = blueprint_library.find('controller.ai.walker')
            controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=new_walker)
            if controller:
                controller.start()
                target_angle = random.uniform(0, 2*math.pi)
                target_dist = random.uniform(10, 20)
                target_loc = spawn_loc + carla.Location(x=target_dist*math.cos(target_angle),
                                                        y=target_dist*math.sin(target_angle))
                controller.go_to_location(target_loc)
                all_spawned_actors.append(controller)
            spawned_pedestrians.append(new_walker)
            all_spawned_actors.append(new_walker)
        return new_walker
    except Exception as e:
        print(f"生成行人异常: {e}")
    return None

def remove_far_actors(ego_location):
    global spawned_vehicles, spawned_pedestrians, all_spawned_actors
    to_remove_v = [v for v in spawned_vehicles if v.get_location().distance(ego_location) > current_remove_distance]
    for v in to_remove_v:
        try:
            v.destroy()
            spawned_vehicles.remove(v)
            all_spawned_actors.remove(v)
        except:
            pass
    to_remove_w = [w for w in spawned_pedestrians if w.get_location().distance(ego_location) > current_remove_distance]
    for w in to_remove_w:
        try:
            for a in all_spawned_actors[:]:
                if hasattr(a, 'parent_id') and a.parent_id == w.id:
                    a.stop()
                    a.destroy()
                    all_spawned_actors.remove(a)
                    break
            w.destroy()
            spawned_pedestrians.remove(w)
            all_spawned_actors.remove(w)
        except:
            pass

def init_trajectory_csv():
    global trajectory_file, trajectory_writer
    try:
        trajectory_file = open(trajectory_csv_path, 'w', newline='')
        trajectory_writer = csv.writer(trajectory_file)
        trajectory_writer.writerow([
            "timestamp", "lat", "lon", "alt", 
            "velocity_x", "velocity_y", "velocity_z", "speed_kmh",
            "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z",
            "compass", "roll", "pitch", "yaw"
        ])
        print(f"📊 轨迹记录文件: {trajectory_csv_path}")
    except Exception as e:
        print(f"创建轨迹文件失败: {e}")

def save_trajectory_point():
    global trajectory_writer, latest_gnss, latest_imu, vehicle
    if trajectory_writer is None:
        return
    try:
        vel = vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        transform = vehicle.get_transform()
        roll = math.radians(transform.rotation.roll)
        pitch = math.radians(transform.rotation.pitch)
        yaw = math.radians(transform.rotation.yaw)
        lat, lon, alt = 0.0, 0.0, 0.0
        if latest_gnss is not None:
            lat = latest_gnss.latitude
            lon = latest_gnss.longitude
            alt = latest_gnss.altitude
        accel_x, accel_y, accel_z = 0.0, 0.0, 0.0
        gyro_x, gyro_y, gyro_z = 0.0, 0.0, 0.0
        compass = 0.0
        if latest_imu is not None:
            accel_x = latest_imu.accelerometer.x
            accel_y = latest_imu.accelerometer.y
            accel_z = latest_imu.accelerometer.z
            gyro_x = latest_imu.gyroscope.x
            gyro_y = latest_imu.gyroscope.y
            gyro_z = latest_imu.gyroscope.z
            compass = latest_imu.compass
        row = [
            time.time(), lat, lon, alt,
            vel.x, vel.y, vel.z, speed_kmh,
            accel_x, accel_y, accel_z,
            gyro_x, gyro_y, gyro_z,
            compass, roll, pitch, yaw
        ]
        trajectory_writer.writerow(row)
        trajectory_file.flush()
        print(f"📌 轨迹点已保存 (时间: {datetime.now().strftime('%H:%M:%S')})")
    except Exception as e:
        print(f"保存轨迹点失败: {e}")

def cycle_weather():
    global current_weather_idx, weather_change_time
    current_weather_idx = (current_weather_idx + 1) % len(weather_names)
    weather_name = weather_names[current_weather_idx]
    world.set_weather(weather_presets[weather_name])
    weather_change_time = time.time()
    print(f"🌤️ 切换天气: {weather_name.upper()}")

def spawn_safe_sensor(bp_name, transform, attach_to, attributes=None, retries=2):
    for attempt in range(retries):
        try:
            bp = blueprint_library.find(bp_name)
            if bp is None:
                print(f"找不到蓝图 {bp_name}")
                return None
            if attributes:
                for key, value in attributes.items():
                    bp.set_attribute(key, str(value))
            actor = world.spawn_actor(bp, transform, attach_to=attach_to)
            if actor:
                return actor
            else:
                print(f"生成 {bp_name} 失败，重试 {attempt+1}/{retries}")
                time.sleep(0.5)
        except Exception as e:
            print(f"传感器 {bp_name} 生成异常 (尝试 {attempt+1}/{retries}): {e}")
            time.sleep(0.5)
    print(f"❌ 传感器 {bp_name} 最终生成失败")
    return None

# 传感器创建
camera_front = spawn_safe_sensor('sensor.camera.rgb',
                                 carla.Transform(carla.Location(x=1.5, z=2.4)),
                                 vehicle,
                                 {'image_size_x': 800, 'image_size_y': 600, 'fov': 110})
camera_follow = spawn_safe_sensor('sensor.camera.rgb',
                                  carla.Transform(carla.Location(x=-5.0, y=0, z=3.0), carla.Rotation(pitch=-10)),
                                  vehicle,
                                  {'image_size_x': 1024, 'image_size_y': 768, 'fov': 90})
lidar = spawn_safe_sensor('sensor.lidar.ray_cast',
                          carla.Transform(carla.Location(x=0, z=2.5)),
                          vehicle,
                          {'range': 100, 'points_per_second': 50000, 'rotation_frequency': 10})
collision_sensor = spawn_safe_sensor('sensor.other.collision',
                                     carla.Transform(),
                                     vehicle)
semantic_camera = spawn_safe_sensor('sensor.camera.semantic_segmentation',
                                    carla.Transform(carla.Location(x=-5.0, y=0, z=3.0), carla.Rotation(pitch=-10)),
                                    vehicle,
                                    {'image_size_x': 1024, 'image_size_y': 768, 'fov': 90})
gnss = spawn_safe_sensor('sensor.other.gnss',
                         carla.Transform(carla.Location(x=0, z=1.0)),
                         vehicle,
                         {'sensor_tick': 1.0})
imu = spawn_safe_sensor('sensor.other.imu',
                        carla.Transform(carla.Location(x=0, z=1.0)),
                        vehicle,
                        {'sensor_tick': 0.05})

if camera_front is None or camera_follow is None or lidar is None or semantic_camera is None:
    print("❌ 核心传感器生成失败，请检查 CARLA 服务器状态")
    sys.exit(1)

# ====================== 获取相机内参（用于车道线投影） ======================
def get_camera_intrinsic(sensor):
    """从相机传感器获取内参矩阵 K"""
    try:
        # 图像尺寸
        w = int(sensor.attributes['image_size_x'])
        h = int(sensor.attributes['image_size_y'])
        fov = float(sensor.attributes['fov'])
        # 计算焦距（像素单位）
        f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
        cx = w / 2.0
        cy = h / 2.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        return K
    except:
        return None

# 等传感器生成后获取内参
time.sleep(1)  # 简单延迟确保传感器已生成
if camera_follow is not None:
    camera_intrinsic = get_camera_intrinsic(camera_follow)
    camera_follow_transform = camera_follow.get_transform()
    print(f"📷 相机内参矩阵: {camera_intrinsic}")

# ====================== 回调函数 ======================
def on_camera_front(data):
    global latest_camera
    try:
        img = np.frombuffer(data.raw_data, dtype=np.uint8)
        img = img.reshape((data.height, data.width, 4))[:, :, :3]
        latest_camera = img
    except Exception as e:
        global error_count
        error_count += 1
        if error_count % 100 == 1:
            print(f"前置相机回调错误: {e}")

def on_camera_follow(data):
    global latest_follow
    try:
        img = np.frombuffer(data.raw_data, dtype=np.uint8)
        img = img.reshape((data.height, data.width, 4))[:, :, :3]
        latest_follow = img
    except Exception as e:
        error_count += 1
        if error_count % 100 == 1:
            print(f"跟随相机回调错误: {e}")

def on_semantic(data):
    global latest_semantic
    try:
        data.convert(carla.ColorConverter.CityScapesPalette)
        img = np.frombuffer(data.raw_data, dtype=np.uint8)
        img = img.reshape((data.height, data.width, 4))[:, :, :3]
        latest_semantic = img
    except Exception as e:
        error_count += 1
        if error_count % 100 == 1:
            print(f"语义相机回调错误: {e}")

def on_lidar(data):
    global latest_lidar
    try:
        latest_lidar = data
    except Exception as e:
        error_count += 1
        if error_count % 100 == 1:
            print(f"雷达回调错误: {e}")

def on_collision(event):
    global collision_cooldown, collision_cooldown_time
    try:
        now = time.time()
        if collision_cooldown and (now - collision_cooldown_time) < COLLISION_COOLDOWN_SEC:
            return
        collision_cooldown = True
        collision_cooldown_time = now
        impulse = event.normal_impulse
        magnitude = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        if latest_camera is not None:
            img_path = os.path.join(collision_folder, f"collision_{ts}_camera.png")
            cv2.imwrite(img_path, latest_camera)
            print(f"💥 碰撞图像: {img_path} (力度={magnitude:.2f})")
        if latest_lidar is not None:
            lidar_path = os.path.join(collision_folder, f"collision_{ts}_lidar.ply")
            latest_lidar.save_to_disk(lidar_path)
            print(f"💥 碰撞点云: {lidar_path}")
    except Exception as e:
        print(f"碰撞保存失败: {e}")

def on_gnss(data):
    global latest_gnss
    latest_gnss = data

def on_imu(data):
    global latest_imu
    latest_imu = data

camera_front.listen(on_camera_front)
camera_follow.listen(on_camera_follow)
semantic_camera.listen(on_semantic)
lidar.listen(on_lidar)
if collision_sensor:
    collision_sensor.listen(on_collision)
if gnss:
    gnss.listen(on_gnss)
if imu:
    imu.listen(on_imu)

# ====================== 功能函数 ======================
def compute_closest_obstacle(lidar_data, ego_location, ego_rotation):
    if lidar_data is None:
        return float('inf')
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    points = points.reshape((-1, 4))[:, :3]
    if len(points) == 0:
        return float('inf')
    yaw_rad = math.radians(ego_rotation.yaw)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    half_angle = math.radians(OBSTACLE_FOV_ANGLE / 2.0)
    min_dist = float('inf')
    for pt in points:
        dx = pt[0] - ego_location.x
        dy = pt[1] - ego_location.y
        dz = pt[2] - ego_location.z
        if abs(dz) > OBSTACLE_MAX_HEIGHT:
            continue
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw
        if local_x <= 0:
            continue
        angle = math.atan2(abs(local_y), local_x)
        if angle > half_angle:
            continue
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def draw_traffic_light(image, vehicle):
    try:
        light = vehicle.get_traffic_light()
        if light is None:
            return
        state = light.get_state()
        if state == carla.TrafficLightState.Red:
            status = "RED"
            color = (0, 0, 255)
        elif state == carla.TrafficLightState.Yellow:
            status = "YELLOW"
            color = (0, 255, 255)
        elif state == carla.TrafficLightState.Green:
            status = "GREEN"
            color = (0, 255, 0)
        else:
            return
        light_loc = light.get_location()
        vehicle_loc = vehicle.get_location()
        distance = vehicle_loc.distance(light_loc)
        if distance > 50.0:
            return
        x, y = 20, 200  # 位置：车速表下方
        cv2.rectangle(image, (x, y), (x+180, y+40), (0,0,0), -1)
        cv2.putText(image, f"Traffic Light: {status}", (x+5, y+25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(image, f"Distance: {distance:.1f}m", (x+5, y+55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    except Exception as e:
        pass

def draw_lane_lines(image, vehicle, world, camera_intrinsic, camera_transform):
    """
    在图像上绘制车道线（左右车道边界）
    参数:
        image: 要绘制的图像
        vehicle: 自车
        world: CARLA world
        camera_intrinsic: 3x3 相机内参矩阵
        camera_transform: 相机相对于世界的变换 (carla.Transform)
    """
    if camera_intrinsic is None or camera_transform is None:
        return
    try:
        # 获取自车所在车道
        map = world.get_map()
        waypoint = map.get_waypoint(vehicle.get_location(), project_to_road=True)
        if waypoint is None:
            return
        
        # 车道宽度的一半（假设标准车道宽度约4米）
        lane_width = 2.0
        
        # 获取左右车道边界点（世界坐标）
        left_points = []
        right_points = []
        
        # 当前车道中心线往前采样几个点（距离：5, 15, 25, 35米）
        for dist in [5, 15, 25, 35]:
            wp_future = waypoint.next(dist)[0] if waypoint.next(dist) else None
            if wp_future is None:
                continue
            # 左边界 = 中心点 + 垂直于前进方向向左的偏移
            left_loc = wp_future.transform.location + carla.Location(
                x= -lane_width * math.sin(math.radians(wp_future.transform.rotation.yaw)),
                y= lane_width * math.cos(math.radians(wp_future.transform.rotation.yaw)),
                z= 0.5
            )
            right_loc = wp_future.transform.location + carla.Location(
                x= lane_width * math.sin(math.radians(wp_future.transform.rotation.yaw)),
                y= -lane_width * math.cos(math.radians(wp_future.transform.rotation.yaw)),
                z= 0.5
            )
            left_points.append(left_loc)
            right_points.append(right_loc)
        
        # 投影到图像坐标并绘制
        def project_to_image(point, cam_transform, K):
            # 将世界坐标点转换到相机坐标系
            point_w = np.array([point.x, point.y, point.z, 1.0])
            # 构造相机到世界的变换矩阵
            cam_rot = cam_transform.rotation
            cam_loc = cam_transform.location
            R = np.array(carla.Transform(carla.Location(), cam_rot).get_matrix())[:3, :3]
            T = np.array([cam_loc.x, cam_loc.y, cam_loc.z])
            # 世界到相机: 先平移后旋转
            point_cam = R.T @ (point_w[:3] - T)
            if point_cam[2] <= 0:
                return None
            u = K[0,0] * point_cam[0] / point_cam[2] + K[0,2]
            v = K[1,1] * point_cam[1] / point_cam[2] + K[1,2]
            return (int(u), int(v))
        
        cam_transform = camera_transform
        K = camera_intrinsic
        left_pixels = [project_to_image(p, cam_transform, K) for p in left_points if project_to_image(p, cam_transform, K) is not None]
        right_pixels = [project_to_image(p, cam_transform, K) for p in right_points if project_to_image(p, cam_transform, K) is not None]
        
        # 绘制线条
        if len(left_pixels) > 1:
            for i in range(len(left_pixels)-1):
                cv2.line(image, left_pixels[i], left_pixels[i+1], (255, 0, 0), 3)  # 蓝色左线
        if len(right_pixels) > 1:
            for i in range(len(right_pixels)-1):
                cv2.line(image, right_pixels[i], right_pixels[i+1], (0, 0, 255), 3)  # 红色右线
    except Exception as e:
        pass

def adjust_dynamic_traffic(speed_kmh):
    """根据速度调整生成参数和移除距离"""
    global current_spawn_interval, current_max_vehicles, current_max_pedestrians, current_remove_distance
    if speed_kmh < LOW_SPEED_THRESHOLD:
        # 低速：拥堵模式，增加生成频率和数量
        current_spawn_interval = 3.0
        current_max_vehicles = 6
        current_max_pedestrians = 8
        current_remove_distance = REMOVE_DISTANCE_BASE
        # print(f"🚦 低速模式: spawn={current_spawn_interval}s, veh={current_max_vehicles}, ped={current_max_pedestrians}")
    elif speed_kmh > HIGH_SPEED_THRESHOLD:
        # 高速：稀疏模式，减少生成数量，增大移除距离
        current_spawn_interval = 8.0
        current_max_vehicles = 2
        current_max_pedestrians = 3
        current_remove_distance = REMOVE_DISTANCE_HIGH
        # print(f"🏎️ 高速模式: spawn={current_spawn_interval}s, veh={current_max_vehicles}, ped={current_max_pedestrians}, remove={current_remove_distance}")
    else:
        # 中速：正常模式
        current_spawn_interval = BASE_SPAWN_INTERVAL
        current_max_vehicles = BASE_MAX_VEHICLES
        current_max_pedestrians = BASE_MAX_PEDESTRIANS
        current_remove_distance = REMOVE_DISTANCE_BASE
        # print(f"🚗 中速模式: spawn={current_spawn_interval}s, veh={current_max_vehicles}, ped={current_max_pedestrians}")

# ====================== 绘制综合仪表盘 ======================
def draw_speedometer(image, vehicle):
    global closest_obstacle_distance, obstacle_warning_active, weather_change_time
    try:
        vel = vehicle.get_velocity()
        speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        max_speed = 120.0
        ratio = min(speed/max_speed, 1.0)
        bar_w, bar_h = 200, 20
        x, y = 20, 20
        cv2.rectangle(image, (x, y), (x+bar_w, y+bar_h), (50,50,50), -1)
        fill = int(bar_w * ratio)
        color = (0,255,0) if speed<80 else (0,165,255) if speed<120 else (0,0,255)
        cv2.rectangle(image, (x, y), (x+fill, y+bar_h), color, -1)
        cv2.putText(image, f"{int(speed)} km/h", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(image, f"Veh:{len(spawned_vehicles)}  Ped:{len(spawned_pedestrians)}", 
                    (x, y+bar_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(image, f"FPS:{fps:.1f} Err:{error_count}", 
                    (x, y+bar_h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        mode_text = "SEMANTIC" if display_mode == "semantic" else "RGB"
        cv2.putText(image, f"Mode: {mode_text} (press S)", (x, y+bar_h+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.putText(image, f"Weather: {weather_names[current_weather_idx].upper()} (press W)", 
                    (x, y+bar_h+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(image, f"Press R to save point", (x, y+bar_h+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        if time.time() - weather_change_time < 2.0:
            hint = f"Weather: {weather_names[current_weather_idx].upper()}"
            cv2.putText(image, hint, (image.shape[1]//2 - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
        if obstacle_warning_active:
            warn_x = image.shape[1] - 250
            warn_y = 30
            if closest_obstacle_distance < OBSTACLE_DANGER_DISTANCE:
                color = (0, 0, 255)
                status = "DANGER!"
            else:
                color = (0, 255, 255)
                status = "WARNING!"
            cv2.rectangle(image, (warn_x-10, warn_y-10), (warn_x+180, warn_y+50), color, -1)
            cv2.putText(image, f"{status} {closest_obstacle_distance:.1f}m", 
                        (warn_x, warn_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        # 红绿灯检测
        draw_traffic_light(image, vehicle)
    except Exception as e:
        pass

# ====================== 等待传感器就绪 ======================
print("等待传感器数据...", end="")
timeout_start = time.time()
while (latest_follow is None or latest_camera is None or latest_lidar is None or latest_semantic is None) and (time.time() - timeout_start < 12):
    time.sleep(0.2)
    print(".", end="", flush=True)
print()
if latest_follow is not None and latest_semantic is not None:
    print("✅ 所有传感器就绪（RGB + 语义）")
else:
    print("⚠️ 部分传感器未就绪，继续运行")

init_trajectory_csv()

# ====================== 主循环 ======================
print(f"每 {SAVE_INTERVAL//60} 分钟自动保存，碰撞自动保存，动态交通已启用（自适应）")
print("🎨 按 S 键切换显示模式（RGB / 语义分割）")
print("⚠️ 激光雷达障碍物警告已开启（前方10米内预警，5米内危险）")
print("🌤️ 按 W 键切换天气（晴天→雨天→雾天→夜晚）")
print("📊 按 R 键保存当前轨迹点（GPS/IMU/车速等）到 CSV")
print("🚦 红绿灯检测 & 车道线可视化已启用")
print("按 Q/ESC 退出")

loop_counter = 0
try:
    while True:
        loop_counter += 1
        now = time.time()
        if loop_counter % 500 == 0:
            print(f"♥ 心跳: 已运行 {loop_counter} 帧, 车辆={len(spawned_vehicles)}, 行人={len(spawned_pedestrians)}")

        try:
            ego_loc = vehicle.get_location()
            ego_rot = vehicle.get_transform().rotation
        except Exception as e:
            print(f"❌ 获取自车位置失败: {e}")
            try:
                world = client.get_world()
                vehicle = world.get_actor(vehicle.id)
                continue
            except:
                break

        # 获取当前车速（用于动态调整）
        vel = vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        adjust_dynamic_traffic(speed_kmh)

        # 障碍物检测
        if latest_lidar is not None:
            closest_obstacle_distance = compute_closest_obstacle(latest_lidar, ego_loc, ego_rot)
            obstacle_warning_active = (closest_obstacle_distance < OBSTACLE_WARNING_DISTANCE)
        else:
            closest_obstacle_distance = float('inf')
            obstacle_warning_active = False

        # 动态生成（使用自适应间隔）
        if now - last_spawn_time >= current_spawn_interval:
            for _ in range(MAX_SPAWN_ATTEMPTS):
                if len(spawned_vehicles) < current_max_vehicles and random.random() < 0.6:
                    spawn_random_vehicle_near(ego_loc)
                if len(spawned_pedestrians) < current_max_pedestrians and random.random() < 0.4:
                    spawn_random_pedestrian_near(ego_loc)
            last_spawn_time = now

        remove_far_actors(ego_loc)

        if collision_cooldown and (now - collision_cooldown_time) >= COLLISION_COOLDOWN_SEC:
            collision_cooldown = False

        frame_count += 1
        if now - last_fps_time >= 1.0:
            fps = frame_count / (now - last_fps_time)
            frame_count = 0
            last_fps_time = now

        # 画面合成
        if display_mode == "semantic" and latest_semantic is not None:
            main_display = latest_semantic.copy()
            draw_speedometer(main_display, vehicle)
            if latest_camera is not None:
                h, w = main_display.shape[:2]
                sw = max(160, w//4)
                sh = int(sw * latest_camera.shape[0] / latest_camera.shape[1])
                small = cv2.resize(latest_camera, (sw, sh))
                x = w - sw - 10
                y = h - sh - 10
                if x > 0 and y > 0:
                    main_display[y:y+sh, x:x+sw] = small
                    cv2.rectangle(main_display, (x-1,y-1), (x+sw+1,y+sh+1), (255,255,255), 2)
            # 在语义图上也可以画车道线（可选）
            # draw_lane_lines(main_display, vehicle, world, camera_intrinsic, camera_follow.get_transform())
        elif latest_follow is not None:
            main_display = latest_follow.copy()
            draw_speedometer(main_display, vehicle)
            # 绘制车道线（仅RGB模式）
            draw_lane_lines(main_display, vehicle, world, camera_intrinsic, camera_follow.get_transform())
            if latest_camera is not None:
                h, w = main_display.shape[:2]
                sw = max(160, w//4)
                sh = int(sw * latest_camera.shape[0] / latest_camera.shape[1])
                small = cv2.resize(latest_camera, (sw, sh))
                x = w - sw - 10
                y = h - sh - 10
                if x > 0 and y > 0:
                    main_display[y:y+sh, x:x+sw] = small
                    cv2.rectangle(main_display, (x-1,y-1), (x+sw+1,y+sh+1), (255,255,255), 2)
        elif latest_camera is not None:
            main_display = latest_camera.copy()
            draw_speedometer(main_display, vehicle)
        else:
            main_display = np.zeros((600,800,3), dtype=np.uint8)
            cv2.putText(main_display, "Waiting for sensors...", (50,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("CARLA", main_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("用户主动退出")
            break
        elif key == ord('s') or key == ord('S'):
            display_mode = "semantic" if display_mode == "rgb" else "rgb"
            print(f"🔮 切换到 {display_mode.upper()} 视图")
        elif key == ord('w') or key == ord('W'):
            cycle_weather()
        elif key == ord('r') or key == ord('R'):
            save_trajectory_point()

        # 定时保存
        if now - last_save_time >= SAVE_INTERVAL:
            if latest_camera is not None and latest_lidar is not None:
                ts = str(int(now))
                cv2.imwrite(os.path.join(image_folder, f"{ts}.png"), latest_camera)
                latest_lidar.save_to_disk(os.path.join(lidar_folder, f"{ts}.ply"))
                if latest_semantic is not None:
                    cv2.imwrite(os.path.join(semantic_folder, f"semantic_{ts}.png"), latest_semantic)
                print(f"💾 定时保存 {ts}")
                last_save_time = now

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n用户中断")
except Exception as e:
    print(f"主循环异常: {e}")
    traceback.print_exc()
finally:
    cv2.destroyAllWindows()
    if trajectory_file:
        trajectory_file.close()
    for actor in all_spawned_actors:
        if actor:
            try:
                if hasattr(actor, 'stop'): actor.stop()
                actor.destroy()
            except:
                pass
    for actor in [camera_front, camera_follow, semantic_camera, lidar, collision_sensor, gnss, imu, vehicle]:
        if actor:
            try:
                if hasattr(actor, 'stop'): actor.stop()
                actor.destroy()
            except:
                pass
    print("✅ 已清理退出")