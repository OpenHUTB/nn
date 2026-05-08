import sys
import glob
import random
import numpy as np
import queue
import cv2
import os
import xml.etree.ElementTree as ET
import time
import pygame  # Import pygame for keyboard input handling

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((400, 300))

# 当前文件目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 向上回到 Git 目录
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

# 拼接 carla 路径
carla_egg_path = os.path.join(
    project_root,
    'Carla',
    'WindowsNoEditor',
    'PythonAPI',
    'carla',
    'dist',
    'carla-0.9.11-py3.7-win-amd64.egg'
)


# Define the path to the CARLA Egg file
sys.path.append(carla_egg_path)

import carla

# Connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(60.0)

# Load a different map
def load_map(map_name):
    return client.load_world(map_name)


def spawn_vehicle_with_retries(world, blueprint, spawn_points, used_spawn_indices=None):
    if used_spawn_indices is None:
        used_spawn_indices = set()

    candidate_indices = [index for index in range(len(spawn_points)) if index not in used_spawn_indices]
    random.shuffle(candidate_indices)

    for index in candidate_indices:
        vehicle = world.try_spawn_actor(blueprint, spawn_points[index])
        if vehicle is not None:
            used_spawn_indices.add(index)
            return vehicle, index

    return None, None


# Function to spawn vehicles
def spawn_vehicles(num_vehicles, world, spawn_points, used_spawn_indices=None):
    vehicle_bp_lib = world.get_blueprint_library().filter('vehicle.*')
    spawned_vehicles = []
    used_indices = used_spawn_indices if used_spawn_indices is not None else set()

    for _ in range(num_vehicles):
        vehicle_bp = random.choice(vehicle_bp_lib)
        vehicle, _ = spawn_vehicle_with_retries(world, vehicle_bp, spawn_points, used_indices)
        if vehicle:
            spawned_vehicles.append(vehicle)
            # print(f"Spawned vehicle: {vehicle.id} at {spawn_point}")
        else:
            print("Failed to spawn vehicle")

    return spawned_vehicles


def get_driving_waypoint(world_map, location):
    return world_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )


def get_road_id_for_location(world_map, location):
    waypoint = get_driving_waypoint(world_map, location)
    if waypoint is None:
        return None
    return waypoint.road_id


def choose_next_ego_spawn_index(world_map, spawn_points, current_location, current_spawn_index):
    current_road_id = get_road_id_for_location(world_map, current_location)
    far_different_road = []
    far_same_road = []
    nearby_different_road = []
    nearby_same_road = []

    candidate_indices = list(range(len(spawn_points)))
    random.shuffle(candidate_indices)

    for index in candidate_indices:
        if index == current_spawn_index:
            continue

        spawn_point = spawn_points[index]
        waypoint = get_driving_waypoint(world_map, spawn_point.location)
        if waypoint is None:
            continue

        is_far_enough = spawn_point.location.distance(current_location) >= EGO_MIN_RESPAWN_DISTANCE
        is_different_road = waypoint.road_id != current_road_id

        if is_far_enough and is_different_road:
            far_different_road.append(index)
        elif is_far_enough:
            far_same_road.append(index)
        elif is_different_road:
            nearby_different_road.append(index)
        else:
            nearby_same_road.append(index)

    for bucket in (far_different_road, far_same_road, nearby_different_road, nearby_same_road):
        if bucket:
            return bucket[0]

    return None


def relocate_ego_vehicle(vehicle, spawn_points, world_map, traffic_manager, current_spawn_index):
    next_spawn_index = choose_next_ego_spawn_index(
        world_map,
        spawn_points,
        vehicle.get_location(),
        current_spawn_index
    )
    if next_spawn_index is None:
        return current_spawn_index, False

    vehicle.set_autopilot(False, traffic_manager.get_port())
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_transform(spawn_points[next_spawn_index])
    vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.auto_lane_change(vehicle, True)
    traffic_manager.vehicle_percentage_speed_difference(vehicle, BASE_SPEED_DIFFERENCE)
    traffic_manager.distance_to_leading_vehicle(vehicle, BASE_FOLLOW_DISTANCE)

    new_road_id = get_road_id_for_location(world_map, spawn_points[next_spawn_index].location)
    print(f"Ego vehicle relocated to spawn index {next_spawn_index}, road_id={new_road_id}")
    return next_spawn_index, True

# Function to spawn walkers
'''def spawn_walkers(num_walkers, world, spawn_points):
    walker_bp_lib = world.get_blueprint_library().filter('walker.pedestrian.*')
    spawned_walkers = []

    walker_control_bp = world.get_blueprint_library().find('controller.ai.walker')

    for _ in range(num_walkers):
        walker_bp = random.choice(walker_bp_lib)
        spawn_point = random.choice(spawn_points)

        walker = world.try_spawn_actor(walker_bp, spawn_point)
        if walker:
            spawned_walkers.append(walker)
            # print(f"Spawned walker: {walker.id} at {spawn_point}")

            walker_control = world.spawn_actor(walker_control_bp, carla.Transform(), walker)
            walker_control.start()

            # Use the Location object for go_to_location
            walker_control.go_to_location(spawn_point.location)
            walker_control.set_max_speed(1.0)

        else:
            print("Failed to spawn walker")

    return spawned_walkers '''

# Define the map you want to load
# 换一个交通标志多的地图
world = client.load_world('Town05')


# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Initialize Traffic Manager
traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)
if hasattr(traffic_manager, 'set_random_device_seed'):
    traffic_manager.set_random_device_seed(int(time.time()))
world_map = world.get_map()


# Fast cruising with a slower profile only for tight urban turns.
BASE_SPEED_DIFFERENCE = -50.0
TURNING_SPEED_DIFFERENCE = 55.0
BASE_FOLLOW_DISTANCE = 4.0
TURNING_FOLLOW_DISTANCE = 6.0
RIGHT_TURN_LOOKAHEAD_DISTANCES = (4.0, 8.0, 12.0, 16.0)
RIGHT_TURN_ANGLE_THRESHOLD = 35.0
RIGHT_STEER_THRESHOLD = 0.15
OVERTAKE_MIN_TRIGGER_DISTANCE = 14.0
OVERTAKE_LANE_CLEAR_DISTANCE = 18.0
OVERTAKE_RELATIVE_SPEED_THRESHOLD = 2.0
OVERTAKE_TARGET_MAX_SPEED = 4.0
OVERTAKE_COOLDOWN = 3.0
EGO_RESPAWN_INTERVAL_SECONDS = 300.0
EGO_MIN_RESPAWN_DISTANCE = 120.0




# Get map spawn points
spawn_points = world_map.get_spawn_points()
used_spawn_indices = set()

# Spawn vehicles and walkers
num_vehicles = 10
vehicles = spawn_vehicles(num_vehicles, world, spawn_points, used_spawn_indices)

for v in vehicles:
    v.set_autopilot(True, traffic_manager.get_port())

#num_walkers = 2
#walkers = spawn_walkers(num_walkers, world, spawn_points)

# Get the blueprint library
bp_lib = world.get_blueprint_library().filter('*')

# Spawn vehicle
vehicle_bp = bp_lib.find('vehicle.audi.a2')
try:
    vehicle, spawn_index = spawn_vehicle_with_retries(world, vehicle_bp, spawn_points, used_spawn_indices)
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle")
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)

# Disable Autopilot for manual control
vehicle.set_autopilot(True, traffic_manager.get_port())
print("自动驾驶已启用")
# 开启自动变道
traffic_manager.auto_lane_change(vehicle, True)
# 设置全局跟车距离
traffic_manager.set_global_distance_to_leading_vehicle(BASE_FOLLOW_DISTANCE)
#设置遵守交通规则
traffic_manager.ignore_lights_percentage(vehicle, 100.0)  # Ignore all traffic lights
#控制自动驾驶速度（加快）
traffic_manager.vehicle_percentage_speed_difference(vehicle,-50)
# 减少跟车距离
traffic_manager.distance_to_leading_vehicle(vehicle, BASE_FOLLOW_DISTANCE)
# Spawn camera
CAMERA_IMAGE_SIZE = 1280
CAMERA_FOV = 65
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(CAMERA_IMAGE_SIZE))
camera_bp.set_attribute('image_size_y', str(CAMERA_IMAGE_SIZE))
camera_bp.set_attribute('fov', str(CAMERA_FOV))

# Adjust camera position and orientation to avoid car front
#camera_init_trans = carla.Transform(carla.Location(x=2, z=2), carla.Rotation(pitch=-10))
#camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

camera_init_trans = carla.Transform(carla.Location(x=1, y=0.25, z=2), carla.Rotation(pitch=-3, yaw=3))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue(maxsize=50)


# Camera listener
def image_callback(image):
    if not image_queue.full():
        image_queue.put(image)

camera.listen(image_callback)


def get_latest_image(image_queue_obj, timeout=1.0):
    try:
        image = image_queue_obj.get(timeout=timeout)
    except queue.Empty:
        return None

    while True:
        try:
            image = image_queue_obj.get_nowait()
        except queue.Empty:
            break

    return image
# 使用相对路径保存记录到的数据
# 当前文件目录
current_dirc = os.path.dirname(os.path.abspath(__file__))

# 向上回到 Git 目录
project_root = os.path.abspath(os.path.join(current_dirc, '..', '..', '..'))

# 拼接 carla 路径
data_path = os.path.join(
    project_root,
    'OutPut',
    'data01'
)

# Directory to save images and XML files
output_dir = data_path
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to get current weather parameters
def get_weather_params(world):
    weather = world.get_weather()
    return {
        'cloudiness': weather.cloudiness,
        'precipitation': weather.precipitation,
        'precipitation_deposits': weather.precipitation_deposits,
        'wind_intensity': weather.wind_intensity,
        'sun_azimuth_angle': weather.sun_azimuth_angle,
        'sun_altitude_angle': weather.sun_altitude_angle,
        'fog_density': weather.fog_density,
        'wetness': weather.wetness
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

# Calculate the camera projection matrix to project from 3D -> 2D
K = build_projection_matrix(image_w, image_h, fov)

# Define the distance threshold for a clearly visible sign
# 扩大检测距离到20米
DISTANCE_THRESHOLD = 60.0  # Example threshold in meters
SIGN_FACE_ALIGNMENT_THRESHOLD = 0.05
SIGN_LOCATION_ROUND_DIGITS = 1
MIN_CAPTURE_AREA = 650
MIN_CAPTURE_WIDTH = 16
MIN_CAPTURE_HEIGHT = 16
MAX_CAPTURE_DISTANCE = 32.0
MIN_VISIBLE_BOX_RATIO = 0.55

# Track each physical sign and only capture it once when it is clear enough.
captured_sign_states = {}


def get_sign_key(location):
    return (
        round(location.x, SIGN_LOCATION_ROUND_DIGITS),
        round(location.y, SIGN_LOCATION_ROUND_DIGITS),
        round(location.z, SIGN_LOCATION_ROUND_DIGITS)
    )


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


def get_signs_bounding_boxes(vehicle_transform, camera_transform, K, world_2_camera):
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

                sign_location_tuple = get_sign_key(obj.location)
                sign_state = captured_sign_states.setdefault(
                    sign_location_tuple,
                    {'captured': False, 'best_area': 0}
                )
                if (
                    camera_dot_product > 0 and
                    not sign_state['captured'] and
                    is_sign_front_visible(obj, camera_location)
                ):
                    verts = [v for v in obj.get_world_vertices(carla.Transform())]
                    x_coords = [get_image_point(v, K, world_2_camera)[0] for v in verts]
                    y_coords = [get_image_point(v, K, world_2_camera)[1] for v in verts]
                    xmin, xmax = int(min(x_coords)), int(max(x_coords))
                    ymin, ymax = int(min(y_coords)), int(max(y_coords))

                    # Larger projected area usually means the sign is closer and clearer.
                    area = (xmax - xmin) * (ymax - ymin)
                    sign_state['best_area'] = max(sign_state['best_area'], area)

                    # Set a threshold for the minimum area to capture the sign
                    # 降低“面积阈值”过滤
                    min_area_threshold = 10  # Adjust this value as needed

                    clipped_xmin = clamp(xmin, 0, image_w - 1)
                    clipped_ymin = clamp(ymin, 0, image_h - 1)
                    clipped_xmax = clamp(xmax, 0, image_w - 1)
                    clipped_ymax = clamp(ymax, 0, image_h - 1)
                    box_width = clipped_xmax - clipped_xmin
                    box_height = clipped_ymax - clipped_ymin
                    clipped_area = box_width * box_height
                    visible_ratio = clipped_area / float(area) if area > 0 else 0.0
                    aspect_ratio = box_width / float(box_height) if box_height != 0 else 0
                    is_clear_enough = (
                        clipped_area >= MIN_CAPTURE_AREA and
                        box_width >= MIN_CAPTURE_WIDTH and
                        box_height >= MIN_CAPTURE_HEIGHT and
                        distance <= MAX_CAPTURE_DISTANCE and
                        visible_ratio >= MIN_VISIBLE_BOX_RATIO
                    )
                    if area > min_area_threshold and 0.5 < aspect_ratio < 2.0 and is_clear_enough:
                        bounding_boxes.append(
                            {
                                'label': 'TrafficSign',
                                'xmin': clipped_xmin,
                                'ymin': clipped_ymin,
                                'xmax': clipped_xmax,
                                'ymax': clipped_ymax
                            }
                        )
                        sign_state['captured'] = True

    return bounding_boxes


# Function to create an XML file for bounding boxes
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

    # Add weather information
    weather_category = get_weather_category(weather_params)
    weather = ET.SubElement(annotation, "weather")
    condition = ET.SubElement(weather, "condition")
    condition.text = str(weather_category)

    # Objects (Bounding boxes)
    for bbox in bboxes:
        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = bbox['label']


        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        ymin = ET.SubElement(bndbox, "ymin")
        xmax = ET.SubElement(bndbox, "xmax")
        ymax = ET.SubElement(bndbox, "ymax")
        xmin.text = str(bbox['xmin'])
        ymin.text = str(bbox['ymin'])
        xmax.text = str(bbox['xmax'])
        ymax.text = str(bbox['ymax'])

    # Convert the XML tree to a string
    tree = ET.ElementTree(annotation)
    xml_file = os.path.join(output_dir, image_name.replace('.png', '.xml'))
    tree.write(xml_file)

# Function to manually compute dot product
def dot_product(v1, v2):
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def vector_length(vector):
    return np.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)


def normalized_dot_product(v1, v2):
    v1_length = vector_length(v1)
    v2_length = vector_length(v2)
    if v1_length == 0.0 or v2_length == 0.0:
        return 0.0
    return dot_product(v1, v2) / (v1_length * v2_length)


def is_sign_front_visible(sign_bbox, camera_location):
    sign_transform = carla.Transform(sign_bbox.location, sign_bbox.rotation)
    sign_forward_vector = sign_transform.get_forward_vector()
    vector_to_camera = camera_location - sign_bbox.location
    # CARLA traffic sign bounding boxes often use a forward vector that points away from
    # the printable face, so we flip it here to keep front-facing signs.
    sign_face_vector = carla.Vector3D(
        x=-sign_forward_vector.x,
        y=-sign_forward_vector.y,
        z=-sign_forward_vector.z
    )
    facing_score = normalized_dot_product(sign_face_vector, vector_to_camera)
    return facing_score > SIGN_FACE_ALIGNMENT_THRESHOLD


def get_speed(vehicle_actor):
    velocity = vehicle_actor.get_velocity()
    return np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def normalize_angle(angle):
    return (angle + 180.0) % 360.0 - 180.0


def get_max_upcoming_turn_angle(waypoint, lookahead_distances):
    if waypoint is None:
        return 0.0

    current_yaw = waypoint.transform.rotation.yaw
    max_turn_angle = 0.0

    for distance in lookahead_distances:
        next_waypoints = waypoint.next(distance)
        if not next_waypoints:
            continue

        next_yaw = next_waypoints[0].transform.rotation.yaw
        delta_yaw = normalize_angle(next_yaw - current_yaw)
        if abs(delta_yaw) > abs(max_turn_angle):
            max_turn_angle = delta_yaw

    return max_turn_angle


def is_right_turn_imminent(vehicle, world_map):
    waypoint = world_map.get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        return False

    vehicle_control = vehicle.get_control()
    max_turn_angle = get_max_upcoming_turn_angle(waypoint, RIGHT_TURN_LOOKAHEAD_DISTANCES)

    # Slow down before entering the junction and while already steering into it.
    return (
        max_turn_angle >= RIGHT_TURN_ANGLE_THRESHOLD or
        (waypoint.is_junction and vehicle_control.steer > RIGHT_STEER_THRESHOLD)
    )


def lane_change_allowed(lane_change, direction):
    if direction == 'left':
        return lane_change in (carla.LaneChange.Left, carla.LaneChange.Both)
    return lane_change in (carla.LaneChange.Right, carla.LaneChange.Both)


def find_blocking_vehicle(vehicle, world, world_map):
    ego_transform = vehicle.get_transform()
    ego_location = ego_transform.location
    ego_forward = ego_transform.get_forward_vector()
    ego_waypoint = world_map.get_waypoint(
        ego_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if ego_waypoint is None or ego_waypoint.is_junction:
        return None

    ego_speed = get_speed(vehicle)
    closest_vehicle = None
    closest_distance = float('inf')

    for other in world.get_actors().filter('vehicle.*'):
        if other.id == vehicle.id:
            continue

        other_location = other.get_location()
        offset = other_location - ego_location
        forward_distance = dot_product(ego_forward, offset)
        if forward_distance <= 0.0 or forward_distance > OVERTAKE_MIN_TRIGGER_DISTANCE:
            continue

        other_waypoint = world_map.get_waypoint(
            other_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if other_waypoint is None:
            continue

        same_lane = (
            other_waypoint.road_id == ego_waypoint.road_id and
            other_waypoint.lane_id == ego_waypoint.lane_id
        )
        if not same_lane:
            continue

        other_speed = get_speed(other)
        if other_speed > OVERTAKE_TARGET_MAX_SPEED and ego_speed - other_speed < OVERTAKE_RELATIVE_SPEED_THRESHOLD:
            continue

        if forward_distance < closest_distance:
            closest_distance = forward_distance
            closest_vehicle = other

    return closest_vehicle


def is_lane_clear_for_overtake(vehicle, world, world_map, target_waypoint):
    ego_location = vehicle.get_location()

    for other in world.get_actors().filter('vehicle.*'):
        if other.id == vehicle.id:
            continue

        other_waypoint = world_map.get_waypoint(
            other.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if other_waypoint is None:
            continue

        same_target_lane = (
            other_waypoint.road_id == target_waypoint.road_id and
            other_waypoint.lane_id == target_waypoint.lane_id
        )
        if not same_target_lane:
            continue

        if other.get_location().distance(ego_location) < OVERTAKE_LANE_CLEAR_DISTANCE:
            return False

    return True


def try_overtake_blocking_vehicle(vehicle, world, traffic_manager, world_map, current_time, last_overtake_time):
    if current_time - last_overtake_time < OVERTAKE_COOLDOWN:
        return last_overtake_time

    ego_waypoint = world_map.get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if ego_waypoint is None or ego_waypoint.is_junction or is_right_turn_imminent(vehicle, world_map):
        return last_overtake_time

    blocking_vehicle = find_blocking_vehicle(vehicle, world, world_map)
    if blocking_vehicle is None:
        return last_overtake_time

    for direction in ('left', 'right'):
        if not lane_change_allowed(ego_waypoint.lane_change, direction):
            continue

        target_waypoint = ego_waypoint.get_left_lane() if direction == 'left' else ego_waypoint.get_right_lane()
        if target_waypoint is None:
            continue

        if target_waypoint.lane_type != carla.LaneType.Driving:
            continue

        if target_waypoint.is_junction or target_waypoint.road_id != ego_waypoint.road_id:
            continue

        if ego_waypoint.lane_id * target_waypoint.lane_id < 0:
            continue

        if not is_lane_clear_for_overtake(vehicle, world, world_map, target_waypoint):
            continue

        traffic_manager.force_lane_change(vehicle, direction == 'right')
        return current_time

    return last_overtake_time


def update_autopilot_safety(vehicle, world, traffic_manager, world_map, current_time, last_overtake_time):
    right_turn_imminent = is_right_turn_imminent(vehicle, world_map)
    traffic_manager.auto_lane_change(vehicle, not right_turn_imminent)

    if right_turn_imminent:
        traffic_manager.vehicle_percentage_speed_difference(vehicle, TURNING_SPEED_DIFFERENCE)
        traffic_manager.distance_to_leading_vehicle(vehicle, TURNING_FOLLOW_DISTANCE)
    else:
        traffic_manager.vehicle_percentage_speed_difference(vehicle, BASE_SPEED_DIFFERENCE)
        traffic_manager.distance_to_leading_vehicle(vehicle, BASE_FOLLOW_DISTANCE)
        last_overtake_time = try_overtake_blocking_vehicle(
            vehicle,
            world,
            traffic_manager,
            world_map,
            current_time,
            last_overtake_time
        )

    return last_overtake_time

# Define a list of possible weather conditions
weather_conditions = [
    'rainy',
    'sunny',
    'night',
    'foggy'
]

# Create a function to set the weather condition based on a given string
def update_weather(world, condition):
    """Update the weather parameters based on the given condition."""
    if condition == 'rainy':
        weather = carla.WeatherParameters(
            cloudiness=80.0,  # High cloudiness
            precipitation=80.0,  # Heavy rain
            precipitation_deposits=80.0,
            wind_intensity=10.0,  # Moderate wind
            sun_azimuth_angle=270.0,  # Sun position could be irrelevant
            sun_altitude_angle=10.0,  # Low sun angle
            fog_density=10.0,  # Light fog
            wetness=70.0  # Wet ground
        )
    elif condition == 'sunny':
        weather = carla.WeatherParameters(
            cloudiness=20.0,  # Slightly cloudy
            precipitation=0.0,  # No precipitation
            precipitation_deposits=0.0,
            wind_intensity=5.0,  # Light wind
            sun_azimuth_angle=180.0,  # Midday sun
            sun_altitude_angle=60.0,  # High sun angle
            fog_density=0.0,  # No fog
            wetness=0.0  # Dry ground
        )
    elif condition == 'night':
        weather = carla.WeatherParameters(
            cloudiness=0.0,  # Overcast
            precipitation=0.0,  # No precipitation
            precipitation_deposits=0.0,
            wind_intensity=3.0,  # Light wind
            sun_azimuth_angle=0.0,  # Sun below horizon
            sun_altitude_angle=-5.0,  # Negative value for night
            fog_density=0.0,  # Light fog
            wetness=0.0  # Dry ground
        )
    elif condition == 'foggy':
        weather = carla.WeatherParameters(
            cloudiness=0.0,  # Overcast
            precipitation=0.0,  # No precipitation
            precipitation_deposits=0.0,
            wind_intensity=3.0,  # Light wind
            sun_azimuth_angle=0.0,  # Sun below horizon
            sun_altitude_angle=0.0,  # Negative value for night
            fog_density=60.0,  # Light fog
            wetness=0.0  # Dry ground
        )
    else:
        raise ValueError("Unknown weather condition")

    world.set_weather(weather)

# Function to categorize weather conditions
def get_weather_category(weather_params):
    # Example thresholds for categorization
    if weather_params['cloudiness'] > 70 or weather_params['precipitation'] > 50:
        return 0  # "rainy"
    elif weather_params['sun_altitude_angle'] > 30:
        return 1  # "sunny"
    elif weather_params['fog_density'] > 50:
        return 2  # "foggy"
    else:
        return 3  # "night"



# Initialize the weather transition settings
weather_transition_interval = 10  # Interval to change weather conditions in seconds
last_weather_change_time = time.time()
current_condition_index = 0

# Manual control function
def handle_input(vehicle):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    control = vehicle.get_control()
    # control = carla.VehicleControl()

    # Define manual control keys
    if keys[pygame.K_w]:
        control.throttle = 1.0  # Forward
    if keys[pygame.K_s]:
        control.brake = 1.0  # Brake
    if keys[pygame.K_a]:
        control.steer = -1.0  # Left
    if keys[pygame.K_d]:
        control.steer = 1.0  # Right
    if keys[pygame.K_r]:
        control.throttle = 1.0  # Throttle in reverse
        control.reverse = True  # Enable reverse gear
    if keys[pygame.K_s] and control.reverse:
        control.brake = 1.0  # Brake in reverse

    vehicle.apply_control(control)

# Define the IoU Calculation Function
def compute_iou(box1, box2):
    x1 = max(box1['xmin'], box2['xmin'])
    y1 = max(box1['ymin'], box2['ymin'])
    x2 = min(box1['xmax'], box2['xmax'])
    y2 = min(box1['ymax'], box2['ymax'])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    box2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])

    # Prevent division by zero
    if box1_area == 0 or box2_area == 0:
        return 0.0

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# Define the Non-Maximum Suppression Function
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


# Variable to track if the last image had bounding boxes
last_image_had_bboxes = False
last_overtake_time = 0.0
last_ego_respawn_time = time.time()


# Start the game loop
try:
    while True:
        world.tick()
        time.sleep(0.033)
        pygame.event.pump()  # Process event queue for keyboard input
        loop_time = time.time()
        last_overtake_time = update_autopilot_safety(
            vehicle,
            world,
            traffic_manager,
            world_map,
            loop_time,
            last_overtake_time
        )

        if loop_time - last_ego_respawn_time >= EGO_RESPAWN_INTERVAL_SECONDS:
            spawn_index, relocated = relocate_ego_vehicle(
                vehicle,
                spawn_points,
                world_map,
                traffic_manager,
                spawn_index
            )
            if relocated:
                while True:
                    try:
                        image_queue.get_nowait()
                    except queue.Empty:
                        break
                last_ego_respawn_time = loop_time

        # Handle manual input
        # handle_input(vehicle)

        # Get the latest image from the queue without blocking the UI thread indefinitely.
        image = get_latest_image(image_queue, timeout=1.0)
        if image is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):
                print("X key pressed")
                break
            continue

        # Automatically change the weather
        current_time = time.time()
        if current_time - last_weather_change_time >= weather_transition_interval:
            # Update the weather condition
            current_condition = weather_conditions[current_condition_index]
            update_weather(world, current_condition)

            # Move to the next weather condition in the sequence
            current_condition_index = (current_condition_index + 1) % len(weather_conditions)
            last_weather_change_time = current_time  # Update the time of last change

        # Reshape the raw data into an RGB array
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        img_rgb = img[:, :, :3]  # Remove alpha channel for PNG
        img_rgb = img_rgb.astype(np.uint8)  # Ensure data type is uint8

        # Get the camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Get the forward vector of the camera
        camera_transform = camera.get_transform()
        camera_forward_vector = camera_transform.get_forward_vector()

        # Retrieve bounding boxes for traffic signs on the right side only
        bboxes = get_signs_bounding_boxes(vehicle.get_transform(), camera_transform, K, world_2_camera)

        # Apply Non-Maximum Suppression
        bboxes = non_maximum_suppression(bboxes)

        # Save the image and XML only if bounding boxes are present
        if bboxes:
            # Create a copy of the image for visualization
            img_rgb_with_bboxes = img_rgb.copy()

            # Draw bounding boxes on the copy
            for bbox in bboxes:
                cv2.rectangle(img_rgb_with_bboxes, (bbox['xmin'], bbox['ymin']), (bbox['xmax'], bbox['ymax']),
                              (0, 0, 255), 2)

            image_name = f"image_{int(time.time())}.png"
            image_path = os.path.join(output_dir, image_name)

            # Save the original image without bounding boxes
            cv2.imwrite(image_path, img_rgb)

            # Get weather parameters
            weather_params = get_weather_params(world)

            # Save the XML file
            create_xml_file(image_name, bboxes, image_w, image_h, weather_params)

            # Display the image with bounding boxes
            cv2.imshow('ImageWindowName', img_rgb_with_bboxes)

        # Check if any bounding boxes are present before displaying the image
        else:
            cv2.imshow('ImageWindowName', img_rgb)

        # Break the loop if the user presses the X key
        key = cv2.waitKey(10) & 0xFF
        if key == ord('x'):
            print("X key pressed")
            break

finally:
    # Cleanup
    cv2.destroyAllWindows()
    vehicle.destroy()
    camera.destroy()
    pygame.quit()
    for vehicle in vehicles:
        vehicle.destroy()
