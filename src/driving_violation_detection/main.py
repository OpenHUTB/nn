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
            # print(f"Spawned vehicle: {vehicle.id} at {spawn_point}")
        else:
            print("Failed to spawn vehicle")

    return spawned_vehicles

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
world_map = world.get_map()


# Conservative driving profile for tighter urban turns.
BASE_SPEED_DIFFERENCE = 10.0
TURNING_SPEED_DIFFERENCE = 55.0
BASE_FOLLOW_DISTANCE = 4.0
TURNING_FOLLOW_DISTANCE = 6.0
RIGHT_TURN_LOOKAHEAD_DISTANCES = (4.0, 8.0, 12.0, 16.0)
RIGHT_TURN_ANGLE_THRESHOLD = 35.0
RIGHT_STEER_THRESHOLD = 0.15




# Get map spawn points
spawn_points = world_map.get_spawn_points()

# Spawn vehicles and walkers
num_vehicles = 10
vehicles = spawn_vehicles(num_vehicles, world, spawn_points)

for v in vehicles:
    v.set_autopilot(True, traffic_manager.get_port())

#num_walkers = 2
#walkers = spawn_walkers(num_walkers, world, spawn_points)

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
print("自动驾驶已启用")
# 开启自动变道
traffic_manager.auto_lane_change(vehicle, False)
# 设置全局跟车距离
traffic_manager.set_global_distance_to_leading_vehicle(BASE_FOLLOW_DISTANCE)
#设置遵守交通规则
traffic_manager.ignore_lights_percentage(vehicle, 0.0)  # Ignore all traffic lights
#控制自动驾驶速度（加快）
traffic_manager.vehicle_percentage_speed_difference(vehicle, -50)
# 减少跟车距离
traffic_manager.distance_to_leading_vehicle(vehicle, BASE_FOLLOW_DISTANCE)
# Spawn camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1024')
camera_bp.set_attribute('image_size_y', '1024')
camera_bp.set_attribute('fov', '70')

# Adjust camera position and orientation to avoid car front
#camera_init_trans = carla.Transform(carla.Location(x=2, z=2), carla.Rotation(pitch=-10))
#camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

camera_init_trans = carla.Transform(carla.Location(x= 1, z=2), carla.Rotation(pitch=-3))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue(maxsize=50)


# Camera listener
def image_callback(image):
    if not image_queue.full():
        image_queue.put(image)

camera.listen(image_callback)
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
DISTANCE_THRESHOLD = 50.0  # Example threshold in meters

# Set to track captured traffic sign locations
captured_sign_locations = set()

# Variable to track the last captured image time
last_capture_time = 0
capture_cooldown = 5  # Seconds to wait before capturing another image of the same sign

def get_signs_bounding_boxes(vehicle_transform, camera_transform, K, world_2_camera):
    global captured_sign_locations, last_capture_time  # Use global set to track captured sign locations
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

                # Use location tuple to check if sign is not already captured
                sign_location_tuple = (round(obj.location.x, 2), round(obj.location.y, 2), round(obj.location.z, 2))
                if camera_dot_product > 0 and sign_location_tuple not in captured_sign_locations:
                    verts = [v for v in obj.get_world_vertices(carla.Transform())]
                    x_coords = [get_image_point(v, K, world_2_camera)[0] for v in verts]
                    y_coords = [get_image_point(v, K, world_2_camera)[1] for v in verts]
                    xmin, xmax = int(min(x_coords)), int(max(x_coords))
                    ymin, ymax = int(min(y_coords)), int(max(y_coords))

                    # Calculate the area of the bounding box
                    area = (xmax - xmin) * (ymax - ymin)

                    # Set a threshold for the minimum area to capture the sign
                    # 降低“面积阈值”过滤
                    min_area_threshold = 10  # Adjust this value as needed

                    # Check if the bounding box is fully within the image frame
                    if xmin >= 0 and ymin >= 0 and xmax < image_w and ymax < image_h:
                        # Check the size and aspect ratio of the bounding box
                        aspect_ratio = (xmax - xmin) / float(ymax - ymin) if (ymax - ymin) != 0 else 0
                        if area > min_area_threshold and 0.5 < aspect_ratio < 2.0:
                            bounding_boxes.append(
                                {'label': 'TrafficSign', 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

                            # Store sign location in the set to avoid multiple captures
                            current_time = time.time()
                            if current_time - last_capture_time > capture_cooldown:
                                captured_sign_locations.add(sign_location_tuple)
                                last_capture_time = current_time

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


def update_autopilot_safety(vehicle, traffic_manager, world_map):
    if is_right_turn_imminent(vehicle, world_map):
        traffic_manager.vehicle_percentage_speed_difference(vehicle, TURNING_SPEED_DIFFERENCE)
        traffic_manager.distance_to_leading_vehicle(vehicle, TURNING_FOLLOW_DISTANCE)
    else:
        traffic_manager.vehicle_percentage_speed_difference(vehicle, BASE_SPEED_DIFFERENCE)
        traffic_manager.distance_to_leading_vehicle(vehicle, BASE_FOLLOW_DISTANCE)

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


# Start the game loop
try:
    while True:
        world.tick()
        time.sleep(0.033)
        pygame.event.pump()  # Process event queue for keyboard input
        update_autopilot_safety(vehicle, traffic_manager, world_map)

        # Handle manual input
        # handle_input(vehicle)

        # Get the latest image from the queue
        image = image_queue.get()

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
