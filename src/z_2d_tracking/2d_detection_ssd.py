import carla

import os
import queue
import random

import cv2
import torch

import numpy as np

from what.models.detection.ssd.mobilenet_v2_ssd_lite import MobileNetV2SSDLite
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.datasets.voc import VOC_CLASS_NAMES

from what.cli.model import *
from what.utils.file import get_file

from utils.projection import *
from utils.world import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Check what_model_list for all available models
index = 7

# Download the model first if not exists
WHAT_MODEL_FILE = what_model_list[index][WHAT_MODEL_FILE_INDEX]
WHAT_MODEL_URL  = what_model_list[index][WHAT_MODEL_URL_INDEX]
WHAT_MODEL_HASH = what_model_list[index][WHAT_MODEL_HASH_INDEX]

if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, WHAT_MODEL_FILE)):
    get_file(WHAT_MODEL_FILE,
             WHAT_MODEL_PATH,
             WHAT_MODEL_URL,
             WHAT_MODEL_HASH)

# Initialize the model
model = MobileNetV2SSDLite(os.path.join(WHAT_MODEL_PATH, WHAT_MODEL_FILE),
                       VOC_CLASS_NAMES,
                       is_test=True,
                       device=device)

def camera_callback(image, rgb_image_queue):
    rgb_image_queue.put(np.reshape(np.copy(image.raw_data),
                        (image.height, image.width, 4)))

# Part 1
client = carla.Client('localhost', 2000)
world = client.get_world()

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True  # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Get the world spectator
spectator = world.get_spectator()

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

# Spawn the ego vehicle
bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# Spawn the camera
camera_bp = bp_lib.find('sensor.camera.rgb')
# [Windows Only] Fixes https://github.com/carla-simulator/carla/issues/6085
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '640')

camera_init_trans = carla.Transform(carla.Location(x=1, z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()
camera.listen(lambda image: camera_callback(image, image_queue))

# Clear existing NPCs
clear_npc(world)
clear_static_vehicle(world)

# Part 2

# Remember the edge pairs
edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
         [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

# Get the world to camera matrix
world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

# Get the attributes from the camera
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()

# Calculate the camera projection matrix to project from 3D -> 2D
K = build_projection_matrix(image_w, image_h, fov)
K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

# Spawn NPCs
for i in range(50):
    vehicle_bp = bp_lib.filter('vehicle')

    # Exclude bicycle
    car_bp = [bp for bp in vehicle_bp if int(
        bp.get_attribute('number_of_wheels')) == 4]
    npc = world.try_spawn_actor(random.choice(
        car_bp), random.choice(spawn_points))

    if npc:
        npc.set_autopilot(True)

vehicle.set_autopilot(True)

# Main loop
while True:
    try:
        world.tick()

        # Move the spectator to the top of the vehicle 
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=50)), carla.Rotation(yaw=-180, pitch=-90)) 
        spectator.set_transform(transform) 

        # Display RGB camera image
        origin_image = image_queue.get()

        # Image preprocessing
        image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

        # Run inference
        images, boxes, labels, probs = model.predict(image, 10, 0.4)

        # Only draw 6: bus, 7: car
        boxes  = torch.Tensor([box.detach().numpy() for box, label in zip(boxes, labels) if label in [6, 7]])
        probs  = torch.Tensor([prob for prob, label in zip(probs, labels) if label in [6, 7]])
        labels = torch.IntTensor([int(7) for label in labels if label in [6, 7]])

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw bounding boxes onto the image
        output = draw_bounding_boxes(image, boxes, labels, model.class_names, probs);

        cv2.imshow('2D SSD MobileNet', image)

        # Quit if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt as e:
        break

clear(world, camera)
cv2.destroyAllWindows()
