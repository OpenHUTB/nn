import carla

import queue
import random

import cv2
import numpy as np

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from sort.sort import Sort

from utils.projection import *
from utils.world import *


def draw_bounding_boxes(image, boxes, labels, class_names, ids):
    if not hasattr(draw_bounding_boxes, "colours"):
        draw_bounding_boxes.colours = np.random.randint(0, 256, size=(32, 3))

    if len(boxes) > 0:
        assert (boxes.shape[1] == 4)

    # Draw bounding boxes and labels
    for i in range(boxes.shape[0]):
        box = boxes[i]
        label = f"{class_names[labels[i]]}: {int(ids[i])}"
        # print(label)

        # Draw bounding boxes
        # cv2.rectangle(  image,
        #                 (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())),
        #                 tuple([int(c) for c in draw_bounding_boxes.colours[int(ids[i]) % 32, :]]),
        #                 4)

        # Draw labels
        cv2.putText(image, label,
                    (int(box[0]+20), int(box[1]+40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    tuple(
                        [int(c) for c in draw_bounding_boxes.colours[int(ids[i]) % 32, :]]),
                    2)  # line type
    return image

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

mot_tracker = Sort(max_age=1,
                   min_hits=3,
                   iou_threshold=0.3)  # create instance of the SORT tracker

# Main loop
while True:
    try:
        world.tick()

        # Move the spectator to the top of the vehicle
        transform = carla.Transform(vehicle.get_transform().transform(
            carla.Location(x=-4, z=50)), carla.Rotation(yaw=-180, pitch=-90))
        spectator.set_transform(transform)

        # Display RGB camera image
        image = image_queue.get()

        # Image preprocessing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get the camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        boxes = []
        for npc in world.get_actors().filter('*vehicle*'):

            # Filter out the ego vehicle
            if npc.id != vehicle.id:

                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)

                # Filter for the vehicles within 50m
                if dist < 50:
                    # Calculate the dot product between the forward vector
                    # of the vehicle and the vector between the vehicle
                    # and the other vehicle. We threshold this dot product
                    # to limit to drawing bounding boxes IN FRONT OF THE CAMERA

                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - vehicle.get_transform().location

                    if forward_vec.dot(ray) > 0:

                        verts = [v for v in bb.get_world_vertices(
                            npc.get_transform())]

                        points_2d = []

                        for vert in verts:
                            ray0 = vert - camera.get_transform().location
                            cam_forward_vec = camera.get_transform().get_forward_vector()

                            if (cam_forward_vec.dot(ray0) > 0):
                                p = get_image_point(vert, K, world_2_camera)
                            else:
                                p = get_image_point(vert, K_b, world_2_camera)

                            points_2d.append(p)

                        for edge in edges:
                            p1 = points_2d[edge[0]]
                            p2 = points_2d[edge[1]]

                            p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                            p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                            # Both points are out of the canvas
                            if not p1_in_canvas and not p2_in_canvas:
                                continue
                            
                            # Draw 3D Bounding Boxes
                            cv2.line(image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)        

                        x_min, x_max, y_min, y_max = get_2d_box_from_3d_edges(
                            points_2d, edges, image_h, image_w)

                        # Exclude very small bounding boxes
                        if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                            if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):
                                boxes.append(
                                    np.array([x_min, y_min, x_max, y_max]))

        # Only draw 2: car, 5: bus, 7: truck
        boxes = np.array(boxes)
        labels = np.array([2] * len(boxes))
        probs = np.array([1.0] * len(boxes))

        if len(boxes) > 0:
            dets = np.concatenate(
                (boxes, probs.reshape((len(probs), -1))), axis=1)

            # Update tracker
            trackers = mot_tracker.update(dets)

            # Draw bounding boxes onto the image
            output = draw_bounding_boxes(
                image, trackers[:, 0: 4], labels, COCO_CLASS_NAMES, trackers[:, 4])

        cv2.imshow('3D Ground Truth SORT', image)

        # Quit if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt as e:
        break

clear(world, camera)
cv2.destroyAllWindows()
