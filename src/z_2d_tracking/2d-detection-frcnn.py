import carla

import os
import queue
import random

import cv2
import torch
import numpy as np

from what.cli.model import *
from what.utils.file import get_file

from what.models.detection.frcnn.faster_rcnn import FasterRCNN
from what.models.detection.datasets.voc import VOC_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes

from utils.box_utils import draw_bounding_boxes
from utils.projection import *
from utils.world import *
from threading import Thread


# Check what_model_list for all available models
index = 8

# Download the model first if not exists
WHAT_MODEL_FILE = what_model_list[index][WHAT_MODEL_FILE_INDEX]
WHAT_MODEL_URL = what_model_list[index][WHAT_MODEL_URL_INDEX]
WHAT_MODEL_HASH = what_model_list[index][WHAT_MODEL_HASH_INDEX]

if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, WHAT_MODEL_FILE)):
    get_file(WHAT_MODEL_FILE,
             WHAT_MODEL_PATH,
             WHAT_MODEL_URL,
             WHAT_MODEL_HASH)

# Faster RCNN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FasterRCNN(device=device)
model.load(os.path.join(WHAT_MODEL_PATH, WHAT_MODEL_FILE), map_location=device)

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
camera_bp.set_attribute('image_size_x', '320')
camera_bp.set_attribute('image_size_y', '320')

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

for i in range(20):
    vehicle_bp = bp_lib.filter('vehicle')

    # Exclude bicycle
    car_bp = [bp for bp in vehicle_bp if int(
        bp.get_attribute('number_of_wheels')) == 4]
    npc = world.try_spawn_actor(random.choice(
        car_bp), random.choice(spawn_points))

    if npc:
        npc.set_autopilot(True)

vehicle.set_autopilot(True)
# 配置
detect_interval = 4 # 每2帧推理一次（可调，越大越流畅）
result_queue = queue.Queue(maxsize=1)  # 存放检测结果
latest_image = None  # 主线程存最新画面
stop_flag = False    # 停止线程标记

# ---------------------- 子线程：只跑模型推理（已加帧控制！） ----------------------
def inference_worker():
    frame_count_sub = 0


      # 子线程帧计数

    while not stop_flag:
        if image_queue.empty():
            continue

        # 取最新图
        origin_image = image_queue.get()

        # ===================== 关键：每隔 N 帧才推理！ =====================
        frame_count_sub += 1
        if frame_count_sub % detect_interval == 0:


            # ---------------------- 下面只有到了指定帧才会跑 ----------------------
            image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            input = np.array(image).transpose((2, 0, 1))
            input = torch.from_numpy(input)[None]

            inputs, boxes, labels, scores = model.predict(input)

            boxes = np.array(boxes)[0]
            boxes = np.array([box for box, label in zip(boxes, labels[0]) if label in [6, 7]])
            scores = np.array([score for score, label in zip(scores[0], labels[0]) if label in [6, 7]])
            labels = np.array([6 for label in labels[0] if label in [6, 7]])

            if not result_queue.full():
                result_queue.put((origin_image, boxes, labels, scores))


# ---------------------- 启动后台推理线程 ----------------------
infer_thread = Thread(target=inference_worker, daemon=True)
infer_thread.start()
# ---------------------- 主线程：只负责仿真 + 显示（丝滑） ----------------------

output = None  # 保存最新画框图

try:
    while True:
        world.tick()



        # Move the spectator to the top of the vehicle
        transform = carla.Transform(vehicle.get_transform().transform(
            carla.Location(x=-4, z=50)), carla.Rotation(yaw=-180, pitch=-90))
        spectator.set_transform(transform)



        # ===================== 主线程：只显示，不推理 =====================
        if not image_queue.empty():
            while image_queue.qsize() > 1:
                image_queue.get()  # 丢掉延迟的旧图
            origin_image = image_queue.get()

            # 从结果队列拿最新检测框（非阻塞，超快）
            if not result_queue.empty():
                img, boxes, labels, scores = result_queue.get()
                output = draw_bounding_boxes(
                    img, boxes, labels, VOC_CLASS_NAMES[1:], scores
                )

            # 显示画面
            cv2.imshow('2D Faster RCNN', output if output is not None else origin_image)

        # 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 安全停止线程

    stop_flag = True
    clear(world, camera)
    cv2.destroyAllWindows()

