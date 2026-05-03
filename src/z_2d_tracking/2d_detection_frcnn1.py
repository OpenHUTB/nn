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

from threading import Thread, Lock

# 共享图片 + 线程锁（保证主线程、子线程不冲突）
shared_image = None
shared_image_lock = Lock()
# 模型下载（不动）
index = 8
WHAT_MODEL_FILE = what_model_list[index][WHAT_MODEL_FILE_INDEX]
WHAT_MODEL_URL = what_model_list[index][WHAT_MODEL_URL_INDEX]
WHAT_MODEL_HASH = what_model_list[index][WHAT_MODEL_HASH_INDEX]

if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, WHAT_MODEL_FILE)):
    get_file(WHAT_MODEL_FILE, WHAT_MODEL_PATH, WHAT_MODEL_URL, WHAT_MODEL_HASH)

# 模型加载
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FasterRCNN(device=device)
model.load(os.path.join(WHAT_MODEL_PATH, WHAT_MODEL_FILE), map_location=device)

def camera_callback(image, rgb_image_queue):
    rgb_image_queue.put(np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)))

# 连接CARLA
client = carla.Client('localhost', 2000)
world = client.get_world()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

spectator = world.get_spectator()
spawn_points = world.get_map().get_spawn_points()
bp_lib = world.get_blueprint_library()

# 主车
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# 相机
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '320')
camera_bp.set_attribute('image_size_y', '320')
camera_init_trans = carla.Transform(carla.Location(x=1, z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

# 队列
image_queue = queue.Queue()
camera.listen(lambda image: camera_callback(image, image_queue))

clear_npc(world)
clear_static_vehicle(world)

# 生成20辆NPC
for i in range(20):
    vehicle_bp = bp_lib.filter('vehicle')
    car_bp = [bp for bp in vehicle_bp if int(bp.get_attribute('number_of_wheels')) == 4]
    npc = world.try_spawn_actor(random.choice(car_bp), random.choice(spawn_points))
    if npc:
        npc.set_autopilot(True)

vehicle.set_autopilot(True)

# ==============================================
# 🔥 关键修复：线程配置
# 子线程 绝对不允许从 image_queue 取图！
# ==============================================
detect_interval = 6
result_queue = queue.Queue(maxsize=1)
thread_image_queue = queue.Queue(maxsize=1)  # 专门给线程的图
stop_flag = False

# ---------------------- 子线程：只推理，不取相机图 ----------------------
# ---------------------- 子线程：只推理，不取图 ----------------------
# ---------------------- 子线程：只推理，不取图 ----------------------
def inference_worker():
    global stop_flag, model, detect_interval
    frame_count_sub = 0

    while not stop_flag:
        # 🔥 修复：如果还没拿到图，直接跳过，不执行！
        if shared_image is None:
            continue

        # 加锁安全读取图片
        with shared_image_lock:
            origin_image = shared_image.copy()

        # 间隔帧控制
        frame_count_sub += 1
        if frame_count_sub % detect_interval != 0:
            continue

        # --------------------- 你原来的、能画框的推理代码 ---------------------
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
# 启动线程
infer_thread = Thread(target=inference_worker, daemon=True)
infer_thread.start()

# ---------------------- 主线程：唯一取图、保证UE4不卡 ----------------------
output = None

try:
    while True:
        world.tick()

        # 视角跟随
        transform = carla.Transform(
            vehicle.get_transform().transform(carla.Location(x=-4, z=50)),
            carla.Rotation(yaw=-180, pitch=-90)
        )
        spectator.set_transform(transform)

        # 主线程 唯一取图
        if not image_queue.empty():
            while image_queue.qsize() > 1:
                image_queue.get()
            origin_image = image_queue.get()

            # 主线程把图存到共享区
            with shared_image_lock:
                shared_image = origin_image.copy()

            # 有结果就画框
            if not result_queue.empty():
                img, boxes, labels, scores = result_queue.get()
                output = draw_bounding_boxes(img, boxes, labels, VOC_CLASS_NAMES[1:], scores)

            cv2.imshow('2D Faster RCNN', output if output is not None else origin_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stop_flag = True
    clear(world, camera)
    cv2.destroyAllWindows()