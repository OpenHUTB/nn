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

from threading import Thread, Event

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

# ---------------------- 子线程：只推理，不取相机图 ----------------------
# 优化后的线程实现
class DetectionThread(Thread):
    def __init__(self, model, detect_interval=3):
        super().__init__(daemon=True)
        self.model = model
        self.detect_interval = detect_interval
        self.image_queue = queue.Queue(maxsize=2)  # 允许2帧缓冲
        self.result_queue = queue.Queue(maxsize=1)
        self.stop_event = Event()
        self.frame_count = 0

    def run(self):
        while not self.stop_event.is_set():
            try:
                # 阻塞等待新图像
                origin_image = self.image_queue.get(timeout=1)
                self.frame_count += 1

                # 间隔推理
                if self.frame_count % self.detect_interval != 0:
                    continue

                # 推理（保持你的原有逻辑）
                image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
                input = np.array(image).transpose((2, 0, 1))
                input = torch.from_numpy(input)[None]
                inputs, boxes, labels, scores = self.model.predict(input)

                boxes = np.array(boxes)[0]
                boxes = np.array([box for box, label in zip(boxes, labels[0]) if label in [6, 7]])
                scores = np.array([score for score, label in zip(scores[0], labels[0]) if label in [6, 7]])
                labels = np.array([6 for label in labels[0] if label in [6, 7]])

                # 非阻塞放入结果（避免推理线程阻塞）
                try:
                    self.result_queue.put((origin_image, boxes, labels, scores), block=False)
                except queue.Full:
                    # 如果队列满，丢弃旧结果
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put((origin_image, boxes, labels, scores), block=False)
                    except queue.Empty:
                        pass

            except queue.Empty:
                continue

    def stop(self):
        self.stop_event.set()
        self.join(timeout=5)


# 使用方式
detector = DetectionThread(model, detect_interval=6)
detector.start()

# 主线程
output = None  # 初始化output，防止未定义引用
try:
    while True:
        world.tick()

        # 检查车辆是否还存在
        if not vehicle or not vehicle.is_alive:
            print("Vehicle destroyed, respawning...")
            vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            if vehicle:
                vehicle.set_autopilot(True)
            continue

        # 视角跟随
        transform = carla.Transform(
            vehicle.get_transform().transform(carla.Location(x=-4, z=50)),
            carla.Rotation(yaw=-180, pitch=-90)
        )
        spectator.set_transform(transform)

        if not image_queue.empty():
            while image_queue.qsize() > 1:
                image_queue.get()
            origin_image = image_queue.get()

            # 发送到推理线程
            try:
                detector.image_queue.put(origin_image, block=False)
            except queue.Full:
                pass  # 丢弃旧帧

            # 获取推理结果
            if not detector.result_queue.empty():
                img, boxes, labels, scores = detector.result_queue.get()
                output = draw_bounding_boxes(img, boxes, labels, VOC_CLASS_NAMES[1:], scores)
            else:
                output = None  # 没有结果时重置output

            cv2.imshow('2D Faster RCNN', output if output is not None else origin_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    detector.stop()
    clear(world, camera)
    cv2.destroyAllWindows()