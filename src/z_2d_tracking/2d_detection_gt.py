import carla

import queue
import random

import cv2
import numpy as np

# COCO 类别名称
COCO_CLASS_NAMES = [
     'car', 'motorcycle', 'airplane', 'bus', 'truck'
]

from utils.box_utils import draw_bounding_boxes
from utils.projection import *
from utils.world import *

def camera_callback(image, rgb_image_queue):
    rgb_image_queue.put(np.reshape(np.copy(image.raw_data),
                        (image.height, image.width, 4)))

# ===================== Part 1: 初始化环境 =====================
client = carla.Client('localhost', 2000)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

spectator = world.get_spectator()
spawn_points = world.get_map().get_spawn_points()
bp_lib = world.get_blueprint_library()

# 生成自车
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# 生成相机
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '640')

camera_init_trans = carla.Transform(carla.Location(x=1, z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

image_queue = queue.Queue()
camera.listen(lambda image: camera_callback(image, image_queue))

clear_npc(world)
clear_static_vehicle(world)

# ===================== Part 2: 3D投影配置 =====================
edges = [[0,1],[1,3],[3,2],[2,0],[0,4],[4,5],
         [5,1],[5,7],[7,6],[6,4],[6,2],[7,3]]

image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()

K = build_projection_matrix(image_w, image_h, fov)
K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

# 随机生成大量交通工具（包含：轿车、巴士、卡车、摩托）
for i in range(60):
    vehicle_bp = bp_lib.filter('vehicle')
    npc = world.try_spawn_actor(random.choice(vehicle_bp), random.choice(spawn_points))
    if npc:
        npc.set_autopilot(True)

vehicle.set_autopilot(True)

# ===================== 主循环：自带交通工具类型识别 =====================
while True:
    try:
        world.tick()

        transform = carla.Transform(vehicle.get_transform().transform(
            carla.Location(x=-4, z=50)), carla.Rotation(yaw=-180, pitch=-90))
        spectator.set_transform(transform)

        image = image_queue.get()
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        boxes = []
        ids = []
        labels = []  # 存储识别出的类型标签

        # 遍历所有交通工具
        for npc in world.get_actors().filter('*vehicle*'):
            if npc.id == vehicle.id:
                continue

            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(vehicle.get_transform().location)

            # 只处理50米内、前方的目标
            if dist < 50:
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - vehicle.get_transform().location

                if forward_vec.dot(ray) > 0:
                    verts = bb.get_world_vertices(npc.get_transform())
                    points_2d = []

                    for vert in verts:
                        ray0 = vert - camera.get_transform().location
                        cam_forward_vec = camera.get_transform().get_forward_vector()

                        if cam_forward_vec.dot(ray0) > 0:
                            p = get_image_point(vert, K, world_2_camera)
                        else:
                            p = get_image_point(vert, K_b, world_2_camera)
                        points_2d.append(p)

                    x_min, x_max, y_min, y_max = get_2d_box_from_3d_edges(points_2d, edges, image_h, image_w)

                    if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                        if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):

                            # ===================== 核心：自动识别交通工具类型 =====================
                            type_id = npc.type_id
                            wheels = int(npc.attributes.get('number_of_wheels', 4))
                            length = 2 * bb.extent.x
                            height = 2 * bb.extent.z

                            # 识别规则
                            if wheels == 2:
                                cls = 1  # motorcycle
                            elif height > 2.7 and length > 9:
                                cls = 3  # bus
                            elif height > 2.2 and length > 6.5:
                                cls = 4  # truck
                            else:
                                cls = 0  # car

                            ids.append(npc.id)
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(cls)

        # 绘制带类型的框
        if len(boxes) > 0:
            boxes = np.array(boxes)
            labels = np.array(labels)
            probs = np.array([1.0]*len(boxes))
            image = draw_bounding_boxes(image, boxes, labels, COCO_CLASS_NAMES, ids)

        cv2.imshow('Carla 2D Tracking (Car/Bus/Truck/Motorcycle)', image)

        if cv2.waitKey(1) == ord('q'):
            break

    except KeyboardInterrupt:
        break

clear(world, camera)
cv2.destroyAllWindows()