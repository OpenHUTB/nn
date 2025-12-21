import carla
import queue
import random
import cv2
import numpy as np
import math
import os
import time  # 新增：计算帧率

# 修复Deep SORT的API弃用问题
import scipy.optimize as opt
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# COCO类别名称
COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# ===================== Deep SORT修复函数 =====================
def linear_assignment(cost_matrix):
    x, y = opt.linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


import deep_sort.utils.linear_assignment as la

la.linear_assignment = linear_assignment


class SimpleBoxEncoder:
    def __init__(self):
        pass

    def __call__(self, image, boxes):
        features = []
        for box in boxes:
            x1, y1, w, h = box
            aspect_ratio = w / h if h != 0 else 1.0
            center_x = (x1 + w / 2) / image.shape[1]
            center_y = (y1 + h / 2) / image.shape[0]
            area = (w * h) / (image.shape[0] * image.shape[1])
            feature = np.array([aspect_ratio, center_x, center_y, area] + [0.0] * 124)
            features.append(feature)
        return np.array(features)


def create_box_encoder(model_filename=None, batch_size=32):
    return SimpleBoxEncoder()


# ===================== 工具函数 =====================
def get_image_point(vertex, K, world_to_camera):
    point_3d = np.array([vertex.x, vertex.y, vertex.z, 1.0])
    point_camera = np.dot(world_to_camera, point_3d)
    point_img = np.dot(K, point_camera[:3])
    point_img = point_img / point_img[2]
    return (point_img[0], point_img[1])


def get_2d_box_from_3d_edges(points_2d, edges, image_h, image_w):
    x_coords = [p[0] for p in points_2d]
    y_coords = [p[1] for p in points_2d]
    x_min = max(0, min(x_coords))
    x_max = min(image_w, max(x_coords))
    y_min = max(0, min(y_coords))
    y_max = min(image_h, max(y_coords))
    return x_min, x_max, y_min, y_max


def point_in_canvas(point, image_h, image_w):
    x, y = point
    return 0 <= x <= image_w and 0 <= y <= image_h


def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    if is_behind_camera:
        K[0, 0] = -K[0, 0]
    return K


def clear_npc(world):
    for actor in world.get_actors().filter('*vehicle*'):
        if actor.attributes.get('role_name') != 'hero':
            actor.destroy()


def clear_static_vehicle(world):
    pass


def clear(world, camera):
    if camera:
        camera.destroy()
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()


# ===================== 【完善】可视化函数 =====================
def draw_bounding_boxes(image, bboxes, labels, class_names, ids):
    """优化框的颜色，不同ID使用不同颜色"""

    # 生成固定的颜色映射（基于ID的哈希值）
    def get_color(track_id):
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())

    for bbox, label, track_id in zip(bboxes, labels, ids):
        x1, y1, x2, y2 = bbox.astype(int)
        color = get_color(track_id)
        # 绘制框和背景
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        class_name = class_names[label] if label < len(class_names) else 'car'
        text = f"ID:{track_id} | {class_name}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image


def draw_info_text(image, speed_kmh, vehicle_count, map_name, fps):
    """新增帧率显示，优化信息排版"""
    image_copy = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    padding = 5

    text_list = [
        f"Map: {map_name}",
        f"Speed: {speed_kmh:.1f} km/h",
        f"Tracked Vehicles: {vehicle_count}",
        f"FPS: {fps:.1f}"  # 新增帧率
    ]

    y_offset = 30
    for text in text_list:
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(
            image_copy,
            (10, y_offset - text_size[1] - padding),
            (10 + text_size[0] + padding * 2, y_offset + padding),
            bg_color, -1
        )
        cv2.putText(image_copy, text, (10 + padding, y_offset), font, font_scale, text_color, font_thickness)
        y_offset += text_size[1] + padding * 3
    return image_copy


def camera_callback(image, rgb_image_queue):
    rgb_image = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    rgb_image_queue.put(rgb_image)


# ===================== 【新增】窗口工具函数 =====================
def init_window(window_name, width, height):
    """初始化窗口：置顶、自适应大小、显示提示"""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 允许调整大小
    cv2.resizeWindow(window_name, width, height)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # 窗口置顶
    # 显示初始提示文字
    init_img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(init_img, "CARLA DeepSORT Tracking", (width // 4, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    cv2.imshow(window_name, init_img)
    cv2.waitKey(1)


def confirm_exit():
    """退出前弹出确认窗口"""
    confirm_img = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.putText(confirm_img, "Quit? (Y/N)", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Confirm Exit", confirm_img)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("Confirm Exit")
    return key == ord('y') or key == ord('Y')


# ===================== 主函数 =====================
def main():
    # 窗口配置
    WINDOW_NAME = "CARLA 2D Tracking (Enhanced Window)"
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    # 初始化CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    spectator = world.get_spectator()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("❌ 无可用生成点！")
        return

    # 生成自车
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020') or bp_lib.filter('vehicle.*')[0]
    spawn_point = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if not vehicle:
        print("❌ 车辆生成失败！")
        return

    # 生成相机
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    camera_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    camera_bp.set_attribute('fov', '90')
    camera_init_trans = carla.Transform(carla.Location(x=1.2, z=2.0), carla.Rotation(pitch=-5))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    # 初始化窗口
    init_window(WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)

    image_queue = queue.Queue(maxsize=2)
    camera.listen(lambda image: camera_callback(image, image_queue))

    clear_npc(world)
    clear_static_vehicle(world)

    # 追踪参数
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
             [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
    K = build_projection_matrix(CAMERA_WIDTH, CAMERA_HEIGHT, 90)
    K_b = build_projection_matrix(CAMERA_WIDTH, CAMERA_HEIGHT, 90, is_behind_camera=True)

    # 生成NPC
    npc_count = 20
    spawned_npcs = 0
    for i in range(npc_count):
        vehicle_bp_list = bp_lib.filter('vehicle')
        car_bp = [bp for bp in vehicle_bp_list if int(bp.get_attribute('number_of_wheels')) == 4]
        if not car_bp:
            continue
        random_spawn = random.choice(spawn_points)
        if random_spawn.location.distance(vehicle.get_location()) < 10.0:
            continue
        npc = world.try_spawn_actor(random.choice(car_bp), random_spawn)
        if npc:
            npc.set_autopilot(True)
            spawned_npcs += 1
    print(f"✅ 生成{spawned_npcs}辆NPC车辆")

    vehicle.set_autopilot(True)

    # 初始化追踪器
    encoder = create_box_encoder()
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
    tracker = Tracker(metric)
    map_name = world.get_map().name.split('/')[-1]

    # 帧率计算变量
    frame_count = 0
    start_time = time.time()
    fps = 0.0

    # 主循环
    while True:
        try:
            world.tick()
            frame_count += 1

            # 计算帧率（每10帧更新一次）
            if frame_count % 10 == 0:
                end_time = time.time()
                fps = 10 / (end_time - start_time)
                start_time = end_time

            # 旁观者视角
            transform = carla.Transform(
                vehicle.get_transform().transform(carla.Location(x=-4, z=50)),
                carla.Rotation(yaw=-180, pitch=-90)
            )
            spectator.set_transform(transform)

            # 获取图像
            if image_queue.empty():
                continue
            image = image_queue.get()
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            image = cv2.flip(image, 1)

            # 3D转2D检测框
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
            boxes = []
            for npc in world.get_actors().filter('*vehicle*'):
                if npc.id != vehicle.id:
                    bb = npc.bounding_box
                    dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                    if dist < 50:
                        forward_vec = vehicle.get_transform().get_forward_vector()
                        ray = npc.get_transform().location - vehicle.get_transform().location
                        if forward_vec.dot(ray) > 0:
                            verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                            points_2d = []
                            for vert in verts:
                                ray0 = vert - camera.get_transform().location
                                cam_forward_vec = camera.get_transform().get_forward_vector()
                                p = get_image_point(vert, K, world_2_camera) if cam_forward_vec.dot(
                                    ray0) > 0 else get_image_point(vert, K_b, world_2_camera)
                                p = (CAMERA_WIDTH - p[0], p[1])
                                points_2d.append(p)
                            x_min, x_max, y_min, y_max = get_2d_box_from_3d_edges(points_2d, edges, CAMERA_HEIGHT,
                                                                                  CAMERA_WIDTH)
                            if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                                if point_in_canvas((x_min, y_min), CAMERA_HEIGHT, CAMERA_WIDTH) and point_in_canvas(
                                        (x_max, y_max), CAMERA_HEIGHT, CAMERA_WIDTH):
                                    boxes.append(np.array([x_min, y_min, x_max, y_max]))

            boxes = np.array(boxes)
            detections = []
            if len(boxes) > 0:
                sort_boxes = boxes.copy()
                for i, box in enumerate(sort_boxes):
                    box[2] -= box[0]
                    box[3] -= box[1]
                    feature = encoder(image, box.reshape(1, -1).copy())
                    detections.append(Detection(box, 1.0, feature[0]))

            # 更新追踪器
            tracker.predict()
            tracker.update(detections)

            # 绘制结果
            bboxes, ids = [], []
            for track in tracker.tracks:
                if track.is_confirmed() and track.time_since_update <= 1:
                    bboxes.append(track.to_tlbr())
                    ids.append(track.track_id)
            bboxes = np.array(bboxes)
            tracked_vehicle_count = len(bboxes)

            if len(bboxes) > 0:
                labels = np.array([2] * len(bboxes))
                image = draw_bounding_boxes(image, bboxes, labels, COCO_CLASS_NAMES, ids)

            # 绘制信息（含帧率）
            velocity = vehicle.get_velocity()
            speed_ms = math.hypot(velocity.x, velocity.y)
            speed_kmh = speed_ms * 3.6
            image = draw_info_text(image, speed_kmh, tracked_vehicle_count, map_name, fps)

            # 显示图像
            cv2.imshow(WINDOW_NAME, image)

            # 按键处理（完善退出逻辑）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if confirm_exit():  # 确认退出
                    break
            elif key == ord('f'):  # F键切换全屏
                current_flag = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                new_flag = cv2.WINDOW_FULLSCREEN if current_flag == 0 else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, new_flag)
            elif key == ord('s'):  # S键保存当前帧
                save_path = f"track_frame_{frame_count}.png"
                cv2.imwrite(save_path, image)
                print(f"💾 帧已保存至 {save_path}")

        except KeyboardInterrupt:
            if confirm_exit():
                break
        except Exception as e:
            print(f"⚠️ 运行错误：{e}")
            continue

    # 清理资源
    clear(world, camera)
    settings.synchronous_mode = False
    world.apply_settings(settings)
    cv2.destroyAllWindows()
    print("✅ 程序正常退出")


if __name__ == '__main__':
    main()