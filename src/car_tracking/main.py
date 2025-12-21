import carla
import queue
import random
import cv2
import numpy as np
import math
import os  # 新增：处理文件路径
# 修复Deep SORT的API弃用问题
import scipy.optimize as opt
# 替换原有的linear_assignment导入
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
# 若你的COCO_CLASS_NAMES不存在，手动定义（避免依赖缺失）
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

# ===================== 修复：替换Deep SORT的linear_assignment（解决弃用警告） =====================
def linear_assignment(cost_matrix):
    """替换原有的linear_assignment，使用scipy的linear_sum_assignment"""
    x, y = opt.linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

# 覆盖deep_sort.utils.linear_assignment中的同名函数
import deep_sort.utils.linear_assignment as la
la.linear_assignment = linear_assignment

# ===================== 修复：重新实现Box Encoder（解决pb文件读取错误） =====================
# 由于mars-small128.pb读取失败，改用简单的特征编码（不依赖TensorFlow，保证运行）
class SimpleBoxEncoder:
    def __init__(self):
        pass
    def __call__(self, image, boxes):
        """生成简单的框特征（替代原有的mars-small128编码，保证追踪功能运行）"""
        features = []
        for box in boxes:
            x1, y1, w, h = box
            # 特征：框的宽高比、中心坐标归一化、面积归一化
            aspect_ratio = w / h if h != 0 else 1.0
            center_x = (x1 + w/2) / image.shape[1]
            center_y = (y1 + h/2) / image.shape[0]
            area = (w * h) / (image.shape[0] * image.shape[1])
            feature = np.array([aspect_ratio, center_x, center_y, area] + [0.0]*124)  # 凑128维
            features.append(feature)
        return np.array(features)

def create_box_encoder(model_filename=None, batch_size=32):
    """替换原有的create_box_encoder，返回简单编码器"""
    return SimpleBoxEncoder()

# ===================== 自定义工具函数（替代utils中的函数，避免依赖缺失） =====================
def get_image_point(vertex, K, world_to_camera):
    """3D点投影到2D图像平面（简化版）"""
    # 3D点（齐次坐标）
    point_3d = np.array([vertex.x, vertex.y, vertex.z, 1.0])
    # 世界到相机
    point_camera = np.dot(world_to_camera, point_3d)
    # 相机到图像（投影矩阵K）
    point_img = np.dot(K, point_camera[:3])
    # 归一化
    point_img = point_img / point_img[2]
    return (point_img[0], point_img[1])

def get_2d_box_from_3d_edges(points_2d, edges, image_h, image_w):
    """从3D边的2D点生成2D边界框"""
    x_coords = [p[0] for p in points_2d]
    y_coords = [p[1] for p in points_2d]
    x_min = max(0, min(x_coords))
    x_max = min(image_w, max(x_coords))
    y_min = max(0, min(y_coords))
    y_max = min(image_h, max(y_coords))
    return x_min, x_max, y_min, y_max

def point_in_canvas(point, image_h, image_w):
    """判断点是否在画布内"""
    x, y = point
    return 0 <= x <= image_w and 0 <= y <= image_h

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """构建投影矩阵（简化版）"""
    focal = w / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    if is_behind_camera:
        K[0, 0] = -K[0, 0]
    return K

def clear_npc(world):
    """清理NPC车辆"""
    for actor in world.get_actors().filter('*vehicle*'):
        if actor.attributes.get('role_name') != 'hero':
            actor.destroy()

def clear_static_vehicle(world):
    """清理静态车辆（空实现，避免报错）"""
    pass

def clear(world, camera):
    """清理资源"""
    if camera:
        camera.destroy()
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()

def draw_bounding_boxes(image, bboxes, labels, class_names, ids):
    """绘制边界框和追踪ID（解决文字显示问题）"""
    for bbox, label, track_id in zip(bboxes, labels, ids):
        x1, y1, x2, y2 = bbox.astype(int)
        # 绘制框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制标签和ID
        class_name = class_names[label] if label < len(class_names) else 'car'
        text = f"{class_name} | ID: {track_id}"
        # 文字背景
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image

# ===================== 可视化信息绘制函数 =====================
def draw_info_text(image, speed_kmh, vehicle_count, map_name):
    """在图像上绘制车速、车辆数量、地图名称等信息"""
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
        f"Tracked Vehicles: {vehicle_count}"
    ]

    y_offset = 30
    for text in text_list:
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(
            image_copy,
            (10, y_offset - text_size[1] - padding),
            (10 + text_size[0] + padding * 2, y_offset + padding),
            bg_color,
            -1
        )
        cv2.putText(
            image_copy,
            text,
            (10 + padding, y_offset),
            font,
            font_scale,
            text_color,
            font_thickness
        )
        y_offset += text_size[1] + padding * 3

    return image_copy

def camera_callback(image, rgb_image_queue):
    """摄像头回调函数"""
    rgb_image_queue.put(np.reshape(np.copy(image.raw_data),
                        (image.height, image.width, 4)))

def main():
    # Part 1: 初始化CARLA环境
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 获取 spectator
    spectator = world.get_spectator()

    # 获取生成点
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("❌ 无可用生成点！")
        return

    # 生成自车
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    if not vehicle_bp:
        vehicle_bp = bp_lib.filter('vehicle.*')[0]
    spawn_point = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if not vehicle:
        print("❌ 车辆生成失败！")
        return

    # 生成摄像头
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '90')

    camera_init_trans = carla.Transform(carla.Location(x=1.2, z=2.0), carla.Rotation(pitch=-5))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    # 图像队列
    image_queue = queue.Queue(maxsize=2)
    camera.listen(lambda image: camera_callback(image, image_queue))

    # 清理现有NPC
    clear_npc(world)
    clear_static_vehicle(world)

    # Part 2: 初始化追踪参数
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
             [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

    # 摄像头参数
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    # 投影矩阵
    K = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

    # 生成NPC
    npc_count = 20
    spawned_npcs = 0
    for i in range(npc_count):
        vehicle_bp_list = bp_lib.filter('vehicle')
        car_bp = [bp for bp in vehicle_bp_list if int(
            bp.get_attribute('number_of_wheels')) == 4]
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

    # Deep SORT（使用修复后的编码器）
    encoder = create_box_encoder("mars-small128.pb", batch_size=32)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
    tracker = Tracker(metric)

    # 地图名称
    map_name = world.get_map().name.split('/')[-1]

    # 主循环
    while True:
        try:
            world.tick()

            # 移动spectator
            transform = carla.Transform(vehicle.get_transform().transform(
                carla.Location(x=-4, z=50)), carla.Rotation(yaw=-180, pitch=-90))
            spectator.set_transform(transform)

            # 获取图像
            if image_queue.empty():
                continue
            image = image_queue.get()

            # 图像预处理：BGRA→BGR→水平翻转（解决反向）
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            image = cv2.flip(image, 1)

            # 更新相机矩阵
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
                                if (cam_forward_vec.dot(ray0) > 0):
                                    p = get_image_point(vert, K, world_2_camera)
                                else:
                                    p = get_image_point(vert, K_b, world_2_camera)
                                # 翻转x坐标
                                p = (image_w - p[0], p[1])
                                points_2d.append(p)

                            x_min, x_max, y_min, y_max = get_2d_box_from_3d_edges(
                                points_2d, edges, image_h, image_w)

                            if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                                if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):
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

            bboxes = []
            ids = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                bboxes.append(bbox)
                ids.append(track.track_id)

            bboxes = np.array(bboxes)
            tracked_vehicle_count = len(bboxes)

            if len(bboxes) > 0:
                labels = np.array([2] * len(bboxes))
                image = draw_bounding_boxes(
                    image, bboxes, labels, COCO_CLASS_NAMES, ids)

            # 计算车速并绘制信息
            velocity = vehicle.get_velocity()
            speed_ms = math.hypot(velocity.x, velocity.y)
            speed_kmh = speed_ms * 3.6
            image = draw_info_text(image, speed_kmh, tracked_vehicle_count, map_name)

            # 显示图像
            cv2.imshow('2D Ground Truth Deep SORT (Fixed All Issues)', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt as e:
            break
        except Exception as e:
            print(f"⚠️ 运行错误：{e}")
            continue

    # 清理资源
    clear(world, camera)
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    cv2.destroyAllWindows()
    print("✅ 程序正常退出")

if __name__ == '__main__':
    main()