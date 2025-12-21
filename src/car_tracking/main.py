# 导入CARLA仿真环境库
import carla
# 导入队列库，用于处理摄像头图像的异步获取
import queue
# 导入随机数库，用于随机生成车辆和位置
import random
# 导入OpenCV库，用于图像处理和显示
import cv2
# 导入数值计算库，用于矩阵和数组操作
import numpy as np
# 导入数学库，用于几何计算和三角函数
import math
# 导入操作系统库，处理文件路径相关操作
import os

# 修复Deep SORT的API弃用问题：导入scipy的优化模块
import scipy.optimize as opt

# 导入Deep SORT的核心模块
from deep_sort import nn_matching  # 近邻匹配度量
from deep_sort.detection import Detection  # 检测结果类
from deep_sort.tracker import Tracker  # 追踪器类

# 定义COCO数据集的类别名称（避免依赖缺失，手动定义）
# 这里主要用到索引2的'car'类别
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
    """
    替换原有的linear_assignment函数，解决scipy旧版本API弃用问题
    使用scipy的linear_sum_assignment实现匈牙利算法，用于匹配检测框和追踪框
    参数：
        cost_matrix: 代价矩阵，元素表示检测框和追踪框的匹配代价
    返回：
        匹配的索引对数组，格式为[(检测框索引, 追踪框索引), ...]
    """
    # linear_sum_assignment返回行和列的索引，对应最小代价匹配
    x, y = opt.linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

# 覆盖deep_sort.utils.linear_assignment中的同名函数，实现无缝替换
import deep_sort.utils.linear_assignment as la
la.linear_assignment = linear_assignment

# ===================== 修复：重新实现Box Encoder（解决pb文件读取错误） =====================
class SimpleBoxEncoder:
    """
    简单的边界框特征编码器（替代原有的mars-small128.pb模型编码）
    解决TensorFlow pb模型文件读取失败的问题，保证追踪功能正常运行
    生成128维的特征向量，与原编码器输出维度一致
    """
    def __init__(self):
        pass

    def __call__(self, image, boxes):
        """
        生成边界框的特征向量
        参数：
            image: 输入图像（np.array），用于获取图像尺寸进行归一化
            boxes: 边界框列表，每个框格式为[x1, y1, w, h]（左上角坐标+宽高）
        返回：
            128维特征向量的数组，形状为(len(boxes), 128)
        """
        features = []
        for box in boxes:
            x1, y1, w, h = box
            # 计算框的宽高比（避免除零错误）
            aspect_ratio = w / h if h != 0 else 1.0
            # 计算框的中心坐标并归一化（相对于图像宽度/高度）
            center_x = (x1 + w/2) / image.shape[1]
            center_y = (y1 + h/2) / image.shape[0]
            # 计算框的面积并归一化（相对于图像总面积）
            area = (w * h) / (image.shape[0] * image.shape[1])
            # 拼接特征：前4维为几何特征，后124维补0，凑成128维
            feature = np.array([aspect_ratio, center_x, center_y, area] + [0.0]*124)
            features.append(feature)
        return np.array(features)

def create_box_encoder(model_filename=None, batch_size=32):
    """
    替换原有的create_box_encoder函数，返回自定义的简单编码器
    参数：
        model_filename: 原模型文件名（此处无用，保留参数以兼容接口）
        batch_size: 批处理大小（此处无用，保留参数以兼容接口）
    返回：
        SimpleBoxEncoder实例
    """
    return SimpleBoxEncoder()

# ===================== 自定义工具函数（替代utils中的函数，避免依赖缺失） =====================
def get_image_point(vertex, K, world_to_camera):
    """
    将3D世界坐标点投影到2D图像平面（简化版投影计算）
    参数：
        vertex: CARLA的3D顶点对象（包含x, y, z属性）
        K: 相机内参矩阵（3x3）
        world_to_camera: 世界坐标到相机坐标的变换矩阵（4x4）
    返回：
        2D图像坐标(x, y)
    """
    # 将3D点转换为齐次坐标（4维）
    point_3d = np.array([vertex.x, vertex.y, vertex.z, 1.0])
    # 世界坐标→相机坐标（矩阵乘法）
    point_camera = np.dot(world_to_camera, point_3d)
    # 相机坐标→图像坐标（使用内参矩阵K，取前3维）
    point_img = np.dot(K, point_camera[:3])
    # 归一化（除以z分量，透视投影）
    point_img = point_img / point_img[2]
    return (point_img[0], point_img[1])

def get_2d_box_from_3d_edges(points_2d, edges, image_h, image_w):
    """
    从3D物体的2D投影点集生成2D边界框（包围盒）
    参数：
        points_2d: 2D投影点列表，每个点为(x, y)
        edges: 3D物体的边索引（此处未使用，保留参数兼容接口）
        image_h: 图像高度
        image_w: 图像宽度
    返回：
        边界框的x_min, x_max, y_min, y_max（限制在图像范围内）
    """
    # 提取所有点的x和y坐标
    x_coords = [p[0] for p in points_2d]
    y_coords = [p[1] for p in points_2d]
    # 计算最小/最大坐标，限制在图像范围内（避免超出画布）
    x_min = max(0, min(x_coords))
    x_max = min(image_w, max(x_coords))
    y_min = max(0, min(y_coords))
    y_max = min(image_h, max(y_coords))
    return x_min, x_max, y_min, y_max

def point_in_canvas(point, image_h, image_w):
    """
    判断2D点是否在图像画布内
    参数：
        point: 2D点(x, y)
        image_h: 图像高度
        image_w: 图像宽度
    返回：
        布尔值，True表示在画布内，False表示超出范围
    """
    x, y = point
    return 0 <= x <= image_w and 0 <= y <= image_h

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """
    构建相机内参矩阵（投影矩阵），基于图像尺寸和视场角
    参数：
        w: 图像宽度
        h: 图像高度
        fov: 相机视场角（水平，单位：度）
        is_behind_camera: 是否处理相机后方的点（翻转焦距符号）
    返回：
        3x3的内参矩阵K
    """
    # 计算焦距：focal = 宽度 / (2 * tan(fov/2))（弧度转换）
    focal = w / (2.0 * math.tan(fov * math.pi / 360.0))
    # 初始化单位矩阵
    K = np.identity(3)
    # 设置x和y方向的焦距
    K[0, 0] = K[1, 1] = focal
    # 设置图像中心坐标（主点）
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    # 处理相机后方的点：翻转焦距符号
    if is_behind_camera:
        K[0, 0] = -K[0, 0]
    return K

def clear_npc(world):
    """
    清理CARLA世界中的NPC车辆（保留主角车辆）
    参数：
        world: CARLA的世界对象
    """
    for actor in world.get_actors().filter('*vehicle*'):
        # 只销毁角色不是hero的车辆（主角车辆role_name为hero）
        if actor.attributes.get('role_name') != 'hero':
            actor.destroy()

def clear_static_vehicle(world):
    """
    清理静态车辆（空实现，避免依赖缺失导致的报错）
    参数：
        world: CARLA的世界对象
    """
    pass

def clear(world, camera):
    """
    清理CARLA仿真环境的资源（销毁相机和所有车辆）
    参数：
        world: CARLA的世界对象
        camera: 相机传感器对象
    """
    # 销毁相机
    if camera:
        camera.destroy()
    # 销毁所有车辆
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()

def draw_bounding_boxes(image, bboxes, labels, class_names, ids):
    """
    在图像上绘制边界框、类别标签和追踪ID
    参数：
        image: 输入图像（np.array）
        bboxes: 边界框列表，每个框为[x1, y1, x2, y2]（左上角+右下角）
        labels: 类别索引列表
        class_names: 类别名称列表
        ids: 追踪ID列表
    返回：
        绘制后的图像
    """
    for bbox, label, track_id in zip(bboxes, labels, ids):
        # 将边界框坐标转换为整数
        x1, y1, x2, y2 = bbox.astype(int)
        # 绘制矩形边界框（绿色，线宽2）
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 获取类别名称（默认car）
        class_name = class_names[label] if label < len(class_names) else 'car'
        # 拼接标签文本：类别 + 追踪ID
        text = f"{class_name} | ID: {track_id}"
        # 计算文本尺寸，绘制文本背景（绿色填充）
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
        # 绘制文本（白色，字体大小0.5，线宽2）
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image

# ===================== 可视化信息绘制函数 =====================
def draw_info_text(image, speed_kmh, vehicle_count, map_name):
    """
    在图像左上角绘制仿真信息：地图名称、车速、追踪车辆数量
    参数：
        image: 输入图像（np.array）
        speed_kmh: 车辆速度（km/h）
        vehicle_count: 追踪到的车辆数量
        map_name: 地图名称
    返回：
        绘制后的图像
    """
    # 复制图像，避免修改原图像
    image_copy = image.copy()
    # 设置字体参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (255, 255, 255)  # 白色文本
    bg_color = (0, 0, 0)  # 黑色背景
    padding = 5  # 文本背景的内边距

    # 定义要显示的文本列表
    text_list = [
        f"Map: {map_name}",          # 地图名称
        f"Speed: {speed_kmh:.1f} km/h",  # 车速（保留1位小数）
        f"Tracked Vehicles: {vehicle_count}"  # 追踪车辆数量
    ]

    # 初始化文本y轴偏移量
    y_offset = 30
    for text in text_list:
        # 计算文本尺寸
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        # 绘制文本背景矩形
        cv2.rectangle(
            image_copy,
            (10, y_offset - text_size[1] - padding),  # 左上角
            (10 + text_size[0] + padding * 2, y_offset + padding),  # 右下角
            bg_color,  # 背景色
            -1  # 填充矩形
        )
        # 绘制文本
        cv2.putText(
            image_copy,
            text,
            (10 + padding, y_offset),  # 文本起始位置
            font,
            font_scale,
            text_color,
            font_thickness
        )
        # 更新y轴偏移量（换行）
        y_offset += text_size[1] + padding * 3

    return image_copy

def camera_callback(image, rgb_image_queue):
    """
    相机传感器的回调函数，将获取的图像存入队列
    参数：
        image: CARLA的图像对象
        rgb_image_queue: 存储图像的队列
    """
    # 将原始图像数据转换为numpy数组（BGRA格式，尺寸为height×width×4）
    rgb_image = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    # 将图像存入队列
    rgb_image_queue.put(rgb_image)

def main():
    """
    主函数：初始化CARLA环境、相机传感器、Deep SORT追踪器，运行主循环实现车辆追踪
    """
    # Part 1: 初始化CARLA环境
    # 连接CARLA服务器（本地地址，端口2000）
    client = carla.Client('localhost', 2000)
    # 设置超时时间（10秒）
    client.set_timeout(10.0)
    # 获取CARLA世界对象
    world = client.get_world()

    # 设置CARLA同步模式（保证仿真步长固定，避免图像和数据不同步）
    settings = world.get_settings()
    settings.synchronous_mode = True  # 启用同步模式
    settings.fixed_delta_seconds = 0.05  # 固定步长（20FPS）
    world.apply_settings(settings)

    # 获取旁观者（用于视角跟随）
    spectator = world.get_spectator()

    # 获取地图的生成点（车辆生成的位置）
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("❌ 无可用生成点！")
        return

    # 生成主角车辆（hero）
    # 获取蓝图库
    bp_lib = world.get_blueprint_library()
    # 选择林肯MKZ 2020车型（如果不存在则选择第一个车辆蓝图）
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    if not vehicle_bp:
        vehicle_bp = bp_lib.filter('vehicle.*')[0]
    # 随机选择一个生成点
    spawn_point = random.choice(spawn_points)
    # 尝试生成车辆
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if not vehicle:
        print("❌ 车辆生成失败！")
        return

    # 生成RGB相机传感器
    camera_bp = bp_lib.find('sensor.camera.rgb')
    # 设置相机参数：图像宽度640，高度480，视场角90度
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '90')

    # 设置相机挂载位置：车辆前方1.2m，上方2.0m，俯仰角-5度（稍微向下看）
    camera_init_trans = carla.Transform(carla.Location(x=1.2, z=2.0), carla.Rotation(pitch=-5))
    # 将相机挂载到主角车辆上
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    # 创建图像队列（用于存储相机回调的图像，最大长度2）
    image_queue = queue.Queue(maxsize=2)
    # 注册相机回调函数
    camera.listen(lambda image: camera_callback(image, image_queue))

    # 清理现有NPC车辆和静态车辆
    clear_npc(world)
    clear_static_vehicle(world)

    # Part 2: 初始化追踪相关参数
    # 3D车辆的边索引（用于生成2D包围盒，此处未实际使用）
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
             [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

    # 获取相机参数（宽度、高度、视场角）
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    # 构建相机内参矩阵（前向和后向）
    K = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

    # 生成NPC车辆（目标：20辆）
    npc_count = 20
    spawned_npcs = 0  # 实际生成的NPC数量
    for i in range(npc_count):
        # 筛选4轮车辆的蓝图
        vehicle_bp_list = bp_lib.filter('vehicle')
        car_bp = [bp for bp in vehicle_bp_list if int(bp.get_attribute('number_of_wheels')) == 4]
        if not car_bp:
            continue
        # 随机选择生成点
        random_spawn = random.choice(spawn_points)
        # 避免生成在主角车辆附近（距离小于10m则跳过）
        if random_spawn.location.distance(vehicle.get_location()) < 10.0:
            continue
        # 尝试生成NPC车辆
        npc = world.try_spawn_actor(random.choice(car_bp), random_spawn)
        if npc:
            # 启用自动驾驶
            npc.set_autopilot(True)
            spawned_npcs += 1
    print(f"✅ 生成{spawned_npcs}辆NPC车辆")

    # 主角车辆启用自动驾驶
    vehicle.set_autopilot(True)

    # 初始化Deep SORT追踪器
    # 创建特征匹配度量（余弦距离，阈值0.2）
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
    # 创建追踪器（使用自定义的编码器）
    tracker = Tracker(metric)
    # 初始化边界框编码器（替换原有的mars-small128编码器）
    encoder = create_box_encoder("mars-small128.pb", batch_size=32)

    # 获取地图名称（提取最后一部分）
    map_name = world.get_map().name.split('/')[-1]

    # 主循环（车辆追踪和可视化）
    while True:
        try:
            # 推进仿真一步（同步模式必须调用）
            world.tick()

            # 移动旁观者视角：车辆后方4m，上方50m，俯视视角（yaw=-180，pitch=-90）
            transform = carla.Transform(
                vehicle.get_transform().transform(carla.Location(x=-4, z=50)),
                carla.Rotation(yaw=-180, pitch=-90)
            )
            spectator.set_transform(transform)

            # 获取相机图像（队列为空则跳过）
            if image_queue.empty():
                continue
            image = image_queue.get()

            # 图像预处理：BGRA→BGR（去除Alpha通道）→水平翻转（解决图像反向问题）
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            image = cv2.flip(image, 1)

            # 更新世界坐标到相机坐标的变换矩阵
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # 检测车辆：遍历所有车辆，生成2D边界框
            boxes = []
            for npc in world.get_actors().filter('*vehicle*'):
                # 跳过主角车辆
                if npc.id != vehicle.id:
                    # 获取车辆的包围盒
                    bb = npc.bounding_box
                    # 计算车辆与主角的距离（只处理50m内的车辆）
                    dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                    if dist < 50:
                        # 筛选主角车辆前方的车辆（点积>0表示同向）
                        forward_vec = vehicle.get_transform().get_forward_vector()
                        ray = npc.get_transform().location - vehicle.get_transform().location
                        if forward_vec.dot(ray) > 0:
                            # 获取车辆包围盒的所有3D顶点（世界坐标）
                            verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                            points_2d = []
                            for vert in verts:
                                # 判断顶点是否在相机前方（点积>0）
                                ray0 = vert - camera.get_transform().location
                                cam_forward_vec = camera.get_transform().get_forward_vector()
                                if (cam_forward_vec.dot(ray0) > 0):
                                    # 相机前方的点：使用前向内参矩阵投影
                                    p = get_image_point(vert, K, world_2_camera)
                                else:
                                    # 相机后方的点：使用后向内参矩阵投影
                                    p = get_image_point(vert, K_b, world_2_camera)
                                # 水平翻转x坐标（匹配图像翻转）
                                p = (image_w - p[0], p[1])
                                points_2d.append(p)

                            # 从3D顶点的2D投影生成2D边界框
                            x_min, x_max, y_min, y_max = get_2d_box_from_3d_edges(
                                points_2d, edges, image_h, image_w)

                            # 过滤小框（面积>100，宽度>20，避免噪声）
                            if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                                # 确保边界框在画布内
                                if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):
                                    # 存储边界框（x1, y1, x2, y2）
                                    boxes.append(np.array([x_min, y_min, x_max, y_max]))

            # 将边界框列表转换为numpy数组
            boxes = np.array(boxes)

            # 初始化检测结果列表
            detections = []
            if len(boxes) > 0:
                # 复制边界框，转换为Deep SORT格式（x1, y1, w, h）
                sort_boxes = boxes.copy()
                for i, box in enumerate(sort_boxes):
                    # 计算宽度和高度（替换x2, y2）
                    box[2] -= box[0]
                    box[3] -= box[1]
                    # 提取边界框的特征（128维）
                    feature = encoder(image, box.reshape(1, -1).copy())
                    # 创建Detection对象（边界框，置信度1.0，特征）
                    detections.append(Detection(box, 1.0, feature[0]))

            # 更新Deep SORT追踪器
            tracker.predict()  # 预测追踪框的位置
            tracker.update(detections)  # 根据检测结果更新追踪器

            # 提取追踪结果（确认的追踪框，更新时间<=1）
            bboxes = []
            ids = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                # 转换为tlbr格式（top, left, bottom, right）
                bbox = track.to_tlbr()
                bboxes.append(bbox)
                ids.append(track.track_id)

            # 转换为numpy数组，获取追踪车辆数量
            bboxes = np.array(bboxes)
            tracked_vehicle_count = len(bboxes)

            # 在图像上绘制边界框和追踪ID
            if len(bboxes) > 0:
                # 类别标签：全部设为2（car）
                labels = np.array([2] * len(bboxes))
                image = draw_bounding_boxes(image, bboxes, labels, COCO_CLASS_NAMES, ids)

            # 计算主角车辆的速度（m/s → km/h，乘以3.6）
            velocity = vehicle.get_velocity()
            speed_ms = math.hypot(velocity.x, velocity.y)  # 计算xy平面的速度大小
            speed_kmh = speed_ms * 3.6
            # 绘制仿真信息（地图名称、车速、追踪数量）
            image = draw_info_text(image, speed_kmh, tracked_vehicle_count, map_name)

            # 显示图像窗口
            cv2.imshow('2D Ground Truth Deep SORT (Fixed All Issues)', image)

            # 按下q键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt as e:
            # 捕获键盘中断（Ctrl+C），退出循环
            break
        except Exception as e:
            # 捕获其他异常，打印错误信息并继续循环
            print(f"⚠️ 运行错误：{e}")
            continue

    # 清理资源
    clear(world, camera)
    # 关闭同步模式
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("✅ 程序正常退出")

# 程序入口
if __name__ == '__main__':
    main()