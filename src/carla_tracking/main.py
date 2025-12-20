import argparse
import carla
import os
import queue
import random
import cv2
import numpy as np
import torch
from collections import deque
import math


# -------------------------- 内置 SORT 跟踪器（深度增强版） --------------------------
class KalmanFilter:
    def __init__(self):
        self.dt = 1.0
        self.x = np.zeros((4, 1))  # [x, y, vx, vy]
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.P = np.eye(4) * 1000
        # 优化过程噪声协方差，基于距离动态调整
        self.Q_base = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 5, 0],
                                [0, 0, 0, 5]])
        self.Q = self.Q_base.copy()
        self.R = np.eye(2) * 5

    def predict(self, distance=None):
        # 基于距离调整过程噪声：远距离目标运动预测不确定性更大
        if distance is not None:
            if distance > 50:
                self.Q = self.Q_base * 2.0
            elif distance > 30:
                self.Q = self.Q_base * 1.5
            else:
                self.Q = self.Q_base * 0.8

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:2]

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x[:2]


class Track:
    def __init__(self, box, track_id):
        self.id = track_id
        self.kf = KalmanFilter()
        self.x1, self.y1, self.x2, self.y2 = box
        self.center = np.array([[(self.x1 + self.x2) / 2], [(self.y1 + self.y2) / 2]])
        self.kf.x[:2] = self.center
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.hits = 1
        self.age = 0
        self.history = deque(maxlen=8)  # 增加历史帧数
        self.history.append(self.center)
        self.distance = None
        self.distance_history = deque(maxlen=5)  # 距离历史，用于平滑
        self.velocity = 0.0  # 基于距离的速度估计
        self.last_distance = None

    def predict(self):
        self.age += 1
        # 使用距离信息优化预测
        center = self.kf.predict(self.distance)

        # 基于距离调整平滑权重：远距离目标更平滑
        smooth_weight = 0.7 if (self.distance and self.distance > 30) else 0.4
        self.history.append(center)

        # 加权平均平滑
        weights = np.linspace(0.1, 1.0, len(self.history))
        weights = weights / weights.sum()
        smoothed_center = np.average(self.history, axis=0, weights=weights)

        self.x1 = smoothed_center[0, 0] - self.width / 2
        self.y1 = smoothed_center[1, 0] - self.height / 2
        self.x2 = self.x1 + self.width
        self.y2 = self.y1 + self.height

        # 速度估计（m/s）
        if self.last_distance and self.distance:
            self.velocity = abs(self.distance - self.last_distance) / self.kf.dt

        return [self.x1, self.y1, self.x2, self.y2]

    def update(self, box, distance=None):
        self.x1, self.y1, self.x2, self.y2 = box
        self.center = np.array([[(self.x1 + self.x2) / 2], [(self.y1 + self.y2) / 2]])
        self.kf.update(self.center)
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.hits += 1
        self.age = 0
        self.history.append(self.center)

        # 距离更新与平滑
        if distance is not None:
            self.last_distance = self.distance
            self.distance_history.append(distance)
            # 中位数滤波去除异常值
            self.distance = np.median(self.distance_history)

    def get_box(self):
        return [self.x1, self.y1, self.x2, self.y2]

    def get_distance(self):
        return self.distance if self.distance else 0.0


class Sort:
    def __init__(self, max_age=8, min_hits=3, iou_threshold=0.4):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1
        self.depths = []
        self.depth_thresholds = {
            'near': 15.0,  # 近距离
            'medium': 30.0,  # 中距离
            'far': 50.0  # 远距离
        }

    def set_depths(self, depths):
        """设置当前检测目标的距离列表，包含预处理"""
        # 深度值预处理：去除异常值
        processed_depths = []
        for d in depths:
            if 0.1 < d < 200:  # 过滤无效深度值
                processed_depths.append(d)
            else:
                processed_depths.append(50.0)  # 默认值
        self.depths = processed_depths

    def update(self, detections):
        if len(detections) == 0:
            for track in self.tracks:
                track.age += 1
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            return np.array([[t.x1, t.y1, t.x2, t.y2, t.id] for t in self.tracks if t.hits >= self.min_hits])

        if len(self.tracks) == 0:
            for i, det in enumerate(detections):
                track = Track(det[:4], self.next_id)
                if i < len(self.depths):
                    track.distance = self.depths[i]
                    track.distance_history.append(self.depths[i])
                self.tracks.append(track)
                self.next_id += 1
            return np.array([[t.x1, t.y1, t.x2, t.y2, t.id] for t in self.tracks])

        track_boxes = np.array([t.get_box() for t in self.tracks])
        iou_matrix = self._iou_batch(track_boxes, detections[:, :4])
        matches, unmatched_tracks, unmatched_dets = self._hungarian_algorithm(iou_matrix)

        # 优化匹配逻辑：基于距离动态调整IOU阈值
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det_distance = self.depths[det_idx] if det_idx < len(self.depths) else None

            # 动态IOU阈值：近距离目标要求更高的IOU
            dynamic_iou_thresh = self.iou_threshold
            if det_distance:
                if det_distance < self.depth_thresholds['near']:
                    dynamic_iou_thresh = 0.5
                elif det_distance > self.depth_thresholds['far']:
                    dynamic_iou_thresh = 0.3

            if iou_matrix[track_idx, det_idx] >= dynamic_iou_thresh:
                self.tracks[track_idx].update(detections[det_idx][:4], det_distance)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].age += 1

        # 基于距离和尺寸的过滤策略
        for det_idx in unmatched_dets:
            box = detections[det_idx][:4]
            width = box[2] - box[0]
            height = box[3] - box[1]
            det_distance = self.depths[det_idx] if det_idx < len(self.depths) else 50.0

            # 动态尺寸过滤：基于距离的透视校正
            # 远距离目标像素尺寸小，需要更小的阈值
            if det_distance < self.depth_thresholds['near']:
                min_size = 10
            elif det_distance < self.depth_thresholds['medium']:
                min_size = 5
            elif det_distance < self.depth_thresholds['far']:
                min_size = 3
            else:
                min_size = 2

            # 宽高比过滤：过滤异常形状
            aspect_ratio = width / height if height > 0 else 0
            if 0.2 < aspect_ratio < 5.0 and width > min_size and height > min_size:
                track = Track(box, self.next_id)
                track.distance = det_distance
                track.distance_history.append(det_distance)
                self.tracks.append(track)
                self.next_id += 1

        # 过滤过旧的跟踪器，基于距离调整最大年龄
        filtered_tracks = []
        for track in self.tracks:
            if track.distance and track.distance > self.depth_thresholds['far']:
                # 远距离目标保留时间更长
                if track.age <= self.max_age + 2:
                    filtered_tracks.append(track)
            else:
                if track.age <= self.max_age:
                    filtered_tracks.append(track)

        self.tracks = filtered_tracks
        return np.array([[t.x1, t.y1, t.x2, t.y2, t.id] for t in self.tracks if t.hits >= self.min_hits])

    def _iou_batch(self, b1, b2):
        b1_x1, b1_y1, b1_x2, b1_y2 = b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

        inter_x1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
        inter_y1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
        inter_x2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
        inter_y2 = np.minimum(b1_y2[:, None], b2_y2[None, :])

        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        base_iou = inter_area / (b1_area[:, None] + b2_area[None, :] - inter_area + 1e-6)

        # 增强的距离加权IOU
        if len(self.depths) > 0 and b2.shape[0] == len(self.depths):
            # 基础距离权重
            distance_weights = np.array([
                1.5 if d < self.depth_thresholds['near'] else
                1.2 if d < self.depth_thresholds['medium'] else
                0.9 if d < self.depth_thresholds['far'] else 0.7
                for d in self.depths
            ])

            # 增加尺寸一致性权重：尺寸变化小的匹配权重更高
            size_weights = np.ones_like(distance_weights)
            track_sizes = np.sqrt((b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1]))[:, None]
            det_sizes = np.sqrt((b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1]))[None, :]
            size_ratio = np.minimum(track_sizes / det_sizes, det_sizes / track_sizes)
            size_weights = np.mean(size_ratio, axis=0)

            # 组合权重
            combined_weights = distance_weights * size_weights
            return base_iou * combined_weights[None, :]

        return base_iou

    def _hungarian_algorithm(self, cost_matrix):
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        matches = np.array(list(zip(row_ind, col_ind)))
        unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in matches[:, 0]]
        unmatched_dets = [i for i in range(cost_matrix.shape[1]) if i not in matches[:, 1]]
        return matches, unmatched_tracks, unmatched_dets


# -------------------------- YOLOv5 检测模型 --------------------------
from ultralytics import YOLO


# -------------------------- 核心工具函数（深度增强版） --------------------------
def draw_bounding_boxes(image, boxes, labels, class_names, track_ids=None, probs=None, distances=None, velocities=None):
    """增强的绘制函数，包含距离、速度和深度可视化"""
    # 创建深度热力图（可选）
    depth_overlay = image.copy()

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        # 严格的边界检查
        x1 = max(0, min(x1, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))

        # 基于距离的颜色渐变：近红远绿
        color = (0, 255, 0)  # 默认绿色
        if distances and i < len(distances) and distances[i]:
            distance = distances[i]
            # 距离颜色映射 (0-50m: 红->黄->绿)
            if distance < 15:
                r, g = 255, int(255 * (distance / 15))
                color = (0, g, r)
            elif distance < 30:
                r, g = int(255 * (1 - (distance - 15) / 15)), 255
                color = (0, g, r)
            elif distance < 50:
                b, g = int(255 * ((distance - 30) / 20)), 255
                color = (b, g, 0)

        # 绘制边框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 绘制深度相关的填充（半透明）
        if distances and i < len(distances) and distances[i]:
            alpha = 0.2
            cv2.rectangle(depth_overlay, (x1, y1), (x2, y2), color, -1)
            image = cv2.addWeighted(depth_overlay, alpha, image, 1 - alpha, 0)

        # 构建文本信息
        conf_text = f"{probs[i]:.2f}" if (probs is not None and i < len(probs)) else ""
        label_text = f"{class_names[label]} {conf_text}".strip()

        if track_ids and i < len(track_ids):
            label_text += f"_ID:{track_ids[i]}"
        if distances and i < len(distances) and distances[i]:
            label_text += f"_Dist:{distances[i]:.1f}m"
        if velocities and i < len(velocities) and velocities[i]:
            label_text += f"_Speed:{velocities[i]:.1f}m/s"

        # 绘制文本背景
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_bg_x1 = x1
        text_bg_y1 = y1 - text_size[1] - 5
        text_bg_x2 = x1 + text_size[0]
        text_bg_y2 = y1
        # 确保文本背景在图像内
        text_bg_x1 = max(0, text_bg_x1)
        text_bg_y1 = max(0, text_bg_y1)
        text_bg_x2 = min(image.shape[1] - 1, text_bg_x2)

        cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
        cv2.putText(image, label_text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image


def preprocess_depth_image(depth_image):
    """深度图像预处理：去噪、校准、归一化"""
    # 1. 去除异常值
    depth_image = np.clip(depth_image, 0.1, 200.0)

    # 2. 中值滤波去噪
    depth_image = cv2.medianBlur(depth_image.astype(np.float32), 3)

    # 3. 伽马校正，增强近距离细节
    gamma = 0.7
    depth_image = np.power(depth_image / np.max(depth_image), gamma) * np.max(depth_image)

    return depth_image


def build_projection_matrix(w, h, fov):
    """构建相机投影矩阵"""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    return np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])


def clear_npc(world):
    """清理所有NPC车辆"""
    if not world:
        return
    for actor in world.get_actors().filter('vehicle.*'):
        try:
            actor.destroy()
        except:
            pass


def clear_static_vehicle(world):
    """清理静态车辆"""
    if not world:
        return
    for actor in world.get_actors().filter('static.vehicle.*'):
        try:
            actor.destroy()
        except:
            pass


def clear(world, camera, depth_camera=None):
    """清理所有资源"""
    if camera:
        try:
            camera.destroy()
        except:
            pass
    if depth_camera:
        try:
            depth_camera.destroy()
        except:
            pass
    clear_npc(world)
    clear_static_vehicle(world)


# -------------------------- CARLA 相关核心函数（深度增强版） --------------------------
def camera_callback(image, rgb_image_queue):
    """相机回调函数"""
    rgb_image_queue.put(np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)))


def depth_camera_callback(image, depth_queue):
    """增强的深度相机回调：预处理深度数据"""
    # 原始深度数据处理
    depth_data = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    # CARLA深度值转换：根据不同版本调整
    # 方法1：新格式 (depth = (R + G * 256 + B * 256^2) / (256^3 - 1) * 1000)
    depth_channel = (depth_data[:, :, 2] + depth_data[:, :, 1] * 256 + depth_data[:, :, 0] * 256 ** 2)
    depth_in_meters = depth_channel / (256 ** 3 - 1) * 1000.0

    # 预处理
    depth_in_meters = preprocess_depth_image(depth_in_meters)

    depth_queue.put(depth_in_meters)


def setup_carla_client(host='localhost', port=2000):
    """连接CARLA服务器并设置同步模式"""
    client = carla.Client(host, port)
    client.set_timeout(15.0)  # 增加超时时间
    world = client.get_world()

    # 优化同步设置
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)

    return world, client


def spawn_ego_vehicle(world):
    """生成主车辆 - 优化版：选择交通密度高的位置"""
    if not world:
        return None

    bp_lib = world.get_blueprint_library()

    # 优先使用林肯MKZ，如果没有则选择其他小型车辆
    try:
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    except:
        # 选择小型车辆，更容易看到其他车
        small_vehicles = [
            'vehicle.audi.a2',
            'vehicle.audi.tt',
            'vehicle.bmw.grandtourer',
            'vehicle.dodge.charger_police',
            'vehicle.ford.mustang',
            'vehicle.mercedes.coupe',
            'vehicle.mini.cooperst',
            'vehicle.nissan.micra',
            'vehicle.nissan.patrol',
            'vehicle.seat.leon',
            'vehicle.toyota.prius',
            'vehicle.volkswagen.t2'
        ]

        available_vehicles = []
        for vehicle_name in small_vehicles:
            try:
                bp = bp_lib.find(vehicle_name)
                if bp:
                    available_vehicles.append(bp)
            except:
                pass

        if available_vehicles:
            vehicle_bp = random.choice(available_vehicles)
        else:
            vehicle_bp = random.choice(
                [bp for bp in bp_lib.filter('vehicle') if int(bp.get_attribute('number_of_wheels')) == 4])

    # 获取所有生成点
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("警告：没有可用的车辆生成点！")
        return None

    # 分析道路连接性，选择交通流量大的点
    best_spawn_points = []

    # 获取地图拓扑
    topology = world.get_map().get_topology()

    # 分析每个生成点的连接性
    for spawn_point in spawn_points:
        nearby_junctions = 0
        nearby_spawns = 0

        for other_point in spawn_points:
            if other_point != spawn_point:
                dist = math.hypot(
                    other_point.location.x - spawn_point.location.x,
                    other_point.location.y - spawn_point.location.y
                )
                if dist < 50.0:  # 50米内有其他生成点
                    nearby_spawns += 1

        # 优先选择附近有其他生成点的位置
        if nearby_spawns >= 2:
            best_spawn_points.append(spawn_point)

    # 如果没有找到最佳点，使用所有生成点
    if not best_spawn_points:
        best_spawn_points = spawn_points

    # 随机打乱最佳点
    random.shuffle(best_spawn_points)

    # 尝试生成车辆
    vehicle = None
    for spawn_point in best_spawn_points:
        # 检查生成点周围是否有障碍物
        nearby_actors = world.get_actors().filter('vehicle.*')
        too_close = False

        for actor in nearby_actors:
            dist = math.hypot(
                actor.get_location().x - spawn_point.location.x,
                actor.get_location().y - spawn_point.location.y
            )
            if dist < 3.0:  # 减小安全距离到3米
                too_close = True
                break

        if not too_close:
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle:
                print(f"主车辆生成成功！位置：{spawn_point.location}")
                break

    if vehicle:
        vehicle.set_autopilot(True)
        # 设置适中的速度
        try:
            vehicle.set_velocity(carla.Vector3D(30.0, 0, 0))
        except:
            pass
        # 禁用物理碰撞，避免自车被撞停
        vehicle.set_simulate_physics(False)
        print(f"主车辆生成成功！车型：{vehicle_bp.id}")
    else:
        # 最后尝试强制生成
        for spawn_point in random.sample(spawn_points, min(5, len(spawn_points))):
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                vehicle.set_simulate_physics(False)
                print(f"强制生成主车辆成功！位置：{spawn_point.location}")
                break

    if not vehicle:
        print("警告：无法生成主车辆！尝试手动生成...")
        # 尝试生成在原点
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True)
            vehicle.set_simulate_physics(False)
            print(f"在原点生成主车辆成功！")

    return vehicle


def spawn_camera(world, vehicle):
    """生成RGB相机传感器"""
    if not world or not vehicle:
        return None, None, None
    bp_lib = world.get_blueprint_library()
    camera_bp = bp_lib.find('sensor.camera.rgb')

    # 优化相机参数
    camera_bp.set_attribute('image_size_x', '800')  # 提高分辨率
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '100')  # 增加FOV，扩大视野范围
    camera_bp.set_attribute('sensor_tick', '0.05')
    camera_bp.set_attribute('gamma', '2.2')  # 伽马校正

    # 优化相机位置：前向视野更开阔
    camera_init_trans = carla.Transform(carla.Location(x=1.5, z=1.8), carla.Rotation(pitch=-5, yaw=0))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    image_queue = queue.Queue(maxsize=3)  # 限制队列大小，避免内存泄漏
    camera.listen(lambda image: camera_callback(image, image_queue))

    return camera, image_queue, camera_bp


def spawn_depth_camera(world, vehicle):
    """生成深度相机（与RGB相机精确同步）"""
    if not world or not vehicle:
        return None, None
    bp_lib = world.get_blueprint_library()
    depth_bp = bp_lib.find('sensor.camera.depth')

    # 与RGB相机参数完全匹配
    depth_bp.set_attribute('image_size_x', '800')
    depth_bp.set_attribute('image_size_y', '600')
    depth_bp.set_attribute('fov', '100')
    depth_bp.set_attribute('sensor_tick', '0.05')

    # 相同的安装位置和角度
    camera_init_trans = carla.Transform(carla.Location(x=1.5, z=1.8), carla.Rotation(pitch=-5, yaw=0))
    depth_camera = world.spawn_actor(depth_bp, camera_init_trans, attach_to=vehicle)

    depth_queue = queue.Queue(maxsize=3)
    depth_camera.listen(lambda image: depth_camera_callback(image, depth_queue))

    return depth_camera, depth_queue


def get_target_distance(depth_image, box, use_median=True):
    """增强的目标距离计算：多种采样策略"""
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth_image.shape[1] - 1, x2), min(depth_image.shape[0] - 1, y2)

    # 空框检查
    if x1 >= x2 or y1 >= y2:
        return 50.0

    # 提取ROI
    depth_roi = depth_image[y1:y2, x1:x2]
    valid_depths = depth_roi[depth_roi > 0.1]

    if len(valid_depths) == 0:
        return 50.0

    # 多种距离计算策略
    if use_median:
        # 中位数：抗异常值
        return np.median(valid_depths)
    else:
        # 中心加权平均：更准确
        h, w = depth_roi.shape
        cx, cy = w // 2, h // 2
        # 生成距离权重（中心权重高）
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        weights = 1 - (dist_from_center / max_dist)
        weights = weights * (depth_roi > 0.1)

        if np.sum(weights) > 0:
            return np.sum(depth_roi * weights) / np.sum(weights)
        else:
            return np.mean(valid_depths)


def spawn_npcs(world, count=25, ego_vehicle=None):
    """优化的NPC生成：确保主车辆周围有足够车辆"""
    if not world:
        return

    print(f"开始生成 {count} 辆NPC车辆...")

    bp_lib = world.get_blueprint_library()

    # 使用多种小型车辆增加多样性
    small_vehicle_names = [
        'vehicle.audi.a2',
        'vehicle.audi.tt',
        'vehicle.bmw.grandtourer',
        'vehicle.dodge.charger_police',
        'vehicle.ford.mustang',
        'vehicle.mercedes.coupe',
        'vehicle.mini.cooperst',
        'vehicle.nissan.micra',
        'vehicle.nissan.patrol',
        'vehicle.seat.leon',
        'vehicle.toyota.prius',
        'vehicle.volkswagen.t2',
        'vehicle.carlamotors.carlacola',
        'vehicle.citroen.c3',
        'vehicle.harley-davidson.low_rider',
        'vehicle.jeep.wrangler_rubicon',
        'vehicle.yamaha.yzf'
    ]

    # 筛选可用的小型车辆蓝图
    small_vehicle_bps = []
    for name in small_vehicle_names:
        try:
            bp = bp_lib.find(name)
            if bp:
                small_vehicle_bps.append(bp)
        except:
            continue

    # 如果没有找到小型车辆，使用所有四轮车辆
    if not small_vehicle_bps:
        small_vehicle_bps = [bp for bp in bp_lib.filter('vehicle') if int(bp.get_attribute('number_of_wheels')) == 4]

    if not small_vehicle_bps:
        print("警告：没有可用的NPC车辆蓝图！")
        return

    # 获取所有生成点
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("警告：没有可用的生成点！")
        return

    # 获取主车辆位置
    ego_location = None
    if ego_vehicle:
        ego_location = ego_vehicle.get_location()

    # 排序生成点：如果主车辆存在，优先在主车辆附近生成
    if ego_location:
        # 计算每个生成点到主车辆的距离
        spawn_points_with_dist = []
        for spawn_point in spawn_points:
            dist = math.hypot(
                spawn_point.location.x - ego_location.x,
                spawn_point.location.y - ego_location.y
            )
            spawn_points_with_dist.append((spawn_point, dist))

        # 按距离排序（近的在前）
        spawn_points_with_dist.sort(key=lambda x: x[1])
        sorted_spawn_points = [sp[0] for sp in spawn_points_with_dist]
    else:
        # 随机打乱生成点
        random.shuffle(spawn_points)
        sorted_spawn_points = spawn_points

    spawned_count = 0
    npc_vehicles = []

    # 第一轮：在主车辆附近生成（50米内）
    if ego_location:
        for spawn_point in sorted_spawn_points[:30]:  # 检查前30个最近的生成点
            if spawned_count >= count:
                break

            # 计算到主车辆的距离
            dist_to_ego = math.hypot(
                spawn_point.location.x - ego_location.x,
                spawn_point.location.y - ego_location.y
            )

            # 在主车辆50米内生成
            if dist_to_ego < 50.0 and dist_to_ego > 10.0:  # 10-50米范围内
                # 检查与其他NPC的距离
                too_close = False
                for npc in npc_vehicles:
                    npc_loc = npc.get_location()
                    dist = math.hypot(
                        npc_loc.x - spawn_point.location.x,
                        npc_loc.y - spawn_point.location.y
                    )
                    if dist < 8.0:  # 减少最小距离到8米
                        too_close = True
                        break

                if not too_close:
                    try:
                        # 随机选择车辆类型
                        vehicle_bp = random.choice(small_vehicle_bps)

                        # 设置随机的车辆颜色
                        if vehicle_bp.has_attribute('color'):
                            colors = vehicle_bp.get_attribute('color').recommended_values
                            if colors:
                                vehicle_bp.set_attribute('color', random.choice(colors))

                        npc = world.try_spawn_actor(vehicle_bp, spawn_point)

                        if npc:
                            npc.set_autopilot(True)

                            # 设置不同的速度，增加交通真实感
                            try:
                                speed = random.uniform(20.0, 40.0)
                                # 通过设置目标速度来控制车速
                                traffic_manager = world.get_trafficmanager()
                                traffic_manager.distance_to_leading_vehicle(npc, 5.0)
                                traffic_manager.vehicle_percentage_speed_difference(npc, random.uniform(-30, 10))
                            except:
                                pass

                            npc_vehicles.append(npc)
                            spawned_count += 1
                            print(f"生成NPC {spawned_count}/{count} - 距离主车辆: {dist_to_ego:.1f}米")
                    except Exception as e:
                        print(f"生成NPC失败: {str(e)}")

    # 第二轮：如果数量不够，在更远的地方生成
    if spawned_count < count:
        print(f"第一轮生成 {spawned_count} 辆，开始第二轮生成...")

        for spawn_point in sorted_spawn_points:
            if spawned_count >= count:
                break

            # 跳过已经检查过的生成点
            skip = False
            for npc in npc_vehicles:
                npc_loc = npc.get_location()
                dist = math.hypot(
                    npc_loc.x - spawn_point.location.x,
                    npc_loc.y - spawn_point.location.y
                )
                if dist < 8.0:
                    skip = True
                    break

            if skip:
                continue

            # 如果太靠近主车辆，跳过
            if ego_location:
                dist_to_ego = math.hypot(
                    spawn_point.location.x - ego_location.x,
                    spawn_point.location.y - ego_location.y
                )
                if dist_to_ego < 10.0:
                    continue

            try:
                vehicle_bp = random.choice(small_vehicle_bps)

                if vehicle_bp.has_attribute('color'):
                    colors = vehicle_bp.get_attribute('color').recommended_values
                    if colors:
                        vehicle_bp.set_attribute('color', random.choice(colors))

                npc = world.try_spawn_actor(vehicle_bp, spawn_point)

                if npc:
                    npc.set_autopilot(True)

                    try:
                        traffic_manager = world.get_trafficmanager()
                        traffic_manager.distance_to_leading_vehicle(npc, 5.0)
                        traffic_manager.vehicle_percentage_speed_difference(npc, random.uniform(-30, 10))
                    except:
                        pass

                    npc_vehicles.append(npc)
                    spawned_count += 1
                    print(f"生成NPC {spawned_count}/{count}")
            except Exception as e:
                print(f"生成NPC失败: {str(e)}")

    print(f"成功生成 {spawned_count} 辆NPC车辆")

    # 启动交通管理器，增加交通密度和交互
    try:
        traffic_manager = world.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.global_percentage_speed_difference(0.0)
        traffic_manager.set_synchronous_mode(True)
        print("交通管理器配置完成")
    except:
        pass


def load_detection_model(model_type):
    """加载YOLOv5检测模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 更新模型路径，添加yolov5mu和yolov5su
    model_paths = {
        'yolov5s': r"D:\yolo\yolov5s.pt",
        'yolov5su': r"D:\yolo\yolov5su.pt",
        'yolov5m': r"D:\yolo\yolov5m.pt",
        'yolov5mu': r"D:\yolo\yolov5mu.pt",
        'yolov5x': r"D:\yolo\yolov5x.pt"
    }

    if model_type not in model_paths:
        # 尝试使用默认模型
        if 'su' in model_type.lower():
            model_type = 'yolov5su'
        elif 'mu' in model_type.lower():
            model_type = 'yolov5mu'
        else:
            model_type = 'yolov5m'

    # 检查模型文件是否存在
    if not os.path.exists(model_paths.get(model_type, '')):
        print(f"警告：模型文件 {model_paths.get(model_type, '')} 不存在！")
        # 尝试使用默认模型
        fallback_models = ['yolov5s', 'yolov5m', 'yolov5x']
        for fallback in fallback_models:
            if os.path.exists(model_paths[fallback]):
                print(f"使用备用模型：{fallback}")
                model_type = fallback
                break

    if model_type not in model_paths or not os.path.exists(model_paths[model_type]):
        raise FileNotFoundError(f"无法找到模型文件：{model_paths.get(model_type, '未知')}")

    # 加载模型并优化
    model = YOLO(model_paths[model_type])
    model.to(device)

    # 模型优化设置
    if device == 'cuda':
        model.half()  # 使用半精度提高速度

    # 预热模型
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    if device == 'cuda':
        dummy_input = dummy_input.half()
    _ = model(dummy_input)

    print(f"检测模型加载成功（设备：{device}，模型：{model_type}，路径：{model_paths[model_type]}）")
    return model, model.names


def setup_tracker(tracker_type):
    """初始化跟踪器"""
    if tracker_type == 'sort':
        # 优化跟踪器参数
        return Sort(max_age=8, min_hits=2, iou_threshold=0.4), None
    else:
        raise ValueError(f"不支持的跟踪器类型：{tracker_type}")


# -------------------------- 主函数（深度增强版） --------------------------
def main():
    parser = argparse.ArgumentParser(description='CARLA 目标检测与跟踪（深度增强版）')
    parser.add_argument('--model', type=str, default='yolov5mu',
                        choices=['yolov5s', 'yolov5su', 'yolov5m', 'yolov5mu', 'yolov5x'],
                        help='检测模型（yolov5mu为中型优化版）')
    parser.add_argument('--tracker', type=str, default='sort', choices=['sort'], help='跟踪器（仅保留稳定的SORT）')
    parser.add_argument('--host', type=str, default='localhost', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='检测置信度阈值')  # 降低阈值检测更多车辆
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IOU阈值')
    parser.add_argument('--use-depth', action='store_true', default=True, help='启用深度相机辅助跟踪（默认启用）')
    parser.add_argument('--show-depth', action='store_true', help='显示深度图像窗口')
    parser.add_argument('--npc-count', type=int, default=30, help='NPC车辆数量（增加到30辆）')  # 增加数量
    args = parser.parse_args()

    # 初始化变量
    world = None
    camera = None
    depth_camera = None
    vehicle = None
    image_queue = None
    depth_queue = None
    detection_history = deque(maxlen=5)  # 增加历史帧数
    frame_count = 0
    fps_history = deque(maxlen=30)

    # 跟踪统计
    detection_stats = {
        'total_detections': 0,
        'total_frames': 0,
        'max_vehicles_per_frame': 0,
        'no_detection_streak': 0
    }

    try:
        # 1. 初始化设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"当前使用设备：{device}")
        print(f"CUDA可用: {torch.cuda.is_available()}")

        if device == 'cuda':
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")

        # 2. 连接CARLA服务器
        print(f"正在连接CARLA服务器 {args.host}:{args.port}...")
        world, client = setup_carla_client(args.host, args.port)
        spectator = world.get_spectator()
        print("CARLA服务器连接成功！")
        print(f"地图: {world.get_map().name}")

        # 3. 清理环境
        print("清理环境...")
        clear_npc(world)
        clear_static_vehicle(world)
        print("环境清理完成！")

        # 4. 生成主车辆
        print("生成主车辆...")
        vehicle = spawn_ego_vehicle(world)
        if not vehicle:
            print("无法生成主车辆，程序退出！")
            return

        # 获取主车辆位置信息
        ego_location = vehicle.get_location()
        print(f"主车辆位置: X={ego_location.x:.1f}, Y={ego_location.y:.1f}, Z={ego_location.z:.1f}")

        # 5. 生成相机传感器
        print("生成相机传感器...")
        camera, image_queue, camera_bp = spawn_camera(world, vehicle)
        depth_camera = None
        depth_queue = None

        if args.use_depth:
            depth_camera, depth_queue = spawn_depth_camera(world, vehicle)
            print("深度相机传感器生成成功！")

        if not camera:
            print("无法生成相机，程序退出！")
            return
        print("RGB相机传感器生成成功！")

        # 6. 生成NPC车辆 - 优化版：在主车辆周围生成
        print(f"生成 {args.npc_count} 辆NPC车辆...")
        spawn_npcs(world, count=args.npc_count, ego_vehicle=vehicle)

        # 等待NPC车辆稳定
        print("等待NPC车辆初始化...")
        for _ in range(5):
            world.tick()

        # 7. 加载检测模型和跟踪器
        print("加载检测模型和跟踪器...")
        model, class_names = load_detection_model(args.model)
        tracker, _ = setup_tracker(args.tracker)

        # 8. 主循环
        print("\n开始目标检测与跟踪（按 'q' 键退出程序，按 'r' 重新生成NPC）")
        print("=" * 50)
        import time

        # FPS控制
        frame_time_target = 1.0 / 20  # 目标20FPS

        while True:
            start_time = time.time()
            frame_count += 1
            detection_stats['total_frames'] += 1

            # 同步CARLA世界
            world.tick()

            # 移动视角跟随主车辆
            ego_transform = vehicle.get_transform()
            spectator_transform = carla.Transform(
                ego_transform.transform(carla.Location(x=-10, z=12)),  # 更远的视角，看到更多车辆
                carla.Rotation(yaw=ego_transform.rotation.yaw - 180, pitch=-30)
            )
            spectator.set_transform(spectator_transform)

            # 获取图像
            if image_queue.empty():
                time.sleep(0.001)
                continue

            origin_image = image_queue.get()
            image = cv2.cvtColor(origin_image, cv2.COLOR_BGRA2RGB)
            height, width, _ = image.shape

            # 获取深度图像
            depth_image = None
            depths = []

            if args.use_depth and not depth_queue.empty():
                depth_image = depth_queue.get()

                # 显示深度图像
                if args.show_depth:
                    # 深度图像可视化
                    depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                    depth_vis = depth_vis.astype(np.uint8)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    cv2.imshow('Depth Image', depth_vis)

            # 目标检测 - 优化推理参数
            try:
                results = model(image, conf=args.conf_thres, iou=args.iou_thres,
                                device=device, imgsz=640, verbose=False)
            except Exception as e:
                print(f"检测模型推理出错: {e}")
                results = []

            boxes, labels, probs = [], [], []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    # 只保留车辆相关类别 (2: car, 3: motorcycle, 5: bus, 7: truck)
                    if cls in [2, 3, 5, 7]:
                        box_width = x2 - x1
                        box_height = y2 - y1

                        # 动态尺寸过滤 - 放宽限制
                        min_size = 6  # 降低最小尺寸
                        if args.use_depth and depth_image is not None:
                            rough_distance = get_target_distance(depth_image, [x1, y1, x2, y2])
                            if rough_distance > 30:
                                min_size = 4
                            elif rough_distance > 50:
                                min_size = 2
                            else:
                                min_size = 8

                        # 宽高比检查
                        aspect_ratio = box_width / max(box_height, 1)

                        if (box_width > min_size and box_height > min_size and
                                0.3 < aspect_ratio < 3.0):  # 放宽宽高比限制
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls)
                            probs.append(conf)

                            # 计算目标距离
                            if args.use_depth and depth_image is not None:
                                dist = get_target_distance(depth_image, [x1, y1, x2, y2], use_median=True)
                                depths.append(dist)

            # 更新检测统计
            detection_stats['total_detections'] += len(boxes)
            detection_stats['max_vehicles_per_frame'] = max(
                detection_stats['max_vehicles_per_frame'], len(boxes)
            )

            if len(boxes) == 0:
                detection_stats['no_detection_streak'] += 1
            else:
                detection_stats['no_detection_streak'] = 0

            # 转换为numpy数组
            boxes = np.array(boxes) if boxes else np.array([])
            labels = np.array(labels) if labels else np.array([])
            probs = np.array(probs) if probs else np.array([])

            # 多帧投票优化
            detection_history.append((boxes, labels, probs))

            if len(detection_history) >= 3:
                combined_boxes = []
                combined_labels = []
                combined_probs = []
                combined_depths = []

                for i in range(len(boxes)):
                    current_box = boxes[i]
                    current_label = labels[i]
                    count = 1

                    # 检查历史帧
                    for prev_boxes, prev_labels, _ in list(detection_history)[:-1]:
                        if len(prev_boxes) == 0:
                            continue
                        ious = Sort()._iou_batch(np.array([current_box]), prev_boxes)[0]
                        max_iou_idx = np.argmax(ious)
                        if ious[max_iou_idx] > 0.3 and prev_labels[max_iou_idx] == current_label:
                            count += 1

                    if count >= 2:  # 至少两帧检测到
                        combined_boxes.append(current_box)
                        combined_labels.append(current_label)
                        combined_probs.append(probs[i])
                        if args.use_depth and i < len(depths):
                            combined_depths.append(depths[i])

                boxes = np.array(combined_boxes) if combined_boxes else np.array([])
                labels = np.array(combined_labels) if combined_labels else np.array([])
                probs = np.array(combined_probs) if combined_probs else np.array([])
                depths = combined_depths

            # 目标跟踪
            track_distances = []
            track_velocities = []

            if args.tracker == 'sort' and len(boxes) > 0:
                dets = np.hstack([boxes, probs.reshape(-1, 1)]) if len(probs) > 0 else boxes

                if args.use_depth and len(depths) > 0:
                    tracker.set_depths(depths)

                track_results = tracker.update(dets)

                track_boxes = []
                track_ids = []

                for track in track_results:
                    x1, y1, x2, y2, track_id = track
                    track_boxes.append([x1, y1, x2, y2])
                    track_ids.append(int(track_id))

                    # 获取距离和速度
                    track_obj = next((t for t in tracker.tracks if t.id == track_id), None)
                    if track_obj:
                        track_distances.append(track_obj.get_distance())
                        track_velocities.append(track_obj.velocity)
                    else:
                        track_distances.append(None)
                        track_velocities.append(None)

                # 绘制跟踪结果
                if track_boxes:
                    image = draw_bounding_boxes(
                        image, track_boxes,
                        labels=[2] * len(track_boxes),  # 统一使用car类别
                        class_names=class_names,
                        track_ids=track_ids,
                        probs=[0.9] * len(track_boxes),
                        distances=track_distances,
                        velocities=track_velocities
                    )

                    # 显示检测到的车辆数量
                    cv2.putText(image, f'Vehicles: {len(track_boxes)}', (width - 200, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif len(boxes) > 0:
                # 无跟踪时绘制检测结果
                image = draw_bounding_boxes(
                    image, boxes, labels, class_names,
                    probs=probs,
                    distances=depths if args.use_depth else None
                )

                cv2.putText(image, f'Detections: {len(boxes)}', (width - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 计算FPS
            fps = 1.0 / (time.time() - start_time)
            fps_history.append(fps)
            avg_fps = np.mean(fps_history) if fps_history else fps

            # 显示信息
            info = [
                f"FPS: {avg_fps:.1f}",
                f"Frame: {frame_count}",
                f"Tracks: {len(tracker.tracks)}",
                f"Detections: {len(boxes)}",
                f"Model: {args.model}",
                "Press 'q' to quit"
            ]

            y_pos = 30
            for line in info:
                cv2.putText(image, line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_pos += 25

            # 显示结果
            window_name = f'CARLA {args.model} + {args.tracker} (Depth Enhanced)'
            cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # 每30帧打印一次统计信息
            if frame_count % 30 == 0:
                print(
                    f"[Frame {frame_count}] FPS: {avg_fps:.1f}, Detections: {len(boxes)}, Tracks: {len(tracker.tracks)}")

            # 退出检查
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户触发退出程序...")
                break
            elif key == ord('r'):
                # 重新生成NPC（调试功能）
                print("重新生成NPC车辆...")
                clear_npc(world)
                spawn_npcs(world, count=args.npc_count, ego_vehicle=vehicle)
                detection_stats['no_detection_streak'] = 0
                print("NPC重新生成完成")
            elif key == ord('d'):
                # 切换深度显示
                args.show_depth = not args.show_depth
                print(f"深度显示: {'开启' if args.show_depth else '关闭'}")

            # 长时间无检测时自动补充NPC
            if detection_stats['no_detection_streak'] > 50 and frame_count > 100:
                print("长时间无检测，自动补充NPC车辆...")
                # 在主车辆附近补充一些NPC
                spawn_points = world.get_map().get_spawn_points()
                if spawn_points:
                    bp_lib = world.get_blueprint_library()
                    vehicle_bps = [bp for bp in bp_lib.filter('vehicle') if
                                   int(bp.get_attribute('number_of_wheels')) == 4]

                    added = 0
                    for spawn_point in spawn_points[:20]:
                        if added >= 5:
                            break
                        distance = math.hypot(
                            spawn_point.location.x - ego_location.x,
                            spawn_point.location.y - ego_location.y
                        )
                        if 20.0 < distance < 80.0:
                            try:
                                npc = world.try_spawn_actor(random.choice(vehicle_bps), spawn_point)
                                if npc:
                                    npc.set_autopilot(True)
                                    added += 1
                                    print(f"补充生成NPC车辆 {added}/5")
                            except:
                                pass
                detection_stats['no_detection_streak'] = 0

            # FPS限制
            elapsed = time.time() - start_time
            if elapsed < frame_time_target:
                time.sleep(frame_time_target - elapsed)

    except KeyboardInterrupt:
        print("\n用户中断程序...")
    except Exception as e:
        import traceback
        print(f"程序运行出错：{str(e)}")
        traceback.print_exc()
        print("\n报错提示：")
        print("1. 确保CARLA服务器已启动")
        print("2. 确保carla包版本与服务器一致")
        print("3. 确保已安装所有依赖")
        print("4. 确保YOLO模型路径正确")
        print("5. 检查是否有其他程序占用端口")
    finally:
        # 清理资源
        print("\n正在清理资源...")
        clear(world, camera, depth_camera)

        if world:
            # 恢复CARLA设置
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.substepping = False
            world.apply_settings(settings)

        # 关闭窗口
        cv2.destroyAllWindows()

        # 打印最终统计
        print("\n" + "=" * 50)
        print("程序运行统计：")
        print(f"总帧数: {frame_count}")
        print(f"平均FPS: {np.mean(fps_history) if fps_history else 0:.1f}")
        print(f"总检测次数: {detection_stats['total_detections']}")
        print(f"平均每帧检测: {detection_stats['total_detections'] / max(1, detection_stats['total_frames']):.1f}")
        print(f"最大单帧车辆数: {detection_stats['max_vehicles_per_frame']}")
        print("=" * 50)
        print("资源清理完成，程序正常退出！")


if __name__ == "__main__":
    main()