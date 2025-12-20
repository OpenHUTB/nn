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
    """生成主车辆"""
    if not world:
        return None
    bp_lib = world.get_blueprint_library()
    try:
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    except:
        vehicle_bp = random.choice(
            [bp for bp in bp_lib.filter('vehicle') if int(bp.get_attribute('number_of_wheels')) == 4])

    # 选择安全的生成点
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("警告：没有可用的车辆生成点！")
        return None

    # 检查生成点周围是否有障碍物
    vehicle = None
    for spawn_point in spawn_points:
        # 检查生成点周围2米内是否有其他车辆
        nearby_actors = world.get_actors().filter('vehicle.*')
        too_close = False
        for actor in nearby_actors:
            dist = math.hypot(
                actor.get_location().x - spawn_point.location.x,
                actor.get_location().y - spawn_point.location.y
            )
            if dist < 2.0:
                too_close = True
                break

        if not too_close:
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle:
                break

    if vehicle:
        vehicle.set_autopilot(True)
        # 禁用物理碰撞，避免自车被撞停
        vehicle.set_simulate_physics(False)
        print("主车辆生成成功！")
    else:
        print("警告：无法生成主车辆！")
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
    camera_bp.set_attribute('fov', '80')  # 适度减小FOV，减少畸变
    camera_bp.set_attribute('sensor_tick', '0.05')
    camera_bp.set_attribute('gamma', '2.2')  # 伽马校正

    # 优化相机位置：略微向前，减少自车遮挡
    camera_init_trans = carla.Transform(carla.Location(x=1.8, z=1.6), carla.Rotation(pitch=-5))
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
    depth_bp.set_attribute('fov', '80')
    depth_bp.set_attribute('sensor_tick', '0.05')

    # 相同的安装位置和角度
    camera_init_trans = carla.Transform(carla.Location(x=1.8, z=1.6), carla.Rotation(pitch=-5))
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


def spawn_npcs(world, count=15):
    """优化的NPC生成：避免拥堵"""
    if not world:
        return
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle')
    car_bp = [bp for bp in vehicle_bp if int(bp.get_attribute('number_of_wheels')) == 4]

    spawn_points = world.get_map().get_spawn_points()
    if not car_bp or not spawn_points:
        print("警告：无法生成NPC车辆！")
        return

    # 打乱生成点顺序，避免集中生成
    random.shuffle(spawn_points)

    spawned_count = 0
    for i, spawn_point in enumerate(spawn_points):
        if spawned_count >= count:
            break

        # 检查生成点密度
        nearby_actors = world.get_actors().filter('vehicle.*')
        too_close = False
        for actor in nearby_actors:
            dist = math.hypot(
                actor.get_location().x - spawn_point.location.x,
                actor.get_location().y - spawn_point.location.y
            )
            if dist < 5.0:
                too_close = True
                break

        if not too_close:
            npc = world.try_spawn_actor(random.choice(car_bp), spawn_point)
            if npc:
                npc.set_autopilot(True)
                # 设置不同的驾驶行为
                try:
                    npc.set_attribute('speed', str(random.uniform(20, 50)))
                except:
                    pass
                spawned_count += 1

    print(f"成功生成 {spawned_count} 辆NPC车辆")


def load_detection_model(model_type):
    """加载YOLOv5检测模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_paths = {
        'yolov5s': r"D:\yolo\yolov5s.pt",
        'yolov5m': r"D:\yolo\yolov5m.pt",
        'yolov5x': r"D:\yolo\yolov5x.pt"
    }

    if model_type not in model_paths:
        raise ValueError(f"不支持的模型类型：{model_type}")

    # 加载模型并优化
    model = YOLO(model_paths[model_type])
    model.to(device)

    # 模型优化设置
    if device == 'cuda':
        model = torch.jit.optimize_for_inference(model)

    print(f"检测模型加载成功（设备：{device}，路径：{model_paths[model_type]}）")
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
    parser.add_argument('--model', type=str, default='yolov5m', choices=['yolov5s', 'yolov5m', 'yolov5x'],
                        help='检测模型（yolov5x精度最高，yolov5s最轻便）')
    parser.add_argument('--tracker', type=str, default='sort', choices=['sort'], help='跟踪器（仅保留稳定的SORT）')
    parser.add_argument('--host', type=str, default='localhost', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='检测置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IOU阈值')
    parser.add_argument('--use-depth', action='store_true', default=True, help='启用深度相机辅助跟踪（默认启用）')
    parser.add_argument('--show-depth', action='store_true', help='显示深度图像窗口')
    parser.add_argument('--npc-count', type=int, default=15, help='NPC车辆数量')
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

    try:
        # 1. 初始化设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"当前使用设备：{device}")
        print(f"CUDA可用: {torch.cuda.is_available()}")

        # 2. 连接CARLA服务器
        print(f"正在连接CARLA服务器 {args.host}:{args.port}...")
        world, client = setup_carla_client(args.host, args.port)
        spectator = world.get_spectator()
        print("CARLA服务器连接成功！")

        # 3. 清理环境
        clear_npc(world)
        clear_static_vehicle(world)
        print("环境清理完成！")

        # 4. 生成主车辆
        vehicle = spawn_ego_vehicle(world)
        if not vehicle:
            print("无法生成主车辆，程序退出！")
            return

        # 5. 生成相机传感器
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

        # 6. 生成NPC车辆
        spawn_npcs(world, count=args.npc_count)

        # 7. 加载检测模型和跟踪器
        model, class_names = load_detection_model(args.model)
        tracker, _ = setup_tracker(args.tracker)

        # 8. 主循环
        print("开始目标检测与跟踪（按 'q' 键退出程序）")
        import time

        while True:
            start_time = time.time()
            frame_count += 1

            # 同步CARLA世界
            world.tick()

            # 移动视角
            ego_transform = vehicle.get_transform()
            spectator_transform = carla.Transform(
                ego_transform.transform(carla.Location(x=-8, z=10)),
                carla.Rotation(yaw=ego_transform.rotation.yaw - 180, pitch=-35)
            )
            spectator.set_transform(spectator_transform)

            # 获取图像
            if image_queue.empty():
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

            # 目标检测
            results = model(image, conf=args.conf_thres, iou=args.iou_thres, device=device)

            boxes, labels, probs = [], [], []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    # 只保留车辆相关类别
                    if cls in [2, 3, 5, 7]:
                        box_width = x2 - x1
                        box_height = y2 - y1

                        # 动态尺寸过滤
                        min_size = 8
                        if args.use_depth and depth_image is not None:
                            rough_distance = get_target_distance(depth_image, [x1, y1, x2, y2])
                            if rough_distance > 30:
                                min_size = 3
                            elif rough_distance > 50:
                                min_size = 2

                        if box_width > min_size and box_height > min_size:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls)
                            probs.append(conf)

                            # 计算目标距离
                            if args.use_depth and depth_image is not None:
                                dist = get_target_distance(depth_image, [x1, y1, x2, y2], use_median=True)
                                depths.append(dist)

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
                        if ious[max_iou_idx] > 0.4 and prev_labels[max_iou_idx] == current_label:
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
                        labels=[2] * len(track_boxes),
                        class_names=class_names,
                        track_ids=track_ids,
                        probs=[0.9] * len(track_boxes),
                        distances=track_distances,
                        velocities=track_velocities
                    )
            elif len(boxes) > 0:
                # 无跟踪时绘制检测结果
                image = draw_bounding_boxes(
                    image, boxes, labels, class_names,
                    probs=probs,
                    distances=depths if args.use_depth else None
                )

            # 计算FPS
            fps = 1.0 / (time.time() - start_time)
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)

            # 显示FPS
            cv2.putText(image, f'FPS: {avg_fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 显示跟踪器状态
            cv2.putText(image, f'Tracks: {len(tracker.tracks)}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 显示结果
            cv2.imshow(f'CARLA {args.model} + {args.tracker} (Depth Enhanced)',
                       cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户触发退出程序...")
                break

    except Exception as e:
        import traceback
        print(f"程序运行出错：{str(e)}")
        traceback.print_exc()
        print(
            "报错提示：1. 确保CARLA服务器已启动；2. 确保carla包版本与服务器一致；3. 确保已安装所有依赖；4. 确保YOLO模型路径正确")
    finally:
        # 清理资源
        print("正在清理资源...")
        clear(world, camera, depth_camera)

        if world:
            # 恢复CARLA设置
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.substepping = False
            world.apply_settings(settings)

        # 关闭窗口
        cv2.destroyAllWindows()
        print(f"总共处理 {frame_count} 帧")
        print("资源清理完成，程序正常退出！")


if __name__ == "__main__":
    main()