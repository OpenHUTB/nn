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
import time


# -------------------------- 优化版SORT跟踪器 --------------------------
class KalmanFilter:
    def __init__(self):
        self.dt = 1.0
        self.x = np.zeros((4, 1))
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1000
        self.Q_base = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 5, 0],
                                [0, 0, 0, 5]], dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32) * 5
        self.Q = self.Q_base.copy()
        self.dist_thresholds = [30, 50]

    def predict(self, distance=None):
        if distance is not None:
            if distance > self.dist_thresholds[1]:
                self.Q = self.Q_base * 2.0
            elif distance > self.dist_thresholds[0]:
                self.Q = self.Q_base * 1.5
            else:
                self.Q = self.Q_base * 0.8

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P
        return self.x[:2]


class Track:
    __slots__ = ('id', 'kf', 'x1', 'y1', 'x2', 'y2', 'center', 'width', 'height',
                 'hits', 'age', 'history', 'distance', 'distance_history',
                 'velocity', 'last_distance')

    def __init__(self, box, track_id):
        self.id = track_id
        self.kf = KalmanFilter()
        self.x1, self.y1, self.x2, self.y2 = box
        self.center = np.array([[(self.x1 + self.x2) / 2], [(self.y1 + self.y2) / 2]], dtype=np.float32)
        self.kf.x[:2] = self.center
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.hits = 1
        self.age = 0
        self.history = deque(maxlen=8)
        self.history.append(self.center)
        self.distance = None
        self.distance_history = deque(maxlen=5)
        self.velocity = 0.0
        self.last_distance = None

    def predict(self):
        self.age += 1
        center = self.kf.predict(self.distance)

        if self.history:
            smooth_weight = 0.7 if (self.distance and self.distance > 30) else 0.4
            self.history.append(center)

            n = len(self.history)
            if n > 1:
                weights = np.linspace(0.1, 1.0, n)
                weights /= weights.sum()
                history_array = np.array([h.flatten() for h in self.history])
                smoothed_center = np.average(history_array, axis=0, weights=weights)
                smoothed_center = smoothed_center.reshape(2, 1)
            else:
                smoothed_center = center
        else:
            smoothed_center = center

        self.x1 = smoothed_center[0, 0] - self.width / 2
        self.y1 = smoothed_center[1, 0] - self.height / 2
        self.x2 = self.x1 + self.width
        self.y2 = self.y1 + self.height

        if self.last_distance is not None and self.distance is not None:
            self.velocity = abs(self.distance - self.last_distance) / self.kf.dt

        return [self.x1, self.y1, self.x2, self.y2]

    def update(self, box, distance=None):
        self.x1, self.y1, self.x2, self.y2 = box
        self.center = np.array([[(self.x1 + self.x2) / 2], [(self.y1 + self.y2) / 2]], dtype=np.float32)
        self.kf.update(self.center)
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.hits += 1
        self.age = 0
        self.history.append(self.center)

        if distance is not None:
            self.last_distance = self.distance
            self.distance_history.append(distance)
            if self.distance_history:
                self.distance = float(np.median(self.distance_history))

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
        self.depth_thresholds = {'near': 15.0, 'medium': 30.0, 'far': 50.0}
        from scipy.optimize import linear_sum_assignment
        self.linear_sum_assignment = linear_sum_assignment

    def set_depths(self, depths):
        if depths:
            depths_array = np.array(depths, dtype=np.float32)
            valid_mask = (depths_array > 0.1) & (depths_array < 200)
            self.depths = np.where(valid_mask, depths_array, 50.0)
        else:
            self.depths = []

    def update(self, detections):
        if len(detections) == 0:
            for track in self.tracks:
                track.age += 1
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            if self.tracks and self.min_hits > 0:
                return np.array([[t.x1, t.y1, t.x2, t.y2, t.id]
                                 for t in self.tracks if t.hits >= self.min_hits])
            return np.array([])

        if len(self.tracks) == 0:
            tracks_to_return = []
            for i, det in enumerate(detections):
                track = Track(det[:4], self.next_id)
                if i < len(self.depths):
                    track.distance = self.depths[i]
                    track.distance_history.append(self.depths[i])
                self.tracks.append(track)
                tracks_to_return.append([track.x1, track.y1, track.x2, track.y2, track.id])
                self.next_id += 1
            return np.array(tracks_to_return) if tracks_to_return else np.array([])

        track_boxes = np.array([t.get_box() for t in self.tracks], dtype=np.float32)
        iou_matrix = self._iou_batch(track_boxes, detections[:, :4].astype(np.float32))

        if iou_matrix.size > 0:
            matches, unmatched_tracks, unmatched_dets = self._hungarian_algorithm(iou_matrix)

            for track_idx, det_idx in matches:
                if iou_matrix[track_idx, det_idx] >= self._get_dynamic_iou_threshold(det_idx):
                    det_distance = self.depths[det_idx] if det_idx < len(self.depths) else None
                    self.tracks[track_idx].update(detections[det_idx][:4], det_distance)

        for track_idx in range(len(self.tracks)):
            if track_idx in unmatched_tracks:
                self.tracks[track_idx].age += 1

        for det_idx in unmatched_dets:
            box = detections[det_idx][:4]
            if self._is_valid_detection(box, det_idx):
                track = Track(box, self.next_id)
                if det_idx < len(self.depths):
                    track.distance = self.depths[det_idx]
                    track.distance_history.append(self.depths[det_idx])
                self.tracks.append(track)
                self.next_id += 1

        self.tracks = [t for t in self.tracks if self._should_keep_track(t)]

        valid_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
        if valid_tracks:
            return np.array([[t.x1, t.y1, t.x2, t.y2, t.id] for t in valid_tracks])
        return np.array([])

    def _get_dynamic_iou_threshold(self, det_idx):
        if det_idx < len(self.depths):
            dist = self.depths[det_idx]
            if dist < self.depth_thresholds['near']:
                return 0.5
            elif dist > self.depth_thresholds['far']:
                return 0.3
        return self.iou_threshold

    def _is_valid_detection(self, box, det_idx):
        width = box[2] - box[0]
        height = box[3] - box[1]
        aspect_ratio = width / max(height, 1e-6)

        if det_idx < len(self.depths):
            dist = self.depths[det_idx]
            if dist < self.depth_thresholds['near']:
                min_size = 10
            elif dist < self.depth_thresholds['medium']:
                min_size = 5
            elif dist < self.depth_thresholds['far']:
                min_size = 3
            else:
                min_size = 2
        else:
            min_size = 5

        return (0.2 < aspect_ratio < 5.0 and
                width > min_size and
                height > min_size)

    def _should_keep_track(self, track):
        if track.distance and track.distance > self.depth_thresholds['far']:
            return track.age <= self.max_age + 2
        return track.age <= self.max_age

    def _iou_batch(self, b1, b2):
        if b1.shape[0] == 0 or b2.shape[0] == 0:
            return np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float32)

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

        if len(self.depths) == b2.shape[0]:
            distances = self.depths
            distance_weights = np.ones_like(distances)
            near_mask = distances < self.depth_thresholds['near']
            medium_mask = (distances >= self.depth_thresholds['near']) & (distances < self.depth_thresholds['medium'])
            far_mask = (distances >= self.depth_thresholds['medium']) & (distances < self.depth_thresholds['far'])

            distance_weights[near_mask] = 1.5
            distance_weights[medium_mask] = 1.2
            distance_weights[far_mask] = 0.9
            distance_weights[distances >= self.depth_thresholds['far']] = 0.7

            return base_iou * distance_weights[None, :]
        return base_iou

    def _hungarian_algorithm(self, cost_matrix):
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        row_ind, col_ind = self.linear_sum_assignment(-cost_matrix)
        matches = list(zip(row_ind, col_ind))

        all_rows = set(range(cost_matrix.shape[0]))
        all_cols = set(range(cost_matrix.shape[1]))
        matched_rows = set(row_ind)
        matched_cols = set(col_ind)

        unmatched_tracks = list(all_rows - matched_rows)
        unmatched_dets = list(all_cols - matched_cols)

        return matches, unmatched_tracks, unmatched_dets


# -------------------------- YOLOv5检测模型 --------------------------
from ultralytics import YOLO


def load_detection_model(model_type):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_paths = {
        'yolov5s': r"D:\yolo\yolov5s.pt",
        'yolov5su': r"D:\yolo\yolov5su.pt",
        'yolov5m': r"D:\yolo\yolov5m.pt",
        'yolov5mu': r"D:\yolo\yolov5mu.pt",
        'yolov5x': r"D:\yolo\yolov5x.pt"
    }

    if model_type not in model_paths:
        if 'su' in model_type.lower():
            model_type = 'yolov5su'
        elif 'mu' in model_type.lower():
            model_type = 'yolov5mu'
        else:
            model_type = 'yolov5m'

    model_path = model_paths.get(model_type)
    if not model_path or not os.path.exists(model_path):
        for key, path in model_paths.items():
            if os.path.exists(path):
                model_type = key
                model_path = path
                print(f"使用备用模型：{model_type}")
                break

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"无法找到模型文件")

    model = YOLO(model_path)
    model.to(device)

    if device == 'cuda':
        model.half()

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640, device=device)
        if device == 'cuda':
            dummy_input = dummy_input.half()
        _ = model(dummy_input)

    print(f"模型加载成功：{model_type} (设备：{device})")
    return model, model.names


# -------------------------- 性能监控面板函数 --------------------------
def draw_performance_panel(image, timings, fps, frame_count):
    """
    在图像上绘制性能监控面板
    """
    h, w = image.shape[:2]

    # 计算各阶段耗时（最近10帧平均）
    def get_avg_time(key, default=0.0):
        if key in timings and timings[key]:
            recent = timings[key][-10:]  # 取最近10帧
            return np.mean(recent) if recent else default
        return default

    # 计算各阶段耗时（毫秒）
    carla_time = get_avg_time('carla_tick') * 1000
    image_time = get_avg_time('image_get') * 1000
    depth_time = get_avg_time('depth_get') * 1000
    detection_time = get_avg_time('detection') * 1000
    tracking_time = get_avg_time('tracking') * 1000
    drawing_time = get_avg_time('drawing') * 1000
    display_time = get_avg_time('display') * 1000
    total_time = get_avg_time('total') * 1000

    # 面板位置和尺寸
    panel_x = 10  # 左上角x坐标
    panel_y = 10  # 左上角y坐标
    panel_width = 280  # 面板宽度
    panel_height = 200  # 面板高度

    # 创建半透明面板背景
    panel_bg = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    panel_bg[:] = (20, 20, 20)  # 深灰色背景

    # 叠加面板到图像上
    x1, y1 = panel_x, panel_y
    x2, y2 = panel_x + panel_width, panel_y + panel_height

    # 确保面板在图像范围内
    if x2 <= w and y2 <= h:
        # 半透明叠加
        alpha = 0.7
        image[y1:y2, x1:x2] = cv2.addWeighted(
            image[y1:y2, x1:x2], 1 - alpha, panel_bg, alpha, 0
        )

        # 绘制标题和分隔线
        title = "性能监控面板"
        cv2.putText(image, title, (panel_x + 10, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.line(image, (panel_x, panel_y + 25),
                 (panel_x + panel_width, panel_y + 25), (100, 100, 100), 1)

        # 绘制性能指标
        y_offset = 45
        line_height = 20

        # FPS和帧数
        cv2.putText(image, f"FPS: {fps:.1f}", (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, f"Frame: {frame_count}", (panel_x + 120, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        y_offset += line_height

        # CARLA同步
        color = (0, 255, 0) if carla_time < 5.0 else (0, 165, 255) if carla_time < 10.0 else (0, 0, 255)
        cv2.putText(image, f"CARLA: {carla_time:.1f}ms", (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 图像获取
        color = (0, 255, 0) if image_time < 1.0 else (0, 165, 255) if image_time < 3.0 else (0, 0, 255)
        cv2.putText(image, f"Image: {image_time:.1f}ms", (panel_x + 120, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y_offset += line_height

        # 深度获取
        if depth_time > 0:
            color = (0, 255, 0) if depth_time < 2.0 else (0, 165, 255) if depth_time < 5.0 else (0, 0, 255)
            cv2.putText(image, f"Depth: {depth_time:.1f}ms", (panel_x + 10, panel_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y_offset += line_height

        # 目标检测
        color = (0, 255, 0) if detection_time < 10.0 else (0, 165, 255) if detection_time < 20.0 else (0, 0, 255)
        cv2.putText(image, f"Detection: {detection_time:.1f}ms", (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y_offset += line_height

        # 目标跟踪
        color = (0, 255, 0) if tracking_time < 5.0 else (0, 165, 255) if tracking_time < 10.0 else (0, 0, 255)
        cv2.putText(image, f"Tracking: {tracking_time:.1f}ms", (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 结果绘制
        color = (0, 255, 0) if drawing_time < 2.0 else (0, 165, 255) if drawing_time < 5.0 else (0, 0, 255)
        cv2.putText(image, f"Drawing: {drawing_time:.1f}ms", (panel_x + 120, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y_offset += line_height

        # 显示输出
        color = (0, 255, 0) if display_time < 2.0 else (0, 165, 255) if display_time < 5.0 else (0, 0, 255)
        cv2.putText(image, f"Display: {display_time:.1f}ms", (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y_offset += line_height

        # 总帧时间
        color = (0, 255, 0) if total_time < 50.0 else (0, 165, 255) if total_time < 100.0 else (0, 0, 255)
        cv2.putText(image, f"Total: {total_time:.1f}ms", (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y_offset += line_height

        # 绘制简易性能柱状图
        bar_x = panel_x + 10
        bar_y = panel_y + y_offset + 5
        bar_width = 250
        bar_height = 15

        # 绘制背景
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        # 计算各阶段占比
        stages = [carla_time, image_time, depth_time, detection_time, tracking_time, drawing_time, display_time]
        stage_names = ['C', 'I', 'D', 'Det', 'T', 'Draw', 'Disp']
        stage_colors = [
            (0, 200, 200),  # CARLA - 青色
            (200, 200, 0),  # Image - 黄色
            (200, 0, 200),  # Depth - 紫色
            (255, 100, 0),  # Detection - 橙色
            (0, 150, 255),  # Tracking - 浅蓝色
            (100, 255, 100),  # Drawing - 浅绿色
            (255, 100, 100)  # Display - 浅红色
        ]

        # 绘制各阶段柱状图
        if total_time > 0:
            accumulated = 0
            for i, (stage_time, color, name) in enumerate(zip(stages, stage_colors, stage_names)):
                if stage_time > 0:
                    stage_width = int((stage_time / total_time) * bar_width)
                    if stage_width > 0:
                        # 绘制阶段条
                        stage_x1 = bar_x + accumulated
                        stage_x2 = stage_x1 + stage_width
                        cv2.rectangle(image, (stage_x1, bar_y), (stage_x2, bar_y + bar_height), color, -1)

                        # 绘制阶段名称
                        if stage_width > 15:
                            cv2.putText(image, name, (stage_x1 + 2, bar_y + 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                        accumulated += stage_width

    return image


# -------------------------- 核心工具函数 --------------------------
def draw_bounding_boxes(image, boxes, labels, class_names, **kwargs):
    track_ids = kwargs.get('track_ids')
    probs = kwargs.get('probs')
    distances = kwargs.get('distances')
    velocities = kwargs.get('velocities')

    result = image.copy()
    h, w = image.shape[:2]

    # 颜色缓存
    color_cache = {}

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x1 >= x2 or y1 >= y2:
            continue

        # 颜色计算
        color = (0, 255, 0)
        if distances and i < len(distances) and distances[i]:
            dist = distances[i]
            if dist not in color_cache:
                if dist < 15:
                    r, g = 255, int(255 * (dist / 15))
                    color_cache[dist] = (0, g, r)
                elif dist < 30:
                    r, g = int(255 * (1 - (dist - 15) / 15)), 255
                    color_cache[dist] = (0, g, r)
                elif dist < 50:
                    b, g = int(255 * ((dist - 30) / 20)), 255
                    color_cache[dist] = (b, g, 0)
                else:
                    color_cache[dist] = (0, 255, 0)
            color = color_cache[dist]

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # 构建文本信息
        text_parts = [
            class_names.get(labels[i], f"cls{labels[i]}") if i < len(labels) else "",
            f"{probs[i]:.2f}" if probs and i < len(probs) else "",
            f"ID:{track_ids[i]}" if track_ids and i < len(track_ids) else "",
            f"D:{distances[i]:.1f}m" if distances and i < len(distances) and distances[i] else "",
            f"S:{velocities[i]:.1f}m/s" if velocities and i < len(velocities) and velocities[i] else ""
        ]

        # 过滤空字符串并拼接
        label_text = " ".join(filter(None, text_parts))

        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_bg_y1 = max(0, y1 - text_size[1] - 5)
        text_bg_y2 = max(0, y1)

        cv2.rectangle(result, (x1, text_bg_y1),
                      (x1 + text_size[0], text_bg_y2), color, -1)
        cv2.putText(result, label_text, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return result


def preprocess_depth_image(depth_image):
    if depth_image is None:
        return None

    # 如果深度图像是float16类型，转换为float32进行处理
    if depth_image.dtype == np.float16:
        depth_image = depth_image.astype(np.float32)

    depth_image = np.clip(depth_image, 0.1, 200.0)

    if depth_image.shape[0] > 3 and depth_image.shape[1] > 3:
        # 使用较小的高斯模糊，减少计算量
        depth_image = cv2.GaussianBlur(depth_image, (3, 3), 0.5)

    max_val = np.max(depth_image)
    if max_val > 0:
        depth_image = np.power(depth_image / max_val, 0.7) * max_val

    return depth_image


def get_target_distance(depth_image, box, use_median=True):
    if depth_image is None:
        return 50.0

    # 如果深度图像是float16类型，转换为float32进行计算
    if depth_image.dtype == np.float16:
        depth_image = depth_image.astype(np.float32)

    x1, y1, x2, y2 = map(int, box)

    h, w = depth_image.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x1 >= x2 or y1 >= y2:
        return 50.0

    depth_roi = depth_image[y1:y2, x1:x2]
    valid_mask = depth_roi > 0.1
    valid_depths = depth_roi[valid_mask]

    if valid_depths.size == 0:
        return 50.0

    if use_median:
        return float(np.median(valid_depths))
    else:
        h_roi, w_roi = depth_roi.shape
        if h_roi == 0 or w_roi == 0:
            return float(np.mean(valid_depths))

        cy, cx = h_roi // 2, w_roi // 2
        y_coords, x_coords = np.ogrid[:h_roi, :w_roi]
        dist_from_center = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2) + 1e-6
        weights = 1 - (dist_from_center / max_dist)
        weights = weights * valid_mask

        if np.sum(weights) > 0:
            return float(np.sum(depth_roi * weights) / np.sum(weights))
        else:
            return float(np.mean(valid_depths))


# -------------------------- CARLA相关函数（修正版） --------------------------
def setup_carla_client(host='localhost', port=2000):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    return world, client


def spawn_ego_vehicle(world):
    if not world:
        return None

    bp_lib = world.get_blueprint_library()

    # 优先使用林肯MKZ
    vehicle_bp = None
    try:
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    except:
        pass

    if not vehicle_bp:
        small_vehicles = [
            'vehicle.audi.a2',
            'vehicle.audi.tt',
            'vehicle.toyota.prius',
            'vehicle.volkswagen.t2',
            'vehicle.nissan.patrol',
            'vehicle.mercedes.coupe'
        ]
        for vehicle_name in small_vehicles:
            try:
                vehicle_bp = bp_lib.find(vehicle_name)
                if vehicle_bp:
                    break
            except:
                continue

    if not vehicle_bp:
        vehicle_bp = random.choice([bp for bp in bp_lib.filter('vehicle.*')])

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        return None

    # 随机选择生成点
    random.shuffle(spawn_points)

    for spawn_point in spawn_points[:10]:
        # 检查是否有其他车辆
        too_close = False
        for actor in world.get_actors().filter('vehicle.*'):
            actor_loc = actor.get_location()
            dist = math.hypot(
                actor_loc.x - spawn_point.location.x,
                actor_loc.y - spawn_point.location.y
            )
            if dist < 5.0:
                too_close = True
                break

        if not too_close:
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                vehicle.set_simulate_physics(False)
                return vehicle

    # 强制生成
    if spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
        if vehicle:
            vehicle.set_autopilot(True)
            vehicle.set_simulate_physics(False)
            return vehicle

    return None


def spawn_npcs(world, count=30, ego_vehicle=None):
    """修正的NPC生成函数 - 确保在主车辆周围生成足够车辆"""
    if not world:
        return []

    print(f"正在生成 {count} 辆NPC车辆...")

    # 先检查当前已有的车辆
    existing_vehicles = world.get_actors().filter('vehicle.*')
    print(f"当前已有车辆数量: {len(existing_vehicles)}")

    bp_lib = world.get_blueprint_library()

    # 创建一个包含所有车辆蓝图的列表
    vehicle_blueprints = list(bp_lib.filter('vehicle.*'))
    if not vehicle_blueprints:
        print("错误：没有找到任何车辆蓝图！")
        return []

    print(f"可用的车辆蓝图数量: {len(vehicle_blueprints)}")

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("警告：没有可用的生成点！")
        return []

    print(f"可用的生成点数量: {len(spawn_points)}")

    ego_location = ego_vehicle.get_location() if ego_vehicle else None
    ego_transform = ego_vehicle.get_transform() if ego_vehicle else None

    spawned_count = 0
    npc_vehicles = []

    # 只尝试每个生成点一次，避免无限循环
    random.shuffle(spawn_points)

    for i, spawn_point in enumerate(spawn_points):
        if spawned_count >= count:
            break

        if i % 10 == 0:
            print(f"尝试生成点 {i + 1}/{len(spawn_points)}...")

        # 检查与主车辆的距离
        if ego_location:
            dist_to_ego = math.hypot(
                spawn_point.location.x - ego_location.x,
                spawn_point.location.y - ego_location.y
            )

            # 只生成在合理距离范围内的车辆（30-80米）
            if dist_to_ego < 30.0 or dist_to_ego > 80.0:
                continue

        # 检查是否太靠近已生成的NPC
        too_close = False
        for npc in npc_vehicles:
            try:
                npc_loc = npc.get_location()
                dist = math.hypot(
                    npc_loc.x - spawn_point.location.x,
                    npc_loc.y - spawn_point.location.y
                )
                if dist < 5.0:  # 5米内不生成
                    too_close = True
                    break
            except:
                continue

        if too_close:
            continue

        try:
            # 随机选择一个车辆蓝图
            vehicle_bp = random.choice(vehicle_blueprints)

            # 设置随机颜色
            if vehicle_bp.has_attribute('color'):
                colors = vehicle_bp.get_attribute('color').recommended_values
                if colors:
                    vehicle_bp.set_attribute('color', random.choice(colors))

            # 尝试生成车辆
            npc = world.try_spawn_actor(vehicle_bp, spawn_point)

            if npc:
                npc.set_autopilot(True)
                npc_vehicles.append(npc)
                spawned_count += 1

                if ego_location:
                    dist_to_ego = math.hypot(
                        spawn_point.location.x - ego_location.x,
                        spawn_point.location.y - ego_location.y
                    )
                    print(f"✓ 生成NPC {spawned_count}/{count} - 距离: {dist_to_ego:.1f}米")
                else:
                    print(f"✓ 生成NPC {spawned_count}/{count}")
        except Exception as e:
            print(f"生成NPC失败: {e}")
            continue

    print(f"最终生成 {spawned_count} 辆NPC车辆")

    # 验证生成的车辆
    valid_npcs = []
    for npc in npc_vehicles:
        try:
            loc = npc.get_location()
            if loc:
                valid_npcs.append(npc)
        except:
            print(f"警告：无法获取NPC位置，可能已销毁")

    print(f"有效NPC车辆: {len(valid_npcs)} 辆")

    # 打印距离统计
    if ego_location and valid_npcs:
        distances = []
        for npc in valid_npcs:
            try:
                npc_loc = npc.get_location()
                dist = math.hypot(
                    npc_loc.x - ego_location.x,
                    npc_loc.y - ego_location.y
                )
                distances.append(dist)
            except:
                continue

        if distances:
            print(
                f"NPC距离统计: 最近{min(distances):.1f}米, 最远{max(distances):.1f}米, 平均{np.mean(distances):.1f}米")
        else:
            print("警告：无法计算NPC距离")

    return valid_npcs


# 优化内存占用：使用更高效的回调函数
def camera_callback(image, rgb_image_queue):
    """优化内存占用的RGB图像回调函数"""
    # 使用array接口避免额外复制，只取RGB通道
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[..., :3]  # 只存储RGB通道，丢弃Alpha通道

    # 使用put_nowait避免阻塞
    try:
        rgb_image_queue.put_nowait(rgb)
    except queue.Full:
        # 丢弃最旧的一帧，放入新帧
        try:
            rgb_image_queue.get_nowait()
        except:
            pass
        rgb_image_queue.put_nowait(rgb)


def depth_camera_callback(image, depth_queue):
    """优化内存占用的深度图像回调函数"""
    # 使用更小的数据类型存储深度图像
    depth_data = np.frombuffer(image.raw_data, dtype=np.uint8)
    depth_data = depth_data.reshape((image.height, image.width, 4))

    # 使用uint16存储深度数据，减少内存占用
    depth_channel = (
            depth_data[..., 2].astype(np.uint16) +
            depth_data[..., 1].astype(np.uint16) * 256 +
            depth_data[..., 0].astype(np.uint16) * 256 ** 2
    )

    # 转换为米为单位，并使用float16存储（相比float32减少50%内存）
    depth_in_meters = depth_channel.astype(np.float16) / (256 ** 3 - 1) * 1000.0

    # 预处理深度图像
    depth_in_meters = preprocess_depth_image(depth_in_meters)

    # 使用put_nowait避免阻塞
    try:
        depth_queue.put_nowait(depth_in_meters)
    except queue.Full:
        # 丢弃最旧的一帧，放入新帧
        try:
            depth_queue.get_nowait()
        except:
            pass
        depth_queue.put_nowait(depth_in_meters)


# -------------------------- 主函数 --------------------------
def main():
    parser = argparse.ArgumentParser(description='CARLA目标检测与跟踪')
    parser.add_argument('--model', type=str, default='yolov5mu',
                        choices=['yolov5s', 'yolov5su', 'yolov5m', 'yolov5mu', 'yolov5x'])
    parser.add_argument('--tracker', type=str, default='sort', choices=['sort'])
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--conf-thres', type=float, default=0.15)
    parser.add_argument('--iou-thres', type=float, default=0.4)
    parser.add_argument('--use-depth', action='store_true', default=True)
    parser.add_argument('--show-depth', action='store_true')
    parser.add_argument('--npc-count', type=int, default=30)
    args = parser.parse_args()

    # 初始化变量
    world = vehicle = camera = depth_camera = None
    image_queue = depth_queue = None
    client = None

    try:
        # 1. 初始化设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")

        # 2. 连接CARLA
        print("连接CARLA服务器...")
        world, client = setup_carla_client(args.host, args.port)
        spectator = world.get_spectator()
        print(f"连接成功，地图: {world.get_map().name}")

        # 3. 清理环境
        print("清理环境...")
        for actor in world.get_actors().filter('vehicle.*'):
            try:
                actor.destroy()
            except:
                pass

        # 4. 生成主车辆
        print("生成主车辆...")
        vehicle = spawn_ego_vehicle(world)
        if not vehicle:
            print("主车辆生成失败，程序退出！")
            return

        ego_location = vehicle.get_location()
        print(f"主车辆位置: X={ego_location.x:.1f}, Y={ego_location.y:.1f}")

        # 5. 生成传感器
        print("生成相机传感器...")
        bp_lib = world.get_blueprint_library()

        # RGB相机
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '100')
        camera_bp.set_attribute('sensor_tick', '0.05')

        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.8),
                                           carla.Rotation(pitch=-5, yaw=0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        image_queue = queue.Queue(maxsize=3)
        camera.listen(lambda image: camera_callback(image, image_queue))

        # 深度相机
        depth_camera = None
        depth_queue = None
        if args.use_depth:
            depth_bp = bp_lib.find('sensor.camera.depth')
            depth_bp.set_attribute('image_size_x', '800')
            depth_bp.set_attribute('image_size_y', '600')
            depth_bp.set_attribute('fov', '100')
            depth_bp.set_attribute('sensor_tick', '0.05')

            depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
            depth_queue = queue.Queue(maxsize=3)
            depth_camera.listen(lambda image: depth_camera_callback(image, depth_queue))
            print("深度相机传感器生成成功！")

        print("RGB相机传感器生成成功！")

        # 6. 生成NPC车辆
        print(f"生成 {args.npc_count} 辆NPC车辆...")
        npc_vehicles = spawn_npcs(world, count=args.npc_count, ego_vehicle=vehicle)

        if len(npc_vehicles) < args.npc_count // 2:
            print(
                f"警告：只成功生成了 {len(npc_vehicles)} 辆NPC车辆（目标: {args.npc_count} 辆），可能无法获得足够的检测目标")
        else:
            print(f"成功生成 {len(npc_vehicles)} 辆NPC车辆")

        # 配置交通管理器
        try:
            # 获取交通管理器
            traffic_manager = client.get_trafficmanager()

            # 设置交通管理器端口（通常使用默认端口8000）
            tm_port = traffic_manager.get_port()

            # 设置同步模式
            traffic_manager.set_synchronous_mode(True)

            # 设置全局参数
            traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            traffic_manager.global_percentage_speed_difference(0.0)

            # 设置所有NPC车辆的交通管理器参数
            for npc in npc_vehicles:
                # 重新设置自动驾驶，使用交通管理器
                npc.set_autopilot(True, tm_port)

                # 设置与前方车辆的距离
                traffic_manager.distance_to_leading_vehicle(npc, random.uniform(2.0, 5.0))
                # 设置速度差
                traffic_manager.vehicle_percentage_speed_difference(npc, random.uniform(-10.0, 10.0))

            print("交通管理器配置完成")
        except Exception as e:
            print(f"交通管理器配置失败: {e}")

        # 等待NPC车辆稳定
        print("等待NPC车辆初始化...")
        for _ in range(10):
            world.tick()

        # 7. 加载检测模型和跟踪器
        print("加载检测模型和跟踪器...")
        model, class_names = load_detection_model(args.model)
        tracker = Sort()

        # 8. 主循环
        print("\n开始目标检测与跟踪（按 'q' 键退出程序，按 'r' 重新生成NPC）")
        print("=" * 50)

        frame_count = 0
        detection_stats = {'total_detections': 0, 'total_frames': 0, 'max_vehicles_per_frame': 0}

        # 性能监控变量
        timings = {
            'carla_tick': [],
            'image_get': [],
            'depth_get': [],
            'detection': [],
            'tracking': [],
            'drawing': [],
            'display': [],
            'total': []
        }

        while True:
            frame_start = time.time()
            frame_count += 1
            detection_stats['total_frames'] += 1

            # 同步CARLA世界
            tick_start = time.time()
            world.tick()
            timings['carla_tick'].append(time.time() - tick_start)

            # 移动视角跟随主车辆
            ego_transform = vehicle.get_transform()
            spectator_transform = carla.Transform(
                ego_transform.transform(carla.Location(x=-10, z=12)),
                carla.Rotation(yaw=ego_transform.rotation.yaw - 180, pitch=-30)
            )
            spectator.set_transform(spectator_transform)

            # 获取图像
            image_start = time.time()
            if image_queue.empty():
                time.sleep(0.001)
                continue

            origin_image = image_queue.get()
            image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)  # 注意：已经去掉了Alpha通道
            height, width, _ = image.shape
            timings['image_get'].append(time.time() - image_start)

            # 获取深度图像
            depth_start = time.time()
            depth_image = None
            if args.use_depth and depth_queue and not depth_queue.empty():
                depth_image = depth_queue.get()

                if args.show_depth:
                    # 转换为8位用于显示
                    if depth_image.dtype == np.float16:
                        depth_vis = depth_image.astype(np.float32)
                    else:
                        depth_vis = depth_image.copy()

                    depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
                    depth_vis = depth_vis.astype(np.uint8)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    cv2.imshow('Depth Image', depth_vis)
            timings['depth_get'].append(time.time() - depth_start)

            # 目标检测
            detection_start = time.time()
            boxes, labels, probs, depths = [], [], [], []

            try:
                results = model(image, conf=args.conf_thres, iou=args.iou_thres,
                                device=device, imgsz=640, verbose=False)

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        # 只保留车辆相关类别
                        if cls in [2, 3, 5, 7]:
                            box_width = x2 - x1
                            box_height = y2 - y1

                            # 尺寸过滤
                            min_size = 6
                            if depth_image is not None:
                                rough_distance = get_target_distance(depth_image, [x1, y1, x2, y2])
                                if rough_distance > 30:
                                    min_size = 4
                                elif rough_distance > 50:
                                    min_size = 2
                                else:
                                    min_size = 8

                            aspect_ratio = box_width / max(box_height, 1)

                            if (box_width > min_size and box_height > min_size and
                                    0.3 < aspect_ratio < 3.0):
                                boxes.append([x1, y1, x2, y2])
                                labels.append(cls)
                                probs.append(conf)

                                # 计算目标距离
                                if depth_image is not None:
                                    dist = get_target_distance(depth_image, [x1, y1, x2, y2], use_median=True)
                                    depths.append(dist)
            except Exception as e:
                print(f"检测模型推理出错: {e}")

            timings['detection'].append(time.time() - detection_start)

            # 更新检测统计
            detection_stats['total_detections'] += len(boxes)
            detection_stats['max_vehicles_per_frame'] = max(
                detection_stats['max_vehicles_per_frame'], len(boxes)
            )

            # 目标跟踪
            tracking_start = time.time()
            if boxes:
                boxes_np = np.array(boxes, dtype=np.float32)
                probs_np = np.array(probs, dtype=np.float32).reshape(-1, 1)
                dets = np.hstack([boxes_np, probs_np]) if probs_np.size > 0 else boxes_np

                if depths:
                    tracker.set_depths(depths)

                track_results = tracker.update(dets)

                if track_results.size > 0:
                    track_boxes = []
                    track_ids = []
                    track_distances = []
                    track_velocities = []

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
                    drawing_start = time.time()
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

                        cv2.putText(image, f'Vehicles: {len(track_boxes)}', (width - 200, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    timings['drawing'].append(time.time() - drawing_start)
            else:
                timings['tracking'].append(0.0)
                timings['drawing'].append(0.0)

            timings['tracking'].append(time.time() - tracking_start)

            # 计算FPS
            total_time = time.time() - frame_start
            fps = 1.0 / total_time if total_time > 0 else 0
            timings['total'].append(total_time)

            # 绘制性能监控面板
            image = draw_performance_panel(image, timings, fps, frame_count)

            # 显示其他信息
            info = [
                f"FPS: {fps:.1f}",
                f"Frame: {frame_count}",
                f"Tracks: {len(tracker.tracks)}",
                f"Detections: {len(boxes)}",
                f"Model: {args.model}",
                "Press 'q' to quit"
            ]

            # 调整信息位置，避免与性能面板重叠
            y_pos = 220
            for line in info:
                cv2.putText(image, line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_pos += 25

            # 显示结果
            display_start = time.time()
            window_name = f'CARLA {args.model} + {args.tracker}'
            cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            timings['display'].append(time.time() - display_start)

            # 每50帧打印一次性能统计
            if frame_count % 50 == 0 and frame_count > 0:
                print("\n" + "=" * 60)
                print("性能监控统计 (最近50帧平均值):")
                print("-" * 60)
                print(f"CARLA同步: {np.mean(timings['carla_tick'][-50:]):.4f}s")
                print(f"图像获取:   {np.mean(timings['image_get'][-50:]):.4f}s")
                print(f"深度获取:   {np.mean(timings['depth_get'][-50:]):.4f}s")
                print(f"目标检测:   {np.mean(timings['detection'][-50:]):.4f}s (瓶颈)")
                print(f"目标跟踪:   {np.mean(timings['tracking'][-50:]):.4f}s")
                print(f"结果绘制:   {np.mean(timings['drawing'][-50:]):.4f}s")
                print(f"显示输出:   {np.mean(timings['display'][-50:]):.4f}s")
                print(f"总帧时间:   {np.mean(timings['total'][-50:]):.4f}s")
                print(f"理论FPS:    {1.0 / np.mean(timings['total'][-50:]):.1f}")
                print("=" * 60)

                # 自动识别瓶颈并提供建议
                bottleneck_times = {
                    '检测': np.mean(timings['detection'][-50:]),
                    '跟踪': np.mean(timings['tracking'][-50:]),
                    '绘制': np.mean(timings['drawing'][-50:]),
                    'CARLA': np.mean(timings['carla_tick'][-50:])
                }
                bottleneck = max(bottleneck_times, key=bottleneck_times.get)
                if bottleneck_times[bottleneck] > 0.05:  # 超过50ms
                    print(f"⚠️  性能瓶颈: {bottleneck} ({bottleneck_times[bottleneck]:.3f}s)")
                    if bottleneck == '检测':
                        print("建议: 降低模型分辨率或使用更轻量模型")
                    elif bottleneck == '跟踪':
                        print("建议: 减少跟踪目标数量或简化跟踪算法")
                    elif bottleneck == '绘制':
                        print("建议: 简化绘制内容或减少信息显示")
                    elif bottleneck == 'CARLA':
                        print("建议: 减少NPC数量或降低仿真频率")

            # 退出检查
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户触发退出程序...")
                break
            elif key == ord('r'):
                # 重新生成NPC
                print("重新生成NPC车辆...")
                for actor in world.get_actors().filter('vehicle.*'):
                    if actor != vehicle:
                        try:
                            actor.destroy()
                        except:
                            pass
                npc_vehicles = spawn_npcs(world, count=args.npc_count, ego_vehicle=vehicle)
                print("NPC重新生成完成")

                # 重新配置交通管理器
                try:
                    traffic_manager = client.get_trafficmanager()
                    tm_port = traffic_manager.get_port()
                    traffic_manager.set_synchronous_mode(True)

                    for npc in npc_vehicles:
                        npc.set_autopilot(True, tm_port)
                        traffic_manager.distance_to_leading_vehicle(npc, random.uniform(2.0, 5.0))
                        traffic_manager.vehicle_percentage_speed_difference(npc, random.uniform(-10.0, 10.0))

                    print("交通管理器重新配置完成")
                except Exception as e:
                    print(f"交通管理器重新配置失败: {e}")

            # FPS控制
            elapsed = time.time() - frame_start
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)

    except KeyboardInterrupt:
        print("\n用户中断程序...")
    except Exception as e:
        import traceback
        print(f"程序运行出错：{str(e)}")
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n正在清理资源...")

        # 先停止传感器监听
        if camera:
            try:
                camera.stop()
                camera.destroy()
            except:
                pass

        if depth_camera:
            try:
                depth_camera.stop()
                depth_camera.destroy()
            except:
                pass

        # 清理队列中的残留数据
        if image_queue:
            while not image_queue.empty():
                try:
                    image_queue.get_nowait()
                except:
                    break

        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)

        cv2.destroyAllWindows()

        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 打印最终统计
        print("\n" + "=" * 50)
        print("程序运行统计：")
        print(f"总帧数: {frame_count}")

        # 性能统计总结
        if timings['total']:
            print(f"平均FPS: {1.0 / np.mean(timings['total']):.1f}")
            print("\n性能时间分布：")
            print(f"CARLA同步: {np.mean(timings['carla_tick']):.4f}s")
            print(f"图像获取:   {np.mean(timings['image_get']):.4f}s")
            print(f"深度获取:   {np.mean(timings['depth_get']):.4f}s")
            print(f"目标检测:   {np.mean(timings['detection']):.4f}s")
            print(f"目标跟踪:   {np.mean(timings['tracking']):.4f}s")
            print(f"结果绘制:   {np.mean(timings['drawing']):.4f}s")
            print(f"显示输出:   {np.mean(timings['display']):.4f}s")

        print(f"总检测次数: {detection_stats['total_detections']}")
        print(f"平均每帧检测: {detection_stats['total_detections'] / max(1, detection_stats['total_frames']):.1f}")
        print(f"最大单帧车辆数: {detection_stats['max_vehicles_per_frame']}")
        print("=" * 50)
        print("资源清理完成，程序正常退出！")


if __name__ == "__main__":
    main()