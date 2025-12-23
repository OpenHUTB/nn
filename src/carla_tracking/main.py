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
import sys
from pathlib import Path


# -------------------------- 优化版SORT跟踪器 --------------------------
class KalmanFilter:
    def __init__(self):
        self.dt = 1.0
        self.x = np.zeros((6, 1))  # 增加速度和加速度状态
        self.F = np.array([[1, 0, self.dt, 0, 0.5 * self.dt ** 2, 0],
                           [0, 1, 0, self.dt, 0, 0.5 * self.dt ** 2],
                           [0, 0, 1, 0, self.dt, 0],
                           [0, 0, 0, 1, 0, self.dt],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]], dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 1000
        self.Q = np.eye(6, dtype=np.float32) * 0.1
        self.R = np.eye(2, dtype=np.float32) * 5

    def predict(self):
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
                 'velocity', 'last_distance', 'world_location', 'world_velocity')

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
        self.world_location = None  # 在CARLA世界中的位置
        self.world_velocity = None  # 在CARLA世界中的速度

    def predict(self):
        self.age += 1
        center = self.kf.predict()

        if self.history:
            self.history.append(center)

            # 平滑处理
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

        # 更新边界框
        self.x1 = smoothed_center[0, 0] - self.width / 2
        self.y1 = smoothed_center[1, 0] - self.height / 2
        self.x2 = self.x1 + self.width
        self.y2 = self.y1 + self.height

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
                track.predict()
                track.age += 1
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            if self.tracks and self.min_hits > 0:
                return np.array([[t.x1, t.y1, t.x2, t.y2, t.id]
                                 for t in self.tracks if t.hits >= self.min_hits])
            return np.array([])

        # 先预测所有track
        for track in self.tracks:
            track.predict()

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

        for track_idx in unmatched_tracks:
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

        return inter_area / (b1_area[:, None] + b2_area[None, :] - inter_area + 1e-6)

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


# -------------------------- 运动控制器 --------------------------
class VehicleController:
    """车辆运动控制器"""

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.max_speed = 50.0  # 最大速度 km/h
        self.target_speed = 30.0  # 目标速度 km/h
        self.obstacle_distance = 100.0  # 安全距离
        self.last_update = time.time()
        self.control_state = {
            'throttle': 0.0,
            'steer': 0.0,
            'brake': 0.0,
            'reverse': False
        }

    def update_control(self, detected_obstacles):
        """根据检测到的障碍物更新控制"""
        control = carla.VehicleControl()

        # 获取当前速度
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6  # m/s to km/h

        # 基本PID控制器
        speed_error = self.target_speed - speed
        throttle = np.clip(speed_error * 0.01, 0.0, 0.8)

        # 检查障碍物
        brake = 0.0
        if detected_obstacles:
            closest_distance = min([d for d in detected_obstacles if d < 50])
            if closest_distance < 15:  # 紧急制动
                throttle = 0.0
                brake = 1.0
            elif closest_distance < 30:  # 减速
                throttle = np.clip(throttle * 0.3, 0.0, 0.3)
                brake = 0.3

        # 随机转向模拟真实驾驶
        if random.random() < 0.05:  # 5%概率微调方向
            steer = random.uniform(-0.1, 0.1)
        else:
            steer = 0.0

        control.throttle = throttle
        control.steer = steer
        control.brake = brake
        control.hand_brake = False
        control.reverse = False

        self.control_state = {
            'throttle': throttle,
            'steer': steer,
            'brake': brake,
            'speed': speed
        }

        return control

    def set_target_speed(self, speed):
        self.target_speed = np.clip(speed, 0.0, self.max_speed)

    def emergency_stop(self):
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        return control


# -------------------------- NPC车辆运动控制 --------------------------
class NPCManager:
    """NPC车辆管理器"""

    def __init__(self, client):
        self.client = client
        self.traffic_manager = None
        self.npc_vehicles = []

        try:
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_synchronous_mode(True)
            print("交通管理器初始化成功")
        except Exception as e:
            print(f"交通管理器初始化失败: {e}")

    def spawn_npcs(self, world, count=30, ego_vehicle=None):
        """生成NPC车辆并设置自动驾驶"""
        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        if not spawn_points:
            print("没有可用的生成点")
            return []

        # 清理现有NPC
        self.destroy_all_npcs()

        random.shuffle(spawn_points)
        spawned_count = 0

        for spawn_point in spawn_points:
            if spawned_count >= count:
                break

            # 检查是否太靠近主车辆
            if ego_vehicle:
                try:
                    ego_loc = ego_vehicle.get_location()
                    dist = math.sqrt(
                        (spawn_point.location.x - ego_loc.x) ** 2 +
                        (spawn_point.location.y - ego_loc.y) ** 2
                    )
                    if dist < 20.0:  # 离主车辆太近
                        continue
                except:
                    pass

            # 随机选择车辆蓝图
            try:
                vehicle_bp = random.choice(list(bp_lib.filter('vehicle.*')))
            except:
                continue

            # 设置随机颜色
            if vehicle_bp.has_attribute('color'):
                colors = vehicle_bp.get_attribute('color').recommended_values
                if colors:
                    vehicle_bp.set_attribute('color', random.choice(colors))

            # 尝试生成车辆
            npc = world.try_spawn_actor(vehicle_bp, spawn_point)

            if npc:
                # 设置自动驾驶
                if self.traffic_manager:
                    try:
                        npc.set_autopilot(True, self.traffic_manager.get_port())

                        # 设置个性化参数
                        self.traffic_manager.distance_to_leading_vehicle(
                            npc, random.uniform(2.0, 5.0)
                        )
                        self.traffic_manager.vehicle_percentage_speed_difference(
                            npc, random.uniform(-30.0, 30.0)
                        )
                        # 不同版本的CARLA API可能没有set_lane_change_permission
                        try:
                            self.traffic_manager.set_lane_change_permission(npc, False)
                        except AttributeError:
                            # 旧版本API，忽略这个设置
                            pass
                    except Exception as e:
                        print(f"设置NPC自动驾驶失败: {e}")

                self.npc_vehicles.append(npc)
                spawned_count += 1
                print(f"生成NPC {spawned_count}/{count}")

        print(f"总共生成 {len(self.npc_vehicles)} 辆NPC车辆")
        return self.npc_vehicles

    def update_npc_behavior(self):
        """更新NPC行为"""
        for npc in self.npc_vehicles:
            try:
                # 随机更新NPC的速度差异，增加变化性
                if random.random() < 0.01 and self.traffic_manager:  # 1%概率
                    self.traffic_manager.vehicle_percentage_speed_difference(
                        npc, random.uniform(-40.0, 40.0)
                    )
            except:
                pass

    def destroy_all_npcs(self):
        """销毁所有NPC车辆"""
        for npc in self.npc_vehicles:
            try:
                if npc.is_alive:
                    npc.destroy()
            except:
                pass
        self.npc_vehicles.clear()


# -------------------------- 性能监控面板函数 --------------------------
def draw_performance_panel(image, timings, fps, frame_count):
    """
    在图像上绘制性能监控面板
    """
    h, w = image.shape[:2]

    def get_avg_time(key, default=0.0):
        if key in timings and timings[key]:
            recent = timings[key][-10:]
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
    panel_x = 10
    panel_y = 10
    panel_width = 280
    panel_height = 200

    # 创建半透明面板背景
    panel_bg = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    panel_bg[:] = (20, 20, 20)

    # 叠加面板到图像上
    x1, y1 = panel_x, panel_y
    x2, y2 = panel_x + panel_width, panel_y + panel_height

    if x2 <= w and y2 <= h:
        alpha = 0.7
        image[y1:y2, x1:x2] = cv2.addWeighted(
            image[y1:y2, x1:x2], 1 - alpha, panel_bg, alpha, 0
        )

        # 绘制标题
        title = "性能监控面板"
        cv2.putText(image, title, (panel_x + 10, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.line(image, (panel_x, panel_y + 25),
                 (panel_x + panel_width, panel_y + 25), (100, 100, 100), 1)

        # 绘制性能指标
        y_offset = 45
        line_height = 20

        cv2.putText(image, f"FPS: {fps:.1f}", (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, f"Frame: {frame_count}", (panel_x + 120, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        y_offset += line_height

        color = (0, 255, 0) if carla_time < 5.0 else (0, 165, 255) if carla_time < 10.0 else (0, 0, 255)
        cv2.putText(image, f"CARLA: {carla_time:.1f}ms", (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        color = (0, 255, 0) if image_time < 1.0 else (0, 165, 255) if image_time < 3.0 else (0, 0, 255)
        cv2.putText(image, f"Image: {image_time:.1f}ms", (panel_x + 120, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y_offset += line_height

        if depth_time > 0:
            color = (0, 255, 0) if depth_time < 2.0 else (0, 165, 255) if depth_time < 5.0 else (0, 0, 255)
            cv2.putText(image, f"Depth: {depth_time:.1f}ms", (panel_x + 10, panel_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y_offset += line_height

        color = (0, 255, 0) if detection_time < 10.0 else (0, 165, 255) if detection_time < 20.0 else (0, 0, 255)
        cv2.putText(image, f"Detection: {detection_time:.1f}ms", (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y_offset += line_height

        color = (0, 255, 0) if tracking_time < 5.0 else (0, 165, 255) if tracking_time < 10.0 else (0, 0, 255)
        cv2.putText(image, f"Tracking: {tracking_time:.1f}ms", (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        color = (0, 255, 0) if drawing_time < 2.0 else (0, 165, 255) if drawing_time < 5.0 else (0, 0, 255)
        cv2.putText(image, f"Drawing: {drawing_time:.1f}ms", (panel_x + 120, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y_offset += line_height

        color = (0, 255, 0) if display_time < 2.0 else (0, 165, 255) if display_time < 5.0 else (0, 0, 255)
        cv2.putText(image, f"Display: {display_time:.1f}ms", (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y_offset += line_height

        color = (0, 255, 0) if total_time < 50.0 else (0, 165, 255) if total_time < 100.0 else (0, 0, 255)
        cv2.putText(image, f"Total: {total_time:.1f}ms", (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image


# -------------------------- 核心工具函数 --------------------------
def draw_bounding_boxes(image, boxes, labels, class_names, **kwargs):
    track_ids = kwargs.get('track_ids')
    probs = kwargs.get('probs')
    distances = kwargs.get('distances')
    velocities = kwargs.get('velocities')

    result = image.copy()
    h, w = image.shape[:2]

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

    if depth_image.dtype == np.float16:
        depth_image = depth_image.astype(np.float32)

    depth_image = np.clip(depth_image, 0.1, 200.0)

    if depth_image.shape[0] > 3 and depth_image.shape[1] > 3:
        depth_image = cv2.GaussianBlur(depth_image, (3, 3), 0.5)

    max_val = np.max(depth_image)
    if max_val > 0:
        depth_image = np.power(depth_image / max_val, 0.7) * max_val

    return depth_image


def get_target_distance(depth_image, box, use_median=True):
    if depth_image is None:
        return 50.0

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


# -------------------------- CARLA相关函数 --------------------------
def setup_carla_client(host='localhost', port=2000):
    """连接CARLA服务器"""
    try:
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        world = client.get_world()

        # 设置同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        print(f"连接到CARLA服务器 {host}:{port}")
        print(f"地图: {world.get_map().name}")
        print(f"同步模式: {settings.synchronous_mode}")
        print(f"时间步长: {settings.fixed_delta_seconds}")

        return world, client
    except Exception as e:
        print(f"连接CARLA失败: {e}")
        raise


def spawn_ego_vehicle(world, enable_physics=True):
    """生成主车辆"""
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
        try:
            vehicle_bp = random.choice(list(bp_lib.filter('vehicle.*')))
        except:
            print("错误：没有找到可用的车辆蓝图")
            return None

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("错误：没有可用的生成点")
        return None

    # 选择远离其他车辆的生成点
    random.shuffle(spawn_points)

    for spawn_point in spawn_points[:10]:
        # 检查是否有其他车辆
        too_close = False
        for actor in world.get_actors().filter('vehicle.*'):
            try:
                actor_loc = actor.get_location()
                dist = math.hypot(
                    actor_loc.x - spawn_point.location.x,
                    actor_loc.y - spawn_point.location.y
                )
                if dist < 10.0:
                    too_close = True
                    break
            except:
                continue

        if not too_close:
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                if vehicle:
                    print(f"主车辆生成成功: {vehicle_bp.id}")
                    print(f"位置: X={spawn_point.location.x:.1f}, Y={spawn_point.location.y:.1f}")

                    # 启用物理模拟
                    vehicle.set_simulate_physics(enable_physics)

                    return vehicle
            except Exception as e:
                print(f"生成车辆失败: {e}")
                continue

    # 如果找不到合适的位置，强制生成
    if spawn_points:
        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
            if vehicle:
                vehicle.set_simulate_physics(enable_physics)
                print(f"主车辆强制生成: {vehicle_bp.id}")
                return vehicle
        except Exception as e:
            print(f"强制生成车辆失败: {e}")

    return None


# 优化内存占用的回调函数
def camera_callback(image, rgb_image_queue):
    """RGB图像回调函数"""
    try:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb = array[..., :3]  # 只存储RGB通道

        if rgb_image_queue.full():
            try:
                rgb_image_queue.get_nowait()
            except:
                pass
        rgb_image_queue.put(rgb)
    except Exception as e:
        print(f"相机回调错误: {e}")


def depth_camera_callback(image, depth_queue):
    """深度图像回调函数"""
    try:
        depth_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        depth_data = depth_data.reshape((image.height, image.width, 4))

        depth_channel = (
                depth_data[..., 2].astype(np.uint16) +
                depth_data[..., 1].astype(np.uint16) * 256 +
                depth_data[..., 0].astype(np.uint16) * 256 ** 2
        )

        depth_in_meters = depth_channel.astype(np.float16) / (256 ** 3 - 1) * 1000.0
        depth_in_meters = preprocess_depth_image(depth_in_meters)

        if depth_queue.full():
            try:
                depth_queue.get_nowait()
            except:
                pass
        depth_queue.put(depth_in_meters)
    except Exception as e:
        print(f"深度相机回调错误: {e}")


# -------------------------- 主函数 --------------------------
def main():
    # 初始化变量
    world = vehicle = camera = depth_camera = None
    image_queue = depth_queue = None
    client = controller = npc_manager = None
    frame_count = 0

    parser = argparse.ArgumentParser(description='CARLA目标检测与跟踪 - 动态车辆版本')
    parser.add_argument('--model', type=str, default='yolov5m',
                        choices=['yolov5s', 'yolov5su', 'yolov5m', 'yolov5mu', 'yolov5x'])
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--conf-thres', type=float, default=0.15)
    parser.add_argument('--iou-thres', type=float, default=0.4)
    parser.add_argument('--use-depth', action='store_true', default=True)
    parser.add_argument('--show-depth', action='store_true')
    parser.add_argument('--npc-count', type=int, default=20)
    parser.add_argument('--enable-physics', action='store_true', default=True,
                        help='启用物理模拟（使车辆可以碰撞）')
    parser.add_argument('--target-speed', type=float, default=30.0,
                        help='目标速度 (km/h)')
    parser.add_argument('--manual-control', action='store_true',
                        help='手动控制模式（使用键盘WASD控制）')
    args = parser.parse_args()

    try:
        # 1. 初始化设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")

        # 2. 连接CARLA
        print("连接CARLA服务器...")
        world, client = setup_carla_client(args.host, args.port)

        # 获取交通管理器
        traffic_manager = None
        try:
            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            print("交通管理器初始化成功")
        except Exception as e:
            print(f"交通管理器初始化失败，NPC可能无法自动驾驶: {e}")

        spectator = world.get_spectator()

        # 3. 清理环境
        print("清理环境...")
        try:
            for actor in world.get_actors().filter('vehicle.*'):
                try:
                    if actor.is_alive:
                        actor.destroy()
                except:
                    pass
        except:
            pass
        time.sleep(1)

        # 4. 生成主车辆
        print("生成主车辆...")
        vehicle = spawn_ego_vehicle(world, enable_physics=args.enable_physics)
        if not vehicle:
            print("主车辆生成失败，程序退出！")
            return

        # 初始化车辆控制器
        controller = VehicleController(vehicle)
        controller.set_target_speed(args.target_speed)

        if not args.manual_control and traffic_manager:
            try:
                # 设置自动驾驶（用于基础运动）
                vehicle.set_autopilot(True, traffic_manager.get_port())
                traffic_manager.distance_to_leading_vehicle(vehicle, 2.5)
                traffic_manager.vehicle_percentage_speed_difference(vehicle, 0.0)
                print("主车辆自动驾驶已启用")
            except Exception as e:
                print(f"主车辆自动驾驶设置失败: {e}")

        try:
            ego_location = vehicle.get_location()
            print(f"主车辆位置: X={ego_location.x:.1f}, Y={ego_location.y:.1f}, Z={ego_location.z:.1f}")
        except:
            print("无法获取主车辆位置")

        # 5. 生成传感器
        print("生成相机传感器...")
        bp_lib = world.get_blueprint_library()

        # RGB相机
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.05')

        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.8),
                                           carla.Rotation(pitch=-5, yaw=0))
        try:
            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            image_queue = queue.Queue(maxsize=3)
            camera.listen(lambda image: camera_callback(image, image_queue))
            print("RGB相机传感器生成成功！")
        except Exception as e:
            print(f"RGB相机生成失败: {e}")
            return

        # 深度相机
        depth_camera = None
        depth_queue = None
        if args.use_depth:
            try:
                depth_bp = bp_lib.find('sensor.camera.depth')
                depth_bp.set_attribute('image_size_x', '800')
                depth_bp.set_attribute('image_size_y', '600')
                depth_bp.set_attribute('fov', '90')
                depth_bp.set_attribute('sensor_tick', '0.05')

                depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
                depth_queue = queue.Queue(maxsize=3)
                depth_camera.listen(lambda image: depth_camera_callback(image, depth_queue))
                print("深度相机传感器生成成功！")
            except Exception as e:
                print(f"深度相机生成失败: {e}")

        # 6. 生成NPC车辆
        print(f"生成 {args.npc_count} 辆NPC车辆...")
        npc_manager = NPCManager(client)
        npc_vehicles = npc_manager.spawn_npcs(world, count=args.npc_count, ego_vehicle=vehicle)

        if len(npc_vehicles) < args.npc_count // 2:
            print(f"警告：只成功生成了 {len(npc_vehicles)} 辆NPC车辆")

        # 等待NPC车辆稳定
        print("等待NPC车辆初始化...")
        for _ in range(10):
            world.tick()

        # 7. 加载检测模型和跟踪器
        print("加载检测模型和跟踪器...")
        try:
            model, class_names = load_detection_model(args.model)
        except Exception as e:
            print(f"模型加载失败: {e}")
            return

        tracker = Sort()

        # 8. 主循环
        print("\n开始目标检测与跟踪")
        print("=" * 50)
        if args.manual_control:
            print("手动控制模式：使用WASD控制车辆")
            print("W: 加速, S: 刹车/倒车, A: 左转, D: 右转")
            print("Q: 退出程序, R: 重新生成NPC")
        else:
            print("自动控制模式：车辆将自动行驶")
            print("按 'q' 键退出程序，按 'r' 重新生成NPC")

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

        # 手动控制变量
        manual_controls = {
            'throttle': 0.0,
            'brake': 0.0,
            'steer': 0.0,
            'reverse': False
        }

        while True:
            try:
                frame_start = time.time()
                frame_count += 1
                detection_stats['total_frames'] += 1

                # 同步CARLA世界
                tick_start = time.time()
                world.tick()
                timings['carla_tick'].append(time.time() - tick_start)

                # 更新NPC行为
                if npc_manager:
                    npc_manager.update_npc_behavior()

                # 移动视角跟随主车辆
                try:
                    ego_transform = vehicle.get_transform()
                    spectator_transform = carla.Transform(
                        ego_transform.transform(carla.Location(x=-10, z=12)),
                        carla.Rotation(yaw=ego_transform.rotation.yaw - 180, pitch=-30)
                    )
                    spectator.set_transform(spectator_transform)
                except:
                    pass

                # 手动控制
                if args.manual_control:
                    # 读取键盘输入
                    key = cv2.waitKey(1) & 0xFF

                    # 控制逻辑
                    if key == ord('w'):  # 加速
                        manual_controls['throttle'] = min(manual_controls['throttle'] + 0.1, 1.0)
                        manual_controls['brake'] = 0.0
                    elif key == ord('s'):  # 刹车/倒车
                        manual_controls['brake'] = min(manual_controls['brake'] + 0.1, 1.0)
                        manual_controls['throttle'] = 0.0
                    elif key == ord('a'):  # 左转
                        manual_controls['steer'] = max(manual_controls['steer'] - 0.1, -1.0)
                    elif key == ord('d'):  # 右转
                        manual_controls['steer'] = min(manual_controls['steer'] + 0.1, 1.0)
                    else:
                        # 逐渐回正转向
                        if manual_controls['steer'] > 0:
                            manual_controls['steer'] = max(manual_controls['steer'] - 0.05, 0)
                        elif manual_controls['steer'] < 0:
                            manual_controls['steer'] = min(manual_controls['steer'] + 0.05, 0)

                        # 逐渐减速
                        manual_controls['throttle'] = max(manual_controls['throttle'] - 0.05, 0)
                        manual_controls['brake'] = max(manual_controls['brake'] - 0.05, 0)

                    # 应用手动控制
                    control = carla.VehicleControl()
                    control.throttle = manual_controls['throttle']
                    control.brake = manual_controls['brake']
                    control.steer = manual_controls['steer']
                    control.hand_brake = False
                    control.reverse = manual_controls['reverse']
                    try:
                        vehicle.apply_control(control)
                    except:
                        pass

                # 获取图像
                image_start = time.time()
                if image_queue.empty():
                    time.sleep(0.001)
                    continue

                origin_image = image_queue.get()
                image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
                height, width, _ = image.shape
                timings['image_get'].append(time.time() - image_start)

                # 获取深度图像
                depth_start = time.time()
                depth_image = None
                if args.use_depth and depth_queue and not depth_queue.empty():
                    depth_image = depth_queue.get()

                    if args.show_depth:
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
                            if cls in [2, 3, 5, 7]:  # 车辆类别
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

                            # 自动控制模式下的障碍物检测
                            if not args.manual_control and track_distances:
                                # 找到最近的车
                                valid_distances = [d for d in track_distances if d is not None]
                                if valid_distances:
                                    closest_distance = min(valid_distances)
                                    if closest_distance < 20:  # 20米内有障碍物
                                        # 可以在这里添加减速逻辑
                                        pass

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
                    "Press 'q' to quit, 'r' to reset NPCs"
                ]

                # 调整信息位置
                y_pos = 220
                for line in info:
                    cv2.putText(image, line, (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_pos += 25

                # 显示手动控制状态
                if args.manual_control:
                    try:
                        velocity = vehicle.get_velocity()
                        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6  # m/s to km/h
                        control_info = [
                            f"Speed: {speed:.1f} km/h",
                            f"Throttle: {manual_controls['throttle']:.1f}",
                            f"Brake: {manual_controls['brake']:.1f}",
                            f"Steer: {manual_controls['steer']:.1f}"
                        ]
                        for i, line in enumerate(control_info):
                            cv2.putText(image, line, (width - 200, 60 + i * 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
                    except:
                        pass

                # 显示结果
                display_start = time.time()
                window_name = f'CARLA Detection & Tracking - {"Manual" if args.manual_control else "Auto"}'
                cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                timings['display'].append(time.time() - display_start)

                # 每100帧打印一次性能统计
                if frame_count % 100 == 0 and frame_count > 0:
                    print(
                        f"\n[帧数 {frame_count}] FPS: {fps:.1f} | 检测到车辆: {len(boxes)} | 跟踪目标: {len(tracker.tracks)}")

                # 退出检查
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户触发退出程序...")
                    break
                elif key == ord('r'):
                    # 重新生成NPC
                    print("重新生成NPC车辆...")
                    if npc_manager:
                        npc_manager.destroy_all_npcs()
                        npc_vehicles = npc_manager.spawn_npcs(world, count=args.npc_count, ego_vehicle=vehicle)
                        print(f"重新生成 {len(npc_vehicles)} 辆NPC车辆完成")

                # FPS控制
                elapsed = time.time() - frame_start
                if elapsed < 0.05:
                    time.sleep(max(0, 0.05 - elapsed))

            except Exception as e:
                print(f"主循环出错: {e}")
                import traceback
                traceback.print_exc()
                break

    except KeyboardInterrupt:
        print("\n用户中断程序...")
    except Exception as e:
        import traceback
        print(f"程序运行出错：{str(e)}")
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n正在清理资源...")

        # 停止传感器
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

        # 销毁NPC
        if npc_manager:
            npc_manager.destroy_all_npcs()

        # 销毁主车辆
        if vehicle:
            try:
                vehicle.destroy()
            except:
                pass

        # 恢复世界设置
        if world:
            try:
                settings = world.get_settings()
                settings.synchronous_mode = False
                world.apply_settings(settings)
            except:
                pass

        cv2.destroyAllWindows()

        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 打印最终统计
        print("\n" + "=" * 50)
        print("程序运行统计：")
        print(f"总帧数: {frame_count}")

        if timings.get('total'):
            try:
                avg_fps = 1.0 / np.mean(list(timings['total'])) if timings['total'] else 0
                print(f"平均FPS: {avg_fps:.1f}")
            except:
                pass

        if detection_stats:
            print(f"总检测次数: {detection_stats['total_detections']}")
            print(f"平均每帧检测: {detection_stats['total_detections'] / max(1, detection_stats['total_frames']):.1f}")
            print(f"最大单帧车辆数: {detection_stats['max_vehicles_per_frame']}")
        print("=" * 50)
        print("资源清理完成，程序正常退出！")


if __name__ == "__main__":
    main()