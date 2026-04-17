import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


CLASS_INFO = {
    0: {"name": "person", "label": "Pedestrian", "real_height_m": 1.70, "color": (80, 120, 255)},
    1: {"name": "bicycle", "label": "Bicycle", "real_height_m": 1.30, "color": (80, 200, 255)},
    2: {"name": "car", "label": "Car", "real_height_m": 1.55, "color": (255, 170, 80)},
    3: {"name": "school_bus", "label": "SchoolBus", "real_height_m": 3.10, "color": (0, 210, 255)},
    4: {"name": "ambulance", "label": "Ambulance", "real_height_m": 2.80, "color": (80, 80, 255)},
    5: {"name": "fire_truck", "label": "FireTruck", "real_height_m": 3.20, "color": (60, 60, 240)},
}

TRAFFIC_MARK_INFO = {
    6: {"name": "stop_sign", "label": "StopSign"},
    7: {"name": "traffic_light_red", "label": "RedLight"},
}

ALLOWED_CLASS_IDS = list(CLASS_INFO.keys())
SPECIAL_VEHICLE_IDS = {3, 4, 5}

RISK_COLORS = {
    "danger": (0, 0, 255),
    "warning": (0, 255, 255),
    "safe": (0, 255, 0),
}


def iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def xyxy_to_z(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    scale = w * h
    ratio = w / max(h, 1e-6)
    return np.array([cx, cy, scale, ratio], dtype=np.float32).reshape(4, 1)


def x_to_xyxy(state):
    cx, cy, scale, ratio = state[:4].reshape(-1)
    scale = max(scale, 1.0)
    ratio = max(ratio, 1e-3)
    w = np.sqrt(scale * ratio)
    h = scale / max(w, 1e-6)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y2 = int(np.clip(y2, 0, height - 1))
    return x1, y1, x2, y2


def estimate_distance_meters(box, cls_id, frame_height):
    _, y1, _, y2 = box
    pixel_height = max(y2 - y1, 1)
    focal_length_px = frame_height * 1.15
    real_height_m = CLASS_INFO.get(cls_id, CLASS_INFO[2])["real_height_m"]
    return (real_height_m * focal_length_px) / pixel_height


def risk_level_by_distance(distance_m):
    if distance_m < 8.0:
        return "danger", "Danger"
    if distance_m < 16.0:
        return "warning", "Warning"
    return "safe", "Safe"


def draw_label(frame, text, x1, y1, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y1 - th - baseline - 8)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 8, top + th + baseline + 8), color, -1)
    cv2.putText(frame, text, (x1 + 4, top + th + 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_chinese_text(frame, text, position, color=(255, 255, 255), font_size=36):
    font_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    font = None
    for path in font_candidates:
        if os.path.exists(path):
            font = ImageFont.truetype(path, font_size)
            break
    if font is None:
        cv2.putText(frame, text.encode("ascii", "ignore").decode("ascii"), position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


@dataclass
class Detection:
    bbox: np.ndarray
    score: float
    cls_id: int


class SortTrack:
    count = 0

    def __init__(self, detection):
        self.kf = self._create_kalman_filter()
        self.kf.x[:4] = xyxy_to_z(detection.bbox)
        self.track_id = SortTrack.count
        SortTrack.count += 1
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.cls_id = detection.cls_id
        self.score = detection.score
        self.last_distance = None

    @staticmethod
    def _create_kalman_filter():
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        kf.R[2:, 2:] *= 10.0
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        return kf

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_xyxy(self.kf.x)

    def update(self, detection):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.cls_id = detection.cls_id
        self.score = detection.score
        self.kf.update(xyxy_to_z(detection.bbox))

    def current_bbox(self):
        return x_to_xyxy(self.kf.x)


class SortTracker:
    def __init__(self, max_age=8, min_hits=2, iou_threshold=0.2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        predicted_boxes = [track.predict() for track in self.tracks]
        matched, unmatched_det, unmatched_trk = self._associate_detections_to_trackers(detections, predicted_boxes)

        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx])

        for det_idx in unmatched_det:
            self.tracks.append(SortTrack(detections[det_idx]))

        alive_tracks = []
        outputs = []
        for idx, track in enumerate(self.tracks):
            if track.time_since_update <= self.max_age:
                alive_tracks.append(track)
            if track.time_since_update == 0 and (track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                outputs.append(track)
        self.tracks = alive_tracks
        return outputs

    def _associate_detections_to_trackers(self, detections, predicted_boxes):
        if not predicted_boxes:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(predicted_boxes)))

        candidate_pairs = []
        for det_idx, det in enumerate(detections):
            for trk_idx, pred_box in enumerate(predicted_boxes):
                overlap = iou_xyxy(det.bbox, pred_box)
                if overlap >= self.iou_threshold:
                    candidate_pairs.append((overlap, det_idx, trk_idx))
        candidate_pairs.sort(reverse=True, key=lambda item: item[0])

        matched = []
        used_det = set()
        used_trk = set()
        for _, det_idx, trk_idx in candidate_pairs:
            if det_idx in used_det or trk_idx in used_trk:
                continue
            used_det.add(det_idx)
            used_trk.add(trk_idx)
            matched.append((det_idx, trk_idx))

        unmatched_det = [idx for idx in range(len(detections)) if idx not in used_det]
        unmatched_trk = [idx for idx in range(len(predicted_boxes)) if idx not in used_trk]
        return matched, unmatched_det, unmatched_trk


class CourtesyPlanner:
    def __init__(self):
        self.mode = "cruise"
        self.current_speed = 6.0
        self.target_speed = 6.0
        self.lane_offset_px = 0
        self._pedestrian_hold_frames = 0

    def update(self, fused_tracks, scene_meta):
        ped_on_crosswalk = False
        special_vehicle_near = False
        nearest_special_name = None
        special_distance = 1e9

        crosswalk = scene_meta["crosswalk_rect"]
        for track, distance_m in fused_tracks:
            cls_id = track.cls_id
            if cls_id not in CLASS_INFO:
                continue
            x1, y1, x2, y2 = track.current_bbox()
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5

            if cls_id == 0:
                in_crosswalk = (
                    crosswalk[0] <= cx <= crosswalk[2]
                    and crosswalk[1] <= cy <= crosswalk[3]
                    and distance_m < 25.0
                )
                if in_crosswalk:
                    ped_on_crosswalk = True

            if cls_id in SPECIAL_VEHICLE_IDS and distance_m < 35.0:
                special_vehicle_near = True
                if distance_m < special_distance:
                    special_distance = distance_m
                    nearest_special_name = CLASS_INFO[cls_id]["label"]

        red_light_active = scene_meta["red_light_active"]
        stop_sign_active = scene_meta["stop_sign_active"]

        message = "正常巡航"
        if ped_on_crosswalk:
            self.mode = "yield_pedestrian"
            self._pedestrian_hold_frames = 18
            self.target_speed = 0.0
            self.lane_offset_px = 0
            message = "检测到行人，正在礼让"
        elif self._pedestrian_hold_frames > 0:
            self.mode = "yield_pedestrian"
            self._pedestrian_hold_frames -= 1
            self.target_speed = 0.0
            self.lane_offset_px = 0
            message = "行人通过中，继续停车礼让"
        elif red_light_active:
            self.mode = "traffic_stop"
            self.target_speed = 0.0
            self.lane_offset_px = 0
            message = "检测到前方红灯，平稳停车"
        elif stop_sign_active:
            self.mode = "traffic_stop"
            self.target_speed = 0.0
            self.lane_offset_px = 0
            message = "检测到停止标志，平稳停车"
        elif special_vehicle_near:
            self.mode = "yield_special"
            self.target_speed = 2.0
            self.lane_offset_px = 48
            message = f"检测到{nearest_special_name}，正在靠边减速避让"
        else:
            self.mode = "cruise"
            self.target_speed = 6.0
            self.lane_offset_px = 0

        delta = self.target_speed - self.current_speed
        max_step = 0.18
        delta = float(np.clip(delta, -max_step, max_step))
        self.current_speed = float(np.clip(self.current_speed + delta, 0.0, 8.0))
        return {
            "mode": self.mode,
            "message": message,
            "speed_mps": self.current_speed,
            "target_speed_mps": self.target_speed,
            "lane_offset_px": self.lane_offset_px,
        }


class VirtualEnv:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.frame_idx = 0
        self.random = random.Random(2026)
        self.objects = []
        self.red_light_active = False
        self.stop_sign_active = False
        self.crosswalk_rect = (
            int(width * 0.33),
            int(height * 0.59),
            int(width * 0.67),
            int(height * 0.68),
        )
        self._init_background_objects()

    def _init_background_objects(self):
        for cls_id in [2, 2, 1]:
            self.objects.append(
                {
                    "cls_id": cls_id,
                    "cx": self.random.uniform(180, self.width - 180),
                    "cy": self.random.uniform(260, self.height - 120),
                    "vx": self.random.uniform(-1.8, 1.8),
                    "vy": self.random.uniform(-0.6, 0.6),
                    "distance_m": self.random.uniform(16.0, 28.0),
                    "distance_v": self.random.uniform(-0.07, 0.07),
                }
            )

    def _size_from_distance(self, cls_id, distance_m):
        base = CLASS_INFO.get(cls_id, CLASS_INFO[2])["real_height_m"]
        focal_length_px = self.height * 1.1
        box_h = max(24.0, (base * focal_length_px) / max(distance_m, 1.0))
        if cls_id in [2, 4, 5]:
            box_w = box_h * 1.4
        elif cls_id == 3:
            box_w = box_h * 1.9
        elif cls_id == 1:
            box_w = box_h * 0.9
        else:
            box_w = box_h * 0.45
        return box_w, box_h

    def _scripted_objects(self):
        scripted = []
        ped_phase = 90 <= self.frame_idx <= 180
        if ped_phase:
            t = (self.frame_idx - 90) / 90.0
            x = int(self.crosswalk_rect[0] + (self.crosswalk_rect[2] - self.crosswalk_rect[0]) * t)
            scripted.append(
                {
                    "cls_id": 0,
                    "cx": x,
                    "cy": int((self.crosswalk_rect[1] + self.crosswalk_rect[3]) * 0.5),
                    "distance_m": 12.0,
                }
            )

        special_phase = 210 <= self.frame_idx <= 300
        if special_phase:
            cls_cycle = [3, 4, 5]
            cls_id = cls_cycle[(self.frame_idx // 30) % len(cls_cycle)]
            scripted.append(
                {
                    "cls_id": cls_id,
                    "cx": int(self.width * 0.55),
                    "cy": int(self.height * 0.54),
                    "distance_m": 18.0,
                }
            )

        self.red_light_active = 320 <= self.frame_idx <= 380
        self.stop_sign_active = 410 <= self.frame_idx <= 460
        return scripted

    def update(self):
        self.frame_idx += 1
        for obj in self.objects:
            obj["cx"] += obj["vx"]
            obj["cy"] += obj["vy"]
            obj["distance_m"] = float(np.clip(obj["distance_m"] + obj["distance_v"], 10.0, 32.0))
            if obj["cx"] < 100 or obj["cx"] > self.width - 100:
                obj["vx"] *= -1
            if obj["cy"] < 220 or obj["cy"] > self.height - 80:
                obj["vy"] *= -1
            obj["vx"] = float(np.clip(obj["vx"] + self.random.uniform(-0.08, 0.08), -2.2, 2.2))
            obj["vy"] = float(np.clip(obj["vy"] + self.random.uniform(-0.05, 0.05), -1.0, 1.0))

    def get_mock_detections(self):
        detections = []
        all_objects = list(self.objects) + self._scripted_objects()
        for obj in all_objects:
            cls_id = obj["cls_id"]
            box_w, box_h = self._size_from_distance(cls_id, obj["distance_m"])
            jitter_x = self.random.uniform(-6.0, 6.0)
            jitter_y = self.random.uniform(-6.0, 6.0)
            x1 = obj["cx"] - box_w / 2.0 + jitter_x
            y1 = obj["cy"] - box_h / 2.0 + jitter_y
            x2 = obj["cx"] + box_w / 2.0 + jitter_x
            y2 = obj["cy"] + box_h / 2.0 + jitter_y
            detections.append(
                Detection(
                    bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                    score=round(self.random.uniform(0.82, 0.98), 2),
                    cls_id=cls_id,
                )
            )

        if self.red_light_active:
            detections.append(
                Detection(
                    bbox=np.array([self.width * 0.70, self.height * 0.23, self.width * 0.74, self.height * 0.30], dtype=np.float32),
                    score=0.99,
                    cls_id=7,
                )
            )

        if self.stop_sign_active:
            detections.append(
                Detection(
                    bbox=np.array([self.width * 0.23, self.height * 0.32, self.width * 0.28, self.height * 0.40], dtype=np.float32),
                    score=0.97,
                    cls_id=6,
                )
            )
        return detections

    def render(self):
        frame = np.full((self.height, self.width, 3), (28, 28, 28), dtype=np.uint8)
        sky_h = int(self.height * 0.42)
        frame[:sky_h, :, :] = (95, 125, 150)
        road_top = int(self.height * 0.46)
        cv2.rectangle(frame, (0, road_top), (self.width, self.height), (48, 48, 48), -1)

        lane_center = self.width // 2
        cv2.line(frame, (lane_center, road_top), (lane_center, self.height), (0, 220, 255), 4)
        for offset in [240, -240]:
            cv2.line(frame, (lane_center + offset, self.height), (lane_center + offset // 2, road_top), (160, 160, 160), 2)

        x1, y1, x2, y2 = self.crosswalk_rect
        stripe_w = max(1, (x2 - x1) // 10)
        for idx in range(10):
            if idx % 2 == 0:
                cv2.rectangle(frame, (x1 + idx * stripe_w, y1), (x1 + (idx + 1) * stripe_w, y2), (240, 240, 240), -1)

        cv2.rectangle(frame, (int(self.width * 0.68), int(self.height * 0.17)), (int(self.width * 0.75), int(self.height * 0.34)), (35, 35, 35), -1)
        if self.red_light_active:
            cv2.circle(frame, (int(self.width * 0.715), int(self.height * 0.23)), 20, (0, 0, 255), -1)
        else:
            cv2.circle(frame, (int(self.width * 0.715), int(self.height * 0.23)), 20, (80, 80, 80), -1)

        if self.stop_sign_active:
            oct_center = (int(self.width * 0.255), int(self.height * 0.36))
            size = 26
            octagon = np.array(
                [
                    [oct_center[0] - size // 2, oct_center[1] - size],
                    [oct_center[0] + size // 2, oct_center[1] - size],
                    [oct_center[0] + size, oct_center[1] - size // 2],
                    [oct_center[0] + size, oct_center[1] + size // 2],
                    [oct_center[0] + size // 2, oct_center[1] + size],
                    [oct_center[0] - size // 2, oct_center[1] + size],
                    [oct_center[0] - size, oct_center[1] + size // 2],
                    [oct_center[0] - size, oct_center[1] - size // 2],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(frame, [octagon], (0, 0, 220))
            cv2.putText(frame, "STOP", (oct_center[0] - 24, oct_center[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        for obj in list(self.objects) + self._scripted_objects():
            cls_id = obj["cls_id"]
            color = CLASS_INFO[cls_id]["color"]
            box_w, box_h = self._size_from_distance(cls_id, obj["distance_m"])
            x1 = int(obj["cx"] - box_w / 2.0)
            y1 = int(obj["cy"] - box_h / 2.0)
            x2 = int(obj["cx"] + box_w / 2.0)
            y2 = int(obj["cy"] + box_h / 2.0)

            if cls_id == 0:
                cv2.circle(frame, (int(obj["cx"]), y1 + 10), 10, color, -1)
                cv2.rectangle(frame, (int(obj["cx"]) - 9, y1 + 22), (int(obj["cx"]) + 9, y2), color, -1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(frame, (x1 + 10, y1 + 8), (x2 - 10, y1 + 24), (235, 235, 235), -1)
                if cls_id == 4:
                    cv2.line(frame, (x1 + 16, y1 + 36), (x2 - 16, y1 + 36), (0, 0, 255), 3)
                if cls_id == 3:
                    cv2.putText(frame, "BUS", (x1 + 12, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        return frame

    def scene_meta(self):
        return {
            "frame_idx": self.frame_idx,
            "crosswalk_rect": self.crosswalk_rect,
            "red_light_active": self.red_light_active,
            "stop_sign_active": self.stop_sign_active,
        }


class PerceptionDemo:
    def __init__(self, model_path="yolov8n.pt", load_model=False):
        self.model_path = model_path
        self.detector = YOLO(model_path) if load_model else None
        self.sort_tracker = SortTracker()
        self.planner = CourtesyPlanner()

    def detect_with_yolo(self, frame):
        if self.detector is None:
            self.detector = YOLO(self.model_path)
        results = self.detector(frame, classes=[0, 1, 2], verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in [0, 1, 2]:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].item())
                detections.append(Detection(bbox=np.array([x1, y1, x2, y2], dtype=np.float32), score=score, cls_id=cls_id))
        return detections

    def annotate_tracks(self, frame, tracks, scene_meta):
        h, w = frame.shape[:2]
        fused_tracks = []
        for track in tracks:
            box = clamp_box(track.current_bbox(), w, h)
            distance_m = estimate_distance_meters(box, track.cls_id, h)
            track.last_distance = distance_m
            fused_tracks.append((track, distance_m))

        control_info = self.planner.update(fused_tracks, scene_meta)

        for track, distance_m in fused_tracks:
            box = clamp_box(track.current_bbox(), w, h)
            x1, y1, x2, y2 = box
            risk_key, risk_text = risk_level_by_distance(distance_m)
            color = RISK_COLORS[risk_key]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            info = CLASS_INFO.get(track.cls_id, {"label": f"Class{track.cls_id}"})
            label = f"ID {track.track_id} {info['label']} {distance_m:.1f}m {risk_text}"
            draw_label(frame, label, x1, y1, color)

        overlay_h = 88
        cv2.rectangle(frame, (0, 0), (w, overlay_h), (20, 20, 20), -1)
        status = f"Mode: {control_info['mode']}   Speed: {control_info['speed_mps']:.2f} m/s   Target: {control_info['target_speed_mps']:.2f} m/s"
        cv2.putText(frame, status, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        if control_info["mode"] in ["yield_pedestrian", "traffic_stop", "yield_special"]:
            frame = draw_chinese_text(frame, control_info["message"], (20, 46), (0, 230, 255), 34)
        else:
            frame = draw_chinese_text(frame, "道路正常，继续巡航", (20, 46), (120, 255, 120), 34)

        ego_x = int(w * 0.5 + control_info["lane_offset_px"])
        ego_y = int(h * 0.86)
        cv2.rectangle(frame, (ego_x - 42, ego_y - 22), (ego_x + 42, ego_y + 22), (255, 255, 255), 2)
        cv2.putText(frame, "EGO", (ego_x - 24, ego_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def process_frame(self, frame, detections=None, scene_meta=None):
        if detections is None:
            detections = self.detect_with_yolo(frame)
        if scene_meta is None:
            scene_meta = {
                "frame_idx": 0,
                "crosswalk_rect": (0, 0, 0, 0),
                "red_light_active": False,
                "stop_sign_active": False,
            }

        track_dets = [d for d in detections if d.cls_id in ALLOWED_CLASS_IDS]
        tracks = self.sort_tracker.update(track_dets)
        return self.annotate_tracks(frame.copy(), tracks, scene_meta)


def generate_effect_image(output_path=None, target_frame=125):
    env = VirtualEnv()
    demo = PerceptionDemo(load_model=False)
    final = None
    for _ in range(target_frame):
        env.update()
        frame = env.render()
        detections = env.get_mock_detections()
        final = demo.process_frame(frame, detections=detections, scene_meta=env.scene_meta())

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "outputs", "courtesy_driving_demo.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final)
    return output_path


def main():
    env = VirtualEnv()
    demo = PerceptionDemo(load_model=False)
    cv2.namedWindow("Driverless Car Courtesy Demo", cv2.WINDOW_NORMAL)

    while True:
        env.update()
        frame = env.render()
        detections = env.get_mock_detections()
        annotated = demo.process_frame(frame, detections=detections, scene_meta=env.scene_meta())
        cv2.imshow("Driverless Car Courtesy Demo", annotated)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    output_path = generate_effect_image()
    print(f"Effect image saved: {output_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
