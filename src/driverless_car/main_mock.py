import os
import random
import argparse
from dataclasses import dataclass

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO


ALLOWED_CLASSES = {
    0: {"name": "person", "label": "Person", "real_height_m": 1.7},
    2: {"name": "car", "label": "Car", "real_height_m": 1.6},
}
ALLOWED_CLASS_IDS = list(ALLOWED_CLASSES.keys())

RISK_COLORS = {
    "danger": (0, 0, 255),
    "warning": (0, 255, 255),
    "safe": (0, 255, 0),
}

DRAW_OBJECT_COLORS = {
    0: (80, 80, 240),
    1: (70, 190, 255),
    2: (255, 160, 60),
}

DEFAULT_EFFECT_IMAGE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "docs",
        "driverless_car",
        "camera_radar_fusion_demo.png",
    )
)


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
    real_height_m = ALLOWED_CLASSES.get(cls_id, ALLOWED_CLASSES[2])["real_height_m"]
    return (real_height_m * focal_length_px) / pixel_height


def risk_level_by_distance(distance_m):
    if distance_m < 8.0:
        return "danger", "Danger"
    if distance_m < 16.0:
        return "warning", "Warning"
    return "safe", "Safe"


def draw_label(frame, text, x1, y1, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    frame_w = frame.shape[1]
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y1 - th - baseline - 10)
    left = int(np.clip(x1, 0, max(frame_w - tw - 8, 0)))
    cv2.rectangle(frame, (left, top), (left + tw + 8, top + th + baseline + 8), color, -1)
    cv2.putText(frame, text, (left + 4, top + th + 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


@dataclass
class Detection:
    bbox: np.ndarray
    score: float
    cls_id: int


@dataclass
class RadarMeasurement:
    position: np.ndarray
    distance_m: float
    velocity_mps: float


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
        matched, unmatched_det, unmatched_trk = self._associate_detections_to_trackers(
            detections, predicted_boxes
        )

        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx])

        for det_idx in unmatched_det:
            self.tracks.append(SortTrack(detections[det_idx]))

        alive_tracks = []
        outputs = []
        for idx, track in enumerate(self.tracks):
            if idx in unmatched_trk:
                pass
            if track.time_since_update <= self.max_age:
                alive_tracks.append(track)
            if track.time_since_update == 0 and (
                track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
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


class VirtualEnv:
    def __init__(self, width=1024, height=768):
        self.width = width
        self.height = height
        self.objects = []
        self._init_objects()

    def _init_objects(self):
        preset_classes = [2, 0, 2, 0]
        for cls_id in preset_classes:
            distance_m = random.uniform(6.0, 22.0)
            self.objects.append(
                {
                    "cls_id": cls_id,
                    "cx": random.uniform(140, self.width - 140),
                    "cy": random.uniform(120, self.height - 120),
                    "vx": random.uniform(-2.0, 2.0),
                    "vy": random.uniform(-1.6, 1.6),
                    "distance_m": distance_m,
                    "distance_v": random.uniform(-0.08, 0.08),
                }
            )

    def _size_from_distance(self, cls_id, distance_m):
        base = ALLOWED_CLASSES[cls_id]["real_height_m"]
        focal_length_px = self.height * 1.15
        box_h = max(28.0, (base * focal_length_px) / max(distance_m, 1.0))
        if cls_id == 2:
            box_w = box_h * 1.45
        elif cls_id == 1:
            box_w = box_h * 0.85
        else:
            box_w = box_h * 0.45
        return box_w, box_h

    def update(self):
        for obj in self.objects:
            obj["cx"] += obj["vx"]
            obj["cy"] += obj["vy"]
            obj["distance_m"] = float(np.clip(obj["distance_m"] + obj["distance_v"], 4.5, 26.0))
            if obj["cx"] < 90 or obj["cx"] > self.width - 90:
                obj["vx"] *= -1
            if obj["cy"] < 90 or obj["cy"] > self.height - 90:
                obj["vy"] *= -1
            obj["vx"] += random.uniform(-0.12, 0.12)
            obj["vy"] += random.uniform(-0.08, 0.08)
            obj["vx"] = float(np.clip(obj["vx"], -2.8, 2.8))
            obj["vy"] = float(np.clip(obj["vy"], -2.2, 2.2))

    def get_mock_detections(self):
        detections = []
        for obj in self.objects:
            box_w, box_h = self._size_from_distance(obj["cls_id"], obj["distance_m"])
            jitter_x = random.uniform(-8.0, 8.0)
            jitter_y = random.uniform(-7.0, 7.0)
            x1 = obj["cx"] - box_w / 2.0 + jitter_x
            y1 = obj["cy"] - box_h / 2.0 + jitter_y
            x2 = obj["cx"] + box_w / 2.0 + jitter_x
            y2 = obj["cy"] + box_h / 2.0 + jitter_y
            detections.append(
                Detection(
                    bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                    score=round(random.uniform(0.78, 0.97), 2),
                    cls_id=obj["cls_id"],
                )
            )
        return detections

    def get_mock_radar_measurements(self):
        radar_points = []
        ego_center_x = self.width / 2.0
        for obj in self.objects:
            radar_x = obj["cx"] + random.uniform(-10.0, 10.0)
            radar_y = obj["cy"] + random.uniform(-8.0, 8.0)
            lateral_offset = (radar_x - ego_center_x) / max(self.width / 2.0, 1.0)
            radar_distance = obj["distance_m"] + random.uniform(-0.35, 0.35)
            relative_speed = -(obj["distance_v"] * 12.0)
            radar_points.append(
                RadarMeasurement(
                    position=np.array([radar_x, radar_y], dtype=np.float32),
                    distance_m=max(0.5, radar_distance),
                    velocity_mps=relative_speed + lateral_offset * 0.6,
                )
            )
        return radar_points

    def render(self):
        frame = np.full((self.height, self.width, 3), 24, dtype=np.uint8)
        cv2.rectangle(frame, (0, self.height // 2), (self.width, self.height), (45, 45, 45), -1)
        lane_center = self.width // 2
        cv2.line(frame, (lane_center, self.height // 2), (lane_center, self.height), (0, 220, 255), 4)
        cv2.line(frame, (lane_center - 180, self.height), (lane_center - 60, self.height // 2), (180, 180, 180), 2)
        cv2.line(frame, (lane_center + 180, self.height), (lane_center + 60, self.height // 2), (180, 180, 180), 2)

        for obj in self.objects:
            box_w, box_h = self._size_from_distance(obj["cls_id"], obj["distance_m"])
            x1 = int(obj["cx"] - box_w / 2.0)
            y1 = int(obj["cy"] - box_h / 2.0)
            x2 = int(obj["cx"] + box_w / 2.0)
            y2 = int(obj["cy"] + box_h / 2.0)
            color = DRAW_OBJECT_COLORS[obj["cls_id"]]
            if obj["cls_id"] == 0:
                cv2.circle(frame, (int(obj["cx"]), y1 + 12), 10, color, -1)
                cv2.rectangle(frame, (int(obj["cx"]) - 8, y1 + 24), (int(obj["cx"]) + 8, y2), color, -1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(frame, (x1 + 10, y1 + 8), (x2 - 10, y1 + 24), (255, 255, 255), -1)
        return frame


class PerceptionDemo:
    def __init__(self, model_path="yolov8n.pt", load_model=False):
        self.model_path = model_path
        self.detector = YOLO(model_path) if load_model else None
        self.sort_tracker = SortTracker()

    def detect_with_yolo(self, frame):
        if self.detector is None:
            self.detector = YOLO(self.model_path)
        results = self.detector(frame, classes=ALLOWED_CLASS_IDS, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in ALLOWED_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].item())
                detections.append(
                    Detection(
                        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                        score=score,
                        cls_id=cls_id,
                    )
                )
        return detections

    def fuse_radar_with_tracks(self, tracks, radar_measurements):
        fused_results = []
        used_radar = set()
        for track in tracks:
            box = track.current_bbox()
            x1, y1, x2, y2 = box
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
            diagonal = max(np.linalg.norm([x2 - x1, y2 - y1]), 1.0)
            best_idx = None
            best_score = float("inf")
            for idx, radar in enumerate(radar_measurements):
                if idx in used_radar:
                    continue
                score = np.linalg.norm(center - radar.position)
                if score < best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None and best_score <= diagonal * 0.8:
                used_radar.add(best_idx)
                fused_results.append((track, radar_measurements[best_idx]))
            else:
                fused_results.append((track, None))
        return fused_results

    def annotate_tracks(self, frame, fused_tracks):
        frame_h, frame_w = frame.shape[:2]
        for track, radar in fused_tracks:
            box = clamp_box(track.current_bbox(), frame_w, frame_h)
            camera_distance = estimate_distance_meters(box, track.cls_id, frame_h)
            distance_m = radar.distance_m if radar is not None else camera_distance
            track.last_distance = distance_m
            risk_key, risk_text = risk_level_by_distance(distance_m)
            color = RISK_COLORS[risk_key]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            info = ALLOWED_CLASSES[track.cls_id]
            range_source = "Radar" if radar is not None else "Camera"
            label = (
                f"ID {track.track_id} {info['label']} "
                f"{range_source} {distance_m:.1f}m {risk_text} conf {track.score:.2f}"
            )
            draw_label(frame, label, x1, y1, color)
            if radar is not None:
                radar_point = tuple(radar.position.astype(int))
                box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.circle(frame, radar_point, 6, (255, 255, 0), -1)
                cv2.line(frame, radar_point, box_center, (255, 255, 0), 2)
                radar_text = f"Speed {radar.velocity_mps:+.1f}m/s"
                cv2.putText(
                    frame,
                    radar_text,
                    (x1, min(frame_h - 10, y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        legend_items = [
            "Camera: detect person/car",
            "Radar: measure target distance",
            "Fusion: overlay radar range on camera box",
        ]
        for idx, text in enumerate(legend_items):
            cv2.putText(
                frame,
                text,
                (22, 34 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return frame

    def process_frame(self, frame, detections=None, radar_measurements=None):
        if detections is None:
            detections = self.detect_with_yolo(frame)
        if radar_measurements is None:
            radar_measurements = []
        tracks = self.sort_tracker.update(detections)
        fused_tracks = self.fuse_radar_with_tracks(tracks, radar_measurements)
        return self.annotate_tracks(frame.copy(), fused_tracks)

def generate_effect_image(output_path=None, num_frames=18, seed=7):
    random.seed(seed)
    np.random.seed(seed)
    env = VirtualEnv()
    demo = PerceptionDemo(load_model=False)
    final_frame = None
    for _ in range(num_frames):
        env.update()
        frame = env.render()
        detections = env.get_mock_detections()
        radar_measurements = env.get_mock_radar_measurements()
        final_frame = demo.process_frame(
            frame,
            detections=detections,
            radar_measurements=radar_measurements,
        )

    if output_path is None:
        output_path = DEFAULT_EFFECT_IMAGE_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final_frame)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Driverless car camera-radar fusion mock demo")
    parser.add_argument("--save-only", action="store_true", help="Only export the effect image without opening the demo window")
    parser.add_argument("--output", type=str, default=None, help="Custom output image path")
    parser.add_argument("--frames", type=int, default=18, help="Warm-up frames before saving the effect image")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used for deterministic demo output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.save_only:
        output_path = generate_effect_image(output_path=args.output, num_frames=args.frames, seed=args.seed)
        print(f"效果图已保存: {output_path}")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    env = VirtualEnv()
    demo = PerceptionDemo(load_model=False)
    cv2.namedWindow("Driverless Car Camera Radar Fusion Demo", cv2.WINDOW_NORMAL)

    while True:
        env.update()
        frame = env.render()
        detections = env.get_mock_detections()
        radar_measurements = env.get_mock_radar_measurements()
        annotated = demo.process_frame(
            frame,
            detections=detections,
            radar_measurements=radar_measurements,
        )
        cv2.imshow("Driverless Car Camera Radar Fusion Demo", annotated)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    output_path = generate_effect_image(output_path=args.output, num_frames=args.frames, seed=args.seed)
    print(f"效果图已保存: {output_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
