import cv2
import numpy as np
import argparse
import csv
import glob
import sys
import time
import os
import math
from pathlib import Path
from collections import deque
from yolo_det import ObjectDetector


# ================= 配置区 =================

# --- 车道线检测参数 ---
CANNY_LOW, CANNY_HIGH = 30, 100
ROI_TOP, ROI_HEIGHT = 0.40, 0.60

# --- HLS颜色阈值 ---
WHITE_LOWER = np.array([0, 160, 0], dtype=np.uint8)
WHITE_UPPER = np.array([180, 255, 100], dtype=np.uint8)

YELLOW_LOWER = np.array([15, 80, 80], dtype=np.uint8)
YELLOW_UPPER = np.array([35, 255, 255], dtype=np.uint8)

# --- 车道线平滑 ---
HISTORY_LEN = 10

# --- YOLO检测参数 ---
SKIP_FRAMES = 3

# --- 转向显示参数 ---
STEER_SENSITIVITY = 1.5

# --- 危险判断参数 ---
RISK_WARNING_SCORE = 45.0
RISK_DANGER_SCORE = 70.0

TTC_WARNING = 3.0      # TTC小于3秒，进入预警
TTC_DANGER = 1.5       # TTC小于1.5秒，进入危险

IOU_MATCH_THRESHOLD = 0.25
TRACK_HISTORY_LEN = 8
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
TRACK_MAX_AGE = 1.2    # 秒，超过该时间未匹配则删除目标轨迹


# ============================================================
# 黑匣子：危险事件记录
# ============================================================

class EventLogger:
    """危险事件抓拍"""

    def __init__(self, save_dir="events"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.last_save_time = 0
        self.cooldown = 2.0

    def log_danger(self, frame, obj_name, risk_score=None, ttc=None):
        now = time.time()

        if now - self.last_save_time > self.cooldown:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            safe_name = str(obj_name).replace("/", "_").replace("\\", "_")
            filename = f"{self.save_dir}/danger_{timestamp}_{safe_name}.jpg"

            cv2.imwrite(filename, frame)

            if risk_score is not None and ttc is not None and np.isfinite(ttc):
                print(f"📸 危险已抓拍: {filename} | 对象: {obj_name} | 风险分数: {risk_score:.1f} | TTC: {ttc:.2f}s")
            elif risk_score is not None:
                print(f"📸 危险已抓拍: {filename} | 对象: {obj_name} | 风险分数: {risk_score:.1f}")
            else:
                print(f"📸 危险已抓拍: {filename}")

            self.last_save_time = now
            return True

        return False


# ============================================================
# 车道线检测系统
# ============================================================

class LaneSystem:
    """车道线系统：颜色过滤 + 霍夫线检测 + 加权拟合 + 历史平滑"""

    def __init__(self):
        self.left_history = deque(maxlen=HISTORY_LEN)
        self.right_history = deque(maxlen=HISTORY_LEN)

        self.last_stable_left = None
        self.last_stable_right = None
        self.vertices = None

    def color_filter(self, frame):
        """HLS颜色空间过滤白线和黄线"""
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        white_mask = cv2.inRange(hls, WHITE_LOWER, WHITE_UPPER)
        yellow_mask = cv2.inRange(hls, YELLOW_LOWER, YELLOW_UPPER)

        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        return combined_mask

    def get_lane_info(self, frame):
        """
        返回:
        lane_layer: 车道区域图层
        deviation: 车辆相对车道中心偏移比例
        angle: 估算转向角
        """

        if frame is None:
            return None, 0, 0

        h, w = frame.shape[:2]

        color_mask = self.color_filter(frame)
        edges = cv2.Canny(color_mask, CANNY_LOW, CANNY_HIGH)

        if self.vertices is None:
            top_w = w * ROI_TOP
            self.vertices = np.array([[
                (0, h),
                (int(w * 0.5 - top_w / 2), int(h * ROI_HEIGHT)),
                (int(w * 0.5 + top_w / 2), int(h * ROI_HEIGHT)),
                (w, h)
            ]], dtype=np.int32)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self.vertices, 255)
        roi = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=30,
            maxLineGap=100
        )

        curr_l_fit, curr_r_fit = self.process_lines(lines)

        if curr_l_fit is not None:
            self.left_history.append(curr_l_fit)

        if curr_r_fit is not None:
            self.right_history.append(curr_r_fit)

        smooth_left = np.mean(self.left_history, axis=0) if self.left_history else self.last_stable_left
        smooth_right = np.mean(self.right_history, axis=0) if self.right_history else self.last_stable_right

        if smooth_left is not None:
            self.last_stable_left = smooth_left

        if smooth_right is not None:
            self.last_stable_right = smooth_right

        y_min = int(h * ROI_HEIGHT) + 40

        l_pts = self.make_pts(smooth_left, y_min, h) if smooth_left is not None else None
        r_pts = self.make_pts(smooth_right, y_min, h) if smooth_right is not None else None

        deviation = 0
        angle = 0
        lane_layer = np.zeros_like(frame)

        if l_pts and r_pts:
            pts = np.array([l_pts[0], l_pts[1], r_pts[1], r_pts[0]], dtype=np.int32)
            cv2.fillPoly(lane_layer, [pts], (0, 255, 0))

            lane_center = (l_pts[0][0] + r_pts[1][0]) / 2
            screen_center = w / 2

            deviation = (lane_center - screen_center) / w

            l_slope = smooth_left[0]
            r_slope = smooth_right[0]
            avg_slope = (l_slope + r_slope) / 2

            angle = -math.degrees(math.atan(avg_slope))

        return lane_layer, deviation, angle

    def process_lines(self, lines):
        """线段分类与加权拟合"""

        left_lines, right_lines = [], []
        left_weights, right_weights = [], []

        if lines is None:
            return None, None

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if abs(slope) < 0.3 or abs(slope) > 3.0:
                continue

            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

        left_fit = np.average(left_lines, axis=0, weights=left_weights) if left_lines else None
        right_fit = np.average(right_lines, axis=0, weights=right_weights) if right_lines else None

        return left_fit, right_fit

    def make_pts(self, line, y1, y2):
        if line is None:
            return None

        s, i = line

        if abs(s) < 1e-2:
            return None

        try:
            x1 = int((y1 - i) / s)
            x2 = int((y2 - i) / s)

            if x1 < -2000 or x1 > 4000 or x2 < -2000 or x2 > 4000:
                return None

            return ((x1, y1), (x2, y2))

        except Exception:
            return None


# ============================================================
# 危险判断系统：核心优化部分
# ============================================================

class RiskAssessor:
    """
    综合风险评估系统。

    不再只使用 width_ratio 判断危险，而是综合：
    1. 目标框面积
    2. 目标框宽度
    3. 目标是否在本车道区域
    4. 目标是否靠近画面中心
    5. 目标底部是否靠近画面下方
    6. 目标框面积增长速度
    7. TTC 碰撞时间估计
    """

    def __init__(self):
        self.tracks = {}
        self.next_track_id = 1

    @staticmethod
    def clamp(x, low=0.0, high=1.0):
        return max(low, min(high, x))

    @staticmethod
    def box_area(box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def box_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = RiskAssessor.box_area(box_a)
        area_b = RiskAssessor.box_area(box_b)

        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0

        return inter_area / union

    def cleanup_old_tracks(self, now):
        """删除长时间未匹配的轨迹"""
        remove_ids = []

        for track_id, track in self.tracks.items():
            if now - track["last_time"] > TRACK_MAX_AGE:
                remove_ids.append(track_id)

        for track_id in remove_ids:
            del self.tracks[track_id]

    def match_track(self, det_box, det_class, now):
        """用 IoU 匹配已有目标轨迹"""

        best_id = None
        best_iou = 0.0

        for track_id, track in self.tracks.items():
            if now - track["last_time"] > TRACK_MAX_AGE:
                continue

            if track["class"] != det_class:
                continue

            iou = self.box_iou(det_box, track["box"])

            if iou > best_iou:
                best_iou = iou
                best_id = track_id

        if best_id is not None and best_iou >= IOU_MATCH_THRESHOLD:
            return best_id

        return None

    def update_track(self, det, now):
        """更新目标轨迹并返回 track_id"""

        x1, y1, x2, y2 = det["box"]
        det_box = (int(x1), int(y1), int(x2), int(y2))
        det_class = det.get("class", "object")

        track_id = self.match_track(det_box, det_class, now)

        if track_id is None:
            track_id = self.next_track_id
            self.next_track_id += 1

            self.tracks[track_id] = {
                "class": det_class,
                "box": det_box,
                "last_time": now,
                "history": deque(maxlen=TRACK_HISTORY_LEN)
            }

        area = self.box_area(det_box)

        self.tracks[track_id]["box"] = det_box
        self.tracks[track_id]["last_time"] = now
        self.tracks[track_id]["history"].append((now, det_box, area))

        return track_id

    def estimate_ttc(self, track_id):
        """
        基于目标框面积增长估算 TTC。

        原理：
        如果目标框面积持续变大，说明目标可能正在接近。
        面积增长越快，TTC 越小，危险程度越高。

        注意：
        这是单目视频下的近似 TTC，不是严格物理距离。
        """

        track = self.tracks.get(track_id)

        if track is None:
            return float("inf")

        history = list(track["history"])

        if len(history) < 2:
            return float("inf")

        t0, _, area0 = history[0]
        t1, _, area1 = history[-1]

        dt = t1 - t0

        if dt <= 0.1:
            return float("inf")

        if area0 <= 0 or area1 <= 0:
            return float("inf")

        if area1 <= area0:
            return float("inf")

        growth_rate = (math.log(area1 + 1.0) - math.log(area0 + 1.0)) / dt

        if growth_rate <= 1e-6:
            return float("inf")

        ttc = 1.0 / growth_rate

        return min(ttc, 99.0)

    def lane_overlap_score(self, box, lane_layer):
        """
        判断目标框是否落在绿色车道区域内。

        返回:
        None: 没有有效车道区域
        0~1: 目标与本车道区域的重叠程度
        """

        if lane_layer is None:
            return None

        lane_mask = lane_layer[:, :, 1] > 0

        if np.count_nonzero(lane_mask) < 100:
            return None

        h, w = lane_mask.shape[:2]

        x1, y1, x2, y2 = box

        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w - 1, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h - 1, int(y2)))

        if x2 <= x1 or y2 <= y1:
            return None

        # 只看目标框的下半部分，因为车辆和行人接地点更能说明其所在车道
        y_start = int(y1 + 0.45 * (y2 - y1))
        roi = lane_mask[y_start:y2, x1:x2]

        if roi.size == 0:
            return None

        overlap_ratio = np.count_nonzero(roi) / roi.size

        return self.clamp(overlap_ratio, 0.0, 1.0)

    def class_weight(self, class_name):
        """不同类别赋予不同风险权重"""

        name = str(class_name).lower()

        high_risk = ["person", "pedestrian", "bicycle", "motorcycle", "bike"]
        vehicle_risk = ["car", "truck", "bus", "van"]

        if name in high_risk:
            return 1.20

        if name in vehicle_risk:
            return 1.00

        return 0.85

    def ttc_score(self, ttc):
        """把 TTC 转换成 0~1 风险分数"""

        if not np.isfinite(ttc):
            return 0.0

        if ttc <= 1.2:
            return 1.0

        if ttc <= 2.0:
            return 0.80

        if ttc <= 3.0:
            return 0.60

        if ttc <= 4.5:
            return 0.35

        return 0.0

    def assess(self, detections, frame_shape, lane_layer=None, now=None):
        """
        对 YOLO 检测结果进行风险评估。

        输入:
        detections: YOLO输出目标列表
        frame_shape: frame.shape
        lane_layer: 车道区域图层
        now: 当前时间

        输出:
        enriched_dets: 带风险分数、TTC、危险等级的检测结果
        """

        if now is None:
            now = time.time()

        self.cleanup_old_tracks(now)

        h, w = frame_shape[:2]
        enriched_dets = []

        for det in detections:
            if "box" not in det:
                continue

            x1, y1, x2, y2 = det["box"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            det = det.copy()
            det["box"] = (x1, y1, x2, y2)

            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h

            width_ratio = box_w / max(w, 1)
            area_ratio = box_area / max(w * h, 1)

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # 1. 目标大小分数
            size_score = self.clamp(area_ratio / 0.08)

            # 2. 目标宽度分数
            width_score = self.clamp(width_ratio / 0.25)

            # 3. 目标是否靠近画面中心
            center_score = self.clamp(1.0 - abs(cx - w / 2) / (w * 0.45))

            # 4. 目标底部是否靠近画面下方
            bottom_score = self.clamp((y2 / h - 0.35) / 0.65)

            # 5. 目标是否在本车道区域
            lane_score = self.lane_overlap_score((x1, y1, x2, y2), lane_layer)

            if lane_score is None:
                # 没检测到稳定车道线时，用画面中心近似替代
                lane_score = center_score * 0.65
            else:
                # 有车道区域时，同时考虑车道重叠和中心位置
                lane_score = max(lane_score, center_score * 0.40)

            # 6. 目标跟踪与 TTC
            track_id = self.update_track(det, now)
            ttc = self.estimate_ttc(track_id)
            ttc_s = self.ttc_score(ttc)

            # 7. 类别风险权重
            cls_name = det.get("class", "object")
            cls_weight = self.class_weight(cls_name)

            # 8. 综合风险分数
            risk_score = (
                20.0 * size_score +
                15.0 * width_score +
                25.0 * lane_score +
                15.0 * center_score +
                10.0 * bottom_score +
                15.0 * ttc_s
            )

            risk_score *= cls_weight
            risk_score = max(0.0, min(100.0, risk_score))

            # 9. 风险等级判断
            risk_level = "SAFE"

            danger_by_score = risk_score >= RISK_DANGER_SCORE
            danger_by_ttc = np.isfinite(ttc) and ttc <= TTC_DANGER and lane_score >= 0.45
            danger_by_close = width_ratio >= 0.35 and lane_score >= 0.55

            warning_by_score = risk_score >= RISK_WARNING_SCORE
            warning_by_ttc = np.isfinite(ttc) and ttc <= TTC_WARNING and lane_score >= 0.35

            if danger_by_score or danger_by_ttc or danger_by_close:
                risk_level = "DANGER"
            elif warning_by_score or warning_by_ttc:
                risk_level = "WARNING"

            det["track_id"] = track_id
            det["risk_score"] = risk_score
            det["risk_level"] = risk_level
            det["ttc"] = ttc
            det["width_ratio"] = width_ratio
            det["area_ratio"] = area_ratio
            det["center_score"] = center_score
            det["lane_score"] = lane_score
            det["bottom_score"] = bottom_score

            enriched_dets.append(det)

        return enriched_dets


# ============================================================
# 可视化函数
# ============================================================

def draw_detection_with_risk(display, det):
    """绘制带风险等级的目标框"""

    x1, y1, x2, y2 = det["box"]
    cls_name = det.get("class", "object")
    track_id = det.get("track_id", -1)

    risk_level = det.get("risk_level", "SAFE")
    risk_score = det.get("risk_score", 0.0)
    ttc = det.get("ttc", float("inf"))

    if risk_level == "DANGER":
        color = (0, 0, 255)
        thickness = 3
    elif risk_level == "WARNING":
        color = (0, 255, 255)
        thickness = 2
    else:
        color = (0, 255, 0)
        thickness = 2

    cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

    label = f"{cls_name} ID:{track_id} {risk_level} {risk_score:.0f}"

    cv2.putText(
        display,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )

    if np.isfinite(ttc):
        ttc_text = f"TTC:{ttc:.1f}s"
    else:
        ttc_text = "TTC:--"

    cv2.putText(
        display,
        ttc_text,
        (x1, min(display.shape[0] - 10, y2 + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        1
    )

    if risk_level == "DANGER":
        cv2.putText(
            display,
            "BRAKE!",
            (x1, max(45, y1 - 35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3
        )
    elif risk_level == "WARNING":
        cv2.putText(
            display,
            "WARNING",
            (x1, max(45, y1 - 35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


def draw_dashboard(img, deviation, steer_angle, fps, status, max_risk_score=0.0, min_ttc=float("inf")):
    """绘制仪表盘"""

    h, w = img.shape[:2]

    cv2.rectangle(img, (0, h - 90), (w, h), (0, 0, 0), -1)

    # 虚拟方向盘
    center = (w // 2, h - 45)
    radius = 30

    display_angle = steer_angle * 4 * STEER_SENSITIVITY
    display_angle = max(-90, min(90, display_angle))

    rad = math.radians(display_angle - 90)

    end_x = int(center[0] + radius * math.cos(rad))
    end_y = int(center[1] + radius * math.sin(rad))

    cv2.circle(img, center, radius, (200, 200, 200), 2)
    cv2.line(img, center, (end_x, end_y), (0, 0, 255), 3)

    cv2.putText(
        img,
        "STEER",
        (center[0] - 22, center[1] + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )

    # 偏移指示条
    cv2.rectangle(
        img,
        (w // 2 - 100, h - 82),
        (w // 2 + 100, h - 77),
        (50, 50, 50),
        -1
    )

    marker_x = int(w // 2 + deviation * w)
    marker_x = max(w // 2 - 100, min(w // 2 + 100, marker_x))

    dev_color = (0, 255, 0) if abs(deviation) < 0.05 else (0, 0, 255)

    cv2.circle(img, (marker_x, h - 79), 6, dev_color, -1)

    # 状态颜色
    if status == "DANGER":
        status_color = (0, 0, 255)
    elif status == "WARNING":
        status_color = (0, 255, 255)
    else:
        status_color = (0, 255, 0)

    # 左侧数据
    cv2.putText(
        img,
        f"FPS: {fps:.1f}",
        (20, h - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1
    )

    cv2.putText(
        img,
        f"RISK: {max_risk_score:.1f}",
        (20, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2
    )

    # 右侧状态
    if np.isfinite(min_ttc):
        ttc_text = f"TTC: {min_ttc:.1f}s"
    else:
        ttc_text = "TTC: --"

    cv2.putText(
        img,
        ttc_text,
        (w - 190, h - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1
    )

    cv2.putText(
        img,
        f"STATUS: {status}",
        (w - 190, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2
    )


# ============================================================
# 主程序
# ============================================================

class FrameSource:
    """Read frames from either a video file/camera or a CARLA RGB image directory."""

    def __init__(self, source, input_kind="auto", fps=20.0):
        self.source = source
        self.input_kind = input_kind
        self.fps = fps
        self.cap = None
        self.image_paths = []
        self.index = 0
        self.mode = self._resolve_mode()

        if self.mode == "images":
            self.image_paths = self._collect_images(Path(source))
            if not self.image_paths:
                raise ValueError(f"No image frames found in {source}")
            first = cv2.imread(str(self.image_paths[0]))
            if first is None:
                raise ValueError(f"Cannot read first image frame: {self.image_paths[0]}")
            self.width = first.shape[1]
            self.height = first.shape[0]
            self.total_frames = len(self.image_paths)
        else:
            video_source = int(source) if isinstance(source, str) and source.isdigit() else source
            self.cap = cv2.VideoCapture(video_source)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video source {source}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _resolve_mode(self):
        if self.input_kind in ("video", "images"):
            return self.input_kind
        return "images" if Path(str(self.source)).is_dir() else "video"

    @staticmethod
    def _collect_images(source_dir):
        paths = []
        for pattern in IMAGE_EXTS:
            paths.extend(glob.glob(str(source_dir / pattern)))
        return [Path(p) for p in sorted(paths)]

    def read(self):
        if self.mode == "images":
            if self.index >= len(self.image_paths):
                return False, None
            frame = cv2.imread(str(self.image_paths[self.index]))
            self.index += 1
            return frame is not None, frame
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()


class CarlaTestReporter:
    """Save CARLA/video test artifacts: annotated video, screenshots, and per-frame CSV."""

    def __init__(self, source, output_dir=DEFAULT_OUTPUT_DIR, fps=20.0, frame_size=None, report_name="carla_test_report.csv", save_every=0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir = self.output_dir / "screenshots"
        self.screenshot_dir.mkdir(exist_ok=True)
        self.events_dir = self.output_dir / "events"
        self.events_dir.mkdir(exist_ok=True)

        source_name = Path(str(source)).stem or "camera"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.video_path = self.output_dir / f"{source_name}_risk_{timestamp}.mp4"
        self.report_path = self.output_dir / report_name
        self.save_every = max(0, int(save_every))
        self.first_screenshot = None
        self.rows = []
        self.writer = None

        if frame_size is not None:
            self.writer = cv2.VideoWriter(
                str(self.video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(float(fps), 1.0),
                frame_size,
            )

    def write_frame(self, frame, frame_idx):
        if self.writer is not None:
            self.writer.write(frame)
        if self.first_screenshot is None:
            self.save_screenshot(frame, frame_idx, "preview")
        if self.save_every and frame_idx % self.save_every == 0:
            self.save_screenshot(frame, frame_idx, "auto")

    def save_screenshot(self, frame, frame_idx, suffix="manual"):
        path = self.screenshot_dir / f"frame_{frame_idx:06d}_{suffix}.jpg"
        cv2.imwrite(str(path), frame)
        if self.first_screenshot is None:
            self.first_screenshot = path
        return path

    def add_row(self, frame_idx, fps, status, detections, max_risk_score, min_ttc):
        self.rows.append({
            "frame": frame_idx,
            "fps": f"{fps:.2f}",
            "status": status,
            "detections": len(detections),
            "max_risk_score": f"{max_risk_score:.2f}",
            "min_ttc": "" if not np.isfinite(min_ttc) else f"{min_ttc:.2f}",
        })

    def release(self):
        if self.writer is not None:
            self.writer.release()
        with open(self.report_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["frame", "fps", "status", "detections", "max_risk_score", "min_ttc"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


def draw_test_overlay(display, frame_idx, total_frames, source_mode, report_path):
    h, w = display.shape[:2]
    progress = frame_idx / total_frames if total_frames > 0 else 0.0
    frame_text = f"{frame_idx}/{total_frames}" if total_frames > 0 else f"{frame_idx}/?"
    progress_text = f"{progress * 100:.1f}%" if total_frames > 0 else "N/A"

    cv2.rectangle(display, (0, 0), (w, 42), (0, 0, 0), -1)
    cv2.putText(display, f"SOURCE: {source_mode.upper()}", (18, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(display, f"FRAME: {frame_text}", (190, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(display, f"PROGRESS: {progress_text}", (360, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(display, f"CSV: {Path(report_path).name}", (570, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 255), 1)


def parse_args():
    parser = argparse.ArgumentParser(description="AutoPilot V3.2 risk optimized lane visualization.")
    parser.add_argument("source", nargs="?", default="sample.hevc", help="Video path, camera index, or CARLA RGB image directory")
    parser.add_argument("--input-kind", choices=["auto", "video", "images"], default="auto", help="Input type; use images for CARLA exported RGB frames")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for test artifacts")
    parser.add_argument("--report-csv", default="carla_test_report.csv", help="Per-frame CSV report name")
    parser.add_argument("--no-display", action="store_true", help="Run without OpenCV window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means all frames")
    parser.add_argument("--save-every", type=int, default=0, help="Save a screenshot every N frames")
    parser.add_argument("--fps", type=float, default=20.0, help="FPS for image-directory inputs")
    return parser.parse_args()


def run_pipeline(args):
    frame_source = FrameSource(args.source, input_kind=args.input_kind, fps=args.fps)
    total_frames = args.max_frames if args.max_frames > 0 else frame_source.total_frames
    reporter = CarlaTestReporter(
        args.source,
        output_dir=args.output_dir,
        fps=frame_source.fps,
        frame_size=(frame_source.width, frame_source.height),
        report_name=args.report_csv,
        save_every=args.save_every,
    )

    lane_sys = LaneSystem()
    yolo_sys = ObjectDetector()
    risk_sys = RiskAssessor()
    logger = EventLogger(save_dir=str(reporter.events_dir))

    print("AutoPilot V3.2: risk optimized system with CARLA test support")
    print(f"Input: {args.source}")
    print(f"Input mode: {frame_source.mode}")
    print(f"Output video: {reporter.video_path}")
    print(f"CSV report: {reporter.report_path}")

    frame_count = 0
    current_dets = []
    danger_frames = 0
    warning_frames = 0
    fps_values = []

    while True:
        t_start = time.time()
        ret, frame = frame_source.read()
        if not ret:
            break

        display = frame.copy()
        h, w = frame.shape[:2]

        lane_layer, deviation, steer_angle = lane_sys.get_lane_info(frame)
        if lane_layer is not None:
            display = cv2.addWeighted(display, 1.0, lane_layer, 0.35, 0)

        should_detect = frame_count % (SKIP_FRAMES + 1) == 0
        if should_detect:
            _, raw_dets = yolo_sys.detect(frame)
            current_dets = risk_sys.assess(raw_dets, frame.shape, lane_layer, now=time.time())

        is_danger = False
        has_warning = False
        max_risk_score = 0.0
        min_ttc = float("inf")
        danger_obj = None
        danger_score = None
        danger_ttc = None

        for det in current_dets:
            draw_detection_with_risk(display, det)
            risk_level = det.get("risk_level", "SAFE")
            risk_score = det.get("risk_score", 0.0)
            ttc = det.get("ttc", float("inf"))
            max_risk_score = max(max_risk_score, risk_score)
            if np.isfinite(ttc):
                min_ttc = min(min_ttc, ttc)
            if risk_level == "DANGER":
                is_danger = True
                danger_obj = det.get("class", "object")
                danger_score = risk_score
                danger_ttc = ttc
            elif risk_level == "WARNING":
                has_warning = True

        if is_danger:
            danger_frames += 1
            saved = logger.log_danger(display, danger_obj, danger_score, danger_ttc)
            if saved:
                cv2.putText(display, "SNAPSHOT SAVED", (w // 2 - 130, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        elif has_warning:
            warning_frames += 1

        fps = 1.0 / max(time.time() - t_start, 1e-6)
        fps_values.append(fps)
        status = "DANGER" if is_danger else ("WARNING" if has_warning else "CRUISING")

        draw_dashboard(display, deviation, steer_angle, fps, status, max_risk_score=max_risk_score, min_ttc=min_ttc)
        frame_count += 1
        draw_test_overlay(display, frame_count, total_frames, frame_source.mode, reporter.report_path)
        reporter.add_row(frame_count, fps, status, current_dets, max_risk_score, min_ttc)
        reporter.write_frame(display, frame_count)

        if total_frames > 0 and frame_count % max(1, int(frame_source.fps)) == 0:
            print(f"\rProcessed {frame_count}/{total_frames} frames ({frame_count / total_frames * 100:.1f}%)", end="")

        if not args.no_display:
            cv2.imshow("AutoPilot V3.2 Risk Optimized", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                reporter.save_screenshot(display, frame_count, "manual")

        if args.max_frames > 0 and frame_count >= args.max_frames:
            break

    frame_source.release()
    reporter.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print()
    print(f"Done. Processed frames: {frame_count}")
    print(f"Average FPS: {np.mean(fps_values) if fps_values else 0.0:.1f}")
    print(f"Warning frames: {warning_frames}")
    print(f"Danger frames: {danger_frames}")
    print(f"Saved video: {reporter.video_path}")
    print(f"Saved CSV: {reporter.report_path}")
    if reporter.first_screenshot is not None:
        print(f"Saved screenshot: {reporter.first_screenshot}")


def main():
    args = parse_args()
    run_pipeline(args)


def legacy_main():
    source = sys.argv[1] if len(sys.argv) > 1 else "sample.hevc"

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return

    lane_sys = LaneSystem()
    yolo_sys = ObjectDetector()
    risk_sys = RiskAssessor()
    logger = EventLogger()

    print("🚀 AutoPilot V3.2: 启动危险判断优化版系统...")
    print("说明：危险判断已由单一 width_ratio 阈值升级为综合 Risk Score + TTC 判断。")
    print("按 q 退出。")

    frame_count = 0
    current_dets = []

    while True:
        t_start = time.time()

        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        display = frame.copy()
        h, w = frame.shape[:2]

        # =====================================================
        # A. 车道线检测
        # =====================================================

        lane_layer, deviation, steer_angle = lane_sys.get_lane_info(frame)

        if lane_layer is not None:
            display = cv2.addWeighted(display, 1.0, lane_layer, 0.35, 0)

        # =====================================================
        # B. YOLO目标检测 + 风险评估
        # =====================================================

        should_detect = frame_count % (SKIP_FRAMES + 1) == 0

        if should_detect:
            _, raw_dets = yolo_sys.detect(frame)

            current_dets = risk_sys.assess(
                detections=raw_dets,
                frame_shape=frame.shape,
                lane_layer=lane_layer,
                now=time.time()
            )

        # =====================================================
        # C. 绘制目标框与风险信息
        # =====================================================

        is_danger = False
        has_warning = False

        max_risk_score = 0.0
        min_ttc = float("inf")

        danger_obj = None
        danger_score = None
        danger_ttc = None

        for det in current_dets:
            draw_detection_with_risk(display, det)

            risk_level = det.get("risk_level", "SAFE")
            risk_score = det.get("risk_score", 0.0)
            ttc = det.get("ttc", float("inf"))

            max_risk_score = max(max_risk_score, risk_score)

            if np.isfinite(ttc):
                min_ttc = min(min_ttc, ttc)

            if risk_level == "DANGER":
                is_danger = True
                danger_obj = det.get("class", "object")
                danger_score = risk_score
                danger_ttc = ttc

            elif risk_level == "WARNING":
                has_warning = True

        # =====================================================
        # D. 危险抓拍
        # =====================================================

        if is_danger:
            saved = logger.log_danger(display, danger_obj, danger_score, danger_ttc)

            if saved:
                cv2.putText(
                    display,
                    "SNAPSHOT SAVED",
                    (w // 2 - 130, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2
                )

        # =====================================================
        # E. 仪表盘
        # =====================================================

        fps = 1.0 / max(time.time() - t_start, 1e-6)

        if is_danger:
            status = "DANGER"
        elif has_warning:
            status = "WARNING"
        else:
            status = "CRUISING"

        draw_dashboard(
            display,
            deviation=deviation,
            steer_angle=steer_angle,
            fps=fps,
            status=status,
            max_risk_score=max_risk_score,
            min_ttc=min_ttc
        )

        cv2.imshow("AutoPilot V3.2 Risk Optimized", display)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
