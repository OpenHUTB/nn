# detection_engine.py
# 功能：封装 YOLOv8 模型的加载与推理逻辑，提供干净的检测接口
# 修改：增加距离估计和危险等级显示，增加目标跟踪及检测框平滑滤波（指数移动平均），并打印目标位移

from ultralytics import YOLO
import io
import sys
import os
import cv2
import numpy as np


class ModelLoadError(Exception):
    """模型加载失败专用异常"""
    pass


class DetectionEngine:
    """
    目标检测引擎类。
    负责加载 YOLO 模型并对输入图像帧执行推理，
    同时屏蔽模型内部的冗余打印输出（如进度条、日志等），
    使主程序输出更整洁。
    """

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.25):
        """
        初始化检测引擎。

        参数:
            model_path (str): YOLO 模型文件路径或名称（如 'yolov8n.pt'）
            conf_threshold (float): 置信度阈值，低于此值的检测结果将被过滤
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = self._load_model()
        self.tracker = None
        self.enable_tracking = True
        self.smoothing_alpha = 0.3

    def _load_model(self):
        """加载模型并静默输出（仅在加载时屏蔽）"""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            model = YOLO(self.model_path)
            return model
        except FileNotFoundError as e:
            raise ModelLoadError(f"Model file not found: {self.model_path}") from e
        except RuntimeError as e:
            msg = str(e)
            if "CUDA out of memory" in msg:
                raise ModelLoadError("GPU memory insufficient. Try using CPU or a smaller model.") from e
            raise ModelLoadError(f"Runtime error: {msg}") from e
        except Exception as e:
            raise ModelLoadError(f"Unexpected error: {e}") from e
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def _estimate_distance(self, box_height, known_height=1.6, focal_length=700):
        if box_height < 1:
            return 999.9
        return (known_height * focal_length) / box_height

    def _get_danger_level(self, distance):
        if distance < 10:
            return "DANGER"
        elif distance < 20:
            return "WARNING"
        else:
            return "SAFE"

    def detect(self, frame):
        """
        对单帧图像执行目标检测。
        返回 (annotated_frame, results)
        """
        # 保存原始输出流
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            # 仅静音 YOLO 推理时的输出
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            # 立即恢复标准输出，以便 print 能正常显示
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            annotated_frame = frame.copy()

            if results[0].boxes is not None:
                raw_boxes = results[0].boxes
                boxes_xyxy = raw_boxes.xyxy.cpu().numpy()
                confs = raw_boxes.conf.cpu().numpy()
                clses = raw_boxes.cls.cpu().numpy()

                detections = []
                for i in range(len(boxes_xyxy)):
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    detections.append((x1, y1, x2, y2, confs[i], int(clses[i])))

                if self.enable_tracking:
                    if self.tracker is None:
                        self.tracker = SimpleTracker(smoothing_alpha=self.smoothing_alpha)
                    tracked_dets = self.tracker.update(detections)
                else:
                    tracked_dets = [d + (None,) for d in detections]

                for det in tracked_dets:
                    x1, y1, x2, y2, conf, cls, obj_id = det
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    class_name = self.model.names.get(cls, f"class_{cls}")
                    label = f"{class_name} {conf:.2f}"
                    if obj_id is not None:
                        label += f" ID:{obj_id}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    box_height = y2 - y1
                    distance = self._estimate_distance(box_height)
                    danger = self._get_danger_level(distance)
                    info_text = f"{danger} {distance:.1f}m"
                    cv2.putText(annotated_frame, info_text, (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            return annotated_frame, results

        except Exception as e:
            print(f"⚠️ Warning: Detection failed on current frame: {e}")
            return frame.copy(), []
        finally:
            # 确保在任何情况下都恢复标准输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class SimpleTracker:
    """基于 IoU 匹配的目标跟踪器，支持指数移动平均平滑，并打印目标位移。"""

    def __init__(self, iou_threshold=0.3, max_lost=5, smoothing_alpha=0.3, history_len=10):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.smoothing_alpha = smoothing_alpha
        self.history_len = history_len
        self.next_id = 1
        self.tracks = []  # each: {id, box, smoothed_box, centers, lost_count}

    def update(self, detections):
        if not detections:
            for t in self.tracks:
                t['lost_count'] += 1
            self._remove_lost_tracks()
            return []

        # 计算 IoU 矩阵
        iou_matrix = []
        for trk in self.tracks:
            trk_box = trk['box']
            row = [self._box_iou(trk_box, det[:4]) for det in detections]
            iou_matrix.append(row)

        # 贪心匹配
        matched_track_idx = []
        matched_det_idx = []
        used_track = set()
        used_det = set()
        pairs = [(i, j) for i in range(len(self.tracks)) for j in range(len(detections))]
        pairs.sort(key=lambda x: iou_matrix[x[0]][x[1]], reverse=True)

        for track_idx, det_idx in pairs:
            if track_idx in used_track or det_idx in used_det:
                continue
            if iou_matrix[track_idx][det_idx] >= self.iou_threshold:
                matched_track_idx.append(track_idx)
                matched_det_idx.append(det_idx)
                used_track.add(track_idx)
                used_det.add(det_idx)

        # 更新匹配的轨迹
        for trk_idx, det_idx in zip(matched_track_idx, matched_det_idx):
            current_box = detections[det_idx][:4]
            self.tracks[trk_idx]['box'] = current_box

            # 计算中心点并记录历史
            x1, y1, x2, y2 = current_box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers = self.tracks[trk_idx]['centers']
            centers.append((cx, cy))
            if len(centers) > self.history_len:
                centers.pop(0)

            # 打印位移（如果有前一帧）
            if len(centers) >= 2:
                prev_cx, prev_cy = centers[-2]
                diff = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                print(f"ID {self.tracks[trk_idx]['id']} 位移: {diff:.1f} 像素")

            # 指数移动平均平滑框
            old_smoothed = self.tracks[trk_idx]['smoothed_box']
            alpha = self.smoothing_alpha
            new_smoothed = tuple(alpha * current_box[i] + (1 - alpha) * old_smoothed[i] for i in range(4))
            self.tracks[trk_idx]['smoothed_box'] = new_smoothed
            self.tracks[trk_idx]['lost_count'] = 0

        # 未匹配的检测 → 新轨迹
        for j, det in enumerate(detections):
            if j not in used_det:
                box = det[:4]
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                self.tracks.append({
                    'id': self.next_id,
                    'box': box,
                    'smoothed_box': box,
                    'centers': [(cx, cy)],
                    'lost_count': 0
                })
                self.next_id += 1

        # 未匹配的轨迹 lost_count +1
        for i in range(len(self.tracks)):
            if i not in used_track:
                self.tracks[i]['lost_count'] += 1

        self._remove_lost_tracks()

        # 组装输出（使用平滑框）
        output = []
        for trk_idx, det_idx in zip(matched_track_idx, matched_det_idx):
            det = detections[det_idx]
            smooth_box = self.tracks[trk_idx]['smoothed_box']
            output.append(smooth_box + det[4:6] + (self.tracks[trk_idx]['id'],))
        for j, det in enumerate(detections):
            if j not in used_det:
                new_id = self.next_id - 1
                output.append(det[:4] + det[4:6] + (new_id,))
        return output

    def _remove_lost_tracks(self):
        self.tracks = [t for t in self.tracks if t['lost_count'] <= self.max_lost]

    @staticmethod
    def _box_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0