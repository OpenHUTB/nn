# detection_engine.py
# 功能：封装 YOLOv8 模型的加载与推理逻辑，提供干净的检测接口
# 修改：增加距离估计和危险等级显示

from ultralytics import YOLO
import io
import sys
import os
import cv2  # 新增导入
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
        # 加载 YOLO 模型（在初始化时完成，避免每次检测重复加载）
        self.model = self._load_model()
        self._last_error_time = 0.0
        self._error_count = 0

    def _load_model(self):
        """
        私有方法：加载 YOLO 模型，并抑制其标准输出和错误输出。
        
        原因：YOLO 在加载模型或首次推理时会自动打印信息（如设备、尺寸等），
        这些信息在 GUI 或自动化脚本中属于干扰。通过临时重定向 stdout/stderr 来静默加载。
        
        返回:
            YOLO: 已加载的模型实例
        """
        # 保存原始的标准输出和错误流
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        # 临时重定向到 StringIO 缓冲区（丢弃所有输出）
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        model = None
        try:
            # 判断是本地文件还是官方模型名
            if os.path.isfile(self.model_path):
                model = YOLO(self.model_path)
            else:
                # 尝试作为 Ultralytics 官方模型加载（会自动下载）
                model = YOLO(self.model_path)
        except FileNotFoundError as e:
            raise ModelLoadError(f"Model file not found: {self.model_path}") from e
        except RuntimeError as e:
            msg = str(e)
            if "CUDA out of memory" in msg:
                raise ModelLoadError(
                    "GPU memory insufficient. Try using CPU or a smaller model (e.g., yolov8n.pt)."
                ) from e
            elif "AssertionError" in msg and ("model" in msg or "state_dict" in msg):
                raise ModelLoadError(f"Corrupted or incompatible model weights: {self.model_path}") from e
            else:
                raise ModelLoadError(f"Runtime error during model loading: {msg}") from e
        except Exception as e:
            raise ModelLoadError(f"Unexpected error loading YOLO model: {e}") from e
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        if model is None:
            raise ModelLoadError(f"Unexpected error loading YOLO model: {self.model_path}")
        try:
            model.fuse()
        except Exception:
            pass
        return model

    def _estimate_distance(self, box_height, known_height=1.6, focal_length=700):
        """根据检测框高度估算距离（米）"""
        if box_height < 1:
            return 999.9
        return (known_height * focal_length) / box_height

    def _get_danger_level(self, distance):
        """根据距离判定危险等级"""
        if distance < 10:
            return "DANGER"
        elif distance < 20:
            return "WARNING"
        else:
            return "SAFE"

    def _color_for_class(self, class_id: int):
        palette = (
            (255, 56, 56),
            (255, 157, 151),
            (255, 112, 31),
            (255, 178, 29),
            (207, 210, 49),
            (72, 249, 10),
            (146, 204, 23),
            (61, 219, 134),
            (26, 147, 52),
            (0, 212, 187),
            (44, 153, 168),
            (0, 194, 255),
            (52, 69, 147),
            (100, 115, 255),
            (0, 24, 236),
            (132, 56, 255),
            (82, 0, 133),
            (203, 56, 255),
            (255, 149, 200),
            (255, 55, 199),
        )
        return palette[class_id % len(palette)]

    def _annotate(self, frame: np.ndarray, result):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return frame
        try:
            box_count = len(boxes)
        except Exception:
            box_count = 0
        if box_count == 0:
            return frame

        annotated_frame = frame.copy()
        names = getattr(result, "names", None) or getattr(self.model, "names", {}) or {}

        xyxy = getattr(boxes, "xyxy", None)
        cls = getattr(boxes, "cls", None)
        conf = getattr(boxes, "conf", None)
        if xyxy is None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0]) if hasattr(box, "cls") else 0
                score = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                name = names.get(class_id, str(class_id))
                color = self._color_for_class(class_id)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    f"{name} {score:.2f}",
                    (x1, max(0, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                box_height = y2 - y1
                distance = self._estimate_distance(box_height)
                danger = self._get_danger_level(distance)
                cv2.putText(
                    annotated_frame,
                    f"{danger} {distance:.1f}m",
                    (x1, max(0, y1 - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
            return annotated_frame

        try:
            xyxy_np = xyxy
            if hasattr(xyxy_np, "cpu"):
                xyxy_np = xyxy_np.cpu()
            if hasattr(xyxy_np, "numpy"):
                xyxy_np = xyxy_np.numpy()
            xyxy_np = np.asarray(xyxy_np, dtype=np.float32)
        except Exception:
            xyxy_np = None

        try:
            cls_np = cls
            if hasattr(cls_np, "cpu"):
                cls_np = cls_np.cpu()
            if hasattr(cls_np, "numpy"):
                cls_np = cls_np.numpy()
            cls_np = np.asarray(cls_np, dtype=np.int32)
        except Exception:
            cls_np = None

        try:
            conf_np = conf
            if hasattr(conf_np, "cpu"):
                conf_np = conf_np.cpu()
            if hasattr(conf_np, "numpy"):
                conf_np = conf_np.numpy()
            conf_np = np.asarray(conf_np, dtype=np.float32)
        except Exception:
            conf_np = None

        if xyxy_np is None or cls_np is None or conf_np is None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0]) if hasattr(box, "cls") else 0
                score = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                name = names.get(class_id, str(class_id))
                color = self._color_for_class(class_id)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    f"{name} {score:.2f}",
                    (x1, max(0, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                box_height = y2 - y1
                distance = self._estimate_distance(box_height)
                danger = self._get_danger_level(distance)
                cv2.putText(
                    annotated_frame,
                    f"{danger} {distance:.1f}m",
                    (x1, max(0, y1 - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
            return annotated_frame

        xyxy_int = xyxy_np.astype(np.int32, copy=False)
        for (x1, y1, x2, y2), class_id, score in zip(xyxy_int, cls_np, conf_np):
            x1i = int(x1)
            y1i = int(y1)
            x2i = int(x2)
            y2i = int(y2)
            class_id_int = int(class_id)
            score_float = float(score)
            name = names.get(class_id_int, str(class_id_int))
            color = self._color_for_class(class_id_int)
            cv2.rectangle(annotated_frame, (x1i, y1i), (x2i, y2i), color, 2)
            cv2.putText(
                annotated_frame,
                f"{name} {score_float:.2f}",
                (x1i, max(0, y1i - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
            box_height = y2i - y1i
            distance = self._estimate_distance(box_height)
            danger = self._get_danger_level(distance)
            cv2.putText(
                annotated_frame,
                f"{danger} {distance:.1f}m",
                (x1i, max(0, y1i - 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        return annotated_frame

    def detect(self, frame):
        """
        对单帧图像执行目标检测。

        参数:
            frame (np.ndarray): 输入图像，格式为 HWC（高度, 宽度, 通道），BGR 或 RGB 均可（YOLO 内部会处理）

        返回:
            tuple:
                - annotated_frame (np.ndarray): 带有检测框、标签和置信度的可视化图像（HWC, BGR 格式）
                - results (List[ultralytics.engine.results.Results]): 原始检测结果对象列表（通常长度为1）
        """
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            result0 = results[0] if results else None
            if result0 is None:
                return frame, []
            annotated_frame = self._annotate(frame, result0)
            return annotated_frame, results
        except Exception as e:
            import time

            now = time.time()
            self._error_count += 1
            if now - self._last_error_time >= 2.0:
                print(f"⚠️ Warning: Detection failed on current frame: {e} (errors={self._error_count})")
                self._last_error_time = now
            return frame, []

    def detect_batch(self, sources):
        results = self.model(sources, conf=self.conf_threshold, verbose=False)
        annotated_frames = []
        for r in results:
            base = getattr(r, "orig_img", None)
            if base is None:
                annotated_frames.append(None)
                continue
            annotated_frames.append(self._annotate(base, r))
        return annotated_frames, results

