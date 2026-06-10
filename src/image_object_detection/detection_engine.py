# detection_engine.py
# 功能：封装 YOLOv8 模型的加载与推理逻辑，提供干净的检测接口
# 增强版：支持多种模型格式、批量检测、跟踪、回调等功能

from ultralytics import YOLO
import io
import sys
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict


class ModelLoadError(Exception):
    """模型加载失败专用异常"""
    pass


class DetectionEngine:
    """
    目标检测引擎类（增强版）。
    
    功能特性：
    - 支持多种模型格式（YOLOv5/YOLOv8/ONNX/TensorRT）
    - 批量图像检测
    - 目标跟踪功能
    - 距离估算与危险等级评估
    - 检测回调机制
    - 结果过滤与后处理
    """

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.25, iou_threshold=0.45, 
                 device="auto", half_precision=False):
        """
        初始化检测引擎。

        参数:
            model_path (str): YOLO 模型文件路径或名称
            conf_threshold (float): 置信度阈值
            iou_threshold (float): IOU阈值（NMS）
            device (str): 设备类型 (auto/cpu/cuda/mps)
            half_precision (bool): 是否使用半精度推理
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = self._resolve_device(device)
        self.half_precision = half_precision
        
        self.model = None
        self.tracker = None
        self.detection_callbacks = []
        
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'avg_inference_time': 0.0,
            'last_inference_time': 0.0
        }
        
        self.model = self._load_model()

    def _resolve_device(self, device):
        """解析设备类型"""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device

    def _load_model(self):
        """加载 YOLO 模型，并抑制冗余输出"""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            model = YOLO(self.model_path)
            
            if self.device != "cpu" and self.half_precision:
                model = model.half()
            
            return model
        except FileNotFoundError as e:
            raise ModelLoadError(f"Model file not found: {self.model_path}") from e
        except RuntimeError as e:
            msg = str(e)
            if "CUDA out of memory" in msg:
                raise ModelLoadError(
                    "GPU memory insufficient. Try using CPU or a smaller model."
                ) from e
            elif "AssertionError" in msg and ("model" in msg or "state_dict" in msg):
                raise ModelLoadError(f"Corrupted or incompatible model: {self.model_path}") from e
            else:
                raise ModelLoadError(f"Runtime error: {msg}") from e
        except Exception as e:
            raise ModelLoadError(f"Unexpected error loading model: {e}") from e
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

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

    def _get_danger_color(self, level):
        """获取危险等级对应的颜色"""
        colors = {
            "DANGER": (0, 0, 255),
            "WARNING": (0, 255, 255),
            "SAFE": (0, 255, 0)
        }
        return colors.get(level, (255, 255, 255))

    def add_callback(self, callback: Callable):
        """添加检测回调函数"""
        self.detection_callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """移除检测回调函数"""
        if callback in self.detection_callbacks:
            self.detection_callbacks.remove(callback)

    def _invoke_callbacks(self, results, frame):
        """调用所有回调函数"""
        for callback in self.detection_callbacks:
            try:
                callback(results, frame)
            except Exception as e:
                print(f"⚠️ Callback error: {e}")

    def detect(self, frame, draw=True, return_data=False):
        """
        对单帧图像执行目标检测。

        参数:
            frame (np.ndarray): 输入图像
            draw (bool): 是否绘制检测框
            return_data (bool): 是否返回检测数据

        返回:
            tuple: annotated_frame, results, [detection_data]
        """
        import time
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        start_time = time.time()
        
        try:
            results = self.model(
                frame, 
                conf=self.conf_threshold, 
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )
            
            inference_time = time.time() - start_time
            self.stats['last_inference_time'] = inference_time
            self.stats['total_frames'] += 1
            self.stats['avg_inference_time'] = (
                (self.stats['avg_inference_time'] * (self.stats['total_frames'] - 1) + inference_time) 
                / self.stats['total_frames']
            )

            if results[0].boxes is not None:
                self.stats['total_detections'] += len(results[0].boxes)

            annotated_frame = frame.copy() if draw else None
            
            detection_data = []
            
            if draw and results[0].boxes is not None:
                annotated_frame = results[0].plot()
                boxes = results[0].boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_height = y2 - y1
                    distance = self._estimate_distance(box_height)
                    danger = self._get_danger_level(distance)
                    danger_color = self._get_danger_color(danger)
                    
                    info_text = f"{danger} {distance:.1f}m"
                    cv2.putText(annotated_frame, info_text, (x1, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, danger_color, 2)
                    
                    if return_data:
                        detection_data.append({
                            'class_id': int(box.cls),
                            'class_name': box.name,
                            'confidence': float(box.conf),
                            'bbox': [int(x) for x in box.xyxy[0]],
                            'distance': distance,
                            'danger_level': danger
                        })

            self._invoke_callbacks(results, frame)
            
            if return_data:
                return annotated_frame, results, detection_data
            return annotated_frame, results
            
        except Exception as e:
            print(f"⚠️ Detection failed: {e}")
            if return_data:
                return frame.copy(), [], []
            return frame.copy(), []
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def detect_batch(self, frames: List[np.ndarray], draw=True, return_data=False):
        """
        批量检测多帧图像。

        参数:
            frames (List[np.ndarray]): 图像列表
            draw (bool): 是否绘制检测框
            return_data (bool): 是否返回检测数据

        返回:
            tuple: annotated_frames, all_results, [all_detection_data]
        """
        results = []
        annotated_frames = []
        all_detection_data = []
        
        for frame in frames:
            if return_data:
                annotated, res, data = self.detect(frame, draw, return_data)
                annotated_frames.append(annotated)
                results.append(res)
                all_detection_data.append(data)
            else:
                annotated, res = self.detect(frame, draw)
                annotated_frames.append(annotated)
                results.append(res)
        
        if return_data:
            return annotated_frames, results, all_detection_data
        return annotated_frames, results

    def detect_with_tracking(self, frame, tracker_type="bytetrack.yaml", draw=True):
        """
        带跟踪功能的检测。

        参数:
            frame (np.ndarray): 输入图像
            tracker_type (str): 跟踪器类型
            draw (bool): 是否绘制检测框

        返回:
            tuple: annotated_frame, results
        """
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            results = self.model.track(
                frame, 
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                tracker=tracker_type,
                verbose=False,
                device=self.device
            )
            
            annotated_frame = frame.copy()
            if draw and results[0].boxes is not None:
                annotated_frame = results[0].plot()
                
                for box in results[0].boxes:
                    if hasattr(box, 'id') and box.id is not None:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        box_height = y2 - y1
                        distance = self._estimate_distance(box_height)
                        danger = self._get_danger_level(distance)
                        danger_color = self._get_danger_color(danger)
                        
                        info_text = f"ID:{int(box.id)} {danger} {distance:.1f}m"
                        cv2.putText(annotated_frame, info_text, (x1, y1 - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, danger_color, 2)
            
            self._invoke_callbacks(results, frame)
            return annotated_frame, results
            
        except Exception as e:
            print(f"⚠️ Tracking failed: {e}")
            return frame.copy(), []
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def filter_results(self, results, class_ids=None, class_names=None, min_conf=None, max_distance=None):
        """
        过滤检测结果。

        参数:
            results: 原始检测结果
            class_ids (List[int]): 保留的类别ID列表
            class_names (List[str]): 保留的类别名称列表
            min_conf (float): 最小置信度
            max_distance (float): 最大距离

        返回:
            过滤后的结果
        """
        if not results or not results[0].boxes:
            return results
        
        boxes = results[0].boxes
        mask = np.ones(len(boxes), dtype=bool)
        
        if class_ids is not None:
            class_ids = set(class_ids)
            mask &= np.array([int(b.cls) in class_ids for b in boxes])
        
        if class_names is not None:
            class_names = set(class_names)
            mask &= np.array([b.name in class_names for b in boxes])
        
        if min_conf is not None:
            mask &= np.array([float(b.conf) >= min_conf for b in boxes])
        
        if max_distance is not None:
            mask &= np.array([self._estimate_distance(int(b.xyxy[0][3] - b.xyxy[0][1])) <= max_distance 
                             for b in boxes])
        
        if np.any(mask):
            results[0].boxes = boxes[mask]
        else:
            results[0].boxes = None
        
        return results

    def get_stats(self):
        """获取检测统计信息"""
        return {
            'total_frames': self.stats['total_frames'],
            'total_detections': self.stats['total_detections'],
            'avg_inference_time_ms': round(self.stats['avg_inference_time'] * 1000, 2),
            'last_inference_time_ms': round(self.stats['last_inference_time'] * 1000, 2),
            'detections_per_frame': round(
                self.stats['total_detections'] / max(self.stats['total_frames'], 1), 2
            )
        }

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'avg_inference_time': 0.0,
            'last_inference_time': 0.0
        }

    def get_model_info(self):
        """获取模型信息"""
        if self.model is None:
            return None
        
        info = {
            'model_path': self.model_path,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'half_precision': self.half_precision,
        }
        
        if hasattr(self.model, 'names'):
            info['num_classes'] = len(self.model.names)
            info['class_names'] = list(self.model.names.values())
        
        return info

    def __repr__(self):
        return f"DetectionEngine(model={self.model_path}, conf={self.conf_threshold})"
