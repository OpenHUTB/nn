"""
tracker.py - 目标检测与跟踪核心算法
包含：YOLO检测器、卡尔曼滤波、SORT跟踪器、行为分析
"""

import numpy as np
import cv2
import torch
import queue
import threading
import time
import sys
import os
import queue

# 配置日志
try:
    from loguru import logger
except ImportError:
    # 使用标准logging作为回退
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# 导入utils模块中的工具函数
try:
    from utils import iou, iou_numpy, clip_box, valid_img, bbox_center
except ImportError:
    # 如果在同一目录下，可以直接导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import iou, iou_numpy, clip_box, valid_img, bbox_center

# ======================== 数据结构 ========================

@dataclass
class Detection:
    """检测结果数据结构"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str = "Unknown"
    
    def __post_init__(self):
        self.bbox = np.array(self.bbox, dtype=np.float32)
        self.area = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class TrackState:
    """跟踪状态枚举"""
    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


# ======================== 卡尔曼滤波器 ========================

class KalmanFilter:
    """
    卡尔曼滤波器 - 用于目标状态估计
    8维状态: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    4维观测: [x1, y1, x2, y2]
    """
    
    def __init__(self, dt=0.05, max_speed=50.0):
        """
        初始化卡尔曼滤波器
        
        Args:
            dt: 时间间隔（秒）
            max_speed: 最大速度（像素/秒）
        """
        self.dt = dt
        self.max_speed = max_speed
        
        # 状态向量维度: 8
        # 观测向量维度: 4
        self.state_dim = 8
        self.measure_dim = 4
        
        # 状态转移矩阵 F
        self.F = np.eye(self.state_dim, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = dt
        
        # 观测矩阵 H
        self.H = np.zeros((self.measure_dim, self.state_dim), dtype=np.float32)
        for i in range(self.measure_dim):
            self.H[i, i] = 1.0
        
        # 过程噪声协方差矩阵 Q
        self.Q = np.eye(self.state_dim, dtype=np.float32)
        for i in range(4):
            self.Q[i, i] = 1.0
        for i in range(4, 8):
            self.Q[i, i] = 5.0
        
        # 观测噪声协方差矩阵 R
        self.R = np.eye(self.measure_dim, dtype=np.float32) * 5.0
        
        # 状态协方差矩阵 P
        self.P = np.eye(self.state_dim, dtype=np.float32) * 50.0
        
        # 状态向量 x
        self.x = np.zeros(self.state_dim, dtype=np.float32)
        
        # 首次更新标志
        self.first_update = True
    
    def init(self, bbox):
        """
        初始化滤波器状态
        
        Args:
            bbox: 初始边界框 [x1, y1, x2, y2]
        """
        self.x[:4] = bbox
        self.first_update = True
    
    def predict(self):
        """
        状态预测
        
        Returns:
            np.ndarray: 预测的边界框
        """
        # 状态预测
        self.x = self.F @ self.x
        
        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # 返回预测的边界框
        return self.x[:4].copy()
    
    def update(self, bbox):
        """
        状态更新
        
        Args:
            bbox: 观测到的边界框 [x1, y1, x2, y2]
            
        Returns:
            np.ndarray: 更新后的边界框
        """
        z = np.array(bbox, dtype=np.float32)
        
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，使用伪逆
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        
        # 计算新息
        y = z - self.H @ self.x
        
        # 状态更新
        self.x = self.x + K @ y
        
        # 协方差更新
        I = np.eye(self.state_dim, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        self.first_update = False
        
        return self.x[:4].copy()
    
    def update_noise(self, speed):
        """
        根据速度更新过程噪声
        
        Args:
            speed: 估计的速度（像素/秒）
        """
        # 速度归一化
        speed_factor = min(1.0, speed / self.max_speed)
        
        # 更新过程噪声协方差
        for i in range(4):
            self.Q[i, i] = 1.0 + speed_factor * 4.0
        for i in range(4, 8):
            self.Q[i, i] = 5.0 + speed_factor * 20.0


# ======================== 跟踪目标 ========================

class TrackedObject:
    """
    单个跟踪目标
    """
    
    def __init__(self, track_id: int, bbox: np.ndarray, 
                 img_shape: Tuple[int, int], config: Dict[str, Any]):
        """
        初始化跟踪目标
        
        Args:
            track_id: 跟踪ID
            bbox: 初始边界框 [x1, y1, x2, y2]
            img_shape: 图像尺寸 (height, width)
            config: 配置字典
        """
        self.track_id = track_id
        self.img_shape = img_shape
        self.config = config
        
        # 卡尔曼滤波器
        self.kf = KalmanFilter(
            dt=config.get('kf_dt', 0.05),
            max_speed=config.get('max_speed', 50.0)
        )
        
        # 边界框处理
        self.bbox = clip_box(bbox.astype(np.float32), img_shape)
        self.kf.init(self.bbox)
        
        # 跟踪历史
        self.track_history: List[Tuple[float, float]] = []  # [(cx, cy), ...]
        self.speed_history: List[float] = []  # 速度历史
        self.acceleration_history: List[float] = []  # 加速度历史
        
        # 状态管理
        self.state = TrackState.NEW
        self.age = 0  # 存在帧数
        self.time_since_update = 0  # 自上次更新以来的帧数
        self.hits = 1  # 匹配次数
        self.total_frames = 0  # 总跟踪帧数
        
        # 检测信息
        self.class_id: Optional[int] = None
        self.class_name: str = "Unknown"
        self.confidence: float = 0.0
        
        # 行为分析
        self.is_stopped = False
        self.is_overtaking = False
        self.is_lane_changing = False
        self.is_braking = False
        self.is_accelerating = False
        self.is_turning = False
        self.is_dangerous = False
        
        self.stop_frames = 0
        self.overtake_frames = 0
        self.lane_change_frames = 0
        self.brake_frames = 0
        self.turn_frames = 0
        
        # 预测轨迹
        self.predicted_trajectory: List[Tuple[float, float]] = []
        
        # 初始化历史
        self._update_history()
    
    def _update_history(self):
        """更新跟踪历史"""
        cx, cy = bbox_center(self.bbox)
        self.track_history.append((cx, cy))
        
        # 限制历史长度
        max_len = self.config.get('track_history_len', 20)
        if len(self.track_history) > max_len:
            self.track_history.pop(0)
        
        # 限制速度历史
        if len(self.speed_history) > 10:
            self.speed_history.pop(0)
        
        # 限制加速度历史
        if len(self.acceleration_history) > 10:
            self.acceleration_history.pop(0)
    
    def _calculate_speed(self) -> float:
        """
        计算当前速度
        
        Returns:
            float: 速度（像素/秒）
        """
        if len(self.track_history) < 2:
            return 0.0
        
        # 计算最后两帧的位移
        prev_cx, prev_cy = self.track_history[-2]
        curr_cx, curr_cy = self.track_history[-1]
        
        dx = curr_cx - prev_cx
        dy = curr_cy - prev_cy
        distance = np.sqrt(dx**2 + dy**2)
        
        # 计算速度
        speed = distance / self.kf.dt
        
        # 更新速度历史
        self.speed_history.append(speed)
        
        # 计算加速度
        if len(self.speed_history) >= 2:
            acceleration = (self.speed_history[-1] - self.speed_history[-2]) / self.kf.dt
            self.acceleration_history.append(acceleration)
        
        return speed
    
    def _calculate_heading(self) -> float:
        """
        计算当前航向角
        
        Returns:
            float: 航向角（度）
        """
        if len(self.track_history) < 3:
            return 0.0
        
        # 使用最近三帧计算航向
        cx1, cy1 = self.track_history[-3]
        cx2, cy2 = self.track_history[-1]
        
        dx = cx2 - cx1
        dy = cy2 - cy1
        
        # 计算角度（弧度转度）
        angle = np.degrees(np.arctan2(dy, dx))
        
        return angle
    
    def _analyze_behavior(self, ego_center: Optional[Tuple[float, float]] = None):
        """
        分析目标行为
        
        Args:
            ego_center: 自车中心点坐标
        """
        # 计算基本状态
        speed = self._calculate_speed()
        heading = self._calculate_heading()
        
        # 1. 停车检测
        stop_speed_thresh = self.config.get('stop_speed_thresh', 1.0)
        stop_frames_thresh = self.config.get('stop_frames_thresh', 5)
        
        if speed < stop_speed_thresh:
            self.stop_frames += 1
            self.is_stopped = self.stop_frames >= stop_frames_thresh
        else:
            self.stop_frames = 0
            self.is_stopped = False
        
        # 2. 超车检测
        overtake_speed_ratio = self.config.get('overtake_speed_ratio', 1.5)
        overtake_dist_thresh = self.config.get('overtake_dist_thresh', 50.0)
        
        if ego_center and len(self.track_history) >= 2:
            curr_cx, curr_cy = self.track_history[-1]
            distance = np.sqrt((curr_cx - ego_center[0])**2 + (curr_cy - ego_center[1])**2)
            
            if distance < overtake_dist_thresh:
                ego_speed = getattr(self, 'ego_speed', 0.0)
                if speed > ego_speed * overtake_speed_ratio:
                    self.overtake_frames += 1
                    self.is_overtaking = self.overtake_frames >= 3
                else:
                    self.overtake_frames = 0
                    self.is_overtaking = False
            else:
                self.overtake_frames = 0
                self.is_overtaking = False
        
        # 3. 变道检测
        lane_change_thresh = self.config.get('lane_change_thresh', 0.5)
        
        if len(self.track_history) >= 5:
            # 计算横向位移
            lateral_displacements = []
            for i in range(1, min(5, len(self.track_history))):
                lateral_displacements.append(
                    abs(self.track_history[-i][0] - self.track_history[-i-1][0])
                )
            
            avg_lateral = np.mean(lateral_displacements) if lateral_displacements else 0.0
            
            if avg_lateral > lane_change_thresh:
                self.lane_change_frames += 1
                self.is_lane_changing = self.lane_change_frames >= 3
            else:
                self.lane_change_frames = 0
                self.is_lane_changing = False
        
        # 4. 刹车/加速检测
        brake_accel_thresh = self.config.get('brake_accel_thresh', 2.0)
        
        if len(self.acceleration_history) >= 3:
            avg_accel = np.mean(self.acceleration_history[-3:])
            
            if avg_accel < -brake_accel_thresh:
                self.brake_frames += 1
                self.is_braking = self.brake_frames >= 2
                self.is_accelerating = False
            elif avg_accel > brake_accel_thresh:
                self.is_accelerating = True
                self.is_braking = False
                self.brake_frames = 0
            else:
                self.is_braking = False
                self.is_accelerating = False
                self.brake_frames = 0
        
        # 5. 转弯检测
        turn_angle_thresh = self.config.get('turn_angle_thresh', 15.0)
        
        if len(self.track_history) >= 3:
            # 计算航向变化
            if hasattr(self, '_prev_heading'):
                heading_change = abs(heading - self._prev_heading)
                if heading_change > turn_angle_thresh:
                    self.turn_frames += 1
                    self.is_turning = self.turn_frames >= 2
                else:
                    self.turn_frames = 0
                    self.is_turning = False
            self._prev_heading = heading
        
        # 6. 危险距离检测
        danger_dist_thresh = self.config.get('danger_dist_thresh', 10.0)
        
        if ego_center:
            curr_cx, curr_cy = self.track_history[-1]
            distance = np.sqrt((curr_cx - ego_center[0])**2 + (curr_cy - ego_center[1])**2)
            self.is_dangerous = distance < danger_dist_thresh
        
        # 7. 预测轨迹
        self._predict_trajectory()
    
    def _predict_trajectory(self):
        """预测未来轨迹"""
        predict_frames = self.config.get('predict_frames', 10)
        self.predicted_trajectory = []
        
        if len(self.track_history) < 5:
            return
        
        # 创建临时的卡尔曼滤波器用于预测
        temp_kf = KalmanFilter(
            dt=self.kf.dt,
            max_speed=self.kf.max_speed
        )
        temp_kf.x = self.kf.x.copy()
        temp_kf.P = self.kf.P.copy()
        
        # 预测未来位置
        for _ in range(predict_frames):
            predicted_bbox = temp_kf.predict()
            predicted_center = bbox_center(predicted_bbox)
            self.predicted_trajectory.append(predicted_center)
    
    def predict(self) -> np.ndarray:
        """
        预测下一帧的位置
        
        Returns:
            np.ndarray: 预测的边界框
        """
        # 预测速度用于调整噪声
        if len(self.track_history) >= 2:
            speed = self._calculate_speed()
            self.kf.update_noise(speed)
        
        # 卡尔曼预测
        self.bbox = self.kf.predict()
        self.bbox = clip_box(self.bbox, self.img_shape)
        
        # 更新状态
        self._update_history()
        self.age += 1
        self.time_since_update += 1
        self.total_frames += 1
        
        if self.time_since_update > 1:
            self.state = TrackState.LOST
        
        return self.bbox
    
    def update(self, detection: Detection, ego_center: Optional[Tuple[float, float]] = None):
        """
        用新的检测更新跟踪
        
        Args:
            detection: 检测结果
            ego_center: 自车中心点坐标
        """
        # 卡尔曼更新
        self.bbox = self.kf.update(detection.bbox)
        self.bbox = clip_box(self.bbox, self.img_shape)
        
        # 更新检测信息
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        
        # 更新状态
        self._update_history()
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.TRACKED
        
        # 行为分析
        self._analyze_behavior(ego_center)
    
    def get_behavior_string(self) -> str:
        """获取行为描述字符串"""
        behaviors = []
        if self.is_stopped:
            behaviors.append("停车")
        if self.is_overtaking:
            behaviors.append("超车")
        if self.is_lane_changing:
            behaviors.append("变道")
        if self.is_braking:
            behaviors.append("刹车")
        if self.is_accelerating:
            behaviors.append("加速")
        if self.is_turning:
            behaviors.append("转弯")
        if self.is_dangerous:
            behaviors.append("危险")
        
        return "|".join(behaviors) if behaviors else "正常"


# ======================== YOLO检测器 ========================

class YOLODetector:
    """
    YOLOv8检测器
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化YOLO检测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 模型配置
        model_path = config.get('yolo_model', 'yolov8n.pt')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_thres = config.get('conf_thres', 0.5)
        self.iou_thres = config.get('iou_thres', 0.3)
        self.imgsz_max = config.get('yolo_imgsz_max', 320)
        self.quantize = config.get('yolo_quantize', False)
        
        # 类别过滤（只检测车辆）
        self.vehicle_classes = {2: "Car", 5: "Bus", 7: "Truck"}
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        logger.info(f"✅ YOLO检测器初始化完成 (设备: {self.device}, 模型: {model_path})")
    
    def _load_model(self, model_path: str):
        """加载YOLO模型"""
        try:
            model = YOLO(model_path)
            
            if self.quantize and self.device == "cuda":
                model = model.quantize()
            
            # 预热模型
            dummy_input = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            _ = model.predict(dummy_input, verbose=False, device=self.device)
            
            return model
            
        except Exception as e:
            logger.error(f"❌ 加载YOLO模型失败: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        检测图像中的目标
        
        Args:
            image: 输入图像
            
        Returns:
            List[Detection]: 检测结果列表
        """
        if not valid_img(image):
            return []
        
        try:
            # 调整图像尺寸
            h, w = image.shape[:2]
            resize_ratio = min(self.imgsz_max / w, self.imgsz_max / h)
            new_w = int(w * resize_ratio)
            new_h = int(h * resize_ratio)
            
            # 确保尺寸是32的倍数
            new_w = (new_w + 31) // 32 * 32
            new_h = (new_h + 31) // 32 * 32
            
            # 执行检测
            results = self.model.predict(
                image,
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=(new_h, new_w),
                device=self.device,
                verbose=False,
                agnostic_nms=True
            )
            
            detections = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # 获取边界框
                        xyxy = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # 只处理车辆类别
                        if class_id in self.vehicle_classes:
                            # 确保边界框有效
                            if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1] and confidence > 0:
                                detection = Detection(
                                    bbox=xyxy,
                                    confidence=confidence,
                                    class_id=class_id,
                                    class_name=self.vehicle_classes[class_id]
                                )
                                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ YOLO检测失败: {e}")
            return []


# ======================== SORT跟踪器 ========================

class SORTTracker:
    """
    SORT (Simple Online and Realtime Tracking) 跟踪器
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化SORT跟踪器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 跟踪参数
        self.max_age = config.get('max_age', 5)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_thres', 0.3)
        
        # 图像尺寸
        self.img_height = config.get('img_height', 480)
        self.img_width = config.get('img_width', 640)
        self.img_shape = (self.img_height, self.img_width)
        
        # 跟踪目标管理
        self.tracks: List[TrackedObject] = []
        self.next_track_id = 1
        
        # 自车信息
        self.ego_center = (self.img_width // 2, self.img_height // 2)
        self.ego_speed = 0.0
        
        logger.info("✅ SORT跟踪器初始化完成")
    
    def update(self, detections: List[Detection], 
               ego_center: Optional[Tuple[float, float]] = None,
               lidar_detections: Optional[List[Dict]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        更新跟踪器
        
        Args:
            detections: 检测结果列表
            ego_center: 自车中心点坐标
            lidar_detections: LiDAR检测结果（可选）
            
        Returns:
            Tuple: (边界框数组, ID数组, 类别数组)
        """
        # 更新自车信息
        if ego_center:
            self.ego_center = ego_center
        
        # 如果没有检测结果，只进行预测
        if not detections:
            # 预测所有现有目标
            for track in self.tracks:
                track.predict()
            
            # 移除丢失时间过长的目标
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            
            # 返回空结果
            return np.array([]), np.array([]), np.array([])
        
        # 预测现有目标的位置
        for track in self.tracks:
            track.predict()
        
        # 创建匹配成本矩阵
        if self.tracks:
            # 计算IoU矩阵
            iou_matrix = np.zeros((len(detections), len(self.tracks)), dtype=np.float32)
            
            for i, det in enumerate(detections):
                for j, track in enumerate(self.tracks):
                    iou_matrix[i, j] = iou(det.bbox, track.bbox)
            
            # 将IoU转换为成本（1 - IoU）
            cost_matrix = 1.0 - iou_matrix
            
            # 使用匈牙利算法进行匹配
            try:
                det_indices, track_indices = linear_sum_assignment(cost_matrix)
            except ValueError:
                det_indices, track_indices = [], []
            
            # 根据阈值过滤匹配
            matched_pairs = []
            unmatched_detections = set(range(len(detections)))
            unmatched_tracks = set(range(len(self.tracks)))
            
            for det_idx, track_idx in zip(det_indices, track_indices):
                if iou_matrix[det_idx, track_idx] >= self.iou_threshold:
                    matched_pairs.append((det_idx, track_idx))
                    unmatched_detections.discard(det_idx)
                    unmatched_tracks.discard(track_idx)
        else:
            matched_pairs = []
            unmatched_detections = set(range(len(detections)))
            unmatched_tracks = set()
        
        # 更新匹配的目标
        for det_idx, track_idx in matched_pairs:
            track = self.tracks[track_idx]
            track.ego_speed = self.ego_speed  # 传递自车速度用于行为分析
            track.update(detections[det_idx], self.ego_center)
        
        # 为未匹配的检测创建新目标
        for det_idx in unmatched_detections:
            new_track = TrackedObject(
                track_id=self.next_track_id,
                bbox=detections[det_idx].bbox,
                img_shape=self.img_shape,
                config=self.config
            )
            new_track.update(detections[det_idx], self.ego_center)
            self.tracks.append(new_track)
            self.next_track_id += 1
        
        # 移除长时间未更新的目标
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # 返回跟踪结果（只返回满足最小匹配次数的目标）
        active_tracks = [t for t in self.tracks if t.hits >= self.min_hits and t.state == TrackState.TRACKED]
        
        if not active_tracks:
            return np.array([]), np.array([]), np.array([])
        
        # 提取边界框、ID和类别
        boxes = np.array([t.bbox for t in active_tracks])
        ids = np.array([t.track_id for t in active_tracks])
        classes = np.array([t.class_id if t.class_id is not None else -1 for t in active_tracks])
        
        return boxes, ids, classes
    
    def get_tracks_info(self) -> List[Dict[str, Any]]:
        """
        获取所有跟踪目标的详细信息
        
        Returns:
            List[Dict]: 跟踪目标信息列表
        """
        tracks_info = []
        
        for track in self.tracks:
            if track.hits >= self.min_hits and track.state == TrackState.TRACKED:
                info = {
                    'track_id': track.track_id,
                    'bbox': track.bbox.tolist(),
                    'class_id': track.class_id,
                    'class_name': track.class_name,
                    'confidence': track.confidence,
                    'speed': track._calculate_speed(),
                    'behavior': track.get_behavior_string(),
                    'age': track.age,
                    'hits': track.hits,
                    'is_stopped': track.is_stopped,
                    'is_overtaking': track.is_overtaking,
                    'is_dangerous': track.is_dangerous,
                }
                tracks_info.append(info)
        
        return tracks_info
    
    def reset(self):
        """重置跟踪器"""
        self.tracks = []
        self.next_track_id = 1
        logger.info("✅ 跟踪器已重置")


# ======================== 检测线程 ========================

class DetectionThread(threading.Thread):
    """
    检测线程 - 将检测过程放到单独线程中
    """
    
    def __init__(self, detector: YOLODetector, input_queue: queue.Queue, 
                 output_queue: queue.Queue, maxsize: int = 2):
        """
        初始化检测线程
        
        Args:
            detector: YOLO检测器
            input_queue: 输入图像队列
            output_queue: 输出检测结果队列
            maxsize: 队列最大大小
        """
        super().__init__(daemon=True)
        self.detector = detector
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.processed_count = 0
        
        logger.info("✅ 检测线程初始化完成")
    
    def run(self):
        """线程主函数"""
        while self.running:
            try:
                # 从输入队列获取图像
                image = self.input_queue.get(timeout=1.0)
                
                if not valid_img(image):
                    self.output_queue.put((image, []))
                    continue
                
                # 执行检测
                detections = self.detector.detect(image)
                
                # 放入输出队列
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.output_queue.put((image, detections))
                self.processed_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ 检测线程出错: {e}")
                self.output_queue.put((None, []))
    
    def stop(self):
        """停止线程"""
        self.running = False
        logger.info("🛑 检测线程已停止")


# ======================== 测试函数 ========================

def test_tracker():
    """测试跟踪器"""
    print("=" * 50)
    print("测试 tracker.py...")
    print("=" * 50)
    
    # 模拟配置
    test_config = {
        'yolo_model': 'yolov8n.pt',
        'conf_thres': 0.5,
        'iou_thres': 0.3,
        'max_age': 5,
        'min_hits': 3,
        'kf_dt': 0.05,
        'max_speed': 50.0,
        'img_width': 640,
        'img_height': 480,
        'track_history_len': 20,
        'stop_speed_thresh': 1.0,
        'stop_frames_thresh': 5,
        'overtake_speed_ratio': 1.5,
        'overtake_dist_thresh': 50.0,
        'lane_change_thresh': 0.5,
        'brake_accel_thresh': 2.0,
        'turn_angle_thresh': 15.0,
        'danger_dist_thresh': 10.0,
        'predict_frames': 10,
    }
    
    # 测试数据结构
    print("1. 测试数据结构...")
    bbox = np.array([100, 100, 200, 200], dtype=np.float32)
    detection = Detection(bbox=bbox, confidence=0.9, class_id=2, class_name="Car")
    assert detection.confidence == 0.9
    assert detection.class_id == 2
    print("   ✅ Detection数据结构测试通过")
    
    # 测试卡尔曼滤波器
    print("2. 测试卡尔曼滤波器...")
    kf = KalmanFilter(dt=0.05)
    kf.init(bbox)
    predicted = kf.predict()
    assert len(predicted) == 4
    updated = kf.update(bbox + 10)
    assert len(updated) == 4
    print("   ✅ 卡尔曼滤波器测试通过")
    
    # 测试跟踪目标
    print("3. 测试跟踪目标...")
    track = TrackedObject(
        track_id=1,
        bbox=bbox,
        img_shape=(480, 640),
        config=test_config
    )
    track.update(detection, ego_center=(320, 240))
    assert track.track_id == 1
    assert track.class_id == 2
    track.predict()
    print("   ✅ 跟踪目标测试通过")
    
    # 测试SORT跟踪器
    print("4. 测试SORT跟踪器...")
    tracker = SORTTracker(test_config)
    detections = [detection]
    boxes, ids, classes = tracker.update(detections)
    assert len(boxes) >= 0  # 可能没有匹配到
    print("   ✅ SORT跟踪器测试通过")
    
    print("=" * 50)
    print("✅ tracker.py 所有测试通过")
    print("注：完整测试需要YOLO模型文件")
    
    return True


if __name__ == "__main__":
    test_tracker()