"""
DeepSORT 目标跟踪器
用于稳定追踪障碍物，解决单帧漏检/误检问题
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class KalmanBoxTracker:
    """
    卡尔曼滤波器用于跟踪边界框
    基于 Sort 算法中的实现
    """
    count = 0

    def __init__(self, bbox):
        """
        初始化卡尔曼滤波器
        bbox: [x1, y1, x2, y2]
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # 测量噪声矩阵
        self.kf.R[2:, 2:] *= 10.
        # 过程噪声矩阵
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        # 过程噪声
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = deque(maxlen=20)
        self.hits = 0

    def update(self, bbox):
        """
        更新卡尔曼滤波器
        """
        self.time_since_update = 0
        self.history = deque(maxlen=20)
        self.hits += 1
        self.kf.update(self._convert_bbox_to_z(bbox))

    def predict(self):
        """
        预测下一帧位置
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        self.time_since_update += 1
        return self.history[-1]

    def get_state(self):
        """
        获取当前边界框状态
        """
        return self._convert_x_to_bbox(self.kf.x)

    @staticmethod
    def _convert_bbox_to_z(bbox):
        """
        将 [x1,y1,x2,y2] 转换为 [x,y,a,h]
        x,y 是中心点, a 是宽高比, h 是高度
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        a = w / h
        return np.array([x, y, a, h]).reshape((4, 1))

    @staticmethod
    def _convert_x_to_bbox(x, score=None):
        """
        将 [x,y,a,h] 转换回 [x1,y1,x2,y2]
        """
        w = x[2] * x[3]
        h = x[3]
        if score is None:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))[0]
        else:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))[0]


class KalmanFilter:
    """
    简化的卡尔曼滤波器实现
    """
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # 状态向量 x
        self.x = np.zeros((dim_x, 1))
        # 状态协方差矩阵 P
        self.P = np.eye(dim_x)
        # 状态转移矩阵 F
        self.F = np.eye(dim_x)
        # 观测矩阵 H
        self.H = np.zeros((dim_z, dim_x))
        # 观测噪声矩阵 R
        self.R = np.eye(dim_z)
        # 过程噪声矩阵 Q
        self.Q = np.eye(dim_x)
        
    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
    def update(self, z):
        z = np.array(z).reshape((self.dim_z, 1))
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(self.dim_x) - np.dot(K, self.H), self.P)


class DeepSORTTracker:
    """
    DeepSORT 目标跟踪器
    使用卡尔曼滤波 + 匈牙利算法进行目标关联
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        初始化跟踪器
        max_age: 最大未更新帧数，超过则删除轨迹
        min_hits: 最小命中帧数，才认为跟踪有效
        iou_threshold: IOU匹配阈值
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []  # KalmanBoxTracker 列表
        self.frame_count = 0
        
    def update(self, detections, confidences=None):
        """
        更新跟踪器
        detections: Nx4 数组, 每行 [x1, y1, x2, y2]
        confidences: N 数组, 置信度(可选)
        返回: Mx5 数组, 每行 [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1
        
        # 预测所有轨迹的下一位置
        for track in self.tracks:
            track.predict()
        
        # 匹配检测框与跟踪轨迹
        matched, unmatched_dets, unmatched_trks = self._associate(detections)
        
        # 更新匹配的轨迹
        for det_idx, track_idx in matched:
            self.tracks[track_idx].update(detections[det_idx])
        
        # 为未匹配的检测创建新轨迹
        for det_idx in unmatched_dets:
            self.tracks.append(KalmanBoxTracker(detections[det_idx]))
        
        # 删除未匹配的长时间未更新轨迹
        i = len(self.tracks)
        for track in reversed(self.tracks):
            i -= 1
            if track.time_since_update > self.max_age:
                self.tracks.pop(i)
        
        # 返回有效跟踪结果
        result = []
        for track in self.tracks:
            if track.time_since_update < self.max_age and track.hits >= self.min_hits:
                bbox = track.get_state()
                result.append([bbox[0], bbox[1], bbox[2], bbox[3], track.id])
        
        return np.array(result) if result else np.empty((0, 5))
    
    def _associate(self, detections):
        """
        使用匈牙利算法匹配检测框与跟踪轨迹
        基于IOU计算代价矩阵
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        # 计算IOU代价矩阵
        iou_matrix = np.zeros((len(detections), len(self.tracks)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, track in enumerate(self.tracks):
                iou_matrix[d, t] = self._iou(det, track.get_state())
        
        # 匈牙利算法求解
        matched_indices = []
        if iou_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] < self.iou_threshold:
                    continue
                matched_indices.append([r, c])
        
        unmatched_detections = [d for d in range(len(detections)) if d not in [m[0] for m in matched_indices]]
        unmatched_tracks = [t for t in range(len(self.tracks)) if t not in [m[1] for m in matched_indices]]
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    @staticmethod
    def _iou(bbox1, bbox2):
        """
        计算两个边界框的IOU
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def get_track_history(self, track_id):
        """获取指定轨迹的历史"""
        for track in self.tracks:
            if track.id == track_id:
                return list(track.history)
        return []
    
    def reset(self):
        """重置跟踪器"""
        self.tracks = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
