# main.py
# CARLA目标检测与跟踪系统 - 优化版
# 主要功能：在CARLA仿真环境中实现实时目标检测、多目标跟踪和自动驾驶控制

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
import json
import logging
from pathlib import Path
from scipy.optimize import linear_sum_assignment  # 匈牙利算法用于目标匹配


# -------------------------- 日志配置 --------------------------
def setup_logging():
    """设置日志系统 - 同时输出到控制台和文件"""
    # 创建logger实例
    logger = logging.getLogger("carla_tracking")
    logger.setLevel(logging.INFO)  # 设置日志级别

    # 创建控制台处理器 - 输出到终端
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建文件处理器 - 输出到文件
    file_handler = logging.FileHandler("carla_tracking.log", mode='w')
    file_handler.setLevel(logging.DEBUG)  # 文件日志更详细

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()  # 初始化全局logger


# -------------------------- 配置管理 --------------------------
class ConfigManager:
    """配置管理器 - 统一管理程序所有配置参数"""

    # 使用单例模式思想，集中管理所有配置参数
    def __init__(self, config_file=None):
        self.config = self._load_default_config()  # 加载默认配置

        if config_file and os.path.exists(config_file):
            self.load_config(config_file)  # 从文件加载用户配置

    def _load_default_config(self):
        """加载默认配置 - 程序运行的基准参数"""
        return {
            'carla': {  # CARLA仿真环境配置
                'host': 'localhost',
                'port': 2000,
                'timeout': 10.0,
                'sync_mode': True,  # 同步模式确保时间步一致
                'fixed_delta_seconds': 0.05  # 时间步长（20Hz）
            },
            'detection': {  # 目标检测配置
                'conf_thres': 0.15,  # 置信度阈值
                'iou_thres': 0.4,  # NMS的IOU阈值
                'model_type': 'yolov5m',  # YOLO模型类型
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
            },
            'tracking': {  # 目标跟踪配置
                'max_age': 30,  # 最大丢失帧数
                'min_hits': 3,  # 最小命中次数
                'iou_threshold': 0.3,  # 匹配阈值
                'use_acceleration': True,  # 是否使用加速度模型
                'use_adaptive_threshold': True,  # 自适应阈值
                'depth_aware': True,  # 使用深度信息
                'min_confidence': 0.1  # 最小置信度
            },
            'vehicle': {  # 主车辆控制配置
                'max_speed': 50.0,  # 最大速度(km/h)
                'target_speed': 30.0,  # 目标巡航速度
                'safety_distance': 15.0,  # 安全距离(m)
                'enable_physics': True  # 启用物理模拟
            },
            'npc': {  # NPC车辆配置
                'count': 20,  # NPC数量
                'min_distance': 20.0  # 最小生成距离
            },
            'camera': {  # 相机传感器配置
                'width': 800,  # 图像宽度
                'height': 600,  # 图像高度
                'fov': 90  # 视野角度
            },
            'performance': {  # 性能监控配置
                'show_panel': True,  # 显示性能面板
                'log_interval': 100,  # 日志间隔（帧）
                'max_fps': 30  # 最大FPS限制
            }
        }

    def load_config(self, config_file):
        """从JSON文件加载配置 - 支持增量更新"""
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self._update_config(self.config, user_config)  # 递归合并配置
                logger.info(f"加载配置文件: {config_file}")
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}")

    def _update_config(self, base, update):
        """递归更新配置字典 - 深度合并"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # 递归更新子字典
                self._update_config(base[key], value)
            else:
                # 直接赋值
                base[key] = value

    def save_config(self, config_file):
        """保存配置到JSON文件 - 便于下次使用"""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)  # 美化输出
            logger.info(f"配置文件已保存: {config_file}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")

    def get(self, key, default=None):
        """获取配置值 - 支持点分隔符（如'carla.host'）"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default  # 键不存在时返回默认值
        return value


# -------------------------- 优化的卡尔曼滤波器 --------------------------
class OptimizedKalmanFilter:
    """优化的卡尔曼滤波器 - 用于目标状态估计和预测"""

    # 卡尔曼滤波器是多目标跟踪的核心，用于估计目标的位置、速度等状态
    def __init__(self, dt=1.0, use_acceleration=True):
        self.dt = dt  # 时间步长
        self.use_acceleration = use_acceleration  # 是否使用加速度模型

        if use_acceleration:
            # 8维状态: [x, y, w, h, vx, vy, vw, vh]
            # 包含位置、尺寸及其一阶导数（速度）
            self.state_dim = 8
            self.meas_dim = 4  # 测量维度（只能直接观测位置和尺寸）

            # 状态转移矩阵 F - 描述状态如何随时间变化
            self.F = np.eye(self.state_dim, dtype=np.float32)
            self.F[0, 4] = dt  # x + vx*dt
            self.F[1, 5] = dt  # y + vy*dt
            self.F[2, 6] = dt  # w + vw*dt
            self.F[3, 7] = dt  # h + vh*dt

            # 测量矩阵 H - 将状态空间映射到测量空间
            self.H = np.zeros((self.meas_dim, self.state_dim), dtype=np.float32)
            self.H[0, 0] = 1  # 测量x
            self.H[1, 1] = 1  # 测量y
            self.H[2, 2] = 1  # 测量w
            self.H[3, 3] = 1  # 测量h

            # 初始化状态向量
            self.x = np.zeros((self.state_dim, 1), dtype=np.float32)

            # 初始化协方差矩阵 P - 状态估计的不确定性
            self.P = np.eye(self.state_dim, dtype=np.float32) * 1000  # 初始高不确定性

            # 过程噪声协方差 Q - 系统模型的不确定性
            self.Q = np.eye(self.state_dim, dtype=np.float32) * 0.1
            self.Q[4:, 4:] = np.eye(4) * 0.01  # 速度噪声更小

            # 测量噪声协方差 R - 传感器测量的不确定性
            self.R = np.eye(self.meas_dim, dtype=np.float32) * 5

        else:
            # 简化版：4维状态，仅包含位置和尺寸
            self.state_dim = 4
            self.meas_dim = 4

            # 简化模型
            self.F = np.eye(self.state_dim, dtype=np.float32)
            self.H = np.eye(self.state_dim, dtype=np.float32)
            self.x = np.zeros((self.state_dim, 1), dtype=np.float32)
            self.P = np.eye(self.state_dim, dtype=np.float32) * 1000
            self.Q = np.eye(self.state_dim, dtype=np.float32) * 0.5
            self.R = np.eye(self.state_dim, dtype=np.float32) * 10

        # 预计算转置矩阵 - 优化性能
        self.H_T = self.H.T
        self.F_T = self.F.T

    def predict(self):
        """预测下一时刻状态 - 卡尔曼滤波预测步骤"""
        # 状态预测：x_k = F * x_{k-1}
        self.x = self.F @ self.x

        # 协方差预测：P_k = F * P_{k-1} * F^T + Q
        self.P = self.F @ self.P @ self.F_T + self.Q

        # 返回预测的边界框（位置和尺寸）
        if self.use_acceleration:
            return self.x[:4].flatten()  # 取前4维（x,y,w,h）
        else:
            return self.x.flatten()

    def update(self, measurement):
        """使用测量值更新状态 - 卡尔曼滤波更新步骤"""
        measurement = np.array(measurement, dtype=np.float32).reshape(-1, 1)

        # 计算残差：y = z - H * x
        y = measurement - self.H @ self.x

        # 计算残差协方差：S = H * P * H^T + R
        S = self.H @ self.P @ self.H_T + self.R

        # 计算卡尔曼增益：K = P * H^T * S^{-1}
        K = self.P @ self.H_T @ np.linalg.inv(S)

        # 状态更新：x = x + K * y
        self.x = self.x + K @ y

        # 协方差更新：P = (I - K * H) * P
        I = np.eye(self.state_dim, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        return self.x[:4].flatten()


# -------------------------- 优化的跟踪目标 --------------------------
class OptimizedTrack:
    """优化的跟踪目标 - 表示一个被跟踪的车辆目标"""
    # __slots__优化内存使用，固定属性列表
    __slots__ = ('id', 'kf', 'bbox', 'center', 'width', 'height', 'aspect_ratio',
                 'hits', 'age', 'total_visible_count', 'consecutive_invisible_count',
                 'history', 'distance_history', 'velocity_history',
                 'current_distance', 'velocity', 'confidence', 'class_id',
                 'is_confirmed', 'last_seen')

    def __init__(self, bbox, track_id, class_id=2, use_acceleration=True):
        # 目标标识
        self.id = track_id
        self.class_id = class_id  # 目标类别（2=car, 3=motorcycle等）

        # 初始化卡尔曼滤波器
        self.kf = OptimizedKalmanFilter(use_acceleration=use_acceleration)

        # 初始边界框 [x1, y1, x2, y2]
        self.bbox = np.array(bbox, dtype=np.float32)

        # 边界框中心点
        self.center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                               dtype=np.float32)

        # 尺寸信息
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.aspect_ratio = self.width / max(self.height, 1e-6)  # 宽高比，防止除零

        # 初始化滤波器状态
        if use_acceleration:
            # 设置初始状态：中心点位置和尺寸
            self.kf.x[:4] = np.array([self.center[0], self.center[1],
                                      self.width, self.height],
                                     dtype=np.float32).reshape(-1, 1)
        else:
            self.kf.x = np.array([self.center[0], self.center[1],
                                  self.width, self.height],
                                 dtype=np.float32).reshape(-1, 1)

        # 统计信息
        self.hits = 1  # 成功匹配次数
        self.age = 0  # 目标年龄（帧数）
        self.total_visible_count = 1  # 总可见次数
        self.consecutive_invisible_count = 0  # 连续不可见次数

        # 历史记录（使用deque限制长度）
        self.history = deque(maxlen=30)  # 轨迹历史
        self.history.append(self.center.copy())

        self.distance_history = deque(maxlen=10)  # 距离历史
        self.velocity_history = deque(maxlen=10)  # 速度历史

        # 当前状态
        self.current_distance = None  # 当前距离（米）
        self.velocity = 0.0  # 当前速度
        self.confidence = 1.0  # 跟踪置信度
        self.is_confirmed = False  # 是否确认为有效目标
        self.last_seen = 0  # 最后被看到的帧编号

        self.mark_seen()  # 标记为当前可见

    def mark_seen(self):
        """标记为目标被看到 - 重置不可见计数"""
        self.consecutive_invisible_count = 0
        self.last_seen = self.age

    def mark_missed(self):
        """标记为目标丢失 - 增加不可见计数"""
        self.consecutive_invisible_count += 1

    def predict(self):
        """预测下一时刻状态 - 使用卡尔曼滤波"""
        self.age += 1  # 年龄增加

        # 使用卡尔曼滤波器预测
        predicted_state = self.kf.predict()

        # 解析预测状态
        if self.kf.use_acceleration:
            cx, cy, w, h = predicted_state
        else:
            cx, cy, w, h = predicted_state

        # 确保尺寸合理（最小值限制）
        w = max(w, 1.0)
        h = max(h, 1.0)

        # 更新边界框 [x1, y1, x2, y2]
        self.bbox = np.array([
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2
        ], dtype=np.float32)

        # 更新中心点
        self.center = np.array([cx, cy], dtype=np.float32)
        self.width = w
        self.height = h

        # 添加到历史记录
        self.history.append(self.center.copy())

        return self.bbox

    def update(self, bbox, confidence=1.0, distance=None):
        """使用检测结果更新跟踪目标"""
        # 从边界框计算中心点和尺寸
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        measurement = np.array([cx, cy, w, h], dtype=np.float32)

        # 更新卡尔曼滤波器
        updated_state = self.kf.update(measurement)

        # 解析更新后的状态
        if self.kf.use_acceleration:
            cx, cy, w, h = updated_state
        else:
            cx, cy, w, h = updated_state

        # 确保尺寸合理
        w = max(w, 1.0)
        h = max(h, 1.0)

        # 更新边界框
        self.bbox = np.array([
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2
        ], dtype=np.float32)

        # 更新属性
        self.center = np.array([cx, cy], dtype=np.float32)
        self.width = w
        self.height = h
        self.aspect_ratio = w / max(h, 1e-6)

        # 更新统计信息
        self.hits += 1
        self.total_visible_count += 1
        self.consecutive_invisible_count = 0
        self.last_seen = self.age

        # 更新距离信息
        if distance is not None:
            self.distance_history.append(distance)
            if self.distance_history:
                # 使用中位数减少噪声影响
                self.current_distance = float(np.median(list(self.distance_history)))

        # 更新置信度（指数加权平均）
        alpha = 0.3  # 平滑因子
        self.confidence = alpha * confidence + (1 - alpha) * self.confidence

        # 确认目标（经过多次匹配后确认）
        if self.hits >= 3 and not self.is_confirmed:
            self.is_confirmed = True

        # 添加到历史记录
        self.history.append(self.center.copy())

    def get_velocity(self):
        """计算目标速度 - 基于历史轨迹"""
        if len(self.history) < 2:
            return 0.0

        # 使用最近5个位置计算速度
        recent_history = list(self.history)[-5:]
        if len(recent_history) < 2:
            return 0.0

        distances = []
        for i in range(1, len(recent_history)):
            # 计算相邻帧间的位移
            dx = recent_history[i][0] - recent_history[i - 1][0]
            dy = recent_history[i][1] - recent_history[i - 1][1]
            distance = np.sqrt(dx * dx + dy * dy)
            distances.append(distance)

        if distances:
            avg_distance = np.mean(distances)
            # 假设每秒30帧，将像素位移转换为速度
            self.velocity = avg_distance * 30
            self.velocity_history.append(self.velocity)

            # 使用中值滤波平滑速度（抗噪声）
            if len(self.velocity_history) >= 3:
                self.velocity = float(np.median(list(self.velocity_history)))

        return self.velocity

    def get_similarity_score(self, bbox):
        """计算与给定边界框的相似度分数 - 用于数据关联"""
        # 计算新检测框的中心点和尺寸
        other_cx = (bbox[0] + bbox[2]) / 2
        other_cy = (bbox[1] + bbox[3]) / 2
        other_w = bbox[2] - bbox[0]
        other_h = bbox[3] - bbox[1]

        # 1. 位置相似度（基于距离）
        dx = other_cx - self.center[0]
        dy = other_cy - self.center[1]
        distance = np.sqrt(dx * dx + dy * dy)

        # 自适应距离阈值：基于目标尺寸
        max_distance = max(self.width, self.height) * 3

        if distance > max_distance:
            return 0.0  # 距离太远，完全不相似

        # 距离相似度（指数衰减）
        position_similarity = np.exp(-distance / (max_distance * 0.5))

        # 2. 尺寸相似度
        size_ratio1 = other_w / max(self.width, 1e-6)
        size_ratio2 = other_h / max(self.height, 1e-6)

        # 尺寸变化过大则拒绝
        if size_ratio1 > 2 or size_ratio1 < 0.5 or size_ratio2 > 2 or size_ratio2 < 0.5:
            return 0.0

        size_similarity = np.exp(-abs(size_ratio1 - 1) - abs(size_ratio2 - 1))

        # 3. 长宽比相似度
        other_aspect = other_w / max(other_h, 1e-6)
        aspect_similarity = np.exp(-abs(other_aspect - self.aspect_ratio))

        # 4. 综合相似度（加权平均）
        similarity = 0.5 * position_similarity + 0.3 * size_similarity + 0.2 * aspect_similarity

        return similarity

    @property
    def is_reliable(self):
        """判断目标是否可靠 - 用于结果输出筛选"""
        # 已确认且最近没有丢失太多帧
        return self.is_confirmed and self.consecutive_invisible_count < 5

    @property
    def should_delete(self):
        """判断是否应该删除目标 - 用于跟踪器清理"""
        # 未确认目标：快速删除
        if not self.is_confirmed and self.consecutive_invisible_count >= 3:
            return True
        # 已确认目标：容忍更多丢失帧
        if self.is_confirmed and self.consecutive_invisible_count >= 10:
            return True
        return False


# -------------------------- 优化的SORT跟踪器 --------------------------
class OptimizedSORT:
    """优化的SORT跟踪器 - Simple Online and Realtime Tracking的改进版"""

    # SORT算法核心：检测-预测-匹配的循环

    def __init__(self, config=None):
        # 默认配置
        self.config = {
            'max_age': 30,  # 最大丢失帧数
            'min_hits': 3,  # 最小命中次数
            'iou_threshold': 0.3,  # IOU匹配阈值
            'use_acceleration': True,  # 使用加速度模型
            'use_adaptive_threshold': True,  # 自适应阈值
            'min_confidence': 0.1  # 最小置信度
        }

        # 更新配置
        if config:
            self.config.update(config)

        # 跟踪目标列表
        self.tracks = []
        self.next_id = 1  # 下一个跟踪ID

        # 缓存（优化性能）
        self._iou_cache = {}  # IOU计算结果缓存
        self._prediction_cache = {}  # 预测结果缓存

        logger.info(f"初始化优化SORT跟踪器")

    def update(self, detections, depths=None):
        """更新跟踪器状态 - 主入口函数"""
        # 1. 预处理检测结果
        detections = self._preprocess_detections(detections)

        # 2. 预测所有现有跟踪目标
        self._predict_tracks()

        # 3. 匹配检测和跟踪目标（数据关联）
        matched_pairs, unmatched_detections, unmatched_tracks = self._match_detections(detections)

        # 4. 更新匹配的跟踪目标
        self._update_matched_tracks(matched_pairs, detections, depths)

        # 5. 为未匹配的检测创建新跟踪目标
        self._create_new_tracks(unmatched_detections, detections, depths)

        # 6. 标记未匹配的跟踪目标为丢失
        self._mark_missed_tracks(unmatched_tracks)

        # 7. 清理无效的跟踪目标
        self._cleanup_tracks()

        # 8. 获取输出结果
        output_tracks = self._get_output_tracks()

        return output_tracks

    def _preprocess_detections(self, detections):
        """预处理检测结果 - 过滤低质量检测"""
        if len(detections) == 0:
            return []

        # 转换为numpy数组便于操作
        detections_np = np.array(detections, dtype=np.float32)

        # 过滤低置信度检测
        if detections_np.shape[1] > 4:  # 包含置信度
            conf_mask = detections_np[:, 4] >= self.config['min_confidence']
            detections_np = detections_np[conf_mask]

        # 过滤无效边界框（x1<x2, y1<y2）
        valid_mask = (detections_np[:, 2] > detections_np[:, 0]) & \
                     (detections_np[:, 3] > detections_np[:, 1])

        return detections_np[valid_mask]

    def _predict_tracks(self):
        """预测所有跟踪目标的下一个状态"""
        for track in self.tracks:
            predicted_bbox = track.predict()
            self._prediction_cache[track.id] = predicted_bbox  # 缓存预测结果

    def _match_detections(self, detections):
        """匹配检测和跟踪目标 - 匈牙利算法"""
        if len(self.tracks) == 0 or len(detections) == 0:
            # 无匹配情况
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        # 构建成本矩阵（代价矩阵）
        cost_matrix = self._build_cost_matrix(detections)

        if cost_matrix.size == 0:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        # 使用匈牙利算法进行最优匹配
        try:
            # linear_sum_assignment最小化总成本
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        except Exception as e:
            logger.error(f"匈牙利算法错误: {e}")
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        # 提取匹配对（成本低于阈值）
        matched_pairs = []
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < self._get_adaptive_threshold():
                matched_pairs.append((row, col))

        # 找出未匹配的检测和跟踪目标
        all_rows = set(range(len(self.tracks)))
        all_cols = set(range(len(detections)))
        matched_rows = set([p[0] for p in matched_pairs])
        matched_cols = set([p[1] for p in matched_pairs])

        unmatched_tracks = list(all_rows - matched_rows)
        unmatched_detections = list(all_cols - matched_cols)

        return matched_pairs, unmatched_detections, unmatched_tracks

    def _build_cost_matrix(self, detections):
        """构建成本矩阵 - 行：跟踪目标，列：检测结果"""
        n_tracks = len(self.tracks)
        n_detections = len(detections)

        # 初始化成本矩阵（默认高成本）
        cost_matrix = np.ones((n_tracks, n_detections), dtype=np.float32) * 1000

        # 为每个跟踪目标-检测对计算成本
        for i, track in enumerate(self.tracks):
            predicted_bbox = self._prediction_cache.get(track.id, track.bbox)

            for j, detection in enumerate(detections):
                # 提取检测边界框
                det_bbox = detection[:4]

                # 计算IOU（交并比）
                iou = self._calculate_iou(predicted_bbox, det_bbox)

                # 计算相似度（特征匹配）
                similarity = track.get_similarity_score(det_bbox)

                # 综合成本 = 1 - 相似度 * IOU
                # 成本越低，匹配可能性越高
                cost = 1.0 - similarity * iou

                cost_matrix[i, j] = cost

        return cost_matrix

    def _calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的IOU - 带缓存优化"""
        # 使用缓存避免重复计算
        key = (tuple(bbox1), tuple(bbox2))
        if key in self._iou_cache:
            return self._iou_cache[key]

        # 计算交集矩形
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # 检查是否有交集
        if x2 <= x1 or y2 <= y1:
            self._iou_cache[key] = 0.0
            return 0.0

        # 计算交集面积
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # 计算IOU：交集/并集
        iou = intersection / (area1 + area2 - intersection + 1e-6)  # 加小值防止除零

        # 缓存结果
        self._iou_cache[key] = iou

        return iou

    def _get_adaptive_threshold(self):
        """获取自适应匹配阈值 - 将IOU阈值转换为成本阈值"""
        return 1.0 - self.config['iou_threshold']  # IOU=0.3 -> 成本=0.7

    def _update_matched_tracks(self, matched_pairs, detections, depths):
        """更新匹配的跟踪目标"""
        for track_idx, det_idx in matched_pairs:
            track = self.tracks[track_idx]
            detection = detections[det_idx]

            # 提取检测信息
            bbox = detection[:4]
            confidence = detection[4] if len(detection) > 4 else 1.0

            # 获取深度信息
            depth = depths[det_idx] if depths and det_idx < len(depths) else None

            # 更新跟踪目标
            track.update(bbox, confidence, depth)

    def _create_new_tracks(self, unmatched_detections, detections, depths):
        """为未匹配的检测创建新跟踪目标"""
        for det_idx in unmatched_detections:
            detection = detections[det_idx]

            # 提取检测信息
            bbox = detection[:4]
            confidence = detection[4] if len(detection) > 4 else 1.0
            class_id = int(detection[5]) if len(detection) > 5 else 2  # 默认类别2（car）

            # 获取深度信息
            depth = depths[det_idx] if depths and det_idx < len(depths) else None

            # 创建新跟踪目标
            new_track = OptimizedTrack(
                bbox=bbox,
                track_id=self.next_id,
                class_id=class_id,
                use_acceleration=self.config['use_acceleration']
            )

            # 设置初始距离
            if depth is not None:
                new_track.distance_history.append(depth)
                new_track.current_distance = depth

            # 添加到跟踪列表
            self.tracks.append(new_track)
            self.next_id += 1  # ID自增

    def _mark_missed_tracks(self, unmatched_tracks):
        """标记未匹配的跟踪目标为丢失"""
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

    def _cleanup_tracks(self):
        """清理无效的跟踪目标 - 删除应删除的目标"""
        self.tracks = [track for track in self.tracks if not track.should_delete]

    def _get_output_tracks(self):
        """获取输出跟踪结果 - 只返回可靠目标"""
        output = []

        for track in self.tracks:
            if track.is_reliable:
                output.append([
                    track.bbox[0],  # x1
                    track.bbox[1],  # y1
                    track.bbox[2],  # x2
                    track.bbox[3],  # y2
                    track.id,  # track_id
                    track.class_id,  # class_id
                    track.confidence  # confidence
                ])

        return output

    def get_tracks_info(self):
        """获取跟踪目标详细信息 - 用于调试和可视化"""
        info = []
        for track in self.tracks:
            if track.is_reliable:
                info.append({
                    'id': track.id,
                    'bbox': track.bbox.tolist(),
                    'center': track.center.tolist(),
                    'velocity': track.get_velocity(),
                    'distance': track.current_distance,
                    'confidence': track.confidence,
                    'age': track.age,
                    'hits': track.hits,
                    'history': [h.tolist() for h in track.history]
                })
        return info


# -------------------------- YOLOv5检测模型 --------------------------
from ultralytics import YOLO  # YOLOv8的Python接口


def load_detection_model(model_type):
    """加载检测模型 - YOLOv5/v8"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模型路径映射
    model_paths = {
        'yolov5s': r"D:\yolo\yolov5s.pt",
        'yolov5su': r"D:\yolo\yolov5su.pt",
        'yolov5m': r"D:\yolo\yolov5m.pt",
        'yolov5mu': r"D:\yolo\yolov5mu.pt",
        'yolov5x': r"D:\yolo\yolov5x.pt"
    }

    # 模型类型回退逻辑
    if model_type not in model_paths:
        if 'su' in model_type.lower():
            model_type = 'yolov5su'
        elif 'mu' in model_type.lower():
            model_type = 'yolov5mu'
        else:
            model_type = 'yolov5m'  # 默认中等大小模型

    model_path = model_paths.get(model_type)

    # 检查模型文件是否存在
    if not model_path or not os.path.exists(model_path):
        # 尝试其他可用模型
        for key, path in model_paths.items():
            if os.path.exists(path):
                model_type = key
                model_path = path
                logger.info(f"使用备用模型：{model_type}")
                break

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"无法找到模型文件")

    # 加载模型
    model = YOLO(model_path)
    model.to(device)

    # 半精度浮点数加速（CUDA）
    if device == 'cuda':
        model.half()

    # 预热模型（避免首次推理延迟）
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640, device=device)
        if device == 'cuda':
            dummy_input = dummy_input.half()
        _ = model(dummy_input)

    logger.info(f"模型加载成功：{model_type} (设备：{device})")
    return model, model.names  # 返回模型和类别名称


# -------------------------- 车辆控制器 --------------------------
class VehicleController:
    """车辆运动控制器 - PID控制 + 障碍物避让"""

    def __init__(self, vehicle, config):
        self.vehicle = vehicle
        self.config = config

        # 控制参数
        self.max_speed = config.get('vehicle.max_speed', 50.0)
        self.target_speed = config.get('vehicle.target_speed', 30.0)
        self.safety_distance = config.get('vehicle.safety_distance', 15.0)

        # 控制状态
        self.control_state = {
            'throttle': 0.0,
            'steer': 0.0,
            'brake': 0.0,
            'reverse': False,
            'speed': 0.0
        }

        # PID控制器参数
        self.Kp = 0.01  # 比例项
        self.Ki = 0.001  # 积分项
        self.Kd = 0.005  # 微分项
        self.last_error = 0.0  # 上一次误差
        self.integral = 0.0  # 误差积分

    def update_control(self, detected_obstacles):
        """根据检测到的障碍物更新控制"""
        control = carla.VehicleControl()

        # 获取当前速度（m/s转km/h）
        try:
            velocity = self.vehicle.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6
            self.control_state['speed'] = speed
        except:
            speed = 0.0

        # PID控制速度
        error = self.target_speed - speed
        self.integral += error
        derivative = error - self.last_error

        # PID计算
        throttle_base = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        throttle = np.clip(throttle_base, 0.0, 0.8)  # 限制油门范围

        self.last_error = error

        # 检查障碍物
        brake = 0.0
        if detected_obstacles:
            # 过滤有效距离
            valid_distances = [d for d in detected_obstacles if d is not None and d < 50]
            if valid_distances:
                closest_distance = min(valid_distances)

                # 距离分级处理
                if closest_distance < self.safety_distance * 0.5:  # 紧急制动
                    throttle = 0.0
                    brake = 1.0
                elif closest_distance < self.safety_distance:  # 减速
                    throttle = np.clip(throttle * 0.3, 0.0, 0.3)
                    brake = 0.3
                elif closest_distance < self.safety_distance * 1.5:  # 轻微减速
                    throttle = np.clip(throttle * 0.7, 0.0, 0.5)
                    brake = 0.1

        # 随机转向模拟真实驾驶（避免直线行驶）
        if random.random() < 0.05:  # 5%概率微调方向
            steer = random.uniform(-0.1, 0.1)
        else:
            steer = 0.0

        # 设置控制命令
        control.throttle = throttle
        control.steer = steer
        control.brake = brake
        control.hand_brake = False
        control.reverse = False

        # 更新控制状态
        self.control_state = {
            'throttle': throttle,
            'steer': steer,
            'brake': brake,
            'speed': speed
        }

        return control

    def set_target_speed(self, speed):
        """设置目标速度"""
        self.target_speed = np.clip(speed, 0.0, self.max_speed)

    def emergency_stop(self):
        """紧急停止"""
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        return control


# -------------------------- NPC管理器 --------------------------
class NPCManager:
    """NPC车辆管理器 - 管理非玩家控制车辆"""

    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.traffic_manager = None  # CARLA交通管理器
        self.npc_vehicles = []  # NPC车辆列表

        try:
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_synchronous_mode(True)  # 同步模式
            logger.info("交通管理器初始化成功")
        except Exception as e:
            logger.warning(f"交通管理器初始化失败: {e}")

    def spawn_npcs(self, world, count=30, ego_vehicle=None):
        """生成NPC车辆"""
        bp_lib = world.get_blueprint_library()  # 蓝图库
        spawn_points = world.get_map().get_spawn_points()  # 生成点

        if not spawn_points:
            logger.warning("没有可用的生成点")
            return []

        # 清理现有NPC
        self.destroy_all_npcs()

        random.shuffle(spawn_points)
        spawned_count = 0
        min_distance = self.config.get('npc.min_distance', 20.0)

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
                    if dist < min_distance:
                        continue  # 太靠近，跳过
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

                        # 设置个性化驾驶参数
                        self.traffic_manager.distance_to_leading_vehicle(
                            npc, random.uniform(2.0, 5.0)  # 车距
                        )
                        self.traffic_manager.vehicle_percentage_speed_difference(
                            npc, random.uniform(-30.0, 30.0)  # 速度差异
                        )
                    except Exception as e:
                        logger.warning(f"设置NPC自动驾驶失败: {e}")

                self.npc_vehicles.append(npc)
                spawned_count += 1
                if spawned_count % 10 == 0:
                    logger.info(f"已生成NPC {spawned_count}/{count}")

        logger.info(f"总共生成 {len(self.npc_vehicles)} 辆NPC车辆")
        return self.npc_vehicles

    def update_npc_behavior(self):
        """更新NPC行为 - 随机调整驾驶风格"""
        for npc in self.npc_vehicles:
            try:
                # 随机更新NPC的速度差异（1%概率）
                if random.random() < 0.01 and self.traffic_manager:
                    self.traffic_manager.vehicle_percentage_speed_difference(
                        npc, random.uniform(-40.0, 40.0)
                    )
            except:
                pass

    def destroy_all_npcs(self):
        """销毁所有NPC车辆 - 资源清理"""
        for npc in self.npc_vehicles:
            try:
                if npc.is_alive:
                    npc.destroy()
            except:
                pass
        self.npc_vehicles.clear()


# -------------------------- 工具函数 --------------------------
def preprocess_depth_image(depth_image):
    """预处理深度图像 - 归一化、滤波、增强"""
    if depth_image is None:
        return None

    # 数据类型转换
    if depth_image.dtype == np.float16:
        depth_image = depth_image.astype(np.float32)

    # 限制深度范围（0.1-200米）
    depth_image = np.clip(depth_image, 0.1, 200.0)

    # 高斯模糊降噪
    if depth_image.shape[0] > 3 and depth_image.shape[1] > 3:
        depth_image = cv2.GaussianBlur(depth_image, (3, 3), 0.5)

    # 非线性增强（伽马校正）
    max_val = np.max(depth_image)
    if max_val > 0:
        depth_image = np.power(depth_image / max_val, 0.7) * max_val

    return depth_image


def get_target_distance(depth_image, box, use_median=True):
    """获取目标距离 - 基于深度图像"""
    if depth_image is None:
        return 50.0  # 默认距离

    if depth_image.dtype == np.float16:
        depth_image = depth_image.astype(np.float32)

    # 边界框坐标
    x1, y1, x2, y2 = map(int, box)

    h, w = depth_image.shape
    # 边界检查
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x1 >= x2 or y1 >= y2:
        return 50.0

    # 提取目标区域深度
    depth_roi = depth_image[y1:y2, x1:x2]
    valid_mask = depth_roi > 0.1  # 过滤无效深度
    valid_depths = depth_roi[valid_mask]

    if valid_depths.size == 0:
        return 50.0

    # 使用中位数或均值
    if use_median:
        return float(np.median(valid_depths))  # 中位数抗噪声
    else:
        return float(np.mean(valid_depths))


def draw_bounding_boxes(image, boxes, labels, class_names, **kwargs):
    """绘制边界框 - 带ID、距离、速度等信息"""
    # 提取可选参数
    track_ids = kwargs.get('track_ids')
    probs = kwargs.get('probs')
    distances = kwargs.get('distances')
    velocities = kwargs.get('velocities')

    result = image.copy()
    h, w = image.shape[:2]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # 边界检查
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x1 >= x2 or y1 >= y2:
            continue

        # 根据距离设置颜色（红-橙-绿）
        if distances and i < len(distances) and distances[i] is not None:
            dist = distances[i]
            if dist < 15:
                color = (0, 0, 255)  # 红色，近距离（危险）
            elif dist < 30:
                color = (0, 165, 255)  # 橙色（警告）
            else:
                color = (0, 255, 0)  # 绿色，远距离（安全）
        else:
            color = (0, 255, 0)

        # 绘制边界框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # 构建文本信息
        text_parts = []

        if i < len(labels):
            text_parts.append(class_names.get(labels[i], f"cls{labels[i]}"))

        if probs and i < len(probs):
            text_parts.append(f"{probs[i]:.2f}")  # 置信度

        if track_ids and i < len(track_ids):
            text_parts.append(f"ID:{track_ids[i]}")  # 跟踪ID

        if distances and i < len(distances) and distances[i] is not None:
            text_parts.append(f"D:{distances[i]:.1f}m")  # 距离

        if velocities and i < len(velocities) and velocities[i] is not None:
            text_parts.append(f"V:{velocities[i]:.1f}")  # 速度

        label_text = " ".join(filter(None, text_parts))

        if label_text:
            # 计算文本大小
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_bg_y1 = max(0, y1 - text_size[1] - 5)
            text_bg_y2 = max(0, y1)

            # 绘制文本背景
            cv2.rectangle(result, (x1, text_bg_y1),
                          (x1 + text_size[0] + 5, text_bg_y2), color, -1)
            # 绘制文本
            cv2.putText(result, label_text, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return result


def draw_trajectories(image, tracks_info):
    """绘制目标轨迹 - 显示运动路径"""
    if not tracks_info:
        return image

    result = image.copy()

    for track in tracks_info:
        if 'history' not in track or len(track['history']) < 2:
            continue

        # 为不同目标使用不同颜色（基于ID）
        color_id = track['id'] % 10
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0)
        ]
        color = colors[color_id]

        # 绘制轨迹线
        points = []
        for point in track['history'][-20:]:  # 只绘制最近20个点
            if point:
                x, y = int(point[0]), int(point[1])
                points.append((x, y))

        if len(points) >= 2:
            for i in range(1, len(points)):
                cv2.line(result, points[i - 1], points[i], color, 1)

        # 绘制轨迹点
        for point in points:
            cv2.circle(result, point, 2, color, -1)

    return result


def draw_performance_panel(image, timings, fps, frame_count, config):
    """绘制性能监控面板 - 显示FPS和各阶段耗时"""
    if not config.get('performance.show_panel', True):
        return image

    h, w = image.shape[:2]

    def get_avg_time(key, default=0.0):
        """计算平均耗时"""
        if key in timings and timings[key]:
            # timings[key] 是 deque，需要转换为列表再切片
            timing_list = list(timings[key])
            recent = timing_list[-10:] if len(timing_list) >= 10 else timing_list
            return np.mean(recent) if recent else default
        return default

    # 计算各阶段耗时（毫秒）
    carla_time = get_avg_time('carla_tick') * 1000
    image_time = get_avg_time('image_get') * 1000
    depth_time = get_avg_time('depth_get') * 1000
    detection_time = get_avg_time('detection') * 1000
    tracking_time = get_avg_time('tracking') * 1000
    drawing_time = get_avg_time('drawing') * 1000
    total_time = get_avg_time('total') * 1000

    # 面板位置和尺寸
    panel_x = 10
    panel_y = 10
    panel_width = 300
    panel_height = 220

    # 创建半透明面板背景
    if panel_x + panel_width <= w and panel_y + panel_height <= h:
        overlay = image.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      (20, 20, 20), -1)
        image = cv2.addWeighted(image, 0.3, overlay, 0.7, 0)  # 透明度混合

        # 绘制标题
        cv2.putText(image, "性能监控面板", (panel_x + 10, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.line(image, (panel_x, panel_y + 25),
                 (panel_x + panel_width, panel_y + 25), (100, 100, 100), 1)

        # 绘制性能指标
        y_offset = 45
        line_height = 20

        # FPS和帧数
        fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(image, f"FPS: {fps:.1f}", (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        cv2.putText(image, f"Frame: {frame_count}", (panel_x + 120, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        y_offset += line_height

        # 各阶段时间（分两列显示）
        metrics = [
            ('CARLA', carla_time, 5.0, 10.0),  # 名称, 时间, 良好阈值, 警告阈值
            ('Image', image_time, 1.0, 3.0),
            ('Depth', depth_time, 2.0, 5.0),
            ('Detection', detection_time, 10.0, 20.0),
            ('Tracking', tracking_time, 5.0, 10.0),
            ('Drawing', drawing_time, 2.0, 5.0),
            ('Total', total_time, 50.0, 100.0)
        ]

        for i in range(0, len(metrics), 2):
            name1, time1, good1, warn1 = metrics[i]
            color1 = (0, 255, 0) if time1 < good1 else (0, 165, 255) if time1 < warn1 else (0, 0, 255)
            cv2.putText(image, f"{name1}: {time1:.1f}ms", (panel_x + 10, panel_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 1)

            if i + 1 < len(metrics):
                name2, time2, good2, warn2 = metrics[i + 1]
                color2 = (0, 255, 0) if time2 < good2 else (0, 165, 255) if time2 < warn2 else (0, 0, 255)
                cv2.putText(image, f"{name2}: {time2:.1f}ms", (panel_x + 150, panel_y + y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 1)

            y_offset += line_height

    return image


# -------------------------- CARLA相关函数 --------------------------
def setup_carla_client(config):
    """连接CARLA服务器 - 初始化仿真环境"""
    host = config.get('carla.host', 'localhost')
    port = config.get('carla.port', 2000)
    timeout = config.get('carla.timeout', 10.0)

    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()

        # 设置同步模式
        settings = world.get_settings()
        settings.synchronous_mode = config.get('carla.sync_mode', True)  # 同步模式
        settings.fixed_delta_seconds = config.get('carla.fixed_delta_seconds', 0.05)  # 时间步长
        world.apply_settings(settings)

        logger.info(f"连接到CARLA服务器 {host}:{port}")
        logger.info(f"地图: {world.get_map().name}")
        logger.info(f"同步模式: {settings.synchronous_mode}")
        logger.info(f"时间步长: {settings.fixed_delta_seconds}")

        return world, client
    except Exception as e:
        logger.error(f"连接CARLA失败: {e}")
        raise


def spawn_ego_vehicle(world, config):
    """生成主车辆 - 玩家控制的车辆"""
    bp_lib = world.get_blueprint_library()

    # 优先使用林肯MKZ（较好的驾驶性能）
    vehicle_bp = None
    try:
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    except:
        pass

    if not vehicle_bp:
        # 尝试其他车辆（备选列表）
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
        # 随机选择任意车辆
        try:
            vehicle_bp = random.choice(list(bp_lib.filter('vehicle.*')))
        except:
            logger.error("错误：没有找到可用的车辆蓝图")
            return None

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        logger.error("错误：没有可用的生成点")
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
                if dist < 10.0:  # 10米以内认为太靠近
                    too_close = True
                    break
            except:
                continue

        if not too_close:
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                if vehicle:
                    logger.info(f"主车辆生成成功: {vehicle_bp.id}")

                    # 启用物理模拟
                    enable_physics = config.get('vehicle.enable_physics', True)
                    vehicle.set_simulate_physics(enable_physics)

                    return vehicle
            except Exception as e:
                logger.warning(f"生成车辆失败: {e}")
                continue

    # 如果找不到合适的位置，强制生成
    if spawn_points:
        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
            if vehicle:
                enable_physics = config.get('vehicle.enable_physics', True)
                vehicle.set_simulate_physics(enable_physics)
                logger.info(f"主车辆强制生成: {vehicle_bp.id}")
                return vehicle
        except Exception as e:
            logger.error(f"强制生成车辆失败: {e}")

    return None


# 回调函数
def camera_callback(image, rgb_image_queue):
    """RGB图像回调函数 - 接收相机数据"""
    try:
        # 转换CARLA图像数据为numpy数组
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # RGBA
        rgb = array[..., :3]  # 取RGB通道

        # 线程安全队列操作
        if rgb_image_queue.full():
            try:
                rgb_image_queue.get_nowait()  # 丢弃最旧帧
            except:
                pass
        rgb_image_queue.put(rgb)
    except Exception as e:
        logger.error(f"相机回调错误: {e}")


def depth_camera_callback(image, depth_queue):
    """深度图像回调函数 - 接收深度数据"""
    try:
        depth_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        depth_data = depth_data.reshape((image.height, image.width, 4))

        # CARLA深度数据编码：R + G*256 + B*256^2
        depth_channel = (
                depth_data[..., 2].astype(np.uint16) +
                depth_data[..., 1].astype(np.uint16) * 256 +
                depth_data[..., 0].astype(np.uint16) * 256 ** 2
        )

        # 转换为米（归一化并乘以1000）
        depth_in_meters = depth_channel.astype(np.float16) / (256 ** 3 - 1) * 1000.0
        depth_in_meters = preprocess_depth_image(depth_in_meters)

        if depth_queue.full():
            try:
                depth_queue.get_nowait()
            except:
                pass
        depth_queue.put(depth_in_meters)
    except Exception as e:
        logger.error(f"深度相机回调错误: {e}")


# -------------------------- 主函数 --------------------------
def main():
    """主函数 - 程序入口"""
    # 初始化命令行参数解析
    parser = argparse.ArgumentParser(description='CARLA目标检测与跟踪 - 优化版')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--model', type=str, default='yolov5m', help='模型类型')
    parser.add_argument('--host', type=str, default='localhost', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU阈值')
    parser.add_argument('--use-depth', action='store_true', default=True, help='使用深度信息')
    parser.add_argument('--show-depth', action='store_true', help='显示深度图像')
    parser.add_argument('--npc-count', type=int, default=20, help='NPC数量')
    parser.add_argument('--target-speed', type=float, default=30.0, help='目标速度 (km/h)')
    parser.add_argument('--manual-control', action='store_true', help='手动控制模式')

    args = parser.parse_args()

    # 加载配置管理器
    config_manager = ConfigManager(args.config)

    # 更新命令行参数到配置
    if args.model:
        config_manager.config['detection']['model_type'] = args.model
    if args.host:
        config_manager.config['carla']['host'] = args.host
    if args.port:
        config_manager.config['carla']['port'] = args.port
    if args.conf_thres:
        config_manager.config['detection']['conf_thres'] = args.conf_thres
    if args.iou_thres:
        config_manager.config['detection']['iou_thres'] = args.iou_thres
    if args.target_speed:
        config_manager.config['vehicle']['target_speed'] = args.target_speed
    if args.npc_count:
        config_manager.config['npc']['count'] = args.npc_count

    config = config_manager

    # 初始化变量
    world = vehicle = camera = depth_camera = None
    image_queue = depth_queue = None
    client = controller = npc_manager = None
    frame_count = 0

    try:
        # 1. 连接CARLA
        logger.info("连接CARLA服务器...")
        world, client = setup_carla_client(config)

        spectator = world.get_spectator()  # 观察者视角

        # 2. 清理环境
        logger.info("清理环境...")
        try:
            for actor in world.get_actors().filter('vehicle.*'):
                try:
                    if actor.is_alive:
                        actor.destroy()  # 清理现有车辆
                except:
                    pass
        except:
            pass

        time.sleep(1)  # 等待清理完成

        # 3. 生成主车辆
        logger.info("生成主车辆...")
        vehicle = spawn_ego_vehicle(world, config)
        if not vehicle:
            logger.error("主车辆生成失败，程序退出！")
            return

        # 初始化车辆控制器
        controller = VehicleController(vehicle, config)

        # 4. 生成传感器
        logger.info("生成相机传感器...")
        bp_lib = world.get_blueprint_library()

        # RGB相机
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(config.get('camera.width', 800)))
        camera_bp.set_attribute('image_size_y', str(config.get('camera.height', 600)))
        camera_bp.set_attribute('fov', str(config.get('camera.fov', 90)))
        camera_bp.set_attribute('sensor_tick', '0.05')  # 20Hz

        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.8),
                                           carla.Rotation(pitch=-5, yaw=0))
        try:
            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            image_queue = queue.Queue(maxsize=3)  # 队列大小限制
            camera.listen(lambda image: camera_callback(image, image_queue))
            logger.info("RGB相机传感器生成成功！")
        except Exception as e:
            logger.error(f"RGB相机生成失败: {e}")
            return

        # 深度相机
        depth_camera = None
        depth_queue = None
        if args.use_depth:
            try:
                depth_bp = bp_lib.find('sensor.camera.depth')
                depth_bp.set_attribute('image_size_x', str(config.get('camera.width', 800)))
                depth_bp.set_attribute('image_size_y', str(config.get('camera.height', 600)))
                depth_bp.set_attribute('fov', str(config.get('camera.fov', 90)))
                depth_bp.set_attribute('sensor_tick', '0.05')

                depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
                depth_queue = queue.Queue(maxsize=3)
                depth_camera.listen(lambda image: depth_camera_callback(image, depth_queue))
                logger.info("深度相机传感器生成成功！")
            except Exception as e:
                logger.warning(f"深度相机生成失败: {e}")

        # 5. 生成NPC车辆
        npc_count = config.get('npc.count', 20)
        logger.info(f"生成 {npc_count} 辆NPC车辆...")
        npc_manager = NPCManager(client, config)
        npc_vehicles = npc_manager.spawn_npcs(world, count=npc_count, ego_vehicle=vehicle)

        if len(npc_vehicles) < npc_count // 2:
            logger.warning(f"只成功生成了 {len(npc_vehicles)} 辆NPC车辆")

        # 等待NPC车辆稳定
        logger.info("等待NPC车辆初始化...")
        for _ in range(10):
            world.tick()

        # 6. 加载检测模型和跟踪器
        logger.info("加载检测模型和跟踪器...")
        try:
            model_type = config.get('detection.model_type', 'yolov5m')
            model, class_names = load_detection_model(model_type)
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return

        # 初始化优化跟踪器
        tracking_config = config.get('tracking', {})
        tracker = OptimizedSORT(tracking_config)

        # 7. 主循环
        logger.info("\n开始目标检测与跟踪")
        logger.info("=" * 50)
        if args.manual_control:
            logger.info("手动控制模式：使用WASD控制车辆")
        else:
            logger.info("自动控制模式：车辆将自动行驶")

        # 统计信息
        detection_stats = {
            'total_detections': 0,
            'total_frames': 0,
            'max_vehicles_per_frame': 0
        }

        # 性能监控 - 修复：确保所有timings都是deque
        timings = {
            'carla_tick': deque(maxlen=100),
            'image_get': deque(maxlen=100),
            'depth_get': deque(maxlen=100),
            'detection': deque(maxlen=100),
            'tracking': deque(maxlen=100),
            'drawing': deque(maxlen=100),
            'display': deque(maxlen=100),
            'total': deque(maxlen=100)
        }

        # 手动控制变量
        manual_controls = {
            'throttle': 0.0,
            'brake': 0.0,
            'steer': 0.0,
            'reverse': False
        }

        # 主循环
        while True:
            try:
                frame_start = time.time()
                frame_count += 1
                detection_stats['total_frames'] += 1

                # 同步CARLA世界
                tick_start = time.time()
                world.tick()  # 同步模式下的关键调用
                timings['carla_tick'].append(time.time() - tick_start)

                # 更新NPC行为
                if npc_manager:
                    npc_manager.update_npc_behavior()

                # 移动视角（跟随主车辆）
                try:
                    ego_transform = vehicle.get_transform()
                    spectator_transform = carla.Transform(
                        ego_transform.transform(carla.Location(x=-10, z=12)),  # 后方10米，高12米
                        carla.Rotation(yaw=ego_transform.rotation.yaw - 180, pitch=-30)  # 俯视视角
                    )
                    spectator.set_transform(spectator_transform)
                except:
                    pass

                # 手动控制
                if args.manual_control:
                    key = cv2.waitKey(1) & 0xFF

                    # 控制逻辑
                    if key == ord('w'):  # 加速
                        manual_controls['throttle'] = min(manual_controls['throttle'] + 0.1, 1.0)
                        manual_controls['brake'] = 0.0
                    elif key == ord('s'):  # 减速/刹车
                        manual_controls['brake'] = min(manual_controls['brake'] + 0.1, 1.0)
                        manual_controls['throttle'] = 0.0
                    elif key == ord('a'):  # 左转
                        manual_controls['steer'] = max(manual_controls['steer'] - 0.1, -1.0)
                    elif key == ord('d'):  # 右转
                        manual_controls['steer'] = min(manual_controls['steer'] + 0.1, 1.0)
                    else:
                        # 逐渐回正
                        if manual_controls['steer'] > 0:
                            manual_controls['steer'] = max(manual_controls['steer'] - 0.05, 0)
                        elif manual_controls['steer'] < 0:
                            manual_controls['steer'] = min(manual_controls['steer'] + 0.05, 0)

                        # 逐渐减速
                        manual_controls['throttle'] = max(manual_controls['throttle'] - 0.05, 0)
                        manual_controls['brake'] = max(manual_controls['brake'] - 0.05, 0)

                    # 应用控制
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
                    continue  # 无新图像，跳过

                origin_image = image_queue.get()
                image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)  # CARLA是BGR
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
                        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)  # 伪彩色
                        cv2.imshow('Depth Image', depth_vis)
                timings['depth_get'].append(time.time() - depth_start)

                # 目标检测
                detection_start = time.time()
                boxes, labels, probs, depths = [], [], [], []

                try:
                    conf_thres = config.get('detection.conf_thres', 0.15)
                    iou_thres = config.get('detection.iou_thres', 0.4)
                    device = config.get('detection.device', 'cuda' if torch.cuda.is_available() else 'cpu')

                    # YOLO推理
                    results = model(image, conf=conf_thres, iou=iou_thres,
                                    device=device, imgsz=640, verbose=False)

                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())

                            # 只保留车辆相关类别（COCO数据集）
                            # 2:car, 3:motorcycle, 5:bus, 7:truck
                            if cls in [2, 3, 5, 7]:
                                box_width = x2 - x1
                                box_height = y2 - y1

                                # 尺寸过滤（基于距离自适应）
                                min_size = 6
                                if depth_image is not None:
                                    rough_distance = get_target_distance(depth_image, [x1, y1, x2, y2])
                                    if rough_distance > 30:
                                        min_size = 4  # 远处目标可以更小
                                    elif rough_distance > 50:
                                        min_size = 2
                                    else:
                                        min_size = 8  # 近处目标需要更大

                                aspect_ratio = box_width / max(box_height, 1)

                                # 过滤条件：最小尺寸 + 合理宽高比
                                if (box_width > min_size and box_height > min_size and
                                        0.3 < aspect_ratio < 3.0):
                                    boxes.append([x1, y1, x2, y2])
                                    labels.append(cls)
                                    probs.append(conf)

                                    # 计算目标距离
                                    if depth_image is not None:
                                        dist = get_target_distance(depth_image, [x1, y1, x2, y2])
                                        depths.append(dist)
                except Exception as e:
                    logger.error(f"检测模型推理出错: {e}")

                timings['detection'].append(time.time() - detection_start)

                # 更新统计
                detection_stats['total_detections'] += len(boxes)
                detection_stats['max_vehicles_per_frame'] = max(
                    detection_stats['max_vehicles_per_frame'], len(boxes)
                )

                # 目标跟踪
                tracking_start = time.time()
                if boxes:
                    # 准备检测数据 [x1, y1, x2, y2, conf, class]
                    detections = []
                    for i in range(len(boxes)):
                        det = boxes[i] + [probs[i]] + [labels[i]]
                        detections.append(det)

                    # 更新跟踪器
                    track_results = tracker.update(detections, depths)

                    if track_results:
                        track_boxes = []
                        track_ids = []
                        track_classes = []
                        track_confidences = []
                        track_distances = []
                        track_velocities = []

                        # 获取跟踪目标信息
                        tracks_info = tracker.get_tracks_info()

                        for track in track_results:
                            x1, y1, x2, y2, track_id, class_id, confidence = track
                            track_boxes.append([x1, y1, x2, y2])
                            track_ids.append(int(track_id))
                            track_classes.append(int(class_id))
                            track_confidences.append(float(confidence))

                            # 查找对应的跟踪目标信息
                            track_info = next((t for t in tracks_info if t['id'] == track_id), None)
                            if track_info:
                                track_distances.append(track_info.get('distance'))
                                track_velocities.append(track_info.get('velocity'))
                            else:
                                track_distances.append(None)
                                track_velocities.append(None)

                        # 绘制跟踪结果
                        drawing_start = time.time()
                        if track_boxes:
                            image = draw_bounding_boxes(
                                image, track_boxes,
                                labels=track_classes,
                                class_names=class_names,
                                track_ids=track_ids,
                                probs=track_confidences,
                                distances=track_distances,
                                velocities=track_velocities
                            )

                            # 绘制轨迹
                            image = draw_trajectories(image, tracks_info)

                            cv2.putText(image, f'Vehicles: {len(track_boxes)}', (width - 200, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        timings['drawing'].append(time.time() - drawing_start)

                # 这里确保timing只添加一次
                timings['tracking'].append(time.time() - tracking_start)

                # 计算FPS
                total_time = time.time() - frame_start
                fps = 1.0 / total_time if total_time > 0 else 0
                timings['total'].append(total_time)

                # 绘制性能监控面板
                image = draw_performance_panel(image, timings, fps, frame_count, config)

                # 显示其他信息
                info_lines = [
                    f"FPS: {fps:.1f}",
                    f"Frame: {frame_count}",
                    f"Tracks: {len(tracker.tracks)}",
                    f"Detections: {len(boxes)}",
                    f"Model: {config.get('detection.model_type', 'yolov5m')}",
                    "Press 'q' to quit, 'r' to reset NPCs"
                ]

                y_pos = 250
                for line in info_lines:
                    cv2.putText(image, line, (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_pos += 25

                # 显示手动控制状态
                if args.manual_control:
                    try:
                        velocity = vehicle.get_velocity()
                        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6
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

                # 定期打印统计信息
                log_interval = config.get('performance.log_interval', 100)
                if frame_count % log_interval == 0 and frame_count > 0:
                    avg_fps = 1.0 / np.mean(list(timings['total'])) if timings['total'] else 0
                    logger.info(
                        f"[帧数 {frame_count}] FPS: {avg_fps:.1f} | "
                        f"检测: {len(boxes)} | 跟踪: {len(tracker.tracks)}"
                    )

                # 退出检查
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("用户触发退出程序...")
                    break
                elif key == ord('r'):
                    # 重新生成NPC
                    logger.info("重新生成NPC车辆...")
                    if npc_manager:
                        npc_manager.destroy_all_npcs()
                        npc_vehicles = npc_manager.spawn_npcs(world, count=npc_count, ego_vehicle=vehicle)
                        logger.info(f"重新生成 {len(npc_vehicles)} 辆NPC车辆完成")

                # FPS控制（避免过高占用）
                max_fps = config.get('performance.max_fps', 30)
                elapsed = time.time() - frame_start
                if elapsed < 1.0 / max_fps:
                    time.sleep(max(0, 1.0 / max_fps - elapsed))

            except KeyboardInterrupt:
                logger.info("\n用户中断程序...")
                break
            except Exception as e:
                logger.error(f"主循环出错: {e}")
                import traceback
                traceback.print_exc()
                break

    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源（重要：确保程序退出时释放资源）
        logger.info("\n正在清理资源...")

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

        # 恢复世界设置（关闭同步模式）
        if world:
            try:
                settings = world.get_settings()
                settings.synchronous_mode = False
                world.apply_settings(settings)
            except:
                pass

        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

        # 清理PyTorch缓存（GPU内存）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 保存配置
        config_manager.save_config("carla_tracking_config.json")

        # 打印最终统计
        logger.info("\n" + "=" * 50)
        logger.info("程序运行统计：")
        logger.info(f"总帧数: {frame_count}")

        if timings.get('total'):
            try:
                avg_fps = 1.0 / np.mean(list(timings['total'])) if timings['total'] else 0
                logger.info(f"平均FPS: {avg_fps:.1f}")
            except:
                pass

        if detection_stats:
            logger.info(f"总检测次数: {detection_stats['total_detections']}")
            if detection_stats['total_frames'] > 0:
                avg_det = detection_stats['total_detections'] / detection_stats['total_frames']
                logger.info(f"平均每帧检测: {avg_det:.1f}")
            logger.info(f"最大单帧车辆数: {detection_stats['max_vehicles_per_frame']}")

        logger.info("=" * 50)
        logger.info("资源清理完成，程序正常退出！")


if __name__ == "__main__":
    main()