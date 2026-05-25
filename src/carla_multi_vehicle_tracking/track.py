#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO + DeepSort 多车跟踪系统
兼容 Python 3.8.10 + CARLA 0.9.13 + Windows 10

【新增：轨迹预测+提前碰撞预警模块】

功能说明：
1. 实时跟踪多个车辆目标
2. 记录每辆车的轨迹历史（中心点）
3. 基于轨迹预测未来位置
4. 检测潜在的碰撞风险并预警
"""

from __future__ import print_function, absolute_import
import sys
import os
import argparse
import traceback
from collections import defaultdict
import glob

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch 未安装，将使用 CPU")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("[WARNING] Ultralytics 未安装")

try:
    from deep_sort.deep_sort import DeepSort
    from deep_sort.utils.parser import get_config
    DEEPSORT_AVAILABLE = True
except ImportError as e:
    DEEPSORT_AVAILABLE = False
    print(f"[WARNING] DeepSort 导入失败: {e}")

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    print("[WARNING] CARLA 未安装")

import random

# ==================== 【原有】基础配置常量 ====================
class_id = [2, 3, 5, 7]
class_name = {2: 'car', 3: 'motobike', 5: 'bus', 7: 'truck'}

img_w = 256 * 4
img_h = 256 * 3
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
output_path = "output.mp4"

# ==================== 【新增：轨迹预测+提前碰撞预警】配置变量 ====================
# 以下所有阈值参数均可自由调整

# 【新增：轨迹预测+提前碰撞预警】轨迹历史长度配置
# 保存每个车辆最近多少帧的中心点位置
# 值越大，轨迹越长，但内存占用越高
MAX_TRAJECTORY_LENGTH = 10  # 保存最近10帧中心点

# 【新增：轨迹预测+提前碰撞预警】预测帧数配置
# 基于当前速度，预测未来多少帧的位置
# 值越大，预警越提前，但准确性可能降低
PREDICT_FRAMES = 3  # 预测未来3帧

# 【新增：轨迹预测+提前碰撞预警】预警距离阈值配置
# 当两车预测位置的距离小于 画面宽度 × COLLISION_DISTANCE_RATIO 时触发预警
# 值越小，要求越接近才预警；值越大，稍有接近就预警
COLLISION_DISTANCE_RATIO = 0.1  # 预警距离=画面宽度*0.1 (10%)

# 【新增：轨迹预测+提前碰撞预警】计算实际阈值
COLLISION_DISTANCE_THRESHOLD = int(img_w * COLLISION_DISTANCE_RATIO)

# 【新增：轨迹预测+提前碰撞预警】预警显示配置
COLLISION_WARNING_COLOR = (0, 0, 255)  # 红色 (BGR)
COLLISION_WARNING_TEXT_COLOR = (0, 0, 255)  # 红色 (BGR)
COLLISION_WARNING_BOX_THICKNESS = 3  # 红色边框粗细
COLLISION_WARNING_TEXT_SCALE = 0.8  # 文字大小
COLLISION_WARNING_TEXT_THICKNESS = 2  # 文字粗细
COLLISION_WARNING_TEXT_POSITION = (10, 30)  # 文字位置 (x, y)
COLLISION_WARNING_MESSAGE = "COLLISION WARNING!"  # 预警文字

# ==================== 【新增：轨迹预测+提前碰撞预警】函数定义 ====================

def initialize_trajectory_dict():
    """
    【新增：轨迹预测+提前碰撞预警】
    初始化轨迹历史字典
    
    返回:
        defaultdict: 用于存储每个track_id的轨迹历史列表
    """
    return defaultdict(list)


def update_trajectory(trajectory_dict, tracked_vehicles):
    """
    【新增：轨迹预测+提前碰撞预警】
    更新所有车辆的轨迹历史
    
    参数:
        trajectory_dict: 轨迹历史字典 {track_id: [(cx1,cy1), (cx2,cy2), ...]}
        tracked_vehicles: DeepSort输出的跟踪结果，格式为 [x1, y1, x2, y2, track_id, ...]
    
    功能:
        1. 遍历所有跟踪到的车辆
        2. 计算每辆车的中心点 (cx, cy)
        3. 更新该车辆的轨迹历史（只保留最近MAX_TRAJECTORY_LENGTH个点）
        4. 清理已消失的车辆轨迹
    """
    # 获取当前帧所有活跃的track_id
    current_ids = set()
    
    for output in tracked_vehicles:
        if len(output) >= 5:
            try:
                x1, y1, x2, y2 = map(int, output[0:4])
                track_id = int(output[4])
                
                # 计算中心点
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 添加到当前活跃ID集合
                current_ids.add(track_id)
                
                # 更新轨迹历史
                if track_id in trajectory_dict:
                    # 追加新中心点
                    trajectory_dict[track_id].append((center_x, center_y))
                    # 只保留最近MAX_TRAJECTORY_LENGTH个点
                    if len(trajectory_dict[track_id]) > MAX_TRAJECTORY_LENGTH:
                        trajectory_dict[track_id].pop(0)
                else:
                    # 新车辆，初始化轨迹
                    trajectory_dict[track_id] = [(center_x, center_y)]
                    
            except (ValueError, TypeError, IndexError):
                continue
    
    # 清理已消失的车辆轨迹（可选，节省内存）
    # 这里保留轨迹，以便重新出现时仍有历史数据
    # 如果需要清理，启用下面的代码：
    # disappeared_ids = set(trajectory_dict.keys()) - current_ids
    # for track_id in disappeared_ids:
    #     del trajectory_dict[track_id]


def predict_future_position(trajectory, predict_frames):
    """
    【新增：轨迹预测+提前碰撞预警】
    基于轨迹历史预测未来位置（线性预测）
    
    参数:
        trajectory: 轨迹历史列表 [(cx1,cy1), (cx2,cy2), ...]
        predict_frames: 预测未来多少帧
    
    返回:
        tuple: 预测的未来位置 (future_cx, future_cy)，如果无法预测则返回None
    
    算法说明:
        1. 如果轨迹点少于2个，无法计算速度，返回None
        2. 使用最近两个点的位移计算当前速度 (vx, vy)
        3. 基于当前速度和位置，预测未来predict_frames帧的位置
        4. 公式: future_pos = current_pos + velocity * predict_frames
    """
    if len(trajectory) < 2:
        # 轨迹点不足，无法计算速度
        return None
    
    # 获取最近两个点的位置
    cx_prev, cy_prev = trajectory[-2]
    cx_curr, cy_curr = trajectory[-1]
    
    # 计算速度（每帧的位移）
    vx = cx_curr - cx_prev
    vy = cy_curr - cy_prev
    
    # 预测未来位置
    future_cx = cx_curr + vx * predict_frames
    future_cy = cy_curr + vy * predict_frames
    
    return (future_cx, future_cy)


def check_trajectory_collision(trajectory_dict, predict_frames, collision_threshold, frame_width):
    """
    【新增：轨迹预测+提前碰撞预警】
    检测所有车辆之间的轨迹碰撞风险
    
    参数:
        trajectory_dict: 轨迹历史字典 {track_id: [(cx1,cy1), ...]}
        predict_frames: 预测帧数
        collision_threshold: 碰撞距离阈值（像素）
        frame_width: 画面宽度（用于计算实际阈值）
    
    返回:
        dict: 风险车辆字典 {track_id: [(bbox, other_id), ...]}
    
    功能:
        1. 遍历所有车辆对
        2. 预测每辆车PREDICT_FRAMES帧后的位置
        3. 计算两车预测位置的距离
        4. 如果距离小于阈值，判定为碰撞风险
        5. 打印预警信息到控制台
    """
    collision_risk = {}  # {track_id: [(bbox, other_id, distance), ...]}
    vehicle_info = {}  # {track_id: {'bbox': (x1,y1,x2,y2), 'center': (cx,cy), 'future': (fx,fy)}}
    
    # 第一步：计算所有车辆的预测位置
    for track_id, trajectory in trajectory_dict.items():
        if len(trajectory) >= 2:
            future_pos = predict_future_position(trajectory, predict_frames)
            if future_pos:
                # 使用轨迹中最后一个点作为当前位置
                current_pos = trajectory[-1]
                vehicle_info[track_id] = {
                    'trajectory': trajectory,
                    'current': current_pos,
                    'future': future_pos
                }
    
    # 第二步：两两检测碰撞风险
    track_ids = list(vehicle_info.keys())
    
    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            id1 = track_ids[i]
            id2 = track_ids[j]
            
            vehicle1 = vehicle_info[id1]
            vehicle2 = vehicle_info[id2]
            
            # 计算两车预测位置的距离
            fx1, fy1 = vehicle1['future']
            fx2, fy2 = vehicle2['future']
            
            dx = fx1 - fx2
            dy = fy1 - fy2
            distance = (dx ** 2 + dy ** 2) ** 0.5
            
            # 判断是否碰撞
            if distance < collision_threshold:
                # 【新增：轨迹预测+提前碰撞预警】控制台打印预警信息
                print(f"[提前碰撞预警] 车辆ID:{id1} 和 车辆ID:{id2} 距离过近，存在碰撞风险")
                
                # 添加到风险列表
                if id1 not in collision_risk:
                    collision_risk[id1] = []
                if id2 not in collision_risk:
                    collision_risk[id2] = []
                
                collision_risk[id1].append((id2, distance))
                collision_risk[id2].append((id1, distance))
    
    return collision_risk


def draw_trajectory_prediction(frame, trajectory_dict, collision_risk_ids):
    """
    【新增：轨迹预测+提前碰撞预警】
    在画面上绘制轨迹和预测（可选功能）
    
    参数:
        frame: 视频帧
        trajectory_dict: 轨迹历史字典
        collision_risk_ids: 有碰撞风险的track_id集合
    
    功能:
        1. 为每辆车绘制历史轨迹点
        2. 为风险车辆绘制预测位置
    """
    for track_id, trajectory in trajectory_dict.items():
        if len(trajectory) < 2:
            continue
        
        # 判断是否为风险车辆
        is_risk = track_id in collision_risk_ids
        
        # 选择颜色：风险车辆红色，普通车辆绿色
        color = (0, 0, 255) if is_risk else (0, 255, 0)  # 红或绿
        
        # 绘制历史轨迹点
        for i, (cx, cy) in enumerate(trajectory):
            if i > 0:
                # 绘制轨迹线段
                cx_prev, cy_prev = trajectory[i-1]
                cv2.line(frame, (int(cx_prev), int(cy_prev)), 
                        (int(cx), int(cy)), color, 2)
        
        # 为风险车辆绘制预测位置
        if is_risk and len(trajectory) >= 2:
            future_pos = predict_future_position(trajectory, PREDICT_FRAMES)
            if future_pos:
                fx, fy = future_pos
                # 绘制预测点（大红色圆圈）
                cv2.circle(frame, (int(fx), int(fy)), 10, (0, 0, 255), -1)
                # 添加标签
                cv2.putText(frame, f"PREDICT", (int(fx)-30, int(fy)-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def draw_collision_warning(frame, collision_risk_ids, tracked_vehicles):
    """
    【新增：轨迹预测+提前碰撞预警】
    在视频帧上绘制碰撞预警信息
    
    参数:
        frame: 视频帧
        collision_risk_ids: 有碰撞风险的track_id集合
        tracked_vehicles: 跟踪结果
    
    功能:
        1. 如果有碰撞风险，在画面顶部显示警告文字
        2. 为风险车辆绘制红色边框
        3. 保持普通车辆的原有颜色
    """
    if len(collision_risk_ids) > 0:
        # 绘制顶部警告文字
        cv2.putText(
            frame,
            COLLISION_WARNING_MESSAGE,
            COLLISION_WARNING_TEXT_POSITION,
            cv2.FONT_HERSHEY_SIMPLEX,
            COLLISION_WARNING_TEXT_SCALE,
            COLLISION_WARNING_TEXT_COLOR,
            COLLISION_WARNING_TEXT_THICKNESS,
            cv2.LINE_AA
        )
        
        # 为风险车辆绘制红色边框
        for output in tracked_vehicles:
            if len(output) >= 5:
                try:
                    x1, y1, x2, y2 = map(int, output[0:4])
                    track_id = int(output[4])
                    
                    # 如果是风险车辆，绘制红色边框
                    if track_id in collision_risk_ids:
                        cv2.rectangle(
                            frame, 
                            (x1, y1), 
                            (x2, y2), 
                            COLLISION_WARNING_COLOR, 
                            COLLISION_WARNING_BOX_THICKNESS
                        )
                except (ValueError, TypeError, IndexError):
                    continue
    
    return frame


# ==================== 【原有】主跟踪类 ====================
class VehicleTracker:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
        print(f"[INFO] 使用设备: {self.device}")
        print(f"[INFO] Python 版本: {sys.version}")
        
        # 检查依赖
        self._check_dependencies()
        
        # 加载模型
        self.model = None
        self.deepsort = None
        
        # 【新增：轨迹预测+提前碰撞预警】初始化轨迹历史字典
        self.trajectory_dict = initialize_trajectory_dict()
        
        if ULTRALYTICS_AVAILABLE:
            self._load_yolo_model()
        
        if DEEPSORT_AVAILABLE:
            self._load_deepsort()
        
        print("[INFO] 初始化完成，准备开始跟踪")
    
    def _check_dependencies(self):
        """检查依赖是否完整"""
        print("\n[INFO] 检查依赖...")
        
        deps_status = {
            'PyTorch': TORCH_AVAILABLE,
            'Ultralytics': ULTRALYTICS_AVAILABLE,
            'DeepSort': DEEPSORT_AVAILABLE,
            'CARLA': CARLA_AVAILABLE
        }
        
        for name, available in deps_status.items():
            status = "✓" if available else "✗"
            print(f"  [{status}] {name}")
        
        if not ULTRALYTICS_AVAILABLE:
            print("[ERROR] 必须安装 ultralytics: pip install ultralytics==8.0.150")
        
        if not DEEPSORT_AVAILABLE:
            print("[ERROR] 必须安装 deep_sort 模块")
    
    def _load_yolo_model(self):
        """加载 YOLO 模型"""
        model_paths = [
            'weights/yolov8n.pt',
            'weights/best.pt',
            'yolov8n.pt'
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print(f"[WARNING] 未找到 YOLO 模型文件，使用默认路径: {model_paths[0]}")
            model_path = model_paths[0]
        
        try:
            self.model = YOLO(model_path)
            print(f"[INFO] YOLO 模型加载成功: {model_path}")
        except Exception as e:
            print(f"[ERROR] 加载 YOLO 模型失败: {str(e)}")
            traceback.print_exc()
    
    def _load_deepsort(self):
        """加载 DeepSort 跟踪器"""
        weight_paths = [
            "deep_sort/deep/checkpoint/ckpt.t7",
            "deep_sort/deepSORT/ckpt.t7"
        ]
        
        weight_path = None
        for path in weight_paths:
            if os.path.exists(path):
                weight_path = path
                break
        
        if not weight_path:
            print(f"[ERROR] 未找到 DeepSort 权重文件: {weight_paths[0]}")
            print("[INFO] 跳过 DeepSort，跟踪功能可能受限")
            return
        
        try:
            self.cfg = get_config()
            cfg_path = 'deep_sort/configs/deep_sort.yaml'
            if os.path.exists(cfg_path):
                self.cfg.merge_from_file(cfg_path)
            
            self.deepsort = DeepSort(weight_path, max_age=70)
            print(f"[INFO] DeepSort 加载成功: {weight_path}")
        except Exception as e:
            print(f"[ERROR] 加载 DeepSort 失败: {str(e)}")
            traceback.print_exc()
    
    def yolo_details(self, frame):
        """YOLO 检测"""
        if not self.model:
            return frame, [], [], []
        
        try:
            results = self.model(frame)
            bbox_xyxy = []
            conf_score = []
            cls_id = []
            
            for box in results:
                if hasattr(box, 'boxes') and box.boxes is not None:
                    data_list = box.boxes.data.tolist()
                    for row in data_list:
                        if len(row) >= 6:
                            class_id_val = int(row[5])
                            if class_id_val in class_id:
                                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                                conf = row[4]
                                bbox_xyxy.append([x1, y1, x2, y2])
                                conf_score.append(conf)
                                cls_id.append(class_id_val)
            
            return frame, bbox_xyxy, conf_score, cls_id
        except Exception as e:
            print(f"[ERROR] YOLO 检测失败: {str(e)}")
            return frame, [], [], []
    
    def colour_label(self, label):
        """生成颜色标签"""
        label_colour = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(label_colour)
    
    def draw_bbox(self, frame, output, conf, cls_id, collision_risk_ids=None):
        """
        【原有函数，修改】绘制边界框
        
        参数:
            frame: 视频帧
            output: 跟踪输出 [x1, y1, x2, y2, track_id, ...]
            conf: 置信度
            cls_id: 类别ID
            collision_risk_ids: 【新增参数】有碰撞风险的track_id集合，如果为None则不检查
        
        功能:
            1. 如果车辆在collision_risk_ids中，绘制红色边框
            2. 否则使用原有颜色逻辑
        """
        try:
            x1, y1, x2, y2 = map(int, output[0:4])
            track_id = int(output[4])
            label = class_name.get(cls_id, str(cls_id))
            
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            
            # 【新增：轨迹预测+提前碰撞预警】判断是否为风险车辆
            if collision_risk_ids is not None and track_id in collision_risk_ids:
                # 使用预警颜色（红色）
                colour = COLLISION_WARNING_COLOR
                label = f"[!] {label}"  # 在标签前加[!]标记
            else:
                # 使用原有颜色逻辑
                colour = self.colour_label(track_id)
            
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c_id = f'{label} {track_id}'
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
            cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), colour, -1)
            cv2.putText(frame, c_id, (x1, y1 + t_size[1] + 4), 
                       cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 2)
        except Exception as e:
            print(f"[ERROR] 绘制边界框失败: {str(e)}")
        
        return frame
    
    def process_frame(self, frame):
        """
        【原有函数，修改】处理单帧图像
        
        新增功能:
            1. 更新轨迹历史
            2. 基于轨迹预测碰撞风险
            3. 绘制预警信息
        """
        frame, bbox_xyxy, conf_score, cls_id = self.yolo_details(frame)
        
        if len(bbox_xyxy) > 0 and self.deepsort is not None:
            try:
                outputs = self.deepsort.update(bbox_xyxy, conf_score, frame)
                
                if len(outputs) > 0:
                    # 【新增：轨迹预测+提前碰撞预警】更新轨迹历史
                    update_trajectory(self.trajectory_dict, outputs)
                    
                    # 【新增：轨迹预测+提前碰撞预警】检测碰撞风险
                    collision_risk = check_trajectory_collision(
                        self.trajectory_dict,
                        PREDICT_FRAMES,
                        COLLISION_DISTANCE_THRESHOLD,
                        img_w
                    )
                    
                    # 提取风险车辆ID集合
                    collision_risk_ids = set(collision_risk.keys())
                    
                    # 【新增：轨迹预测+提前碰撞预警】绘制预警信息
                    frame = draw_collision_warning(frame, collision_risk_ids, outputs)
                    
                    # 【原有逻辑，保持不变】绘制边界框（增加了collision_risk_ids参数）
                    min_len = min(len(outputs), len(conf_score), len(cls_id))
                    for i in range(min_len):
                        frame = self.draw_bbox(frame, outputs[i], conf_score[i], cls_id[i], 
                                              collision_risk_ids)
                    
            except Exception as e:
                print(f"[ERROR] DeepSort 更新失败: {str(e)}")
        
        return frame
    
    def run_video_mode(self, video_path):
        """视频文件模式"""
        if not os.path.exists(video_path):
            print(f"[ERROR] 视频文件不存在: {video_path}")
            print("[INFO] 可用视频:")
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                videos = glob.glob(ext)
                for v in videos:
                    print(f"  - {v}")
            return
        
        print(f"[INFO] 视频模式: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频文件: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] 视频信息: {frame_width}x{frame_height} @ {fps}fps")
        
        video_writer = None
        if self.args.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = self.process_frame(frame)
                
                if not self.args.no_display:
                    cv2.imshow('Vehicle Tracking', frame)
                
                if video_writer:
                    video_writer.write(frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"[INFO] 已处理 {frame_count} 帧")
                
                if cv2.waitKey(1) == ord('q'):
                    break
        except KeyboardInterrupt:
            print("[INFO] 用户中断")
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            print(f"[INFO] 视频模式完成，共处理 {frame_count} 帧")
    
    def run_carla_mode(self):
        """CARLA 模拟器模式"""
        if not CARLA_AVAILABLE:
            print("[ERROR] CARLA 未安装，无法使用 CARLA 模式")
            print("[INFO] 请安装: pip install carla==0.9.13")
            return
        
        if not ULTRALYTICS_AVAILABLE:
            print("[ERROR] Ultralytics 未安装，无法进行目标检测")
            return
        
        print("[INFO] CARLA 模式: 尝试连接到 localhost:2000")
        
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(10.0)
            
            try:
                world = client.get_world()
                print("[INFO] ✓ 成功连接到 CARLA 模拟器")
            except Exception as e:
                print(f"[ERROR] 连接 CARLA 失败: {str(e)}")
                print("[INFO] 请确保 CARLA 模拟器已启动")
                print("[INFO] 或使用 --video 参数运行视频模式")
                return
            
            # 获取地图和生成点
            spawn_points = world.get_map().get_spawn_points()
            if not spawn_points:
                print("[ERROR] 无法获取生成点")
                return
            
            print(f"[INFO] 找到 {len(spawn_points)} 个生成点")
            
            # 生成主车辆
            vehicle_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
            vehicle_bp.set_attribute('role_name', 'ego')
            ego_vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            
            if not ego_vehicle:
                print("[ERROR] 无法生成主车辆")
                return
            
            print("[INFO] ✓ 主车辆生成成功")
            
            # 设置相机
            camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(img_w))
            camera_bp.set_attribute('image_size_y', str(img_h))
            camera_bp.set_attribute('fov', '110')
            
            camera_location = carla.Location(2, 0, 1)
            camera_rotation = carla.Rotation(0, 180, 0)
            camera_init_trans = carla.Transform(camera_location, camera_rotation)
            
            # CARLA 0.9.13 附件类型：Rigid, SpringArm
            camera = world.spawn_actor(
                camera_bp, 
                camera_init_trans, 
                attach_to=ego_vehicle,
                attachment_type=carla.AttachmentType.Rigid
            )
            
            print("[INFO] ✓ 相机生成成功")
            
            # 生成 NPC 车辆
            npc_count = 0
            for i in range(20):
                vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle'))
                npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
                if npc:
                    npc.set_autopilot(True)
                    npc_count += 1
            
            print(f"[INFO] ✓ 生成了 {npc_count} 个 NPC 车辆")
            
            # 图像数据
            camera_data = {'image': np.zeros((img_h, img_w, 3), dtype=np.uint8)}
            
            def capture_image(image):
                try:
                    image_data_array = np.array(image.raw_data)
                    image_rgb = image_data_array.reshape((image.height, image.width, 4))[:, :, :3]
                    camera_data['image'] = image_rgb
                except Exception as e:
                    print(f"[ERROR] 图像捕获失败: {str(e)}")
            
            camera.listen(capture_image)
            ego_vehicle.set_autopilot(True)
            
            # 视频写入器
            video_writer = None
            if self.args.save_output:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, 14.0, (img_w, img_h))
            
            print("[INFO] ✓ 开始跟踪... 按 'q' 退出")
            
            frame_count = 0
            try:
                while True:
                    frame = camera_data['image'].copy()
                    frame = self.process_frame(frame)
                    
                    if not self.args.no_display:
                        cv2.imshow('CARLA Tracking', frame)
                    
                    if video_writer:
                        video_writer.write(frame)
                    
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"[INFO] 已处理 {frame_count} 帧")
                    
                    if cv2.waitKey(1) == ord('q'):
                        break
                        
            except KeyboardInterrupt:
                print("[INFO] 用户中断")
            finally:
                print("[INFO] 清理资源...")
                camera.stop()
                camera.destroy()
                ego_vehicle.destroy()
                
                for npc in world.get_actors().filter('vehicle*'):
                    try:
                        npc.destroy()
                    except:
                        pass
                
                if video_writer:
                    video_writer.release()
                
                cv2.destroyAllWindows()
                print("[INFO] ✓ CARLA 模式结束")
        
        except Exception as e:
            print(f"[ERROR] CARLA 运行失败: {str(e)}")
            traceback.print_exc()
    
    def run(self):
        """运行跟踪"""
        if self.args.video:
            self.run_video_mode(self.args.video)
        else:
            self.run_carla_mode()


# ==================== 【原有】主函数 ====================
def parse_args():
    parser = argparse.ArgumentParser(
        description='YOLO + DeepSort 多车跟踪系统\n'
                   '兼容 Python 3.8.10 + CARLA 0.9.13\n\n'
                   '【新增功能】轨迹预测 + 提前碰撞预警',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--video', type=str, default=None,
                       help='视频文件路径（跳过 CARLA）')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示画面')
    parser.add_argument('--save-output', action='store_true',
                       help='保存输出视频')
    
    return parser.parse_args()


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("YOLO + DeepSort 多车跟踪系统")
    print("【新增：轨迹预测+提前碰撞预警模块】")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"平台: {sys.platform}")
    print(f"工作目录: {os.getcwd()}")
    print("=" * 60)
    
    # 显示【新增：轨迹预测+提前碰撞预警】配置
    print("\n【新增：轨迹预测+提前碰撞预警】当前配置:")
    print(f"  - 轨迹长度: {MAX_TRAJECTORY_LENGTH} 帧")
    print(f"  - 预测帧数: {PREDICT_FRAMES} 帧")
    print(f"  - 碰撞阈值: {COLLISION_DISTANCE_RATIO*100:.0f}% 画面宽度 ({COLLISION_DISTANCE_THRESHOLD} 像素)")
    print("=" * 60)


def main():
    """主入口函数"""
    check_environment()
    
    args = parse_args()
    
    try:
        tracker = VehicleTracker(args)
        tracker.run()
    except KeyboardInterrupt:
        print("\n[INFO] 程序被用户中断")
    except Exception as e:
        print(f"\n[FATAL ERROR] 程序异常终止: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
