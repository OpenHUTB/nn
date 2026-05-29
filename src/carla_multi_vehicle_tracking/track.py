#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO + DeepSort 多车跟踪系统
兼容 Python 3.8.10 + CARLA 0.9.13 + Windows 10

【新增：轨迹预测+提前碰撞预警模块】
【新增：车速估算 + 超速报警】
【新增：违章行为检测】

功能说明：
1. 实时跟踪多个车辆目标
2. 记录每辆车的轨迹历史（中心点）
3. 基于轨迹预测未来位置
4. 检测潜在的碰撞风险并预警
5. 估算车辆行驶速度，超速时报警
6. 检测逆行和拥堵违章行为
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

# ==================== 【新增：违章行为检测】配置参数 ====================
# 轨迹字典：保存每个车辆的轨迹
violation_trajectories = {}
# 拥堵判定：画面车辆≥6 辆视为拥堵
CONGESTION_THRESHOLD = 6

# ==================== 【新增：车速估算 + 超速报警】配置参数 ====================
# 以下参数可自由调整

# 帧率：视频帧率
FPS = 30  # 每秒帧数
# 像素与米的换算系数（1像素等于多少米）
PIXEL_TO_METER = 0.1  # 1像素 = 0.1米
# 限速（km/h）
SPEED_LIMIT = 60  # 限速60公里/小时
# 保存车辆轨迹字典（用于存储每个track_id的最近几帧中心点）
vehicle_trajectories = {}

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

# ==================== 【新增：UI界面增强模块】配置参数 ====================
# 以下参数用于控制顶部信息条的显示样式

# UI信息条配置
UI_BAR_HEIGHT = 40  # 信息条高度（像素）
UI_BAR_COLOR = (0, 0, 0)  # 黑色背景 (BGR格式)
UI_BAR_ALPHA = 0.6  # 半透明透明度 (0-1，0=完全透明，1=完全不透明)
UI_TEXT_COLOR = (255, 255, 255)  # 白色文字 (BGR格式)
UI_TEXT_SCALE = 0.6  # 文字大小系数
UI_TEXT_THICKNESS = 1  # 文字线条粗细
UI_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX  # 使用的字体

# ==================== 【新增：UI界面增强模块】函数定义 ====================

def draw_ui_info_bar(frame, vehicle_count, collision_warnings, overspeed_count, retrograde_count, is_congested):
    """
    【新增：UI界面增强模块】
    在画面顶部绘制黑色半透明信息条，显示实时数据
    
    参数:
        frame: 视频帧
        vehicle_count: 当前车辆总数
        collision_warnings: 碰撞预警次数
        overspeed_count: 超速车辆数量
        retrograde_count: 逆行车辆数量
        is_congested: 是否拥堵
    
    返回:
        frame: 绘制了信息条的帧
    """
    # 【新增：UI界面增强模块】获取帧的尺寸
    frame_height, frame_width = frame.shape[:2]
    
    # 【新增：UI界面增强模块】创建黑色半透明背景条
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_width, UI_BAR_HEIGHT), UI_BAR_COLOR, -1)
    
    # 【新增：UI界面增强模块】将半透明背景条叠加到原帧上
    cv2.addWeighted(overlay, UI_BAR_ALPHA, frame, 1 - UI_BAR_ALPHA, 0, frame)
    
    # 【新增：UI界面增强模块】准备信息文本和颜色
    congestion_status = "拥堵" if is_congested else "正常"
    congestion_color = (0, 0, 255) if is_congested else (0, 255, 0)  # 拥堵红色，正常绿色
    
    # 【新增：UI界面增强模块】准备所有要显示的信息项列表
    info_items = [
        f"车辆总数: {vehicle_count}",
        f"碰撞预警: {collision_warnings}",
        f"超速车辆: {overspeed_count}",
        f"逆行车辆: {retrograde_count}",
        f"拥堵状态: {congestion_status}"
    ]
    
    # 【新增：UI界面增强模块】计算每个信息项的文本宽度
    total_width = 0
    text_widths = []
    for item in info_items:
        (w, h), _ = cv2.getTextSize(item, UI_TEXT_FONT, UI_TEXT_SCALE, UI_TEXT_THICKNESS)
        text_widths.append(w)
        total_width += w
    
    # 【新增：UI界面增强模块】计算均匀分布的间距
    spacing = (frame_width - total_width) / (len(info_items) + 1)
    
    # 【新增：UI界面增强模块】开始绘制每个信息项
    current_x = int(spacing)
    text_y = int(UI_BAR_HEIGHT / 2 + UI_TEXT_SCALE * 10)  # 垂直居中
    
    for i, item in enumerate(info_items):
        # 【新增：UI界面增强模块】最后一项（拥堵状态）使用特殊颜色，其他项用白色
        if i == len(info_items) - 1:
            cv2.putText(frame, item, (current_x, text_y), 
                       UI_TEXT_FONT, UI_TEXT_SCALE, congestion_color, UI_TEXT_THICKNESS, cv2.LINE_AA)
        else:
            cv2.putText(frame, item, (current_x, text_y), 
                       UI_TEXT_FONT, UI_TEXT_SCALE, UI_TEXT_COLOR, UI_TEXT_THICKNESS, cv2.LINE_AA)
        
        # 【新增：UI界面增强模块】移动到下一个信息项的位置
        current_x += text_widths[i] + int(spacing)
    
    return frame

# ==================== 【新增：违章行为检测】函数定义 ====================

def update_violation_trajectories(tracked_vehicles, traj_dict):
    """
    【新增：违章行为检测】
    更新车辆轨迹，保存最近8帧中心点
    
    参数:
        tracked_vehicles: DeepSort输出的跟踪结果
        traj_dict: 车辆轨迹字典
    """
    # 遍历所有跟踪车辆
    for output in tracked_vehicles:
        if len(output) >= 5:
            try:
                x1, y1, x2, y2 = map(int, output[0:4])
                track_id = int(output[4])
                
                # 计算中心点
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 更新轨迹字典
                if track_id not in traj_dict:
                    traj_dict[track_id] = []
                traj_dict[track_id].append((center_x, center_y))
                # 只保存最近8帧
                if len(traj_dict[track_id]) > 8:
                    traj_dict[track_id].pop(0)
                    
            except (ValueError, TypeError, IndexError):
                continue

def detect_violations(traj_dict, tracked_vehicles, img_height, congestion_threshold):
    """
    【新增：违章行为检测】
    检测车辆违章行为（逆行和拥堵）
    
    参数:
        traj_dict: 车辆轨迹字典
        tracked_vehicles: DeepSort输出的跟踪结果
        img_height: 画面高度
        congestion_threshold: 拥堵阈值
    
    返回:
        set: retrograde_ids 逆行车辆ID集合
        bool: is_congested 是否拥堵
    """
    retrograde_ids = set()
    vehicle_count = len(tracked_vehicles)
    
    # 判断是否拥堵
    is_congested = vehicle_count >= congestion_threshold
    if is_congested:
        print(f"【拥堵警告】当前区域车辆密集，存在拥堵风险")
    
    # 判断逆行
    for track_id, traj in traj_dict.items():
        if len(traj) >= 2:
            # 计算车辆整体移动方向
            total_dy = 0
            for i in range(1, len(traj)):
                prev_y = traj[i-1][1]
                curr_y = traj[i][1]
                total_dy += (prev_y - curr_y)  # y减少表示向上移动
            
            # 车辆整体向上移动 → 判定逆行
            if total_dy > 0:  # 总位移是向上的
                retrograde_ids.add(track_id)
                print(f"【逆行警告】车辆 ID:{track_id} 存在逆行行为")
    
    return retrograde_ids, is_congested

def draw_violation_warnings(frame, retrograde_ids, is_congested, tracked_vehicles):
    """
    【新增：违章行为检测】
    在画面上绘制违章警告信息
    
    参数:
        frame: 视频帧
        retrograde_ids: 逆行车辆ID集合
        is_congested: 是否拥堵
        tracked_vehicles: 跟踪结果
    
    返回:
        frame: 绘制了警告信息的帧
    """
    # 准备顶部警告文字
    warning_text = ""
    if is_congested and len(retrograde_ids) > 0:
        warning_text = "逆行 / 拥堵状态"
    elif len(retrograde_ids) > 0:
        warning_text = "逆行状态"
    elif is_congested:
        warning_text = "拥堵状态"
    
    # 绘制顶部警告文字
    if warning_text:
        cv2.putText(
            frame,
            warning_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
    
    # 为逆行车辆绘制红色边框
    for output in tracked_vehicles:
        if len(output) >= 5:
            try:
                x1, y1, x2, y2 = map(int, output[0:4])
                track_id = int(output[4])
                
                # 如果是逆行车辆，绘制红色边框
                if track_id in retrograde_ids:
                    cv2.rectangle(
                        frame, 
                        (x1, y1), 
                        (x2, y2), 
                        (0, 0, 255), 
                        3
                    )
            except (ValueError, TypeError, IndexError):
                continue
    
    return frame

# ==================== 【新增：车速估算 + 超速报警】函数定义 ====================

def update_vehicle_trajectories(tracked_vehicles, traj_dict):
    """
    【新增：车速估算 + 超速报警】
    更新车辆轨迹，保存最近5帧中心点
    
    参数:
        tracked_vehicles: DeepSort输出的跟踪结果
        traj_dict: 车辆轨迹字典
    """
    # 遍历所有跟踪车辆
    for output in tracked_vehicles:
        if len(output) >= 5:
            try:
                x1, y1, x2, y2 = map(int, output[0:4])
                track_id = int(output[4])
                
                # 计算中心点
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 更新轨迹字典
                if track_id not in traj_dict:
                    traj_dict[track_id] = []
                traj_dict[track_id].append((center_x, center_y))
                # 只保存最近5帧
                if len(traj_dict[track_id]) > 5:
                    traj_dict[track_id].pop(0)
                    
            except (ValueError, TypeError, IndexError):
                continue

def calculate_speed(traj, fps, pixel_to_meter):
    """
    【新增：车速估算 + 超速报警】
    基于轨迹计算车辆速度
    
    参数:
        traj: 轨迹列表 [(cx1, cy1), (cx2, cy2), ...]
        fps: 帧率
        pixel_to_meter: 像素转米的比例
    
    返回:
        float: 速度(km/h)，如果轨迹不足2帧返回0
    """
    if len(traj) < 2:
        return 0.0
    
    # 计算最近两帧的中心点
    prev_x, prev_y = traj[-2]
    curr_x, curr_y = traj[-1]
    
    # 计算像素距离
    pixel_distance = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5
    
    # 换算成米
    meter_distance = pixel_distance * pixel_to_meter
    
    # 计算每秒移动时间（秒）
    time_seconds = 1 / fps
    
    # 计算米每秒 (m/s)
    speed_ms = meter_distance / time_seconds
    
    # 换算成公里每小时 (km/h)
    speed_kmh = speed_ms * 3.6
    
    return speed_kmh

def estimate_vehicle_speeds(traj_dict, fps, pixel_to_meter, speed_limit):
    """
    【新增：车速估算 + 超速报警】
    估算所有车辆速度，判断是否超速
    
    参数:
        traj_dict: 车辆轨迹字典
        fps: 帧率
        pixel_to_meter: 像素转米的比例
        speed_limit: 限速
    
    返回:
        dict: speed_dict {track_id: speed_kmh}
        set: overspeed_ids {track_id}
    """
    speed_dict = {}
    overspeed_ids = set()
    
    for track_id, traj in traj_dict.items():
        speed = calculate_speed(traj, fps, pixel_to_meter)
        speed_dict[track_id] = speed
        
        # 判断是否超速
        if speed > speed_limit:
            overspeed_ids.add(track_id)
            # 控制台输出超速警告
            print(f"【超速警告】车辆 ID:{track_id} 当前速度：{speed:.1f} km/h")
    
    return speed_dict, overspeed_ids

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
        trajectory_dict: 轨迹历史字典 {track_id: [(cx1,cy1), (cx2,cy2), ...}
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
        
        # 【新增：车速估算 + 超速报警】初始化车辆轨迹字典
        self.vehicle_traj = {}
        
        # 【新增：违章行为检测】初始化违章检测轨迹字典
        self.violation_traj = {}
        
        # 【新增：UI界面增强模块】初始化碰撞预警计数器
        # 用于统计累计碰撞预警次数，在UI信息条中显示
        self.collision_warning_count = 0
        
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
    
    def draw_bbox(self, frame, output, conf, cls_id, collision_risk_ids=None, speed_dict=None, overspeed_ids=None, retrograde_ids=None):
        """
        【原有函数，修改】绘制边界框
        
        参数:
            frame: 视频帧
            output: 跟踪输出 [x1, y1, x2, y2, track_id, ...]
            conf: 置信度
            cls_id: 类别ID
            collision_risk_ids: 【新增参数】有碰撞风险的track_id集合，如果为None则不检查
            speed_dict: 【新增：车速估算 + 超速报警】车辆速度字典
            overspeed_ids: 【新增：车速估算 + 超速报警】超速车辆ID集合
            retrograde_ids: 【新增：违章行为检测】逆行车辆ID集合
        
        功能:
            1. 如果车辆在collision_risk_ids中，绘制红色边框
            2. 如果车辆在overspeed_ids中，绘制红色边框，显示速度
            3. 如果车辆在retrograde_ids中，绘制红色边框
            4. 否则使用原有颜色逻辑，显示速度
        """
        try:
            x1, y1, x2, y2 = map(int, output[0:4])
            track_id = int(output[4])
            label = class_name.get(cls_id, str(cls_id))
            
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            
            # 【新增：车速估算 + 超速报警】判断是否为超速车辆
            is_overspeed = False
            if overspeed_ids is not None and track_id in overspeed_ids:
                is_overspeed = True
            
            # 【新增：轨迹预测+提前碰撞预警】判断是否为碰撞风险车辆
            is_risk = False
            if collision_risk_ids is not None and track_id in collision_risk_ids:
                is_risk = True
            
            # 【新增：违章行为检测】判断是否为逆行车辆
            is_retrograde = False
            if retrograde_ids is not None and track_id in retrograde_ids:
                is_retrograde = True
            
            # 【新增：车速估算 + 超速报警】获取车辆速度
            speed_str = ""
            if speed_dict is not None and track_id in speed_dict:
                speed_str = f" {speed_dict[track_id]:.1f}km/h"
            
            # 确定颜色和标签
            if is_overspeed or is_risk or is_retrograde:
                # 违章车辆使用红色
                colour = (0, 0, 255)
                label = f"[!] {label}"
            else:
                # 普通车辆使用原有颜色
                colour = self.colour_label(track_id)
            
            c_id = f'{label} {track_id}{speed_str}'
            
            t_size = cv2.getTextSize(c_id, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            
            # 根据是否为违章车辆决定边框粗细
            box_thickness = 3 if is_overspeed or is_risk or is_retrograde else 1
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, box_thickness)
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
            3. 【新增：车速估算 + 超速报警】更新车辆轨迹
            4. 【新增：车速估算 + 超速报警】估算车辆速度，判断是否超速
            5. 【新增：违章行为检测】更新违章检测轨迹
            6. 【新增：违章行为检测】检测逆行和拥堵
            7. 绘制预警信息
        """
        frame, bbox_xyxy, conf_score, cls_id = self.yolo_details(frame)
        
        if len(bbox_xyxy) > 0 and self.deepsort is not None:
            try:
                outputs = self.deepsort.update(bbox_xyxy, conf_score, frame)
                
                if len(outputs) > 0:
                    # 【新增：轨迹预测+提前碰撞预警】更新轨迹历史
                    update_trajectory(self.trajectory_dict, outputs)
                    
                    # 【新增：车速估算 + 超速报警】更新车辆轨迹（用于速度计算）
                    update_vehicle_trajectories(outputs, self.vehicle_traj)
                    
                    # 【新增：违章行为检测】更新违章检测轨迹
                    update_violation_trajectories(outputs, self.violation_traj)
                    
                    # 【新增：车速估算 + 超速报警】估算车辆速度，判断是否超速
                    speed_dict, overspeed_ids = estimate_vehicle_speeds(
                        self.vehicle_traj,
                        FPS,
                        PIXEL_TO_METER,
                        SPEED_LIMIT
                    )
                    
                    # 【新增：轨迹预测+提前碰撞预警】检测碰撞风险
                    collision_risk = check_trajectory_collision(
                        self.trajectory_dict,
                        PREDICT_FRAMES,
                        COLLISION_DISTANCE_THRESHOLD,
                        img_w
                    )
                    
                    # 【新增：违章行为检测】检测逆行和拥堵
                    retrograde_ids, is_congested = detect_violations(
                        self.violation_traj,
                        outputs,
                        img_h,
                        CONGESTION_THRESHOLD
                    )
                    
                    # 提取风险车辆ID集合
                    collision_risk_ids = set(collision_risk.keys())
                    
                    # 【新增：UI界面增强模块】更新碰撞预警计数器
                    # 如果本帧检测到碰撞风险，则计数器加1
                    if len(collision_risk_ids) > 0:
                        self.collision_warning_count += 1
                    
                    # 【新增：轨迹预测+提前碰撞预警】绘制预警信息
                    frame = draw_collision_warning(frame, collision_risk_ids, outputs)
                    
                    # 【新增：违章行为检测】绘制违章警告信息
                    frame = draw_violation_warnings(frame, retrograde_ids, is_congested, outputs)
                    
                    # 【原有逻辑，保持不变】绘制边界框（增加了更多参数）
                    min_len = min(len(outputs), len(conf_score), len(cls_id))
                    for i in range(min_len):
                        frame = self.draw_bbox(
                            frame, 
                            outputs[i], 
                            conf_score[i], 
                            cls_id[i], 
                            collision_risk_ids,
                            speed_dict,
                            overspeed_ids,
                            retrograde_ids
                        )
                    
                    # 【新增：UI界面增强模块】绘制顶部信息条
                    # 准备各项统计数据
                    vehicle_count = len(outputs)  # 当前车辆总数
                    overspeed_count = len(overspeed_ids)  # 超速车辆数量
                    retrograde_count = len(retrograde_ids)  # 逆行车辆数量
                    # 调用绘制函数显示信息条
                    frame = draw_ui_info_bar(
                        frame, 
                        vehicle_count, 
                        self.collision_warning_count, 
                        overspeed_count, 
                        retrograde_count, 
                        is_congested
                    )
                    
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
                   '【新增功能】轨迹预测 + 提前碰撞预警\n'
                   '【新增功能】车速估算 + 超速报警\n'
                   '【新增功能】违章行为检测',
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
    print("【新增：车速估算 + 超速报警】")
    print("【新增：违章行为检测】")
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
    
    # 显示【新增：车速估算 + 超速报警】配置
    print("\n【新增：车速估算 + 超速报警】当前配置:")
    print(f"  - 帧率: {FPS} fps")
    print(f"  - 像素转米: {PIXEL_TO_METER} m/px")
    print(f"  - 限速: {SPEED_LIMIT} km/h")
    
    # 显示【新增：违章行为检测】配置
    print("\n【新增：违章行为检测】当前配置:")
    print(f"  - 拥堵阈值: {CONGESTION_THRESHOLD} 辆")
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
