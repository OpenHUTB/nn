#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO + DeepSort 多车跟踪系统
兼容 Python 3.8.10 + CARLA 0.9.13 + Windows 10

【新增：轨迹预测+提前碰撞预警模块】
【新增：车速估算 + 超速报警】
【新增：违章行为检测】
【新增：UI界面增强模块】
【新增：车辆轨迹绘制模块】
【新增：第六模块 车流量统计】
【新增：第七模块 车辆属性识别】

功能说明：
1. 实时跟踪多个车辆目标
2. 记录每辆车的轨迹历史（中心点）
3. 基于轨迹预测未来位置
4. 检测潜在的碰撞风险并预警
5. 估算车辆行驶速度，超速时报警
6. 检测逆行和拥堵违章行为
7. 实时显示统计信息面板
8. 绘制车辆运动轨迹
9. 统计车流量（驶入/驶出）
10. 识别车辆属性（车型、颜色）
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

# ==================== 【新增：第六模块 车流量统计】配置参数 ====================
# 虚拟计数线坐标（画面中下部横向统计线，可自行微调）
COUNT_LINE_Y = 450
# 上下行计数
car_in_count = 0  # 驶入车辆计数
car_out_count = 0  # 驶出车辆计数
total_car_count = 0  # 总车流量
# 记录车辆是否已经统计过，防止重复计数
counted_id = set()  # 已统计的车辆ID集合

# 车流量统计UI显示配置
TRAFFIC_UI_POSITION = (10, 60)  # 左上角位置 (x, y)
TRAFFIC_UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
TRAFFIC_UI_FONT_SCALE = 0.5
TRAFFIC_UI_FONT_THICKNESS = 1
TRAFFIC_UI_TEXT_COLOR = (255, 255, 0)  # 黄色文字 (BGR)
TRAFFIC_UI_BG_COLOR = (50, 50, 50)  # 半透明背景
TRAFFIC_UI_BG_ALPHA = 0.7

# ==================== 【新增：第七模块 车辆属性识别】配置参数 ====================
# 定义车型列表
car_type = ["轿车", "SUV", "面包车", "货车"]
# 定义车身颜色列表
car_color = ["白色", "黑色", "红色", "蓝色", "黄色", "灰色"]
# 建立字典：key=跟踪ID，value=[车型,颜色]，保存每辆车识别结果，避免重复识别
car_attr = {}

# 车辆属性识别UI配置
CAR_ATTR_FONT = cv2.FONT_HERSHEY_SIMPLEX
CAR_ATTR_FONT_SCALE = 0.4
CAR_ATTR_FONT_THICKNESS = 1
CAR_ATTR_TEXT_COLOR = (255, 255, 255)  # 白色文字

# ==================== 【原有】基础配置常量 ====================
# 【新增：车辆轨迹绘制模块】轨迹绘制配置
MAX_TRAJECTORY_POINTS = 15  # 轨迹保留15帧
TRAJECTORY_LINE_WIDTH = 2  # 轨迹线宽度
TRAJECTORY_LINE_ALPHA = 0.8  # 轨迹线透明度

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

def draw_ui_info_bar(frame, vehicle_count, collision_warnings, overspeed_count, retrograde_count, is_congested, car_type_stats=None, car_color_stats=None):
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
        car_type_stats: 【新增：第七模块 车辆属性识别】车型统计字典
        car_color_stats: 【新增：第七模块 车辆属性识别】颜色统计字典
    
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
    
    # 【新增：第七模块 车辆属性识别】添加车型统计信息
    if car_type_stats:
        type_text = "车型:"
        for t in car_type:
            count = car_type_stats.get(t, 0)
            if count > 0:
                type_text += f" {t}{count}"
        info_items.append(type_text)
    
    # 【新增：第七模块 车辆属性识别】添加颜色统计信息
    if car_color_stats and vehicle_count > 0:
        color_text = "颜色:"
        for c in car_color:
            count = car_color_stats.get(c, 0)
            if count > 0:
                ratio = (count / vehicle_count) * 100
                color_text += f" {c}{int(ratio)}%"
        info_items.append(color_text)
    
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
        if i == 4:  # 拥堵状态
            cv2.putText(frame, item, (current_x, text_y), 
                       UI_TEXT_FONT, UI_TEXT_SCALE, congestion_color, UI_TEXT_THICKNESS, cv2.LINE_AA)
        else:
            cv2.putText(frame, item, (current_x, text_y), 
                       UI_TEXT_FONT, UI_TEXT_SCALE, UI_TEXT_COLOR, UI_TEXT_THICKNESS, cv2.LINE_AA)
        
        # 【新增：UI界面增强模块】移动到下一个信息项的位置
        current_x += text_widths[i] + int(spacing)
    
    return frame

# ==================== 【新增：第七模块 车辆属性识别】函数定义 ====================

def initialize_car_attr():
    """
    【新增：第七模块 车辆属性识别】
    初始化车辆属性字典
    
    返回:
        dict: 车辆属性字典 {track_id: [车型, 颜色]}
    """
    return {}

def classify_car_type(bbox):
    """
    【新增：第七模块 车辆属性识别】
    根据检测框长宽比区分车型
    
    参数:
        bbox: 检测框 [x1, y1, x2, y2]
    
    返回:
        str: 车型（轿车/SUV/面包车/货车）
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    if width <= 0 or height <= 0:
        return "轿车"
    
    # 计算框高/框宽比值
    ratio = float(height) / float(width)
    
    # 根据长宽比分类车型
    if ratio < 0.7:
        return "轿车"
    elif 0.7 <= ratio < 1.1:
        return "SUV"
    elif 1.1 <= ratio < 1.5:
        return "面包车"
    else:
        return "货车"

def classify_car_color(frame, bbox):
    """
    【新增：第七模块 车辆属性识别】
    截取车辆框内中心区域像素，统计RGB均值匹配车身主色
    
    参数:
        frame: 视频帧
        bbox: 检测框 [x1, y1, x2, y2]
    
    返回:
        str: 车身颜色（白色/黑色/红色/蓝色/黄色/灰色）
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # 确保边界在图像范围内
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width - 1, x2)
    y2 = min(frame_height - 1, y2)
    
    # 截取中心区域（约占整个框的40%）
    width = x2 - x1
    height = y2 - y1
    margin_x = int(width * 0.3)
    margin_y = int(height * 0.3)
    
    crop_x1 = x1 + margin_x
    crop_y1 = y1 + margin_y
    crop_x2 = x2 - margin_x
    crop_y2 = y2 - margin_y
    
    if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
        # 如果中心区域太小，使用整个框
        crop_x1, crop_y1, crop_x2, crop_y2 = x1, y1, x2, y2
    
    # 提取中心区域
    roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # 计算RGB均值
    avg_b = np.mean(roi[:, :, 0])
    avg_g = np.mean(roi[:, :, 1])
    avg_r = np.mean(roi[:, :, 2])
    
    # 根据RGB均值判断颜色
    total = avg_r + avg_g + avg_b
    
    # 判断白色（高亮度，RGB接近）
    if avg_r > 180 and avg_g > 180 and avg_b > 180:
        return "白色"
    
    # 判断黑色（低亮度）
    if total < 100:
        return "黑色"
    
    # 判断灰色（中等亮度，RGB接近）
    diff_rg = abs(avg_r - avg_g)
    diff_rb = abs(avg_r - avg_b)
    diff_gb = abs(avg_g - avg_b)
    if diff_rg < 30 and diff_rb < 30 and diff_gb < 30 and total > 100 and total < 500:
        return "灰色"
    
    # 判断红色（R明显高于其他）
    if avg_r > avg_g + 30 and avg_r > avg_b + 30:
        return "红色"
    
    # 判断黄色（R和G都较高，B较低）
    if avg_r > 150 and avg_g > 150 and avg_b < 100:
        return "黄色"
    
    # 判断蓝色（B明显高于其他）
    if avg_b > avg_r + 30 and avg_b > avg_g + 30:
        return "蓝色"
    
    # 默认返回灰色
    return "灰色"

def update_car_attr(tracked_vehicles, frame, car_attr_dict):
    """
    【新增：第七模块 车辆属性识别】
    更新车辆属性，同一跟踪ID只在首次出现时识别一次
    
    参数:
        tracked_vehicles: DeepSort输出的跟踪结果
        frame: 视频帧
        car_attr_dict: 车辆属性字典
    
    返回:
        dict: 更新后的车辆属性字典
    """
    for output in tracked_vehicles:
        if len(output) >= 5:
            try:
                x1, y1, x2, y2 = map(int, output[0:4])
                track_id = int(output[4])
                
                # 【新增：第七模块 车辆属性识别】只在首次出现时识别
                if track_id not in car_attr_dict:
                    bbox = [x1, y1, x2, y2]
                    car_type_result = classify_car_type(bbox)
                    car_color_result = classify_car_color(frame, bbox)
                    car_attr_dict[track_id] = [car_type_result, car_color_result]
                    
            except (ValueError, TypeError, IndexError):
                continue
    
    return car_attr_dict

def get_car_type_stats(car_attr_dict):
    """
    【新增：第七模块 车辆属性识别】
    统计各类车型数量
    
    参数:
        car_attr_dict: 车辆属性字典
    
    返回:
        dict: 车型统计 {车型: 数量}
    """
    stats = {t: 0 for t in car_type}
    for attr in car_attr_dict.values():
        if attr[0] in stats:
            stats[attr[0]] += 1
    return stats

def get_car_color_stats(car_attr_dict):
    """
    【新增：第七模块 车辆属性识别】
    统计各色车辆数量
    
    参数:
        car_attr_dict: 车辆属性字典
    
    返回:
        dict: 颜色统计 {颜色: 数量}
    """
    stats = {c: 0 for c in car_color}
    for attr in car_attr_dict.values():
        if attr[1] in stats:
            stats[attr[1]] += 1
    return stats

def draw_car_attr_label(frame, bbox, track_id, car_attr_dict):
    """
    【新增：第七模块 车辆属性识别】
    在车辆目标框右下角标注颜色和车型
    
    参数:
        frame: 视频帧
        bbox: 检测框 [x1, y1, x2, y2]
        track_id: 跟踪ID
        car_attr_dict: 车辆属性字典
    
    返回:
        frame: 绘制了属性标签的帧
    """
    if track_id not in car_attr_dict:
        return frame
    
    x1, y1, x2, y2 = map(int, bbox)
    car_type_result, car_color_result = car_attr_dict[track_id]
    
    # 准备标签文本
    label_text = f"{car_color_result}{car_type_result}"
    
    # 计算文本尺寸
    (text_width, text_height), _ = cv2.getTextSize(label_text, CAR_ATTR_FONT, CAR_ATTR_FONT_SCALE, CAR_ATTR_FONT_THICKNESS)
    
    # 计算标签位置（右下角）
    label_x = x2 - text_width - 5
    label_y = y2 - 5
    
    # 确保标签在画面内
    label_x = max(0, label_x)
    label_y = max(text_height + 5, label_y)
    
    # 绘制背景矩形
    cv2.rectangle(frame, 
                  (label_x - 2, label_y - text_height - 2), 
                  (x2 - 5, y2 - 3), 
                  (0, 0, 0),  # 黑色背景
                  -1)
    
    # 绘制文本
    cv2.putText(frame, label_text, (label_x, label_y),
                CAR_ATTR_FONT, CAR_ATTR_FONT_SCALE,
                CAR_ATTR_TEXT_COLOR, CAR_ATTR_FONT_THICKNESS, cv2.LINE_AA)
    
    return frame

# ==================== 【新增：第六模块 车流量统计】函数定义 ====================

def initialize_traffic_counting():
    """
    【新增：第六模块 车流量统计】
    初始化车流量统计相关变量
    
    返回:
        tuple: (previous_positions, counted_ids, in_count, out_count, total_count)
    """
    return {}, set(), 0, 0, 0

def update_traffic_counting(tracked_vehicles, count_line_y, previous_positions, counted_ids, in_count, out_count, total_count):
    """
    【新增：第六模块 车流量统计】
    更新车流量统计，检测车辆是否穿过虚拟统计线
    
    参数:
        tracked_vehicles: DeepSort输出的跟踪结果
        count_line_y: 虚拟统计线的Y坐标
        previous_positions: 上帧车辆位置字典 {track_id: previous_y}
        counted_ids: 已统计的车辆ID集合
        in_count: 驶入计数
        out_count: 驶出计数
        total_count: 总车流量
    
    返回:
        tuple: (current_positions, in_count, out_count, total_count)
    """
    current_positions = {}
    
    for output in tracked_vehicles:
        if len(output) >= 5:
            try:
                x1, y1, x2, y2 = map(int, output[0:4])
                track_id = int(output[4])
                
                center_y = (y1 + y2) / 2
                current_positions[track_id] = center_y
                
                if track_id in previous_positions and track_id not in counted_ids:
                    prev_y = previous_positions[track_id]
                    curr_y = center_y
                    
                    if prev_y > count_line_y and curr_y <= count_line_y:
                        out_count += 1
                        total_count += 1
                        counted_ids.add(track_id)
                    
                    elif prev_y < count_line_y and curr_y >= count_line_y:
                        in_count += 1
                        total_count += 1
                        counted_ids.add(track_id)
                        
            except (ValueError, TypeError, IndexError):
                continue
    
    return current_positions, in_count, out_count, total_count

def draw_traffic_counting_ui(frame, in_count, out_count, total_count, count_line_y):
    """
    【新增：第六模块 车流量统计】
    在画面上绘制车流量统计信息
    
    参数:
        frame: 视频帧
        in_count: 驶入计数
        out_count: 驶出计数
        total_count: 总车流量
        count_line_y: 统计线Y坐标
    
    返回:
        frame: 绘制了车流量信息的帧
    """
    traffic_text = [
        f"总车流量: {total_count}",
        f"驶入车辆: {in_count}",
        f"驶出车辆: {out_count}"
    ]
    
    frame_height, frame_width = frame.shape[:2]
    max_width = 0
    text_height = 0
    for text in traffic_text:
        (w, h), _ = cv2.getTextSize(text, TRAFFIC_UI_FONT, TRAFFIC_UI_FONT_SCALE, TRAFFIC_UI_FONT_THICKNESS)
        max_width = max(max_width, w)
        text_height += h + 10
    
    bg_x1 = TRAFFIC_UI_POSITION[0] - 5
    bg_y1 = TRAFFIC_UI_POSITION[1] - 20
    bg_x2 = bg_x1 + max_width + 20
    bg_y2 = bg_y1 + text_height + 15
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), TRAFFIC_UI_BG_COLOR, -1)
    cv2.addWeighted(overlay, TRAFFIC_UI_BG_ALPHA, frame, 1 - TRAFFIC_UI_BG_ALPHA, 0, frame)
    
    current_y = TRAFFIC_UI_POSITION[1]
    for text in traffic_text:
        cv2.putText(
            frame,
            text,
            (TRAFFIC_UI_POSITION[0], current_y),
            TRAFFIC_UI_FONT,
            TRAFFIC_UI_FONT_SCALE,
            TRAFFIC_UI_TEXT_COLOR,
            TRAFFIC_UI_FONT_THICKNESS,
            cv2.LINE_AA
        )
        current_y += 25
    
    return frame

# ==================== 【新增：车辆轨迹绘制模块】函数定义 ====================

def initialize_vehicle_path():
    """
    【新增：车辆轨迹绘制模块】
    初始化车辆轨迹字典
    
    返回:
        dict: 存储每辆车的轨迹点 {track_id: [(cx, cy), ...]}
    """
    return {}

def update_vehicle_path(traj_dict, tracked_vehicles, max_points):
    """
    【新增：车辆轨迹绘制模块】
    更新车辆轨迹点，记录最近若干帧的中心点坐标
    
    参数:
        traj_dict: 轨迹字典 {track_id: [(cx, cy), ...]}
        tracked_vehicles: DeepSort输出的跟踪结果
        max_points: 最大保留点数
    """
    for output in tracked_vehicles:
        if len(output) >= 5:
            try:
                x1, y1, x2, y2 = map(int, output[0:4])
                track_id = int(output[4])
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                if track_id not in traj_dict:
                    traj_dict[track_id] = []
                traj_dict[track_id].append((center_x, center_y))
                
                if len(traj_dict[track_id]) > max_points:
                    traj_dict[track_id].pop(0)
                    
            except (ValueError, TypeError, IndexError):
                continue

def draw_vehicle_trajectories(frame, traj_dict, colour_func):
    """
    【新增：车辆轨迹绘制模块】
    在画面上绘制车辆运动轨迹（连线形式）
    
    参数:
        frame: 视频帧
        traj_dict: 轨迹字典 {track_id: [(cx, cy), ...]}
        colour_func: 用于生成车辆颜色的函数
    
    返回:
        frame: 绘制了轨迹的帧
    """
    for track_id, trajectory in traj_dict.items():
        if len(trajectory) < 2:
            continue
        
        colour = colour_func(track_id)
        
        for i in range(1, len(trajectory)):
            pt1 = trajectory[i-1]
            pt2 = trajectory[i]
            
            cv2.line(
                frame, 
                (int(pt1[0]), int(pt1[1])), 
                (int(pt2[0]), int(pt2[1])), 
                colour, 
                TRAJECTORY_LINE_WIDTH
            )
        
        if len(trajectory) > 0:
            start_pt = trajectory[0]
            cv2.circle(
                frame, 
                (int(start_pt[0]), int(start_pt[1])), 
                3,
                colour, 
                -1
            )
    
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
    for output in tracked_vehicles:
        if len(output) >= 5:
            try:
                x1, y1, x2, y2 = map(int, output[0:4])
                track_id = int(output[4])
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                if track_id not in traj_dict:
                    traj_dict[track_id] = []
                traj_dict[track_id].append((center_x, center_y))
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
    
    is_congested = vehicle_count >= congestion_threshold
    if is_congested:
        print(f"【拥堵警告】当前区域车辆密集，存在拥堵风险")
    
    for track_id, traj in traj_dict.items():
        if len(traj) >= 2:
            total_dy = 0
            for i in range(1, len(traj)):
                prev_y = traj[i-1][1]
                curr_y = traj[i][1]
                total_dy += (prev_y - curr_y)
            
            if total_dy > 0:
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
    warning_text = ""
    if is_congested and len(retrograde_ids) > 0:
        warning_text = "逆行 / 拥堵状态"
    elif len(retrograde_ids) > 0:
        warning_text = "逆行状态"
    elif is_congested:
        warning_text = "拥堵状态"
    
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
    
    for output in tracked_vehicles:
        if len(output) >= 5:
            try:
                x1, y1, x2, y2 = map(int, output[0:4])
                track_id = int(output[4])
                
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
    for output in tracked_vehicles:
        if len(output) >= 5:
            try:
                x1, y1, x2, y2 = map(int, output[0:4])
                track_id = int(output[4])
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                if track_id not in traj_dict:
                    traj_dict[track_id] = []
                traj_dict[track_id].append((center_x, center_y))
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
    
    prev_x, prev_y = traj[-2]
    curr_x, curr_y = traj[-1]
    
    pixel_distance = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5
    meter_distance = pixel_distance * pixel_to_meter
    time_seconds = 1 / fps
    speed_ms = meter_distance / time_seconds
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
        
        if speed > speed_limit:
            overspeed_ids.add(track_id)
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
    """
    current_ids = set()
    
    for output in tracked_vehicles:
        if len(output) >= 5:
            try:
                x1, y1, x2, y2 = map(int, output[0:4])
                track_id = int(output[4])
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                current_ids.add(track_id)
                
                if track_id in trajectory_dict:
                    trajectory_dict[track_id].append((center_x, center_y))
                    if len(trajectory_dict[track_id]) > MAX_TRAJECTORY_LENGTH:
                        trajectory_dict[track_id].pop(0)
                else:
                    trajectory_dict[track_id] = [(center_x, center_y)]
                    
            except (ValueError, TypeError, IndexError):
                continue

def predict_future_position(trajectory, predict_frames):
    """
    【新增：轨迹预测+提前碰撞预警】
    基于轨迹历史预测未来位置（线性预测）
    """
    if len(trajectory) < 2:
        return None
    
    cx_prev, cy_prev = trajectory[-2]
    cx_curr, cy_curr = trajectory[-1]
    
    vx = cx_curr - cx_prev
    vy = cy_curr - cy_prev
    
    future_cx = cx_curr + vx * predict_frames
    future_cy = cy_curr + vy * predict_frames
    
    return (future_cx, future_cy)

def check_trajectory_collision(trajectory_dict, predict_frames, collision_threshold, frame_width):
    """
    【新增：轨迹预测+提前碰撞预警】
    检测所有车辆之间的轨迹碰撞风险
    """
    collision_risk = {}
    vehicle_info = {}
    
    for track_id, trajectory in trajectory_dict.items():
        if len(trajectory) >= 2:
            future_pos = predict_future_position(trajectory, predict_frames)
            if future_pos:
                current_pos = trajectory[-1]
                vehicle_info[track_id] = {
                    'trajectory': trajectory,
                    'current': current_pos,
                    'future': future_pos
                }
    
    track_ids = list(vehicle_info.keys())
    
    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            id1 = track_ids[i]
            id2 = track_ids[j]
            
            vehicle1 = vehicle_info[id1]
            vehicle2 = vehicle_info[id2]
            
            fx1, fy1 = vehicle1['future']
            fx2, fy2 = vehicle2['future']
            
            dx = fx1 - fx2
            dy = fy1 - fy2
            distance = (dx ** 2 + dy ** 2) ** 0.5
            
            if distance < collision_threshold:
                print(f"[提前碰撞预警] 车辆ID:{id1} 和 车辆ID:{id2} 距离过近，存在碰撞风险")
                
                if id1 not in collision_risk:
                    collision_risk[id1] = []
                if id2 not in collision_risk:
                    collision_risk[id2] = []
                
                collision_risk[id1].append((id2, distance))
                collision_risk[id2].append((id1, distance))
    
    return collision_risk

def draw_collision_warning(frame, collision_risk_ids, tracked_vehicles):
    """
    【新增：轨迹预测+提前碰撞预警】
    在视频帧上绘制碰撞预警信息
    """
    if len(collision_risk_ids) > 0:
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
        
        for output in tracked_vehicles:
            if len(output) >= 5:
                try:
                    x1, y1, x2, y2 = map(int, output[0:4])
                    track_id = int(output[4])
                    
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
        
        self._check_dependencies()
        
        self.model = None
        self.deepsort = None
        
        self.trajectory_dict = initialize_trajectory_dict()
        self.vehicle_traj = {}
        self.violation_traj = {}
        self.collision_warning_count = 0
        self.vehicle_path = initialize_vehicle_path()
        
        self.traffic_previous_positions = {}
        self.traffic_counted_ids = set()
        self.traffic_in_count = 0
        self.traffic_out_count = 0
        self.traffic_total_count = 0
        
        # 【新增：第七模块 车辆属性识别】初始化车辆属性字典
        self.car_attr = initialize_car_attr()
        
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
        """
        try:
            x1, y1, x2, y2 = map(int, output[0:4])
            track_id = int(output[4])
            label = class_name.get(cls_id, str(cls_id))
            
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            
            is_overspeed = False
            if overspeed_ids is not None and track_id in overspeed_ids:
                is_overspeed = True
            
            is_risk = False
            if collision_risk_ids is not None and track_id in collision_risk_ids:
                is_risk = True
            
            is_retrograde = False
            if retrograde_ids is not None and track_id in retrograde_ids:
                is_retrograde = True
            
            speed_str = ""
            if speed_dict is not None and track_id in speed_dict:
                speed_str = f" {speed_dict[track_id]:.1f}km/h"
            
            if is_overspeed or is_risk or is_retrograde:
                colour = (0, 0, 255)
                label = f"[!] {label}"
            else:
                colour = self.colour_label(track_id)
            
            c_id = f'{label} {track_id}{speed_str}'
            
            t_size = cv2.getTextSize(c_id, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            
            box_thickness = 3 if is_overspeed or is_risk or is_retrograde else 1
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, box_thickness)
            cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), colour, -1)
            cv2.putText(frame, c_id, (x1, y1 + t_size[1] + 4), 
                       cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 2)
            
            # 【新增：第七模块 车辆属性识别】绘制车辆属性标签
            frame = draw_car_attr_label(frame, [x1, y1, x2, y2], track_id, self.car_attr)
            
        except Exception as e:
            print(f"[ERROR] 绘制边界框失败: {str(e)}")
        
        return frame
    
    def process_frame(self, frame):
        """
        【原有函数，修改】处理单帧图像
        """
        frame, bbox_xyxy, conf_score, cls_id = self.yolo_details(frame)
        
        if len(bbox_xyxy) > 0 and self.deepsort is not None:
            try:
                outputs = self.deepsort.update(bbox_xyxy, conf_score, frame)
                
                if len(outputs) > 0:
                    update_trajectory(self.trajectory_dict, outputs)
                    update_vehicle_trajectories(outputs, self.vehicle_traj)
                    update_violation_trajectories(outputs, self.violation_traj)
                    update_vehicle_path(self.vehicle_path, outputs, MAX_TRAJECTORY_POINTS)
                    
                    (self.traffic_previous_positions, 
                     self.traffic_in_count, 
                     self.traffic_out_count, 
                     self.traffic_total_count) = update_traffic_counting(
                        outputs,
                        COUNT_LINE_Y,
                        self.traffic_previous_positions,
                        self.traffic_counted_ids,
                        self.traffic_in_count,
                        self.traffic_out_count,
                        self.traffic_total_count
                    )
                    
                    # 【新增：第七模块 车辆属性识别】更新车辆属性
                    self.car_attr = update_car_attr(outputs, frame, self.car_attr)
                    
                    speed_dict, overspeed_ids = estimate_vehicle_speeds(
                        self.vehicle_traj,
                        FPS,
                        PIXEL_TO_METER,
                        SPEED_LIMIT
                    )
                    
                    collision_risk = check_trajectory_collision(
                        self.trajectory_dict,
                        PREDICT_FRAMES,
                        COLLISION_DISTANCE_THRESHOLD,
                        img_w
                    )
                    
                    retrograde_ids, is_congested = detect_violations(
                        self.violation_traj,
                        outputs,
                        img_h,
                        CONGESTION_THRESHOLD
                    )
                    
                    collision_risk_ids = set(collision_risk.keys())
                    
                    if len(collision_risk_ids) > 0:
                        self.collision_warning_count += 1
                    
                    frame = draw_vehicle_trajectories(frame, self.vehicle_path, self.colour_label)
                    frame = draw_collision_warning(frame, collision_risk_ids, outputs)
                    frame = draw_violation_warnings(frame, retrograde_ids, is_congested, outputs)
                    
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
                    
                    # 【新增：第七模块 车辆属性识别】获取车型和颜色统计
                    car_type_stats = get_car_type_stats(self.car_attr)
                    car_color_stats = get_car_color_stats(self.car_attr)
                    
                    vehicle_count = len(outputs)
                    overspeed_count = len(overspeed_ids)
                    retrograde_count = len(retrograde_ids)
                    frame = draw_ui_info_bar(
                        frame, 
                        vehicle_count, 
                        self.collision_warning_count, 
                        overspeed_count, 
                        retrograde_count, 
                        is_congested,
                        car_type_stats,
                        car_color_stats
                    )
                    
                    frame = draw_traffic_counting_ui(
                        frame,
                        self.traffic_in_count,
                        self.traffic_out_count,
                        self.traffic_total_count,
                        COUNT_LINE_Y
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
            
            spawn_points = world.get_map().get_spawn_points()
            if not spawn_points:
                print("[ERROR] 无法获取生成点")
                return
            
            print(f"[INFO] 找到 {len(spawn_points)} 个生成点")
            
            vehicle_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
            vehicle_bp.set_attribute('role_name', 'ego')
            ego_vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            
            if not ego_vehicle:
                print("[ERROR] 无法生成主车辆")
                return
            
            print("[INFO] ✓ 主车辆生成成功")
            
            camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(img_w))
            camera_bp.set_attribute('image_size_y', str(img_h))
            camera_bp.set_attribute('fov', '110')
            
            camera_location = carla.Location(2, 0, 1)
            camera_rotation = carla.Rotation(0, 180, 0)
            camera_init_trans = carla.Transform(camera_location, camera_rotation)
            
            camera = world.spawn_actor(
                camera_bp, 
                camera_init_trans, 
                attach_to=ego_vehicle,
                attachment_type=carla.AttachmentType.Rigid
            )
            
            print("[INFO] ✓ 相机生成成功")
            
            npc_count = 0
            for i in range(20):
                vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle'))
                npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
                if npc:
                    npc.set_autopilot(True)
                    npc_count += 1
            
            print(f"[INFO] ✓ 生成了 {npc_count} 个 NPC 车辆")
            
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
                   '【新增功能】违章行为检测\n'
                   '【新增功能】UI界面增强\n'
                   '【新增功能】车辆轨迹绘制\n'
                   '【新增功能】第六模块 车流量统计\n'
                   '【新增功能】第七模块 车辆属性识别',
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
    print("【新增：UI界面增强模块】")
    print("【新增：车辆轨迹绘制模块】")
    print("【新增：第六模块 车流量统计】")
    print("【新增：第七模块 车辆属性识别】")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"平台: {sys.platform}")
    print(f"工作目录: {os.getcwd()}")
    print("=" * 60)
    
    print("\n【新增：轨迹预测+提前碰撞预警】当前配置:")
    print(f"  - 轨迹长度: {MAX_TRAJECTORY_LENGTH} 帧")
    print(f"  - 预测帧数: {PREDICT_FRAMES} 帧")
    print(f"  - 碰撞阈值: {COLLISION_DISTANCE_RATIO*100:.0f}% 画面宽度 ({COLLISION_DISTANCE_THRESHOLD} 像素)")
    
    print("\n【新增：车速估算 + 超速报警】当前配置:")
    print(f"  - 帧率: {FPS} fps")
    print(f"  - 像素转米: {PIXEL_TO_METER} m/px")
    print(f"  - 限速: {SPEED_LIMIT} km/h")
    
    print("\n【新增：违章行为检测】当前配置:")
    print(f"  - 拥堵阈值: {CONGESTION_THRESHOLD} 辆")
    
    print("\n【新增：车辆轨迹绘制模块】当前配置:")
    print(f"  - 轨迹保留帧数: {MAX_TRAJECTORY_POINTS} 帧")
    print(f"  - 轨迹线宽度: {TRAJECTORY_LINE_WIDTH} 像素")
    
    print("\n【新增：第六模块 车流量统计】当前配置:")
    print(f"  - 统计线Y坐标: {COUNT_LINE_Y} 像素")
    
    print("\n【新增：第七模块 车辆属性识别】当前配置:")
    print(f"  - 车型列表: {', '.join(car_type)}")
    print(f"  - 颜色列表: {', '.join(car_color)}")
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
