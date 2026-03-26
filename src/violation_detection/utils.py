#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数模块
"""
import cv2
import numpy as np

def calculate_ear(eye_points):
    """
    计算眼睛纵横比 (Eye Aspect Ratio)
    用于检测眨眼和闭眼
    """
    # 计算垂直距离
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    
    # 计算水平距离
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    
    # EAR公式
    ear = (A + B) / (2.0 * C)
    return ear

def detect_hands(image):
    """
    简单的肤色检测，用于手部位置识别
    """
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 肤色范围
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # 肤色掩码
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # 形态学操作去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # 提取手部中心点
    hand_positions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # 过滤小区域
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                hand_positions.append((cx, cy))
    
    return hand_positions

def draw_text_with_background(img, text, position, font_scale=0.7, 
                              color=(0, 0, 255), thickness=2):
    """
    绘制带背景的文字
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness)
    
    x, y = position
    # 绘制背景矩形
    cv2.rectangle(img, (x, y - text_height - 5), 
                  (x + text_width, y + 5), (0, 0, 0), -1)
    # 绘制文字
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
    
    return img