#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Line Detection (精简版)
核心功能：Canny边缘检测 + 霍夫变换实现车道线检测，单视频处理+结果保存
适用环境：Ubuntu (Python 3.10 + OpenCV + numpy)
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path

# ===================== 核心参数（仅保留必要的） =====================
# 车道线检测核心参数
CANNY_LOW_THRESH = 50       # Canny边缘检测低阈值
CANNY_HIGH_THRESH = 150     # Canny边缘检测高阈值
HOUGH_RHO = 1               # 霍夫变换rho步长
HOUGH_THETA = np.pi / 180   # 霍夫变换theta步长
HOUGH_THRESHOLD = 20        # 霍夫变换阈值
HOUGH_MIN_LINE_LEN = 40     # 最小线段长度
HOUGH_MAX_LINE_GAP = 20     # 最大线段间隙

# ===================== 核心函数：车道线检测 =====================
def detect_lane_lines(frame):
    """核心：检测并绘制车道线（Canny + 霍夫变换）"""
    # 1. 灰度化 + 高斯模糊
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Canny边缘检测
    edges = cv2.Canny(blur, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)
    
    # 3. 区域掩码（仅检测图像下半部分，聚焦车道）
    h, w = frame.shape[:2]
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, h), (w//2, h//2), (w, h)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 4. 霍夫变换检测直线
    lines = cv2.HoughLinesP(
        masked_edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN, maxLineGap=HOUGH_MAX_LINE_GAP
    )
    
    # 5. 绘制车道线
    frame_with_lane = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame_with_lane, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    return frame_with_lane

# ===================== 核心函数：视频处理 =====================
def process_video(video_path, max_frames=10):
    """处理单视频：读取帧→检测车道线→保存结果"""
    # 校验视频文件
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 → {video_path}")
        return
    
    # 打开视频流
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 → {video_path}")
        return
    
    # 获取视频基础信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    
    # 初始化结果视频写入器
    video_name = Path(video_path).stem
    result_path = f"{video_name}_lane_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
    
    # 逐帧处理
    count = 0
    print(f"开始处理视频（最大{max_frames}帧）...")
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 核心：检测车道线
        frame_with_lane = detect_lane_lines(frame)
        
        # 写入结果视频
        writer.write(frame_with_lane)
        
        # 实时显示（可选，按Q退出）
        cv2.imshow("Lane Detection", frame_with_lane)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        count += 1
    
    # 释放资源
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"处理完成！结果保存至 → {result_path}")

# ===================== 主函数（精简参数） =====================
if __name__ == "__main__":
    # 简化命令行参数：仅保留输入视频、最大帧数
    parser = argparse.ArgumentParser(description="车道线检测（精简版）")
    parser.add_argument("video_path", type=str, help="输入视频文件路径")
    parser.add_argument("--max-frames", type=int, default=10, help="最大处理帧数（默认10）")
    args = parser.parse_args()
    
    # 执行核心逻辑
    process_video(args.video_path, args.max_frames)
