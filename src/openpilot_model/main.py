#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Line Detection (完整功能版)
核心：全流程车道线检测（无步骤省略）+ 功能齐全 + 高级可视化
步骤包含：颜色阈值过滤→透视变换→边缘检测→车道线拟合→曲率计算→偏移计算→结果可视化
依赖：opencv-python, numpy, matplotlib
"""

import cv2
import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 检测参数（全流程关键参数，不省略）
# 颜色阈值（HLS空间，筛选白色/黄色车道线）
WHITE_LOW = np.array([0, 200, 0], dtype=np.uint8)
WHITE_HIGH = np.array([255, 255, 255], dtype=np.uint8)
YELLOW_LOW = np.array([10, 0, 100], dtype=np.uint8)
YELLOW_HIGH = np.array([40, 255, 255], dtype=np.uint8)
# Canny边缘检测
CANNY_LOW_THRESH = 50
CANNY_HIGH_THRESH = 150
# 滑动窗口拟合参数
WINDOW_NUM = 9
WINDOW_MARGIN = 100
MIN_PIXELS = 50
# 透视变换参数（鸟瞰视角）
SRC_POINTS = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])
DST_POINTS = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
# 物理尺寸转换（像素→米）
YM_PER_PIX = 30 / 720   # 纵向：30米/720像素
XM_PER_PIX = 3.7 / 700  # 横向：3.7米/700像素

# 2. 可视化参数（高级且信息完整）
COLOR_LEFT_LANE = (0, 0, 255)    # 左车道：红
COLOR_RIGHT_LANE = (255, 0, 0)   # 右车道：蓝
COLOR_LANE_AREA = (0, 255, 0)    # 车道区域：绿（半透明）
COLOR_TEXT = (255, 255, 255)     # 文本：白
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
PANEL_ALPHA = 0.7                # 信息面板透明度
FRAME_DELAY = 300                # 每帧显示时间（ms）


def preprocess_frame(frame):
    """完整预处理流程：颜色阈值过滤→灰度→高斯模糊→边缘检测"""
    # 1. 转换为HLS颜色空间，筛选白色/黄色车道线
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hls, WHITE_LOW, WHITE_HIGH)
    yellow_mask = cv2.inRange(hls, YELLOW_LOW, YELLOW_HIGH)
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # 2. 灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 3. 高斯模糊去噪
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. Canny边缘检测
    edges = cv2.Canny(blur, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)
    
    # 5. 融合颜色掩码和边缘检测结果（提升鲁棒性）
    combined = cv2.bitwise_and(edges, color_mask)
    
    return combined


def warp_frame(frame):
    """透视变换：将图像转换为鸟瞰视角，便于检测车道线"""
    h, w = frame.shape[:2]
    M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
    Minv = cv2.getPerspectiveTransform(DST_POINTS, SRC_POINTS)
    warped = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, Minv

def fit_lane_lines(binary_warped):
    """滑动窗口法拟合车道线，计算车道线参数+曲率+偏移"""
    h, w = binary_warped.shape
    
    # 1. 计算底部直方图，找到车道线起始点
    histogram = np.sum(binary_warped[h//2:, :], axis=0)
    midpoint = w // 2
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # 2. 滑动窗口初始化
    window_height = h // WINDOW_NUM
    left_x_current = left_x_base
    right_x_current = right_x_base
    
    # 3. 收集车道线像素点
    left_lane_inds = []
    right_lane_inds = []
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # 4. 滑动窗口遍历
    for window in range(WINDOW_NUM):
        # 计算窗口边界
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_xleft_low = left_x_current - WINDOW_MARGIN
        win_xleft_high = left_x_current + WINDOW_MARGIN
        win_xright_low = right_x_current - WINDOW_MARGIN
        win_xright_high = right_x_current + WINDOW_MARGIN
        
        # 收集窗口内的像素点
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # 更新窗口中心
        if len(good_left_inds) > MIN_PIXELS:
            left_x_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > MIN_PIXELS:
            right_x_current = int(np.mean(nonzerox[good_right_inds]))
    
    # 合并像素点
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # 提取车道线像素坐标
    left_x = nonzerox[left_lane_inds]
    left_y = nonzeroy[left_lane_inds]
    right_x = nonzerox[right_lane_inds]
    right_y = nonzeroy[right_lane_inds]
    
    # 5. 多项式拟合（2次多项式）
    left_fit = np.polyfit(left_y, left_x, 2) if len(left_x) > 0 else None
    right_fit = np.polyfit(right_y, right_x, 2) if len(right_x) > 0 else None
    
    # 6. 生成拟合线的x/y坐标
    ploty = np.linspace(0, h - 1, h)
    left_fitx = None
    right_fitx = None
    if left_fit is not None:
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    if right_fit is not None:
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    
    # 7. 计算曲率（物理尺寸，关键指标不省略）
    left_curverad, right_curverad = calculate_curvature(left_x, left_y, right_x, right_y)
    
    # 8. 计算车道偏移（车辆相对于车道中心的偏移）
    center_offset = calculate_offset(left_fit, right_fit, w, h)
    
    return left_fitx, right_fitx, ploty, left_curverad, right_curverad, center_offset

def calculate_curvature(left_x, left_y, right_x, right_y):
    """计算车道线曲率（单位：米）"""
    h = 720  # 参考高度
    # 转换为物理尺寸
    left_fit_cr = np.polyfit(left_y * YM_PER_PIX, left_x * XM_PER_PIX, 2) if len(left_x) > 0 else None
    right_fit_cr = np.polyfit(right_y * YM_PER_PIX, right_x * XM_PER_PIX, 2) if len(right_x) > 0 else None
    
    # 计算曲率
    left_curverad = ((1 + (2 * left_fit_cr[0] * h * YM_PER_PIX + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0]) if left_fit_cr is not None else 0
    right_curverad = ((1 + (2 * right_fit_cr[0] * h * YM_PER_PIX + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0]) if right_fit_cr is not None else 0
    
    return left_curverad, right_curverad

def calculate_offset(left_fit, right_fit, frame_width, frame_height):
    """计算车辆相对于车道中心的偏移（单位：米）"""
    if left_fit is None or right_fit is None:
        return 0.0
    
    # 计算底部车道线的x坐标
    y_eval = frame_height - 1
    left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
    right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
    
    # 车道中心x坐标
    lane_center_x = (left_x + right_x) / 2
    # 车辆中心x坐标（假设在图像中心）
    car_center_x = frame_width / 2
    
    # 转换为物理偏移
    offset = (car_center_x - lane_center_x) * XM_PER_PIX
    return offset



def draw_lane(frame, warped, Minv, left_fitx, right_fitx, ploty):
    """绘制拟合后的车道线+车道区域"""
    h, w = frame.shape[:2]
    # 1. 创建车道区域蒙版
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # 2. 构建车道区域多边形
    if left_fitx is not None and right_fitx is not None:
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # 绘制车道区域（半透明）
        cv2.fillPoly(color_warp, np.int_([pts]), COLOR_LANE_AREA)
        # 绘制车道线
        cv2.polylines(color_warp, np.int_([pts_left]), isClosed=False, color=COLOR_LEFT_LANE, thickness=10)
        cv2.polylines(color_warp, np.int_([pts_right]), isClosed=False, color=COLOR_RIGHT_LANE, thickness=10)
    
    # 3. 逆透视变换，将车道线投影回原图
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # 4. 融合原图和车道线
    result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
    
    return result


def draw_info_panel(frame, frame_idx, total_frames, fps, left_curv, right_curv, offset):
    """绘制完整信息面板：包含帧数、FPS、曲率、偏移等关键指标"""
    h, w = frame.shape[:2]
    # 黑色半透明背景
    panel = np.zeros_like(frame)
    cv2.rectangle(panel, (10, 10), (400, 250), (0, 0, 0), -1)
    frame = cv2.addWeighted(panel, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0)
    
    # 完整信息文本
    info_texts = [
        f"Frame: {frame_idx}/{total_frames}",
        f"FPS: {fps:.1f}",
        f"Left Lane Curvature: {left_curv:.1f} m",
        f"Right Lane Curvature: {right_curv:.1f} m",
        f"Center Offset: {offset:.2f} m",
        "=== Key Controls ===",
        "Space: Single Step (Next Frame)",
        "P: Pause/Resume",
        "Q: Quit"
    ]
    
    y_offset = 40
    for text in info_texts:
        cv2.putText(
            frame, text, (20, y_offset), FONT,
            FONT_SCALE, COLOR_TEXT, FONT_THICKNESS
        )
        y_offset += 30
    
    return frame


def create_split_screen(original, detected, warped):
    """三分屏：原图 + 检测结果 + 鸟瞰视角（功能齐全，不省略）"""
    h, w = original.shape[:2]
    # 统一尺寸
    detected = cv2.resize(detected, (w, h), interpolation=cv2.INTER_LINEAR)
    warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)  # 转换为彩色
    warped = cv2.resize(warped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 拼接分屏（左：原图，中：检测结果，右：鸟瞰视角）
    top_row = np.hstack((original, detected))
    bottom_row = np.hstack((warped, np.zeros_like(original)))
    split_frame = np.vstack((top_row, bottom_row))[:h*2, :w*2]  # 裁剪为2x2布局
    
    # 添加标题
    cv2.putText(split_frame, "Original", (20, 30), FONT, 1, COLOR_TEXT, 3)
    cv2.putText(split_frame, "Lane Detection", (w + 20, 30), FONT, 1, COLOR_TEXT, 3)
    cv2.putText(split_frame, "Bird's Eye View", (20, h + 30), FONT, 1, COLOR_TEXT, 3)
    
    return split_frame


def process_video(video_path, max_frames=10):
    """完整视频处理流程：预处理→透视变换→拟合→绘制→可视化"""
    # 1. 校验文件
    if not os.path.exists(video_path):
        print(f"Error: Video file not found → {video_path}")
        return
    
    # 2. 打开视频流
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video → {video_path}")
        return
    
    # 3. 获取视频基础信息
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
    
    # 4. 初始化结果保存（支持分屏尺寸）
    video_name = Path(video_path).stem
    result_path = f"{video_name}_lane_detected_full.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(result_path, fourcc, fps, (w*2, h*2))
    
    # 5. 控制变量（功能齐全，交互完整）
    paused = False
    step_mode = False
    frame_idx = 0
    print("=== Lane Detection (Full Version) ===")
    print("Key Controls:")
    print("  Space: Single Step (Next Frame)")
    print("  P: Pause/Resume")
    print("  Q: Quit")
    
    # 6. 创建可调整窗口（高级可视化）
    cv2.namedWindow("Lane Detection (Full Screen)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Detection (Full Screen)", w*2, h*2)
    
    # 7. 全流程处理（步骤无省略）
    while cap.isOpened() and frame_idx < total_frames:
        # 单步模式（方便截图）
        if step_mode:
            key = cv2.waitKey(0) & 0xFF
            if key == ord(' '):
                step_mode = False
            elif key == ord('q'):
                break
            continue
        
        # 读取帧
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 步骤1：预处理
            preprocessed = preprocess_frame(frame)
            
            # 步骤2：透视变换
            warped, Minv = warp_frame(preprocessed)
            
            # 步骤3：车道线拟合
            left_fitx, right_fitx, ploty, left_curv, right_curv, offset = fit_lane_lines(warped)
            
            # 步骤4：绘制车道线
            detected_frame = draw_lane(frame, warped, Minv, left_fitx, right_fitx, ploty)
            
            # 步骤5：绘制信息面板
            start_time = time.time()
            fps_real = 1 / (time.time() - start_time)
            detected_frame = draw_info_panel(
                detected_frame, frame_idx+1, total_frames,
                fps_real, left_curv, right_curv, offset
            )
            
            # 步骤6：分屏可视化
            split_frame = create_split_screen(frame, detected_frame, warped)
            
            # 保存结果
            writer.write(split_frame)
            frame_idx += 1
        
        # 显示窗口
        cv2.imshow("Lane Detection (Full Screen)", split_frame)
        
        # 交互控制（功能齐全）
        key = cv2.waitKey(FRAME_DELAY) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord(' '):
            step_mode = True
    
    # 8. 释放资源（完整收尾，不省略）
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n=== Process Finished ===")
    print(f"Result Video: {result_path}")
    print(f"Key Metrics: Curvature (m), Offset (m)")


def batch_process(input_dir, max_frames=10):
    """批量处理视频，功能扩展不省略"""
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found → {input_dir}")
        return
    
    video_extensions = (".mp4", ".avi", ".mov")
    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"No video files found in {input_dir} (supported: {video_extensions})")
        return
    
    print(f"Batch processing: {len(video_files)} videos")
    for video_file in video_files:
        print(f"\nProcessing: {video_file}")
        process_video(video_file, max_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Detection (Full Version - Complete Steps)")
    parser.add_argument("input", type=str, help="Video file path or directory (batch mode)")
    parser.add_argument("--max-frames", type=int, default=10, help="Max frames to process (default: 10)")
    parser.add_argument("--batch", action="store_true", help="Batch process directory")
    args = parser.parse_args()
    
    # 检查依赖（完整，不省略）
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Error: Install dependencies → pip install opencv-python numpy")
        exit(1)
    
    # 执行处理（功能齐全，支持单/批量）
    if args.batch or os.path.isdir(args.input):
        batch_process(args.input, args.max_frames)
    else:
        process_video(args.input, args.max_frames)
