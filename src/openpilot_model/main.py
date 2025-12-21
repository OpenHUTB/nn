# 1. 进入项目目录（必做！确保路径正确）
cd /home/dacun/nn

# 2. 备份旧文件（防止误操作）
cp main.py main_old.py

# 3. 强制清空并写入最终修复版代码（复制下面整段命令执行）
cat > main.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Line Detection (最终增强版)
核心：分屏显示+速度可控+车道分类+高级可视化
依赖：opencv-python, numpy
"""

import cv2
import numpy as np
import argparse
import os
import time
from pathlib import Path

# ===================== 核心参数 =====================
# 检测参数
CANNY_LOW_THRESH = 50
CANNY_HIGH_THRESH = 150
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 20
HOUGH_MIN_LINE_LEN = 40
HOUGH_MAX_LINE_GAP = 20

# 可视化参数
COLOR_LEFT_LANE = (0, 0, 255)    # 左车道：红
COLOR_RIGHT_LANE = (255, 0, 0)   # 右车道：蓝
COLOR_CENTER_LANE = (0, 255, 0)  # 中心线：绿
COLOR_MASK = (0, 255, 0)         # 车道蒙版：绿（半透明）
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
PANEL_ALPHA = 0.7                # 信息面板透明度
FRAME_DELAY = 500                # 每帧显示时间（ms），越大越慢

# ===================== 核心功能：车道线分类与检测 =====================
def classify_lane_lines(lines, frame_width):
    """分类左/右车道线（基于斜率+位置）"""
    left_lines = []   # 左车道：斜率负+左侧
    right_lines = []  # 右车道：斜率正+右侧
    
    if lines is None:
        return left_lines, right_lines
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:  # 避免除零
            continue
        slope = (y2 - y1) / (x2 - x1)
        mid_x = frame_width / 2
        
        # 分类逻辑（斜率阈值0.3，避免误检）
        if slope < -0.3 and (x1 < mid_x or x2 < mid_x):
            left_lines.append(line[0])
        elif slope > 0.3 and (x1 > mid_x or x2 > mid_x):
            right_lines.append(line[0])
    
    return left_lines, right_lines

def detect_and_draw_lane(frame):
    """核心：检测+分类+绘制+蒙版"""
    h, w = frame.shape[:2]
    frame_copy = frame.copy()
    
    # 1. 预处理：灰度→模糊→边缘检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)
    
    # 2. 区域掩码（聚焦车道区域）
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[(0, h), (w//2, h//2), (w, h)]], np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 3. 霍夫变换检测直线
    lines = cv2.HoughLinesP(
        masked_edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN, maxLineGap=HOUGH_MAX_LINE_GAP
    )
    
    # 4. 分类+绘制车道线
    left_lines, right_lines = classify_lane_lines(lines, w)
    for line in left_lines:
        cv2.line(frame_copy, (line[0], line[1]), (line[2], line[3]), COLOR_LEFT_LANE, 4)
    for line in right_lines:
        cv2.line(frame_copy, (line[0], line[1]), (line[2], line[3]), COLOR_RIGHT_LANE, 4)
    
    # 5. 绘制车道蒙版（半透明填充）
    if left_lines and right_lines:
        # 提取车道线关键点，构建蒙版区域
        left_pts = [(l[0], l[1]) for l in left_lines] + [(l[2], l[3]) for l in left_lines]
        right_pts = [(l[0], l[1]) for l in right_lines] + [(l[2], l[3]) for l in right_lines]
        
        left_bottom = max(left_pts, key=lambda p: p[1])
        right_bottom = max(right_pts, key=lambda p: p[1])
        left_top = min(left_pts, key=lambda p: p[1])
        right_top = min(right_pts, key=lambda p: p[1])
        
        mask_pts = np.array([left_bottom, left_top, right_top, right_bottom], np.int32)
        mask_layer = frame_copy.copy()
        cv2.fillPoly(mask_layer, [mask_pts], COLOR_MASK)
        cv2.addWeighted(mask_layer, 0.2, frame_copy, 0.8, 0, frame_copy)
    
    return frame_copy, len(left_lines), len(right_lines)

# ===================== 高级可视化：分屏+信息面板 =====================
def draw_info_panel(frame, frame_idx, total_frames, fps, left_count, right_count):
    """绘制半透明信息面板（不遮挡画面）"""
    h, w = frame.shape[:2]
    # 黑色半透明背景
    panel = np.zeros_like(frame)
    cv2.rectangle(panel, (10, 10), (350, 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(panel, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0)
    
    # 绘制信息文本
    info_texts = [
        f"Frame: {frame_idx}/{total_frames}",
        f"FPS: {fps:.1f}",
        f"Left Lane: {left_count} lines",
        f"Right Lane: {right_count} lines",
        "=== Key Controls ===",
        "Space: Next Frame (Step)",
        "P: Pause/Resume",
        "Q: Quit"
    ]
    
    y_offset = 40
    for text in info_texts:
        cv2.putText(
            frame, text, (20, y_offset), FONT,
            FONT_SCALE, (255, 255, 255), FONT_THICKNESS
        )
        y_offset += 25
    
    return frame

def create_split_screen(original_frame, detected_frame):
    """修复分屏：强制统一尺寸，确保左右拼接正常"""
    # 统一尺寸（避免尺寸不一致导致分屏失败）
    h, w = original_frame.shape[:2]
    detected_frame = cv2.resize(detected_frame, (w, h), interpolation=cv2.INTER_LINEAR)
    # 拼接分屏（左：原图，右：检测结果）
    split_frame = np.hstack((original_frame, detected_frame))
    # 添加分屏标题
    cv2.putText(split_frame, "Original", (20, 30), FONT, 1, (255, 255, 255), 3)
    cv2.putText(split_frame, "Lane Detection", (w + 20, 30), FONT, 1, (255, 255, 255), 3)
    return split_frame

# ===================== 主处理函数（速度可控+单步播放） =====================
def process_video(video_path, max_frames=10):
    """核心：分屏显示+速度可控+单步播放+截图友好"""
    # 校验文件
    if not os.path.exists(video_path):
        print(f"Error: Video file not found → {video_path}")
        return
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video → {video_path}")
        return
    
    # 视频基础信息
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
    
    # 结果保存（分屏尺寸：宽×2）
    video_name = Path(video_path).stem
    result_path = f"{video_name}_lane_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(result_path, fourcc, fps, (w*2, h))
    
    # 控制变量
    paused = False
    step_mode = False  # 单步模式：按空格走一帧
    frame_idx = 0
    print("=== Lane Detection (Enhanced Version) ===")
    print("Key Controls:")
    print("  Space: Single Step (next frame)")
    print("  P: Pause/Resume")
    print("  Q: Quit")
    
    # 创建可调整大小的窗口
    cv2.namedWindow("Lane Detection (Split Screen)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Detection (Split Screen)", w*2, h)
    
    while cap.isOpened() and frame_idx < total_frames:
        # 单步模式：仅按空格键才处理下一帧
        if step_mode:
            key = cv2.waitKey(0) & 0xFF  # 阻塞等待按键
            if key == ord(' '):
                step_mode = False  # 退出单步，处理下一帧
            elif key == ord('q'):
                break
            continue
        
        # 非单步模式：正常读取帧
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 核心检测
            start_time = time.time()
            detected_frame, left_count, right_count = detect_and_draw_lane(frame)
            fps_real = 1 / (time.time() - start_time)
            
            # 高级可视化
            detected_frame = draw_info_panel(
                detected_frame, frame_idx+1, total_frames,
                fps_real, left_count, right_count
            )
            split_frame = create_split_screen(frame, detected_frame)
            
            # 保存结果
            writer.write(split_frame)
            frame_idx += 1
        
        # 显示分屏窗口
        cv2.imshow("Lane Detection (Split Screen)", split_frame)
        
        # 按键控制（核心：单步/暂停/退出）
        key = cv2.waitKey(FRAME_DELAY) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused  # 切换暂停/继续
        elif key == ord(' '):
            step_mode = True     # 进入单步模式（按空格走一帧）
    
    # 释放资源
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n=== Process Finished ===")
    print(f"Result Video: {result_path}")

# ===================== 入口函数 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Detection (Enhanced + Split Screen)")
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("--max-frames", type=int, default=10, help="Max frames to process (default: 10)")
    args = parser.parse_args()
    
    # 检查依赖
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Error: Install dependencies first → pip install opencv-python numpy")
        exit(1)
    
    process_video(args.video_path, args.max_frames)
EOF
