#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
import time

def lane_detection(frame):
    """基础车道线检测（适配道路视频）"""
    # 预处理：灰度→模糊→边缘检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # 掩码聚焦车道区域（只检测画面下半部分）
    h, w = frame.shape[:2]
    roi_vertices = np.array([[(0, h), (w//2, h//2), (w, h)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 霍夫变换检测车道线
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        minLineLength=40,
        maxLineGap=20
    )
    
    # 绘制车道线（红=左，蓝=右）
    detected = frame.copy()
    if lines is not None:
        mid_x = w / 2
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            # 左车道线（负斜率）、右车道线（正斜率）
            if slope < -0.3 and x1 < mid_x:
                cv2.line(detected, (x1, y1), (x2, y2), (0, 0, 255), 4)
            elif slope > 0.3 and x1 > mid_x:
                cv2.line(detected, (x1, y1), (x2, y2), (255, 0, 0), 4)
    return detected

def main(video_path):
    # 打开视频（强制指定解码器，兼容更多格式）
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return
    
    # 获取视频基础信息
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 强制创建分屏窗口（固定尺寸，避免压缩）
    cv2.namedWindow("Lane Detection - Split Screen", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Detection - Split Screen", w*2, h)
    
    # 视频写入器（保存分屏结果）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("/home/dacun/nn/road_detection.mp4", fourcc, fps, (w*2, h))
    
    print("=== 分屏车道线检测（强制窗口显示）===")
    print(f"视频尺寸：{w}x{h} | 总帧数：{total_frames}")
    print("操作说明：")
    print("  Q键 → 退出")
    print("  P键 → 暂停/继续")
    print("  空格 → 单步播放")
    
    paused = False
    step_mode = False
    frame_idx = 0
    
    # 核心播放循环（强制逐帧显示）
    while True:
        # 单步/暂停逻辑
        if not paused or step_mode:
            ret, frame = cap.read()
            if not ret:
                print(f"\n播放完成！已处理 {frame_idx} 帧")
                break
            
            # 检测车道线 + 生成分屏
            detected_frame = lane_detection(frame)
            split_frame = np.hstack((frame, detected_frame))  # 左右拼接
            
            # 绘制分屏标题
            cv2.putText(split_frame, "原始画面", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(split_frame, "车道线检测", (w+20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            # 保存分屏帧
            out.write(split_frame)
            frame_idx += 1
            step_mode = False
        
        # 强制显示窗口（核心：即使无新帧，也刷新窗口）
        cv2.imshow("Lane Detection - Split Screen", split_frame)
        
        # 按键控制（降低等待时间，确保窗口响应）
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\n手动退出！已处理 {frame_idx} 帧")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'暂停' if paused else '继续'}播放")
        elif key == ord(' '):
            step_mode = True
            paused = True  # 单步时自动暂停
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"分屏检测视频已保存：/home/dacun/nn/road_detection.mp4")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"用法：{sys.argv[0]} <视频路径>")
        sys.exit(1)
    main(sys.argv[1])
