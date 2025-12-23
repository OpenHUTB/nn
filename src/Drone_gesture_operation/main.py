#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import time
import threading


class StableFPSHandRecognizer:
    def __init__(self, target_fps=30):
        # 1. 帧率锁定参数
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps  # 每帧间隔时间（秒）
        self.last_frame_time = time.time()

        # 2. 极简手部检测参数
        self.skin_lower = np.array([0, 10, 10], np.uint8)
        self.skin_upper = np.array([30, 255, 180], np.uint8)
        self.kernel = np.ones((3, 3), np.uint8)

        # 3. 手势缓存（仅2帧，快速响应+稳定）
        self.gesture_buffer = []
        self.stable_gesture = "None"

        # 4. 帧缓存（避免堆积）
        self.frame_queue = []
        self.queue_lock = threading.Lock()

    def capture_frames(self, cap):
        """独立线程采集帧，避免主线程阻塞"""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            with self.queue_lock:
                # 只保留最新1帧，避免堆积
                self.frame_queue = [frame]
            # 采集线程限速，匹配目标帧率
            time.sleep(self.frame_interval * 0.5)

    def process_frame(self, frame):
        """轻量化处理，严格控制耗时"""
        # 1. 快速预处理
        frame = cv.flip(frame, 1)
        frame_small = cv.resize(frame, (160, 120))  # 超小尺寸
        hsv = cv.cvtColor(frame_small, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.skin_lower, self.skin_upper)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel)

        # 2. 快速找轮廓
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        current_gesture = "None"
        if contours:
            cnt = max(contours, key=cv.contourArea)
            if cv.contourArea(cnt) > 1000:
                # 3. 极简分类
                hull = cv.convexHull(cnt)
                solidity = cv.contourArea(cnt) / cv.contourArea(hull)
                current_gesture = "Fist" if solidity > 0.85 else "Point"

        # 4. 稳定手势（仅2帧一致）
        self.gesture_buffer.append(current_gesture)
        if len(self.gesture_buffer) > 2:
            self.gesture_buffer.pop(0)
        if len(set(self.gesture_buffer)) == 1:
            self.stable_gesture = self.gesture_buffer[0]

        # 5. 绘制极简UI（控制绘制耗时）
        cv.putText(frame, f"Gesture: {self.stable_gesture}", (10, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv.putText(frame, f"FPS: {self.target_fps}", (10, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # 拉伸显示（保持清晰）
        frame_show = cv.resize(frame, (640, 480))
        return frame_show

    def run(self):
        """主运行逻辑，帧率锁死"""
        # 1. 摄像头初始化（硬件级优化）
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))  # 快速编码
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # 关闭缓存
        cap.set(cv.CAP_PROP_FPS, self.target_fps)  # 强制摄像头输出目标帧率

        # 2. 启动独立采集线程
        capture_thread = threading.Thread(target=self.capture_frames, args=(cap,), daemon=True)
        capture_thread.start()

        print(f"✅ 帧率锁定 {self.target_fps} 帧 | ESC退出")
        print("💡 把手放在画面中间，握拳=Fist，伸食指=Point")

        # 3. 主线程处理+显示（严格控时）
        while cap.isOpened():
            # 计算当前帧应执行的时间，确保帧率稳定
            current_time = time.time()
            elapsed = current_time - self.last_frame_time

            # 如果耗时不足，等待到目标间隔
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)

            # 读取最新帧
            with self.queue_lock:
                if not self.frame_queue:
                    continue
                frame = self.frame_queue.pop(0)

            # 处理并显示
            frame_show = self.process_frame(frame)
            cv.imshow("Stable FPS Gesture", frame_show)

            # 更新时间戳，确保下一帧同步
            self.last_frame_time = time.time()

            # ESC退出
            if cv.waitKey(1) & 0xFF == 27:
                break

        # 释放资源
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    # 实例化并运行，锁定30帧（可改20/15帧，更低更稳）
    recognizer = StableFPSHandRecognizer(target_fps=30)
    recognizer.run()