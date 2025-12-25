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
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = time.time()

        # 2. 优化后的肤色检测阈值（适配更多肤色/光线）
        self.skin_lower = np.array([0, 20, 70], np.uint8)  # 放宽下界
        self.skin_upper = np.array([20, 255, 255], np.uint8)  # 调整上界
        self.kernel = np.ones((5, 5), np.uint8)  # 更大的核去噪

        # 3. 优化后的手指检测参数（降低阈值，提高识别率）
        self.defect_depth_threshold = 10  # 降低深度阈值
        self.min_defect_distance = 5  # 降低距离阈值
        self.min_contour_area = 500  # 降低最小轮廓面积

        # 4. 手势缓存&帧缓存
        self.gesture_buffer = []
        self.stable_gesture = "None"
        self.frame_queue = []
        self.queue_lock = threading.Lock()

        # 5. 识别区域参数（仅显示边框）
        self.recognition_area = None
        self.area_color = (0, 255, 0)  # 边框颜色（绿色）

    def _init_recognition_area(self, frame_shape):
        """初始化识别区域（调大尺寸，右侧更大范围）"""
        h, w = frame_shape[:2]
        x1 = int(w * 1.5 / 3)  # 左边界左移（从2/3改为1.5/3），扩大宽度
        y1 = int(h * 0.05)  # 上边界上移（从0.1改为0.05），扩大高度
        x2 = w - 10  # 右边界右移（从-20改为-10），减少右侧边距
        y2 = int(h * 0.95)  # 下边界下移（从0.9改为0.95），减少底部边距
        self.recognition_area = (x1, y1, x2, y2)

    def _draw_recognition_area(self, frame):
        """绘制识别区域（仅显示边框，无背景色）"""
        if self.recognition_area is None:
            self._init_recognition_area(frame.shape)
        x1, y1, x2, y2 = self.recognition_area

        # 仅绘制边框（移除半透明背景）
        cv.rectangle(frame, (x1, y1), (x2, y2), self.area_color, 2)

        # 添加区域提示文字（在边框上方）
        cv.putText(frame, "Recognition Area", (x1 + 10, y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, self.area_color, 2)
        return frame

    def _get_roi(self, frame):
        """获取识别区域的ROI（确保坐标有效）"""
        if self.recognition_area is None:
            self._init_recognition_area(frame.shape)
        x1, y1, x2, y2 = self.recognition_area

        # 边界保护
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        return frame[y1:y2, x1:x2], (x1, y1)

    def count_fingers(self, cnt):
        """优化后的手指计数逻辑（更鲁棒）"""
        try:
            # 计算凸包（带坐标）和凸包缺陷
            hull = cv.convexHull(cnt)
            hull_indices = cv.convexHull(cnt, returnPoints=False)
            defects = cv.convexityDefects(cnt, hull_indices)

            if defects is None or len(defects) == 0:
                return 0

            finger_count = 0
            # 遍历缺陷点
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                # 计算缺陷深度（实际像素值）
                depth = d / 256.0

                # 计算角度（过滤误判的缺陷）
                a = np.linalg.norm(np.array(end) - np.array(start))
                b = np.linalg.norm(np.array(far) - np.array(start))
                c = np.linalg.norm(np.array(end) - np.array(far))
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / np.pi

                # 有效缺陷：深度足够 + 角度小于90度（手指间的凹陷）
                if depth > self.defect_depth_threshold and angle < 90:
                    finger_count += 1

            # 缺陷数+1=手指数量（最多5根）
            return min(finger_count + 1, 5)
        except Exception as e:
            print(f"手指计数错误: {e}")
            return 0

    def capture_frames(self, cap):
        """帧采集线程（稳定）"""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            with self.queue_lock:
                self.frame_queue = [frame]  # 只保留最新帧
            time.sleep(self.frame_interval * 0.5)

    def process_frame(self, frame):
        """优化后的帧处理逻辑"""
        # 镜像翻转
        frame = cv.flip(frame, 1)
        # 绘制识别区域（仅边框）
        frame = self._draw_recognition_area(frame)
        # 获取ROI
        roi, (roi_x, roi_y) = self._get_roi(frame)
        current_gesture = "None"

        if roi.size > 0:  # 确保ROI有效
            # 预处理：缩小+转HSV+肤色掩码
            roi_small = cv.resize(roi, (320, 240))  # 适度放大ROI，提高检测精度
            hsv = cv.cvtColor(roi_small, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, self.skin_lower, self.skin_upper)

            # 形态学操作（去噪+填充）
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel)
            mask = cv.dilate(mask, self.kernel, iterations=2)

            # 查找轮廓
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours:
                # 取最大轮廓（手部）
                cnt = max(contours, key=cv.contourArea)
                area = cv.contourArea(cnt)

                if area > self.min_contour_area:
                    # 计算密实度（判断是否握拳）
                    hull = cv.convexHull(cnt)
                    hull_area = cv.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0

                    # 手指计数
                    finger_count = self.count_fingers(cnt)

                    # 可视化调试（可选：在ROI内绘制轮廓）
                    cnt_scaled = cnt * (roi.shape[1] / roi_small.shape[1], roi.shape[0] / roi_small.shape[0])
                    cnt_scaled = cnt_scaled.astype(np.int32)
                    cnt_scaled[:, :, 0] += roi_x
                    cnt_scaled[:, :, 1] += roi_y
                    cv.drawContours(frame, [cnt_scaled], -1, (255, 0, 0), 2)

                    # 手势判断逻辑（优化版）
                    if solidity > 0.8:  # 握拳（密实度高）
                        current_gesture = "stop"
                    elif finger_count == 2:  # 食指+中指
                        current_gesture = "front"
                    elif finger_count >= 4:  # 手掌张开（4-5指）
                        current_gesture = "back"
                    # 其他情况（1/3指）归为None

        # 手势缓存稳定
        self.gesture_buffer.append(current_gesture)
        if len(self.gesture_buffer) > 2:
            self.gesture_buffer.pop(0)
        if len(set(self.gesture_buffer)) == 1:
            self.stable_gesture = self.gesture_buffer[0]

        # 绘制UI
        cv.putText(frame, f"Gesture: {self.stable_gesture}", (10, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv.putText(frame, f"FPS: {self.target_fps}", (10, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # 拉伸显示
        frame_show = cv.resize(frame, (640, 480))
        return frame_show

    def run(self):
        """主运行逻辑"""
        # 摄像头初始化（优化参数）
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # 提高摄像头分辨率
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv.CAP_PROP_FPS, self.target_fps)

        # 启动采集线程
        capture_thread = threading.Thread(target=self.capture_frames, args=(cap,), daemon=True)
        capture_thread.start()

        # 提示信息
        print("=" * 50)
        print(f"✅ 帧率锁定 {self.target_fps} 帧 | ESC退出")
        print("💡 调试提示：")
        print("   1. 把手放在右侧绿色边框的识别区域内（已扩大范围）")
        print("   2. 握拳 → stop | 食指+中指 → front | 手掌张开 → back")
        print("   3. 蓝色轮廓表示检测到的手部区域")
        print("=" * 50)

        # 主循环
        while cap.isOpened():
            # 帧率控制
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)

            # 读取帧
            with self.queue_lock:
                if not self.frame_queue:
                    continue
                frame = self.frame_queue.pop(0)

            # 处理并显示
            frame_show = self.process_frame(frame)
            cv.imshow("Hand Gesture Recognition", frame_show)

            # 更新时间戳
            self.last_frame_time = time.time()

            # ESC退出
            if cv.waitKey(1) & 0xFF == 27:
                break

        # 释放资源
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    # 可降低帧率（如15）提高稳定性
    recognizer = StableFPSHandRecognizer(target_fps=20)
    recognizer.run()