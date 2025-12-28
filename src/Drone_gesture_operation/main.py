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

        # 2. 肤色检测（适配明亮+暗光环境）
        self.skin_lower_bright = np.array([0, 10, 50], np.uint8)
        self.skin_upper_bright = np.array([30, 255, 255], np.uint8)
        self.skin_lower_dark = np.array([0, 5, 15], np.uint8)
        self.skin_upper_dark = np.array([40, 180, 200], np.uint8)
        self.skin_lower = self.skin_lower_dark
        self.skin_upper = self.skin_upper_dark
        self.kernel = np.ones((5, 5), np.uint8)

        # 3. 核心参数（细化两者特征差异，解决重叠问题）
        # 握拳参数（收紧阈值，增加横向/方正特征约束）
        self.fist_solidity = 0.85  # 从0.82升至0.85，收紧密实度，拉大与大拇指差距
        self.fist_area_ratio = 0.75
        # 手指计数参数
        self.defect_depth_threshold = 4
        self.min_contour_area = 300
        # 大拇指识别参数（强化纵向特征，与握拳形成明显差异）
        self.thumb_aspect_ratio = 0.6
        self.thumb_solidity_range = (0.4, 0.82)  # 上限设为0.82，与握拳阈值0.85无重叠
        self.thumb_defect_max = 3

        # 4. 缓存参数
        self.gesture_buffer = []
        self.buffer_size = 3
        self.stable_gesture = "None"
        self.frame_queue = []
        self.queue_lock = threading.Lock()

        # 5. 识别区域
        self.recognition_area = None
        self.area_color = (0, 255, 0)

    def _init_recognition_area(self, frame_shape):
        """初始化识别区域"""
        h, w = frame_shape[:2]
        x1 = int(w * 1.5 / 3)
        y1 = int(h * 0.05)
        x2 = w - 10
        y2 = int(h * 0.95)
        self.recognition_area = (x1, y1, x2, y2)

    def _draw_recognition_area(self, frame):
        """绘制识别区域边框"""
        if self.recognition_area is None:
            self._init_recognition_area(frame.shape)
        x1, y1, x2, y2 = self.recognition_area
        cv.rectangle(frame, (x1, y1), (x2, y2), self.area_color, 2)
        cv.putText(frame, "Recognition Area", (x1 + 10, y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, self.area_color, 2)
        return frame

    def _get_roi(self, frame):
        """获取识别区域ROI"""
        if self.recognition_area is None:
            self._init_recognition_area(frame.shape)
        x1, y1, x2, y2 = self.recognition_area
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2], (x1, y1)

    def analyze_contour(self, cnt):
        """轮廓综合分析（返回多维度特征）"""
        try:
            # 基础特征
            area = cv.contourArea(cnt)
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0

            # 凸包特征
            hull = cv.convexHull(cnt)
            hull_area = cv.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            hull_width = hull[:, 0, 0].max() - hull[:, 0, 0].min() if hull.size > 0 else 0
            hull_height = hull[:, 0, 1].max() - hull[:, 0, 1].min() if hull.size > 0 else 0
            hull_aspect = hull_width / hull_height if hull_height > 0 else 0

            # 缺陷特征
            hull_indices = cv.convexHull(cnt, returnPoints=False)
            defects = cv.convexityDefects(cnt, hull_indices)
            defect_count = 0
            valid_defects = []

            if defects is not None and len(defects) > 0:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    depth = d / 256.0
                    # 计算缺陷角度
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    a = np.linalg.norm(np.array(end) - np.array(start))
                    b = np.linalg.norm(np.array(far) - np.array(start))
                    c = np.linalg.norm(np.array(end) - np.array(far))
                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / np.pi if (b * c) > 0 else 0

                    if depth > self.defect_depth_threshold and angle < 100:
                        valid_defects.append((depth, angle, far))
                        defect_count += 1

            # 质心特征
            M = cv.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + w / 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + h / 2

            return {
                "area": area,
                "aspect_ratio": aspect_ratio,
                "solidity": solidity,
                "hull_aspect": hull_aspect,
                "defect_count": defect_count,
                "valid_defects": valid_defects,
                "cx": cx, "cy": cy,
                "x": x, "y": y, "w": w, "h": h
            }
        except Exception as e:
            print(f"轮廓分析错误: {e}")
            return None

    def is_fist(self, features):
        """优化握拳（stop）判定：增加方正/横向轮廓排除，避免误判大拇指"""
        if not features:
            return False
        # 握拳核心特征：高密实度 + 低缺陷数 + 方正/横向轮廓（h <= w，排除纵向大拇指）
        return (features["solidity"] > self.fist_solidity and
                features["defect_count"] <= 1 and
                abs(features["aspect_ratio"] - 1) < 0.3 and
                features["h"] <= features["w"])  # 新增：握拳高度不大于宽度，排除纵向大拇指

    def is_thumb_up(self, features):
        """强化竖大拇指（up）纵向特征，与握拳形成明显差异"""
        if not features:
            return False
        # 大拇指核心特征：窄高轮廓 + 适中密实度 + 少量缺陷 + 强纵向延伸
        return (features["aspect_ratio"] < self.thumb_aspect_ratio and
                self.thumb_solidity_range[0] < features["solidity"] < self.thumb_solidity_range[1] and
                features["defect_count"] <= self.thumb_defect_max and
                features["hull_aspect"] < 0.7 and
                features["h"] > features["w"] * 1.2)  # 强化纵向：高度大于宽度1.2倍，与握拳形成差距

    def capture_frames(self, cap):
        """帧采集线程"""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            with self.queue_lock:
                self.frame_queue = [frame]
            time.sleep(self.frame_interval * 0.5)

    def process_frame(self, frame):
        """核心处理逻辑：调整手势判断优先级，先up后stop"""
        frame = cv.flip(frame, 1)
        frame = self._draw_recognition_area(frame)
        roi, (roi_x, roi_y) = self._get_roi(frame)
        current_gesture = "None"

        if roi.size > 0:
            # 预处理（暗光增强+去噪）
            roi_small = cv.resize(roi, (400, 300))
            alpha = 1.8
            beta = 40
            roi_enhanced = cv.convertScaleAbs(roi_small, alpha=alpha, beta=beta)
            roi_denoised = cv.GaussianBlur(roi_enhanced, (5, 5), 0)

            # 自适应亮度判断
            gray_roi = cv.cvtColor(roi_small, cv.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray_roi)
            if avg_brightness < 50:
                self.skin_lower = self.skin_lower_dark
                self.skin_upper = self.skin_upper_dark
            else:
                self.skin_lower = self.skin_lower_bright
                self.skin_upper = self.skin_upper_bright

            # 肤色掩码提取+形态学优化
            hsv = cv.cvtColor(roi_denoised, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, self.skin_lower, self.skin_upper)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel, iterations=1)
            mask = cv.morphologyEx(mask, cv.MORPH_DILATE, self.kernel, iterations=2)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel, iterations=2)

            # 找轮廓
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv.contourArea)
                features = self.analyze_contour(cnt)

                if features and features["area"] > self.min_contour_area:
                    # 手动轮廓坐标缩放+偏移
                    scale_x = roi.shape[1] / float(roi_small.shape[1])
                    scale_y = roi.shape[0] / float(roi_small.shape[0])
                    cnt_scaled = cnt.astype(np.float64)
                    cnt_scaled[:, :, 0] *= scale_x
                    cnt_scaled[:, :, 1] *= scale_y
                    cnt_scaled[:, :, 0] += roi_x
                    cnt_scaled[:, :, 1] += roi_y
                    cnt_scaled = cnt_scaled.astype(np.int32)
                    cnt_scaled[:, :, 0] = np.clip(cnt_scaled[:, :, 0], 0, frame.shape[1]-1)
                    cnt_scaled[:, :, 1] = np.clip(cnt_scaled[:, :, 1], 0, frame.shape[0]-1)

                    cv.drawContours(frame, [cnt_scaled], -1, (255, 0, 0), 2)

                    # ========== 核心优化：调整手势判断优先级（先up后stop） ==========
                    # 1. 优先判断竖大拇指（up）- 避免被stop提前拦截
                    if self.is_thumb_up(features):
                        current_gesture = "up"
                    # 2. 再判断握拳（stop）- 此时已排除大拇指，无误判
                    elif self.is_fist(features):
                        current_gesture = "stop"
                    # 3. 其他手势判断
                    elif features["defect_count"] == 1:
                        current_gesture = "front"
                    elif features["defect_count"] >= 3:
                        current_gesture = "back"
                    else:
                        current_gesture = "None"

        # 缓存稳定性增强
        self.gesture_buffer.append(current_gesture)
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)
        if len(set(self.gesture_buffer)) == 1 and len(self.gesture_buffer) == self.buffer_size:
            self.stable_gesture = self.gesture_buffer[0]

        # 绘制UI
        cv.putText(frame, f"Gesture: {self.stable_gesture}", (10, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv.putText(frame, f"FPS: {self.target_fps}", (10, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        frame_show = cv.resize(frame, (640, 480))
        return frame_show

    def run(self):
        """主运行逻辑"""
        # 摄像头初始化
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv.CAP_PROP_FPS, self.target_fps)

        # 启动采集线程
        capture_thread = threading.Thread(target=self.capture_frames, args=(cap,), daemon=True)
        capture_thread.start()

        # 提示信息
        print("=" * 60)
        print(f"✅ 帧率锁定 {self.target_fps} 帧 | ESC退出")
        print("💡 修复up误判为stop（优先级+特征优化）：")
        print("   👍 竖大拇指 → up（优先判断，精准识别）")
        print("   ✊ 握拳 → stop（排除大拇指，无重叠）")
        print("   🤘 食指+中指 → front")
        print("   🖐️  手掌张开 → back")
        print("📌 已解决up与stop的特征重叠问题")
        print("=" * 60)

        # 主循环
        while cap.isOpened():
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            sleep_time = self.frame_interval - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

            # 读取帧
            with self.queue_lock:
                if not self.frame_queue:
                    self.last_frame_time = time.time()
                    continue
                frame = self.frame_queue.pop(0)

            # 处理并显示
            frame_show = self.process_frame(frame)
            cv.imshow("Hand Gesture Recognition (Fix UP→Stop Misjudgment)", frame_show)

            # 更新时间戳
            self.last_frame_time = time.time()

            # ESC退出
            if cv.waitKey(1) & 0xFF == 27:
                break

        # 释放资源
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    recognizer = StableFPSHandRecognizer(target_fps=20)
    recognizer.run()