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

        # 2. 肤色检测（适配明亮+暗光环境，核心优化：新增暗光阈值）
        # 明亮环境阈值（保留原有，适配强光场景）
        self.skin_lower_bright = np.array([0, 10, 50], np.uint8)
        self.skin_upper_bright = np.array([30, 255, 255], np.uint8)
        # 暗光环境阈值（降低S和V下限，放宽H范围，适配弱光场景）
        self.skin_lower_dark = np.array([0, 5, 15], np.uint8)
        self.skin_upper_dark = np.array([40, 180, 200], np.uint8)
        # 默认使用暗光阈值（优先适配弱光，也可通过自适应逻辑切换）
        self.skin_lower = self.skin_lower_dark
        self.skin_upper = self.skin_upper_dark
        self.kernel = np.ones((5, 5), np.uint8)

        # 3. 核心参数（精准适配手势特征，优化暗光下轮廓识别）
        # 握拳参数（稳定识别）
        self.fist_solidity = 0.82  # 降低握拳阈值，提高稳定性
        self.fist_area_ratio = 0.75  # 握拳凸包面积比
        # 手指计数参数
        self.defect_depth_threshold = 8  # 降低深度阈值，提高up识别率
        self.min_contour_area = 300  # 核心优化：从600降至300，适配暗光下小手部轮廓
        # 大拇指识别参数（宽松但精准）
        self.thumb_aspect_ratio = 0.45  # 放宽宽高比
        self.thumb_solidity_range = (0.55, 0.82)  # 刚好卡在握拳阈值下
        self.thumb_defect_max = 2  # 允许2个缺陷（适配不同握法）

        # 4. 缓存参数（增加缓存提升稳定性）
        self.gesture_buffer = []
        self.buffer_size = 3  # 增加缓存到3帧，提升stop稳定性
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

                    if depth > self.defect_depth_threshold and angle < 90:
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
        """稳定识别握拳（stop）"""
        if not features:
            return False
        # 握拳核心特征：高密实度 + 低缺陷数 + 方正轮廓
        return (features["solidity"] > self.fist_solidity and
                features["defect_count"] <= 1 and
                abs(features["aspect_ratio"] - 1) < 0.3)

    def is_thumb_up(self, features):
        """精准识别竖大拇指（up）"""
        if not features:
            return False
        # 大拇指核心特征：
        # 1. 窄高轮廓 2. 密实度在握拳和张开之间 3. 少量缺陷 4. 凸包特征匹配
        return (features["aspect_ratio"] < self.thumb_aspect_ratio and
                self.thumb_solidity_range[0] < features["solidity"] < self.thumb_solidity_range[1] and
                features["defect_count"] <= self.thumb_defect_max and
                features["hull_aspect"] < 0.5)

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
        """核心处理逻辑（暗光增强优化）"""
        frame = cv.flip(frame, 1)
        frame = self._draw_recognition_area(frame)
        roi, (roi_x, roi_y) = self._get_roi(frame)
        current_gesture = "None"

        if roi.size > 0:
            # 预处理（暗光增强：亮度+对比度+去噪+形态学，核心优化）
            roi_small = cv.resize(roi, (400, 300))

            # 步骤1：亮度和对比度增强（解决暗光下图像偏暗、细节不清晰）
            alpha = 1.8  # 对比度增益（>1提升对比度，极暗可调整至2.2）
            beta = 40  # 亮度增益（>0提升亮度，极暗可调整至60）
            roi_enhanced = cv.convertScaleAbs(roi_small, alpha=alpha, beta=beta)

            # 步骤2：高斯模糊去噪（去除暗光下的椒盐噪声，避免干扰轮廓提取）
            roi_denoised = cv.GaussianBlur(roi_enhanced, (5, 5), 0)

            # 步骤3：（可选）自适应亮度判断，自动切换明暗阈值（兼顾所有环境）
            gray_roi = cv.cvtColor(roi_small, cv.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray_roi)
            if avg_brightness < 50:  # 亮度阈值，<50判定为暗光
                self.skin_lower = self.skin_lower_dark
                self.skin_upper = self.skin_upper_dark
            else:  # >50判定为明亮环境
                self.skin_lower = self.skin_lower_bright
                self.skin_upper = self.skin_upper_bright

            # 步骤4：转换HSV并提取肤色掩码（使用适配当前环境的阈值）
            hsv = cv.cvtColor(roi_denoised, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, self.skin_lower, self.skin_upper)

            # 步骤5：优化形态学操作（暗光下增加膨胀迭代，填补手部区域孔洞）
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel, iterations=1)  # 开运算：去除小噪声
            mask = cv.morphologyEx(mask, cv.MORPH_DILATE, self.kernel, iterations=2)  # 膨胀：填补手部孔洞，增强轮廓连续性
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel, iterations=2)  # 闭运算：平滑轮廓边缘，去除残留小空洞

            # 找轮廓
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv.contourArea)
                features = self.analyze_contour(cnt)

                if features and features["area"] > self.min_contour_area:
                    # 绘制轮廓（调试用，可直观看到手部提取效果）
                    cnt_scaled = cnt * (roi.shape[1] / roi_small.shape[1], roi.shape[0] / roi_small.shape[0])
                    cnt_scaled = cnt_scaled.astype(np.int32)
                    cnt_scaled[:, :, 0] += roi_x
                    cnt_scaled[:, :, 1] += roi_y
                    cv.drawContours(frame, [cnt_scaled], -1, (255, 0, 0), 2)

                    # ========== 重构手势判断逻辑（优先级+特征双重验证） ==========
                    # 1. 优先判断握拳（stop）- 双重验证
                    if self.is_fist(features):
                        current_gesture = "stop"
                    # 2. 判断竖大拇指（up）- 专属特征
                    elif self.is_thumb_up(features):
                        current_gesture = "up"
                    # 3. 判断两指（front）- 缺陷数精准匹配
                    elif features["defect_count"] == 1:  # 1个缺陷=2根手指
                        current_gesture = "front"
                    # 4. 判断手掌张开（back）- 多缺陷
                    elif features["defect_count"] >= 3:  # 3个缺陷=4根手指
                        current_gesture = "back"
                    # 5. 其他情况
                    else:
                        current_gesture = "None"

        # 增强缓存稳定性（3帧一致才更新）
        self.gesture_buffer.append(current_gesture)
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)
        # 要求所有缓存帧一致才稳定
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
        """主运行逻辑（修复时间计算错误）"""
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
        print("💡 暗光优化版手势识别（高稳定性）：")
        print("   ✊ 握拳 → stop（高稳定）")
        print("   👍 竖大拇指 → up（精准识别）")
        print("   🤘 食指+中指 → front")
        print("   🖐️  手掌张开 → back")
        print("📌 已适配暗光环境，极暗可调整alpha/beta参数")
        print("=" * 60)

        # 主循环（修复帧率控制）
        while cap.isOpened():
            # 精准帧率控制（确保sleep时间非负）
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            sleep_time = self.frame_interval - elapsed

            # 关键修复：确保sleep时间非负
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
            cv.imshow("Hand Gesture Recognition (Dark Mode Optimized)", frame_show)

            # 更新时间戳
            self.last_frame_time = time.time()

            # ESC退出
            if cv.waitKey(1) & 0xFF == 27:
                break

        # 释放资源
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    # 20帧兼顾流畅度和识别稳定性，暗光下更稳定
    recognizer = StableFPSHandRecognizer(target_fps=20)
    recognizer.run()