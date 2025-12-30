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
        # 3. 核心参数（精准适配手势特征，优化暗光下轮廓识别）
        # 2. 肤色检测（适配更多光线）
        self.skin_lower = np.array([0, 10, 50], np.uint8)
        self.skin_upper = np.array([30, 255, 255], np.uint8)
        self.kernel = np.ones((5, 5), np.uint8)

        # 3. 核心参数（精准适配手势特征）
        # 握拳参数（稳定识别）
        self.fist_solidity = 0.82  # 降低握拳阈值，提高稳定性
        self.fist_area_ratio = 0.75  # 握拳凸包面积比
        # 手指计数参数
        self.defect_depth_threshold = 8  # 降低深度阈值，提高up识别率
        self.min_contour_area = 300  # 核心优化：从600降至300，适配暗光下小手部轮廓
        self.min_contour_area = 600  # 降低最小面积，适配小手掌
        # 大拇指识别参数（宽松但精准）
        self.thumb_aspect_ratio = 0.45  # 放宽宽高比
        self.thumb_solidity_range = (0.55, 0.82)  # 刚好卡在握拳阈值下
        self.thumb_defect_max = 2  # 允许2个缺陷（适配不同握法）

        # 4. 缓存参数（增加缓存提升稳定性）
        self.gesture_buffer = []
        self.buffer_size = 3  # 增加缓存到3帧，提升stop稳定性
        # 2. 优化后的肤色检测阈值
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
        # 5. 识别区域参数（仅显示边框）
        self.recognition_area = None
        self.area_color = (0, 255, 0)  # 边框颜色

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
        """获取识别区域ROI"""
        if self.recognition_area is None:
            self._init_recognition_area(frame.shape)
        x1, y1, x2, y2 = self.recognition_area
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
        self.frame_interval = 1.0 / target_fps  # 每帧间隔时间（秒）
        self.last_frame_time = time.time()

        # 2. 极简手部检测参数
        self.skin_lower = np.array([0, 10, 10], np.uint8)
        self.skin_upper = np.array([30, 255, 180], np.uint8)
        self.kernel = np.ones((3, 3), np.uint8)

        # 新增：手指检测参数
        self.defect_depth_threshold = 20  # 凸包缺陷深度阈值
        self.min_defect_distance = 10  # 缺陷点最小距离
        self.palm_solidity_threshold = 0.6  # 手掌的密实度阈值

        # 3. 手势缓存（仅2帧，快速响应+稳定）
        self.gesture_buffer = []
        self.stable_gesture = "None"

        # 4. 帧缓存（避免堆积）
        self.frame_queue = []
        self.queue_lock = threading.Lock()

    def count_fingers(self, cnt, frame_small):
        """通过凸包缺陷计算手指数量"""
        try:
            # 计算凸包和凸包缺陷
            hull = cv.convexHull(cnt, returnPoints=False)
            defects = cv.convexityDefects(cnt, hull)

            if defects is None:
                return 0

            finger_count = 0
            defect_points = []

            # 遍历所有凸包缺陷
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

                # 有效缺陷：深度足够 + 角度小于90度
                if depth > self.defect_depth_threshold and angle < 90:
                    finger_count += 1

            # 缺陷数+1=手指数量
            return min(finger_count + 1, 5)
        except Exception as e:
            print(f"手指计数错误: {e}")
            return 0

    def capture_frames(self, cap):
        """帧采集线程（稳定）"""
                # 计算缺陷深度（转换为实际像素值）
                depth = d / 256.0

                # 只考虑深度足够的缺陷（手指间的凹陷）
                if depth > self.defect_depth_threshold:
                    # 计算两点间距离，避免重复计数
                    if all(np.linalg.norm(np.array(far) - np.array(p)) > self.min_defect_distance for p in
                           defect_points):
                        defect_points.append(far)
                        finger_count += 1

            # 缺陷数+1 = 手指数量（例如：4个缺陷=5根手指）
            return min(finger_count + 1, 5)  # 最多5根手指
        except:
            return 0

    def capture_frames(self, cap):
        """独立线程采集帧，避免主线程阻塞"""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            with self.queue_lock:
                self.frame_queue = [frame]
            time.sleep(self.frame_interval * 0.5)

    def process_frame(self, frame):
        """核心处理逻辑：调整手势判断优先级，先up后stop"""
        """核心处理逻辑（暗光增强优化）"""
        """核心处理逻辑"""
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
            # 预处理（增强手部轮廓）
            roi_small = cv.resize(roi, (400, 300))
            hsv = cv.cvtColor(roi_small, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, self.skin_lower, self.skin_upper)
            # 形态学操作：先开后闭，保留完整轮廓
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel, iterations=1)
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
                    # 绘制轮廓（调试用，可直观看到手部提取效果）
                    # 绘制轮廓（调试用）
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
            roi_small = cv.resize(roi, (320, 240))  # 适度放大ROI
            hsv = cv.cvtColor(roi_small, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, self.skin_lower, self.skin_upper)

            # 形态学操作（去噪+填充）
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel)
            mask = cv.dilate(mask, self.kernel, iterations=2)

            # 查找轮廓
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours:
                # 取最大轮廓
                cnt = max(contours, key=cv.contourArea)
                area = cv.contourArea(cnt)

                if area > self.min_contour_area:
                    # 计算密实度
                    hull = cv.convexHull(cnt)
                    hull_area = cv.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0

                    # 手指计数
                    finger_count = self.count_fingers(cnt)

                    # 可视化调试
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
                    # 手势判断逻辑
                    if solidity > 0.8:  # 握拳
                        current_gesture = "stop"
                    elif finger_count == 2:  # 食指+中指
                        current_gesture = "front"
                    elif finger_count >= 4:  # 手掌张开
                        current_gesture = "back"
                    # 其他情况（1/3指）归为None

        # 手势缓存稳定
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
            area = cv.contourArea(cnt)

            if area > 1000:
                # 3. 手势分类（修改输出文本映射：Fist→stop, Point→front, Palm→back）
                # 3. 手势分类（修改Point为仅食指+中指（2根手指））
                # 3. 手势分类（新增五指识别）
                hull = cv.convexHull(cnt)
                solidity = cv.contourArea(cnt) / cv.contourArea(hull)

                # 计算手指数量
                finger_count = self.count_fingers(cnt, frame_small)

                # 手势判断逻辑（仅修改输出文本）
                if solidity > 0.85:
                    # 密实度高 = 握拳 → 输出stop
                    current_gesture = "stop"
                elif finger_count == 2:
                    # 仅2根手指 = 食指+中指 → 输出front
                    current_gesture = "front"
                elif finger_count >= 4:
                    # 4-5根手指 = 手掌张开 → 输出back
                    current_gesture = "back"
                elif finger_count == 1:
                    # 1根手指 = 单指（归为None）
                # 手势判断逻辑（核心修改）
                if solidity > 0.85:
                    # 密实度高 = 握拳
                    current_gesture = "Fist"
                elif finger_count == 2:
                    # 仅2根手指 = 食指+中指（Point）
                # 手势判断逻辑
                if solidity > 0.85:
                    # 密实度高 = 握拳
                    current_gesture = "Fist"
                elif finger_count == 1:
                    # 1根手指 = 单指
                    current_gesture = "Point"
                elif finger_count >= 4:
                    # 4-5根手指 = 手掌张开
                    current_gesture = "Palm"
                elif finger_count == 1:
                    # 1根手指 = 单指（归为None或单独分类，这里保持None）
                    current_gesture = "None"
                elif finger_count == 3:
                    # 3根手指 = 归为None
                    current_gesture = "None"
                elif 2 <= finger_count <= 3:
                    # 2-3根手指 = 部分张开（归类为Point）
                    current_gesture = "Point"


# 极简手势识别（仅保留拳头/点手势，极致流畅）
def main():
    # 1. 摄像头初始化（极简参数）
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)  # 极低分辨率，秒杀卡顿
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))  # 快速编码
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # 关闭缓存，降低延迟

    # 2. 固定参数（适配所有摄像头）
    skin_lower = np.array([0, 10, 10], np.uint8)
    skin_upper = np.array([30, 255, 180], np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    last_gesture = "None"
    gesture_count = 0

    print("✅ 极致轻量化手势识别 | ESC退出")
    print("💡 把手放在画面中间，握拳=Fist，伸食指=Point")

    while True:
        # 计时（极简FPS）
        t1 = time.time()

        # 3. 读取帧（跳过缓存帧）
from collections import deque


# ========== 稳定版手势识别（固定ROI+帧验证） ==========
class StableHandRecognizer:
    def __init__(self):
        # 1. 固定检测区域（画面右侧1/3）
        self.roi_x1, self.roi_y1 = 600, 100
        self.roi_x2, self.roi_y2 = 900, 500
        # 2. 肤色范围（扩大兼容）
        self.lower_skin = np.array([0, 20, 30], dtype=np.uint8)
        self.upper_skin = np.array([30, 255, 255], dtype=np.uint8)
        # 3. 手势缓存（连续3帧相同才确认）
        self.gesture_buffer = deque(maxlen=3)
        self.last_gesture = "None"

    def get_roi(self, image):
        """截取固定检测区域"""
        return image[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]

    def process(self, image):
        # 1. 截取ROI
        roi = self.get_roi(image)
        if roi.size == 0:
            return "None", []

        # 2. 预处理（降噪+肤色检测）
        blur = cv.GaussianBlur(roi, (7, 7), 0)
        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_skin, self.upper_skin)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # 3. 找手部轮廓
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.gesture_buffer.append("None")
            return self._get_stable_gesture(), []

        max_contour = max(contours, key=cv.contourArea)
        if cv.contourArea(max_contour) < 5000:
            self.gesture_buffer.append("None")
            return self._get_stable_gesture(), []

        # 4. 手势判断（仅保留拳头/点手势，最稳定）
        hull = cv.convexHull(max_contour)
        solidity = cv.contourArea(max_contour) / cv.contourArea(hull)
        current_gesture = "Fist" if solidity > 0.85 else "Point"
        self.gesture_buffer.append(current_gesture)

        # 5. 稳定输出（连续3帧相同）
        return self._get_stable_gesture(), max_contour

    def _get_stable_gesture(self):
        """只有连续3帧相同才输出"""
        if len(self.gesture_buffer) < 3:
            return self.last_gesture
        if len(set(self.gesture_buffer)) == 1:
            self.last_gesture = self.gesture_buffer[0]
        return self.last_gesture


# ========== 主函数（带ROI框+稳定显示） ==========
def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv.CAP_PROP_EXPOSURE, -6)  # 固定曝光，避免过曝
    recognizer = StableHandRecognizer()
    fps_calc = deque(maxlen=10)

    while True:
        # 计时算FPS
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        frame_small = cv.resize(frame, (160, 120))  # 超小尺寸处理

        # 4. 极简手部检测
        hsv = cv.cvtColor(frame_small, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, skin_lower, skin_upper)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # 5. 找轮廓（只找最大的）
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        current_gesture = "None"
        if contours:
            cnt = max(contours, key=cv.contourArea)
            if cv.contourArea(cnt) > 1000:
                # 3. 极简分类
                # 6. 极简分类（仅拳头/点手势）
                hull = cv.convexHull(cnt)
                solidity = cv.contourArea(cnt) / cv.contourArea(hull)
                current_gesture = "Fist" if solidity > 0.85 else "Point"

        # 4. 稳定手势（仅2帧一致）
        self.gesture_buffer.append(current_gesture)
        if len(self.gesture_buffer) > 2:
            self.gesture_buffer.pop(0)
        if len(set(self.gesture_buffer)) == 1:
            self.stable_gesture = self.gesture_buffer[0]

        # 绘制UI
        # 5. 绘制极简UI（显示修改后的手势文本）
        # 5. 绘制极简UI（仅保留手势和FPS显示）
        # 5. 绘制极简UI（仅保留手势和FPS显示，移除手指数量）
        # 5. 绘制极简UI（控制绘制耗时）
        cv.putText(frame, f"Gesture: {self.stable_gesture}", (10, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv.putText(frame, f"FPS: {self.target_fps}", (10, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # 拉伸
        # 拉伸显示（保持清晰）
        frame_show = cv.resize(frame, (640, 480))
        return frame_show

    def run(self):
        """主运行逻辑"""
        # 摄像头初始化
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        """主运行逻辑（修复时间计算错误）"""
        # 摄像头初始化
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        """主运行逻辑"""
        # 摄像头初始化
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
        print("💡 暗光优化版手势识别（高稳定性）：")
        print("💡 优化版手势识别（高稳定性）：")
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

        # 修改控制台提示文本，匹配新的输出
        print(f"✅ 帧率锁定 {self.target_fps} 帧 | ESC退出")
        print("💡 把手放在画面中间，握拳=stop，伸食指+中指=front，五指张开=back")
        print(f"✅ 帧率锁定 {self.target_fps} 帧 | ESC退出")
        print("💡 把手放在画面中间，握拳=Fist，伸食指+中指=Point，五指张开=Palm")
        print("💡 把手放在画面中间，握拳=Fist，伸食指=Point，五指张开=Palm")
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
            cv.imshow("Hand Gesture Recognition (Fix UP→Stop Misjudgment)", frame_show)

            # 更新时间戳
            cv.imshow("Hand Gesture Recognition (Dark Mode Optimized)", frame_show)

            # 更新时间戳
            cv.imshow("Hand Gesture Recognition (Optimized)", frame_show)

            # 更新时间戳
            cv.imshow("Hand Gesture Recognition", frame_show)

            # 更新时间戳
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
    recognizer = StableFPSHandRecognizer(target_fps=20)
    recognizer.run()
    recognizer.run()
    # 20帧兼顾流畅度和识别稳定性
    recognizer = StableFPSHandRecognizer(target_fps=20)
    recognizer.run()
    # 可降低帧率（如15）提高稳定性
    recognizer = StableFPSHandRecognizer(target_fps=20)
    recognizer.run()
    # 实例化并运行，锁定30帧（可改20/15帧，更低更稳）
    recognizer = StableFPSHandRecognizer(target_fps=30)
    recognizer.run()
    recognizer.run()
    recognizer.run()
    recognizer.run()
        # 7. 稳定输出（连续2帧相同）
        if current_gesture == last_gesture:
            gesture_count += 1
        else:
            gesture_count = 0
            last_gesture = current_gesture
        stable_gesture = last_gesture if gesture_count > 1 else "None"

        # 8. 绘制（极简UI，减少计算）
        cv.putText(frame, f"Gesture: {stable_gesture}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv.putText(frame, f"FPS: {int(1 / (time.time() - t1))}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 9. 显示（拉伸回原尺寸，保持清晰）
        frame_show = cv.resize(frame, (640, 480))
        cv.imshow("Ultra Light Gesture", frame_show)

        if cv.waitKey(1) & 0xFF == 27:
            break

        debug_frame = frame.copy()

        # 1. 绘制ROI框（提示用户把手放在这里）
        cv.rectangle(debug_frame, (recognizer.roi_x1, recognizer.roi_y1),
                     (recognizer.roi_x2, recognizer.roi_y2), (0, 255, 255), 2)
        cv.putText(debug_frame, "Put hand here", (recognizer.roi_x1 + 10, recognizer.roi_y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 2. 识别手势
        gesture, contour = recognizer.process(frame)

        # 3. 绘制结果（固定位置，不闪烁）
        cv.putText(debug_frame, f"Stable Gesture: {gesture}", (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        # 4. 绘制手部轮廓（ROI内）
        if contour is not None and len(contour) > 0:
            # 转换轮廓坐标到全局画面
            contour[:, :, 0] += recognizer.roi_x1
            contour[:, :, 1] += recognizer.roi_y1
            cv.drawContours(debug_frame, [contour], -1, (0, 255, 0), 2)

        # 计算FPS
        fps = 1 / (time.time() - start)
        fps_calc.append(fps)
        cv.putText(debug_frame, f"FPS: {int(np.mean(fps_calc))}", (50, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        # 显示
        cv.imshow("Stable Hand Gesture", debug_frame)
        if cv.waitKey(1) & 0xFF == 27:
            break

# 导入必要的库
import copy  # 深拷贝库，用于数据副本创建
import argparse  # 命令行参数解析库
import itertools  # 迭代工具库，用于数据扁平化
from collections import Counter  # 计数工具，用于手势历史统计
from collections import deque  # 双端队列，用于FPS计算和历史数据存储
import time  # 时间库，用于FPS计算

import cv2 as cv  # OpenCV库，核心视觉处理
import numpy as np  # 数值计算库，用于数组操作


# ========== FPS计算类（还原初始版本，无多余逻辑） ==========
class CvFpsCalc:
    """
    FPS（帧率）计算类
    功能：基于时间戳队列计算实时帧率，缓冲区长度控制计算稳定性
    """

    def __init__(self, buffer_len=10):
        """
        初始化FPS计算器
        :param buffer_len: 时间戳缓冲区长度，默认10帧
        """
        self.buffer_len = buffer_len  # 缓冲区长度
        self.times = deque(maxlen=buffer_len)  # 存储时间戳的双端队列

    def get(self):
        """
        计算并返回当前帧率
        :return: 整数型帧率值（FPS）
        """
        # 记录当前时间戳
        self.times.append(time.perf_counter())
        # 缓冲区数据不足时返回0
        if len(self.times) < 2:
            return 0
        # 帧率计算公式：帧数 / 总时间（秒）
        return int(len(self.times) / (self.times[-1] - self.times[0]))


# ========== 手势分类器（简化版，模拟点手势识别） ==========
class KeyPointClassifier:
    """
    关键点分类器（简化版）
    功能：模拟手势分类，固定返回点手势标识（7）
    """

    def __call__(self, landmark_list):
        """
        分类调用方法
        :param landmark_list: 手部关键点列表（未实际使用）
        :return: 固定返回7（代表点手势）
        """
        return 7  # 模拟点手势分类结果


class PointHistoryClassifier:
    """
    轨迹历史分类器（简化版）
    功能：模拟轨迹分类，固定返回0
    """

    def __call__(self, point_history):
        """
        分类调用方法
        :param point_history: 关键点轨迹历史（未实际使用）
        :return: 固定返回0
        """
        return 0  # 模拟轨迹分类结果


# ========== 命令行参数解析函数 ==========
def get_args():
    """
    解析命令行参数
    :return: 解析后的参数对象
    参数说明：
        --device: 摄像头设备号，默认0（内置摄像头）
        --width: 摄像头采集宽度，默认960像素
        --height: 摄像头采集高度，默认540像素
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加摄像头设备号参数
    parser.add_argument("--device", type=int, default=0)
    # 添加采集宽度参数
    parser.add_argument("--width", type=int, default=960)
    # 添加采集高度参数
    parser.add_argument("--height", type=int, default=540)
    # 解析参数并返回
    return parser.parse_args()


# ========== 辅助函数（仅修复按键响应，不改动核心逻辑） ==========
def select_mode(key, mode):
    """
    按键模式选择函数
    功能：根据按键值切换程序运行模式，兼容ASCII码和字符判断
    :param key: 按键ASCII码值
    :param mode: 当前模式（0=Idle/空闲, 1=Log Keypoint/关键点记录, 2=Log Point History/轨迹记录）
    :return: (数字编号, 切换后的模式)
    """
    number = -1  # 初始化数字编号（0-9按键对应）
    # 判断是否为数字按键（0-9）
    if 48 <= key <= 57:
        number = key - 48  # 转换为数字（ASCII码48对应0）

    # 模式切换逻辑（兼容字符和ASCII码）
    if key == ord('n') or key == 110:  # n键：切换到空闲模式
        mode = 0
    elif key == ord('k') or key == 107:  # k键：切换到关键点记录模式
        mode = 1
    elif key == ord('h') or key == 104:  # h键：切换到轨迹记录模式
# 功能：无人机手势识别模拟程序（适配Python3.13+Windows）
# 说明：移除MediaPipe/TensorFlow依赖，通过模拟手部数据实现核心逻辑展示
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import time

import cv2 as cv
import numpy as np


# ========== FPS计算（还原初始版本，无多余逻辑） ==========
# ===================== 工具类：FPS计算 =====================
class CvFpsCalc:
    """帧率计算类，基于时间队列滑动平均计算FPS"""

    def __init__(self, buffer_len=10):
        self.buffer_len = buffer_len  # 缓存长度（计算最近N帧的平均FPS）
        self.times = deque(maxlen=buffer_len)  # 时间戳队列

    def get(self):
        """获取当前FPS值"""
        self.times.append(time.perf_counter())  # 记录当前时间戳
        if len(self.times) < 2:  # 至少需要2个时间戳才能计算
            return 0
        # 计算平均FPS：帧数 / 总时间
        fps = len(self.times) / (self.times[-1] - self.times[0])
        return int(fps)


# ===================== 模拟分类器（适配原逻辑） =====================
class KeyPointClassifier:
    """关键点分类器（简化版）"""

    def __call__(self, landmark_list):
        # 固定返回7（对应PointGesture点手势，模拟分类结果）
        return 7


class PointHistoryClassifier:
    """轨迹点分类器（简化版）"""

    def __call__(self, point_history):
        # 固定返回0（对应None，模拟分类结果）
        return 0

    # ===================== 参数解析 =====================


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="手部手势识别模拟程序")
    parser.add_argument("--device", type=int, default=0, help="摄像头设备号（默认0）")
    parser.add_argument("--width", type=int, default=960, help="摄像头画面宽度")
    parser.add_argument("--height", type=int, default=540, help="摄像头画面高度")
    return parser.parse_args()


# ===================== 核心辅助函数 =====================
def select_mode(key, mode):
    """根据按键切换操作模式
    Args:
        key: 按键值
        mode: 当前模式（0:空闲 1:记录关键点 2:记录轨迹点）
    Returns:
        number: 按键数字（0-9），-1表示非数字键
        mode: 更新后的模式
    """
    number = -1
    if 48 <= key <= 57:  # 数字键0-9
        number = key - 48
    if key == ord('n'):  # n键：切换到空闲模式
        mode = 0
    if key == ord('k'):  # k键：切换到记录关键点模式
        mode = 1
    if key == ord('h'):  # h键：切换到记录轨迹点模式
# ========== FPS计算 ==========
class CvFpsCalc:
    def __init__(self, buffer_len=10):
        self.buffer_len = buffer_len
        self.times = deque(maxlen=buffer_len)

    def get(self):
        self.times.append(time.perf_counter())
        if len(self.times) < 2:
            return 0
        return int(len(self.times) / (self.times[-1] - self.times[0]))


# ========== 手势分类器（还原初始版本） ==========
# ========== 手势分类器（简化版） ==========
class KeyPointClassifier:
    def __call__(self, landmark_list):
        return 7  # 模拟点手势


class PointHistoryClassifier:
    def __call__(self, point_history):
        return 0


# ========== 参数解析（还原初始版本） ==========
# ========== 参数解析 ==========
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


# ========== 辅助函数（仅修复按键响应，不改动逻辑） ==========
def select_mode(key, mode):
    """仅修复按键捕获，保留初始逻辑"""
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    # 还原初始按键判断，仅增加ASCII码兼容（不影响帧率）
    if key == ord('n') or key == 110:
        mode = 0
    elif key == ord('k') or key == 107:
        mode = 1
    elif key == ord('h') or key == 104:
# ========== 辅助函数 ==========
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    if key == ord('n'):
        mode = 0
    if key == ord('k'):
        mode = 1
    if key == ord('h'):
        mode = 2
    return number, mode


def calc_bounding_rect(image):
    """
    计算手部边界框（模拟）
    功能：以画面中心为基准，生成200×200像素的正方形边界框
    :param image: 输入图像（用于获取宽高）
    :return: 边界框坐标 [x1, y1, x2, y2]
    """
    # 获取图像宽高
    h, w = image.shape[:2]
    # 计算画面中心坐标
    cx, cy = w // 2, h // 2
    # 边界框尺寸（200×200）
    bw, bh = 200, 200
    # 返回边界框坐标（左上x, 左上y, 右下x, 右下y）
    """还原初始逻辑"""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    bw, bh = 200, 200
    """生成模拟手部边界框（屏幕中心固定位置）
    Args:
        image: 摄像头帧画面
    Returns:
        brect: 边界框坐标 [x1, y1, x2, y2]
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2  # 屏幕中心
    bw, bh = 200, 200  # 边界框尺寸
    """模拟手部边界框（屏幕中心）"""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    bw, bh = 200, 200  # 边界框大小
    return [cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2]


def calc_landmark_list(image):
    """
    生成模拟手部关键点列表
    功能：以画面中心为基准，生成21个预设的手部关键点坐标（对应手部骨骼）
    :param image: 输入图像（用于获取宽高）
    :return: 21个关键点的坐标列表 [[x1,y1], [x2,y2], ..., [x21,y21]]
    关键点说明：
        0: 手掌中心
        1-4: 拇指
        5-8: 食指
        9-12: 中指
        13-16: 无名指
        17-20: 小指
    """
    # 获取图像宽高
    h, w = image.shape[:2]
    # 画面中心坐标
    cx, cy = w // 2, h // 2
    # 初始化关键点列表
    landmark_list = []

    # 0: 手掌中心
    landmark_list.append([cx, cy])
    # 1-4: 拇指关键点
    """还原初始逻辑"""
    """生成模拟21个手部关键点（适配原代码21点逻辑）
    Args:
        image: 摄像头帧画面
    Returns:
        landmark_list: 21个关键点坐标列表 [[x1,y1], [x2,y2], ...]
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2  # 屏幕中心（关键点基准位置）
    landmark_list = []

    # 0号点：手掌中心
    landmark_list.append([cx, cy])
    # 1-4号点：拇指
    landmark_list.extend([[cx - 50, cy - 30], [cx - 80, cy - 60], [cx - 100, cy - 90], [cx - 110, cy - 110]])
    # 5-8号点：食指（8号点为指尖，点手势关键）
    landmark_list.extend([[cx + 50, cy - 30], [cx + 80, cy - 60], [cx + 100, cy - 90], [cx + 110, cy - 110]])
    # 9-12号点：中指
    landmark_list.extend([[cx + 30, cy - 10], [cx + 50, cy - 40], [cx + 70, cy - 70], [cx + 80, cy - 90]])
    # 13-16号点：无名指
    landmark_list.extend([[cx + 10, cy + 10], [cx + 20, cy - 20], [cx + 30, cy - 50], [cx + 40, cy - 70]])
    # 17-20号点：小指
    landmark_list.extend([[cx - 10, cy + 10], [cx - 20, cy - 20], [cx - 30, cy - 50], [cx - 40, cy - 70]])
    """模拟21个手部关键点（适配原代码逻辑）"""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    landmark_list = []

    landmark_list.append([cx, cy])
    # 手掌中心（0号点）
    landmark_list.append([cx, cy])

    # 拇指（1-4号点）
    landmark_list.append([cx - 50, cy - 30])
    landmark_list.append([cx - 80, cy - 60])
    landmark_list.append([cx - 100, cy - 90])
    landmark_list.append([cx - 110, cy - 110])
    # 5-8: 食指关键点
    landmark_list.append([cx + 50, cy - 30])
    landmark_list.append([cx + 80, cy - 60])
    landmark_list.append([cx + 100, cy - 90])
    landmark_list.append([cx + 110, cy - 110])
    # 9-12: 中指关键点

    # 食指（5-8号点）
    landmark_list.append([cx + 50, cy - 30])
    landmark_list.append([cx + 80, cy - 60])
    landmark_list.append([cx + 100, cy - 90])
    landmark_list.append([cx + 110, cy - 110])  # 8号点（点手势关键）

    # 中指（9-12号点）
    landmark_list.append([cx + 30, cy - 10])
    landmark_list.append([cx + 50, cy - 40])
    landmark_list.append([cx + 70, cy - 70])
    landmark_list.append([cx + 80, cy - 90])
    # 13-16: 无名指关键点

    # 无名指（13-16号点）
    landmark_list.append([cx + 10, cy + 10])
    landmark_list.append([cx + 20, cy - 20])
    landmark_list.append([cx + 30, cy - 50])
    landmark_list.append([cx + 40, cy - 70])
    # 17-20: 小指关键点

    # 小指（17-20号点）
    landmark_list.append([cx - 10, cy + 10])
    landmark_list.append([cx - 20, cy - 20])
    landmark_list.append([cx - 30, cy - 50])
    landmark_list.append([cx - 40, cy - 70])

    return landmark_list


def pre_process_landmark(landmark_list):
    """
    关键点预处理函数
    功能：归一化关键点坐标（以手掌中心为原点，缩放至-1~1范围）
    :param landmark_list: 原始关键点列表
    :return: 归一化后的一维数组
    """
    # 深拷贝关键点列表（避免修改原数据）
    temp = copy.deepcopy(landmark_list)
    # 空列表直接返回
    if not temp:
        return []
    # 以手掌中心（第一个关键点）为原点
    base_x, base_y = temp[0][0], temp[0][1]
    # 所有关键点减去原点坐标（相对化）
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    # 将二维列表扁平化为一维数组
    temp = list(itertools.chain.from_iterable(temp))
    # 计算最大绝对值（用于缩放）
    max_val = max(map(abs, temp)) if temp else 1
    # 归一化到-1~1范围
    """还原初始逻辑"""
    """关键点预处理：相对坐标转换+归一化（适配原逻辑）
    Args:
        landmark_list: 原始关键点列表
    Returns:
        预处理后的一维归一化列表
    """
    temp = copy.deepcopy(landmark_list)
    if not temp:  # 空值保护
        return []

    # 相对坐标：以0号点（手掌中心）为基准
    temp = copy.deepcopy(landmark_list)
    if not temp:
        return []
    base_x, base_y = temp[0][0], temp[0][1]
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y

    # 一维化 + 归一化（消除尺度影响）
    temp = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, temp)) if temp else 1  # 除零保护
    temp = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, temp)) if temp else 1
    return [x / max_val for x in temp]


def pre_process_point_history(image, point_history):
    """
    轨迹历史预处理函数
    功能：归一化轨迹坐标（以第一个轨迹点为原点，缩放至图像宽高比例）
    :param image: 输入图像（用于获取宽高）
    :param point_history: 轨迹点历史列表
    :return: 归一化后的一维数组
    """
    # 深拷贝轨迹列表
    temp = copy.deepcopy(point_history)
    # 空列表直接返回
    if not temp:
        return []
    # 以第一个轨迹点为原点
    base_x, base_y = temp[0][0], temp[0][1]
    # 获取图像宽高
    image_w, image_h = image.shape[1], image.shape[0]
    # 归一化坐标（相对图像比例）
    for i in range(len(temp)):
        temp[i][0] = (temp[i][0] - base_x) / image_w
        temp[i][1] = (temp[i][1] - base_y) / image_h
    # 扁平化数组并返回
    """还原初始逻辑"""
    """轨迹点预处理：相对坐标转换+归一化（适配原逻辑）
    Args:
        image: 摄像头帧画面
        point_history: 轨迹点历史列表
    Returns:
        预处理后的一维归一化列表
    """
    temp = copy.deepcopy(point_history)
    if not temp:  # 空值保护
        return []

    # 相对坐标：以第一个点为基准
    base_x, base_y = temp[0][0], temp[0][1]
    image_w, image_h = image.shape[1], image.shape[0]
    for i in range(len(temp)):
        temp[i][0] = (temp[i][0] - base_x) / image_w  # 归一化到[0,1]
        temp[i][1] = (temp[i][1] - base_y) / image_h

    # 一维化
    return list(itertools.chain.from_iterable(temp))


# ===================== 绘制函数（UI展示） =====================
def draw_landmarks(image, landmark_list):
    """绘制手部关键点和连线
    Args:
        image: 待绘制的画面
        landmark_list: 关键点列表
    Returns:
        绘制后的画面
    """
    if len(landmark_list) == 0:
        return image

    # 手指连线定义（关键点索引对）
    links = [(2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
             (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
             (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]

    # 绘制连线：黑色粗线+白色细线（立体效果）
    for (p1, p2) in links:
        if p1 < len(landmark_list) and p2 < len(landmark_list):  # 索引保护
            cv.line(image, tuple(landmark_list[p1]), tuple(landmark_list[p2]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_list[p1]), tuple(landmark_list[p2]), (255, 255, 255), 2)

    # 绘制关键点：指尖8号/12号等用大圆点，其余用小圆点
    for i, (x, y) in enumerate(landmark_list):
        size = 8 if i in [4, 8, 12, 16, 20] else 5
        cv.circle(image, (x, y), size, (255, 255, 255), -1)  # 白色填充
        cv.circle(image, (x, y), size, (0, 0, 0), 1)  # 黑色边框
    temp = copy.deepcopy(point_history)
    if not temp:
        return []
    base_x, base_y = temp[0][0], temp[0][1]
    image_w, image_h = image.shape[1], image.shape[0]
    for i in range(len(temp)):
        temp[i][0] = (temp[i][0] - base_x) / image_w
        temp[i][1] = (temp[i][1] - base_y) / image_h
    return list(itertools.chain.from_iterable(temp))


def draw_landmarks(image, landmark_list):
    """
    绘制手部关键点和连线
    功能：在图像上绘制21个关键点（圆）和骨骼连线（线条）
    :param image: 输入图像（画布）
    :param landmark_list: 关键点列表
    :return: 绘制后的图像
    """
    # 空列表直接返回原图像
    if len(landmark_list) == 0:
        return image
    # 定义手部骨骼连线（关键点索引对）
    links = [(2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
             (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
             (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
    # 绘制骨骼连线
    for (p1, p2) in links:
        # 确保索引有效
        if p1 < len(landmark_list) and p2 < len(landmark_list):
            # 绘制黑色粗线（底层）
            cv.line(image, tuple(landmark_list[p1]), tuple(landmark_list[p2]), (0, 0, 0), 6)
            # 绘制白色细线（上层，模拟骨骼）
            cv.line(image, tuple(landmark_list[p1]), tuple(landmark_list[p2]), (255, 255, 255), 2)
    # 绘制关键点（圆）
    for i, (x, y) in enumerate(landmark_list):
        # 指尖关键点（4/8/12/16/20）绘制更大的圆
        size = 8 if i in [4, 8, 12, 16, 20] else 5
        # 白色实心圆（底层）
        cv.circle(image, (x, y), size, (255, 255, 255), -1)
        # 黑色描边（上层）
    """还原初始绘制逻辑（不改动）"""
    if len(landmark_list) == 0:
        return image
    """绘制模拟关键点"""
    if len(landmark_list) == 0:
        return image
    # 绘制手指连线
    links = [(2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
             (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
             (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
    for (p1, p2) in links:
        if p1 < len(landmark_list) and p2 < len(landmark_list):
            cv.line(image, tuple(landmark_list[p1]), tuple(landmark_list[p2]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_list[p1]), tuple(landmark_list[p2]), (255, 255, 255), 2)
    # 绘制关键点
    for i, (x, y) in enumerate(landmark_list):
        size = 8 if i in [4, 8, 12, 16, 20] else 5
        cv.circle(image, (x, y), size, (255, 255, 255), -1)
        cv.circle(image, (x, y), size, (0, 0, 0), 1)
    return image


def draw_bounding_rect(image, brect):
    """
    绘制手部边界框
    功能：在图像上绘制绿色矩形边界框
    :param image: 输入图像
    :param brect: 边界框坐标 [x1, y1, x2, y2]
    :return: 绘制后的图像
    """还原初始逻辑"""
    """绘制手部边界框
    Args:
        image: 待绘制的画面
        brect: 边界框坐标 [x1,y1,x2,y2]
    Returns:
        绘制后的画面
    """
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
    return image


def draw_info_text(image, brect, hand_sign_text, finger_gesture_text):
    """
    绘制手势信息文本
    功能：在边界框上方绘制手势类型文本，在画面左上角绘制轨迹类型文本
    :param image: 输入图像
    :param brect: 边界框坐标
    :param hand_sign_text: 手势类型文本（如Point）
    :param finger_gesture_text: 轨迹类型文本（如None）
    :return: 绘制后的图像
    """
    # 绘制手势类型背景框（绿色）
    cv.rectangle(image, (brect[0], brect[1] - 30), (brect[2], brect[1]), (0, 255, 0), -1)
    # 手势类型文本内容
    info = f"Hand: {hand_sign_text}"
    # 绘制手势类型文本（白色）
    cv.putText(image, info, (brect[0] + 5, brect[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # 绘制轨迹类型文本（红色）
    """还原初始逻辑"""
    """绘制手势信息文本
    Args:
        image: 待绘制的画面
        brect: 边界框坐标
        hand_sign_text: 手部手势文本
        finger_gesture_text: 手指轨迹手势文本
    Returns:
        绘制后的画面
    """
    # 绘制背景框（覆盖边界框上方）
    cv.rectangle(image, (brect[0], brect[1] - 30), (brect[2], brect[1]), (0, 255, 0), -1)
    # 绘制手部手势标签
    info = f"Hand: {hand_sign_text}"
    cv.putText(image, info, (brect[0] + 5, brect[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # 绘制轨迹手势标签
    cv.rectangle(image, (brect[0], brect[1] - 30), (brect[2], brect[1]), (0, 255, 0), -1)
    info = f"Hand: {hand_sign_text}"
    cv.putText(image, info, (brect[0] + 5, brect[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if finger_gesture_text:
        cv.putText(image, f"Gesture: {finger_gesture_text}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return image


def draw_point_history(image, point_history):
    """
    绘制轨迹历史
    功能：在图像上绘制关键点的运动轨迹（渐变大小的绿色圆）
    :param image: 输入图像
    :param point_history: 轨迹点历史列表
    :return: 绘制后的图像
    """
    for i, (x, y) in enumerate(point_history):
        # 非空轨迹点才绘制
        if x != 0 and y != 0:
            # 轨迹点大小随索引递增（模拟轨迹深度）
    """还原初始逻辑"""
    """绘制轨迹点历史（指尖移动轨迹）
    Args:
        image: 待绘制的画面
        point_history: 轨迹点列表
    Returns:
        绘制后的画面
    """
    for i, (x, y) in enumerate(point_history):
        if x != 0 and y != 0:  # 跳过无效点
            # 轨迹点大小随索引递增（视觉层次感）
    for i, (x, y) in enumerate(point_history):
        if x != 0 and y != 0:
            cv.circle(image, (x, y), 2 + i // 2, (0, 255, 0), -1)
    return image


def draw_info(image, fps, mode, number):
    """
    绘制系统信息（FPS/模式/数字）
    功能：在画面左上角绘制帧率、运行模式、数字编号
    :param image: 输入图像
    :param fps: 当前帧率
    :param mode: 当前运行模式
    :param number: 数字编号（0-9）
    :return: 绘制后的图像
    """
    # 绘制帧率（红色）
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    # 模式文本映射
    mode_text = ["Idle", "Log Keypoint", "Log Point History"][mode] if 0 <= mode <= 2 else "Idle"
    # 绘制运行模式（白色）
    cv.putText(image, f"Mode: {mode_text}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # 绘制数字编号（白色，仅当有效时）
    """仅修复模式显示的可视化，不改动绘制逻辑"""
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    mode_text = ["Idle", "Log Keypoint", "Log Point History"][mode] if 0 <= mode <= 2 else "Idle"
    cv.putText(image, f"Mode: {mode_text}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if 0 <= number <= 9:
        cv.putText(image, f"Num: {number}", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


# ========== 主函数（程序入口，仅修复退出/按键BUG，不改动核心逻辑） ==========
def main():
    """
    程序主函数
    执行流程：
    1. 解析命令行参数
    2. 初始化摄像头和核心组件
    3. 主循环：采集图像→处理数据→绘制画面→响应按键
    4. 资源释放与退出
    """
    # 1. 解析命令行参数
    args = get_args()

    # 2. 初始化摄像头（原生模式，无硬件加速）
    cap = cv.VideoCapture(args.device)  # 打开摄像头
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)  # 设置采集宽度
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)  # 设置采集高度

    # 3. 初始化核心组件
    cvFpsCalc = CvFpsCalc(buffer_len=10)  # FPS计算器
    keypoint_classifier = KeyPointClassifier()  # 关键点分类器
    point_history_classifier = PointHistoryClassifier()  # 轨迹分类器

    # 手势标签映射（用于显示）
    keypoint_labels = ["None", "Point", "Fist", "OK", "Peace", "ThumbUp", "ThumbDown", "PointGesture"]
    point_history_labels = ["None", "MoveUp", "MoveDown", "MoveLeft", "MoveRight"]
    # 轨迹历史长度（控制队列大小）
    history_length = 16
    # 初始化轨迹历史队列
    point_history = deque(maxlen=history_length)
    # 初始化手势历史队列（用于统计）
    finger_gesture_history = deque(maxlen=history_length)
    # 初始运行模式（0=Idle）
    mode = 0

    # 打印启动信息
    print("✅ 还原初始版本（30帧）| ESC退出 | n/k/h切换模式")

    try:
        # 4. 主循环（持续采集和处理）
        while True:
            # 4.1 计算当前帧率
            fps = cvFpsCalc.get()

            # 4.2 按键响应（1ms等待，避免卡死）
            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC键：退出主循环
                break

            # 4.3 切换运行模式
            number, mode = select_mode(key, mode)

            # 4.4 采集摄像头图像
            ret, frame = cap.read()
            if not ret:  # 采集失败则退出循环
                break

            # 4.5 图像预处理（镜像翻转+深拷贝）
            frame = cv.flip(frame, 1)  # 水平镜像（符合人眼习惯）
            debug_frame = copy.deepcopy(frame)  # 拷贝图像用于绘制（避免修改原数据）

            # 4.6 核心数据处理
            # 计算手部边界框
            brect = calc_bounding_rect(debug_frame)
            # 生成模拟关键点列表
            landmark_list = calc_landmark_list(debug_frame)
            # 关键点归一化
            pre_landmark = pre_process_landmark(landmark_list)
            # 轨迹历史归一化
            pre_point_history = pre_process_point_history(debug_frame, point_history)

            # 手势分类（模拟）
            hand_sign_id = keypoint_classifier(pre_landmark)
            # 记录食指关键点轨迹
            point_history.append(landmark_list[8] if hand_sign_id == 7 else [0, 0])

            # 轨迹分类（模拟，仅当历史数据足够时）
            finger_gesture_id = 0
            if len(pre_point_history) == history_length * 2:
                finger_gesture_id = point_history_classifier(pre_point_history)
            # 记录手势分类历史
            finger_gesture_history.append(finger_gesture_id)
            # 统计最频繁的手势（模拟分类结果）
            most_common = Counter(finger_gesture_history).most_common(1)

            # 4.7 画面绘制
            debug_frame = draw_bounding_rect(debug_frame, brect)  # 绘制边界框
            debug_frame = draw_landmarks(debug_frame, landmark_list)  # 绘制关键点和连线
            # 绘制手势信息
# ========== 主函数（仅修复退出/按键BUG，不改动核心逻辑） ==========
def main():
    args = get_args()
    # 还原初始摄像头初始化（无多余硬件加速配置）
    """绘制全局信息（FPS、模式、数字）
    Args:
        image: 待绘制的画面
        fps: 当前帧率
        mode: 当前模式
        number: 当前数字
    Returns:
        绘制后的画面
    """
    # 绘制FPS
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    # 绘制模式
    mode_text = ["Idle", "Log Keypoint", "Log Point History"][mode] if 0 <= mode <= 2 else "Idle"
    cv.putText(image, f"Mode: {mode_text}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # 绘制数字
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    mode_text = ["Idle", "Log Keypoint", "Log Point History"][mode] if 0 <= mode <= 2 else "Idle"
    cv.putText(image, f"Mode: {mode_text}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if 0 <= number <= 9:
        cv.putText(image, f"Num: {number}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


# ===================== 主程序入口 =====================
def main():
    # 1. 初始化参数和资源
    args = get_args()
    cap = cv.VideoCapture(args.device)  # 打开摄像头
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)  # 设置宽度
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)  # 设置高度
# ========== 主函数 ==========
def main():
    args = get_args()
    # 初始化摄像头
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # 还原初始初始化逻辑
    # 初始化工具类
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # 初始化标签和历史数据
    keypoint_labels = ["None", "Point", "Fist", "OK", "Peace", "ThumbUp", "ThumbDown", "PointGesture"]
    point_history_labels = ["None", "MoveUp", "MoveDown", "MoveLeft", "MoveRight"]
    history_length = 16  # 轨迹点缓存长度
    point_history = deque(maxlen=history_length)  # 指尖轨迹缓存
    finger_gesture_history = deque(maxlen=history_length)  # 手势分类结果缓存
    mode = 0  # 初始模式：空闲

    # 2. 主循环（摄像头帧处理）
    while True:
        # 计算当前FPS
        fps = cvFpsCalc.get()

        # 按键处理（ESC退出）
        key = cv.waitKey(1) & 0xFF
        if key == 27:
    # 标签和历史数据
    keypoint_labels = ["None", "Point", "Fist", "OK", "Peace", "ThumbUp", "ThumbDown", "PointGesture"]
    point_history_labels = ["None", "MoveUp", "MoveDown", "MoveLeft", "MoveRight"]
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    mode = 0

    print("✅ 还原初始版本（30帧）| ESC退出 | n/k/h切换模式")

    try:
        while True:
            # 还原初始帧率计算
            fps = cvFpsCalc.get()

            # 修复按键响应（仅捕获，无多余逻辑）
            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC退出
                break

            # 还原初始模式切换
            number, mode = select_mode(key, mode)

            # 还原初始帧读取（无多余异常捕获）
            ret, frame = cap.read()
            if not ret:
                break

            # 还原初始镜像+拷贝逻辑
            frame = cv.flip(frame, 1)
            debug_frame = copy.deepcopy(frame)

            # 还原初始核心逻辑（不改动）
            brect = calc_bounding_rect(debug_frame)
            landmark_list = calc_landmark_list(debug_frame)
            pre_landmark = pre_process_landmark(landmark_list)
            pre_point_history = pre_process_point_history(debug_frame, point_history)

            hand_sign_id = keypoint_classifier(pre_landmark)
            point_history.append(landmark_list[8] if hand_sign_id == 7 else [0, 0])

            finger_gesture_id = 0
            if len(pre_point_history) == history_length * 2:
                finger_gesture_id = point_history_classifier(pre_point_history)
            finger_gesture_history.append(finger_gesture_id)
            most_common = Counter(finger_gesture_history).most_common(1)

            # 还原初始绘制逻辑
            debug_frame = draw_bounding_rect(debug_frame, brect)
            debug_frame = draw_landmarks(debug_frame, landmark_list)
            debug_frame = draw_info_text(
                debug_frame, brect,
                keypoint_labels[hand_sign_id] if hand_sign_id < len(keypoint_labels) else "Unknown",
                point_history_labels[most_common[0][0]] if most_common else "Unknown"
            )
            debug_frame = draw_point_history(debug_frame, point_history)  # 绘制轨迹
            debug_frame = draw_info(debug_frame, fps, mode, number)  # 绘制系统信息

            # 4.8 显示画面
            cv.imshow('Hand Gesture Recognition', debug_frame)

    # 捕获Ctrl+C中断（手动终止程序）
    except KeyboardInterrupt:
        pass
    # 最终资源释放（无论是否异常，都执行）
    finally:
        cap.release()  # 释放摄像头资源
        cv.destroyAllWindows()  # 关闭所有OpenCV窗口
        print(f"✅ 退出 | 最终帧率：{fps}")  # 打印退出信息


# 程序入口
            debug_frame = draw_point_history(debug_frame, point_history)
            debug_frame = draw_info(debug_frame, fps, mode, number)

            # 还原初始窗口显示
            cv.imshow('Hand Gesture Recognition', debug_frame)

    except KeyboardInterrupt:
        pass
    finally:
        # 还原初始资源释放
        cap.release()
        cv.destroyAllWindows()
        print(f"✅ 退出 | 最终帧率：{fps}")
    while True:
        # FPS计算
        fps = cvFpsCalc.get()

        # 按键处理
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC退出
            break
        number, mode = select_mode(key, mode)

        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:  # 帧读取失败则退出
            break
        frame = cv.flip(frame, 1)  # 镜像翻转（符合视觉习惯）
        debug_frame = copy.deepcopy(frame)  # 用于绘制的帧副本

        # 3. 核心逻辑：模拟手部数据生成 + 预处理 + 分类
        brect = calc_bounding_rect(debug_frame)  # 生成边界框
        landmark_list = calc_landmark_list(debug_frame)  # 生成关键点
        pre_landmark = pre_process_landmark(landmark_list)  # 关键点预处理
        pre_point_history = pre_process_point_history(debug_frame, point_history)  # 轨迹点预处理

        # 手势分类
        hand_sign_id = keypoint_classifier(pre_landmark)  # 关键点分类
        # 记录指尖轨迹（点手势时记录8号点，否则记录无效点）
        point_history.append(landmark_list[8] if hand_sign_id == 7 else [0, 0])

        # 轨迹手势分类（缓存满16*2个点时分类）
        if not ret:
            break
        frame = cv.flip(frame, 1)  # 镜像显示
        debug_frame = copy.deepcopy(frame)

        # 【核心修改】跳过真实检测，直接模拟手部数据
        brect = calc_bounding_rect(debug_frame)
        landmark_list = calc_landmark_list(debug_frame)

        # 预处理
        pre_landmark = pre_process_landmark(landmark_list)
        pre_point_history = pre_process_point_history(debug_frame, point_history)

        # 手势分类
        hand_sign_id = keypoint_classifier(pre_landmark)
        point_history.append(landmark_list[8] if hand_sign_id == 7 else [0, 0])

        # 手指手势分类
        finger_gesture_id = 0
        if len(pre_point_history) == history_length * 2:
            finger_gesture_id = point_history_classifier(pre_point_history)
        finger_gesture_history.append(finger_gesture_id)
        # 取最频繁的手势分类结果（防抖）
        most_common = Counter(finger_gesture_history).most_common(1)

        # 4. UI绘制
        debug_frame = draw_bounding_rect(debug_frame, brect)  # 绘制边界框
        debug_frame = draw_landmarks(debug_frame, landmark_list)  # 绘制关键点
        debug_frame = draw_info_text(  # 绘制手势信息
        most_common = Counter(finger_gesture_history).most_common(1)

        # 绘制UI
        debug_frame = draw_bounding_rect(debug_frame, brect)
        debug_frame = draw_landmarks(debug_frame, landmark_list)
        debug_frame = draw_info_text(
            debug_frame, brect,
            keypoint_labels[hand_sign_id] if hand_sign_id < len(keypoint_labels) else "Unknown",
            point_history_labels[most_common[0][0]] if most_common else "Unknown"
        )
        debug_frame = draw_point_history(debug_frame, point_history)  # 绘制轨迹
        debug_frame = draw_info(debug_frame, fps, mode, number)  # 绘制全局信息

        # 显示画面
        cv.imshow('Hand Gesture Recognition (ESC to exit)', debug_frame)

    # 3. 资源释放

        # 绘制辅助信息
        debug_frame = draw_point_history(debug_frame, point_history)
        debug_frame = draw_info(debug_frame, fps, mode, number)

        # 显示窗口
        cv.imshow('Hand Gesture Recognition (ESC to exit)', debug_frame)

    # 释放资源
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
