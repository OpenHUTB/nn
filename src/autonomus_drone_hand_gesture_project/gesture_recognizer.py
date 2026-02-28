"""
手势识别器模块
负责识别和分析手势
作者: xiaoshiyuan888
优化版本：改进手势历史记录和平滑算法
"""

import time
import math
import numpy as np
import cv2
from collections import deque, Counter


class EnhancedGestureRecognizer:
    """增强的手势识别器 - 支持性能模式选择"""

    def __init__(self, speech_manager=None, config=None):
        self.speech_manager = speech_manager
        self.config = config

        # 加载性能模式配置
        self.performance_mode = config.get_current_performance_mode()
        self.mode_config = config.get_performance_mode_config(self.performance_mode)

        # 根据性能模式初始化参数
        self.history_size = self.mode_config['history_size']
        self.detection_interval = self.mode_config['detection_interval']
        self.smooth_frames = self.mode_config['smooth_frames']
        self.min_confidence = self.mode_config['min_confidence']
        self.resize_factor = self.mode_config['resize_factor']

        # 改进的手势历史和平滑 - 增加时间戳和权重
        self.gesture_history = deque(maxlen=self.history_size)
        self.confidence_history = deque(maxlen=self.history_size)
        self.position_history = deque(maxlen=self.history_size)
        self.timestamp_history = deque(maxlen=self.history_size)  # 新增：时间戳历史
        self.weight_history = deque(maxlen=self.history_size)     # 新增：权重历史

        self.current_gesture = "Waiting"
        self.current_confidence = 0.0

        # 手势状态追踪
        self.gesture_state = "none"
        self.gesture_stability_counter = 0
        self.last_stable_gesture = "Waiting"
        self.gesture_active_frames = 0
        self.last_gesture_change_time = 0

        # 记录上次播报的手势
        self.last_announced_gesture = None
        self.last_announced_time = 0
        self.last_hand_status_time = 0
        self.gesture_announce_interval = 2.0

        # 手部跟踪和状态
        self.last_hand_position = None
        self.hand_tracking = False
        self.track_window = None
        self.hand_states = deque(maxlen=15)
        self.hand_detected_frames = 0
        self.hand_lost_frames = 0

        # 性能统计
        self.process_times = deque(maxlen=30)
        self.frame_counter = 0
        self.last_performance_report = 0

        # 手势颜色映射
        self.gesture_colors = {
            "Stop": (0, 0, 255),
            "Forward": (0, 255, 0),
            "Up": (255, 255, 0),
            "Down": (255, 0, 255),
            "Left": (255, 165, 0),
            "Right": (0, 165, 255),
            "Waiting": (200, 200, 200),
            "Error": (255, 0, 0),
            "Hover": (255, 255, 255),
            "Grab": (255, 0, 255),
            "Release": (0, 255, 0),
            "RotateCW": (0, 255, 255),
            "RotateCCW": (255, 255, 0),
            "TakePhoto": (255, 165, 0),
            "ReturnHome": (0, 128, 128),
            "AutoFlight": (128, 0, 128),
        }

        # 手势到语音的映射
        self.gesture_speech_map = {
            "Stop": "gesture_stop",
            "Forward": "gesture_forward",
            "Up": "gesture_up",
            "Down": "gesture_down",
            "Left": "gesture_left",
            "Right": "gesture_right",
            "Waiting": "gesture_waiting",
            "Error": "gesture_error",
            "Hover": "gesture_hover",
            "Grab": "gesture_grab",
            "Release": "gesture_release",
            "RotateCW": "gesture_rotate_cw",
            "RotateCCW": "gesture_rotate_ccw",
            "TakePhoto": "gesture_photo",
            "ReturnHome": "gesture_return_home",
            "AutoFlight": "gesture_auto_flight",
        }

        # 手势状态颜色
        self.state_colors = {
            "none": (100, 100, 100),
            "starting": (255, 165, 0),
            "active": (0, 255, 0),
            "ending": (255, 0, 0),
        }

        # 根据性能模式初始化背景减除器
        self.bg_subtractor = None
        if self.mode_config['background_subtraction_enabled']:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=100, varThreshold=25, detectShadows=True
            )

        # 形态学操作核
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 性能监控
        self.avg_process_time = 0
        self.frame_rate = 0
        self.last_fps_check = time.time()

        # 存储手部数据用于轨迹记录
        self.last_hand_data = None

        # 性能模式信息
        self.performance_mode_color = self.mode_config['color']
        self.performance_mode_name = self.mode_config['name']

        # UI渲染器引用（稍后设置）
        self.ui_renderer = None

        print(f"✓ 增强的手势识别器已初始化 - 性能模式: {self.performance_mode_name}")

    def set_ui_renderer(self, ui_renderer):
        """设置UI渲染器"""
        self.ui_renderer = ui_renderer

    def set_performance_mode(self, mode):
        """设置性能模式"""
        self.performance_mode = mode
        self.mode_config = self.config.get_performance_mode_config(mode)

        # 更新参数
        self.history_size = self.mode_config['history_size']
        self.detection_interval = self.mode_config['detection_interval']
        self.smooth_frames = self.mode_config['smooth_frames']
        self.min_confidence = self.mode_config['min_confidence']
        self.resize_factor = self.mode_config['resize_factor']

        # 更新背景减除器
        if self.mode_config['background_subtraction_enabled'] and self.bg_subtractor is None:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=100, varThreshold=25, detectShadows=True
            )
        elif not self.mode_config['background_subtraction_enabled']:
            self.bg_subtractor = None

        # 更新队列大小
        self.gesture_history = deque(maxlen=self.history_size)
        self.confidence_history = deque(maxlen=self.history_size)
        self.position_history = deque(maxlen=self.history_size)
        self.timestamp_history = deque(maxlen=self.history_size)
        self.weight_history = deque(maxlen=self.history_size)

        # 更新显示信息
        self.performance_mode_color = self.mode_config['color']
        self.performance_mode_name = self.mode_config['name']

        print(f"✓ 切换到性能模式: {self.performance_mode_name}")

    def get_skin_mask(self, frame):
        """获取肤色掩码"""
        h_low = self.config.get('gesture', 'skin_lower_h')
        h_high = self.config.get('gesture', 'skin_upper_h')
        s_low = self.config.get('gesture', 'skin_lower_s')
        s_high = self.config.get('gesture', 'skin_upper_s')
        v_low = self.config.get('gesture', 'skin_lower_v')
        v_high = self.config.get('gesture', 'skin_upper_v')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper_skin = np.array([h_high, s_high, v_high], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        return skin_mask, hsv

    def enhance_skin_detection(self, frame, skin_mask):
        """增强肤色检测"""
        if not self.mode_config['skin_detection_enabled']:
            return skin_mask

        if self.bg_subtractor is not None:
            fg_mask = self.bg_subtractor.apply(frame)
            combined_mask = cv2.bitwise_and(skin_mask, fg_mask)
        else:
            combined_mask = skin_mask

        # 根据性能模式决定形态学操作次数
        if self.performance_mode == 'accurate':
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
            combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
        elif self.performance_mode == 'balanced':
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
            combined_mask = cv2.GaussianBlur(combined_mask, (3, 3), 0)

        return combined_mask

    def preprocess_frame(self, frame):
        """预处理帧"""
        if self.resize_factor != 1.0:
            new_width = int(frame.shape[1] * self.resize_factor)
            new_height = int(frame.shape[0] * self.resize_factor)
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized_frame
        return frame

    def find_best_hand_contour(self, mask, frame):
        """找到最佳的手部轮廓"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0.0

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        best_contour = None
        best_score = 0.0
        min_area = self.config.get('gesture', 'min_hand_area') * (self.resize_factor ** 2)
        max_area = self.config.get('gesture', 'max_hand_area') * (self.resize_factor ** 2)

        # 根据性能模式调整检查的轮廓数量
        max_contours = 3 if self.performance_mode != 'fast' else 1

        for contour in contours[:max_contours]:
            area = cv2.contourArea(contour)

            if area < min_area or area > max_area:
                continue

            score = self.rate_contour(contour, frame.shape)

            if score > best_score:
                best_score = score
                best_contour = contour

        return best_contour, best_score

    def rate_contour(self, contour, frame_shape):
        """评估轮廓作为手部的可能性"""
        score = 0.0
        area = cv2.contourArea(contour)
        min_area = self.config.get('gesture', 'min_hand_area') * (self.resize_factor ** 2)
        max_area = self.config.get('gesture', 'max_hand_area') * (self.resize_factor ** 2)

        if min_area < area < max_area:
            area_ratio = min(area / max_area, 1.0)
            score += area_ratio * 0.3

        perimeter = cv2.arcLength(contour, True)
        if perimeter > 100:
            score += 0.2

        if area > 0:
            compactness = perimeter ** 2 / area
            if 12 < compactness < 25:
                compactness_score = 1.0 - abs(compactness - 18) / 6
                score += compactness_score * 0.3

        x, y, w, h = cv2.boundingRect(contour)
        if h > 0:
            aspect_ratio = w / h
            if 0.4 < aspect_ratio < 2.5:
                aspect_score = 1.0 - abs(aspect_ratio - 1.0) / 1.5
                score += aspect_score * 0.2

        return score

    def analyze_hand_features(self, contour, frame_shape):
        """分析手部特征"""
        if contour is None:
            return None, 0.0

        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, 0.0

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        palm_center = (cx, cy)
        palm_radius = int(w * self.config.get('gesture', 'palm_circle_radius_ratio'))

        # 根据性能模式调整轮廓简化程度
        epsilon = self.mode_config['contour_simplify_epsilon'] * cv2.arcLength(contour, True)
        fingers, fingertips, defects = self.analyze_fingers(contour, palm_center, palm_radius, epsilon)

        direction = self.calculate_hand_direction(contour, cx, cy)
        h_img, w_img = frame_shape[:2]
        norm_x = cx / w_img
        norm_y = cy / h_img
        confidence = self.calculate_confidence(area, len(fingers), len(contour), bbox_area)

        result = {
            'contour': contour,
            'center': (cx, cy),
            'bbox': (x, y, x + w, y + h),
            'fingers': fingers,
            'fingertips': fingertips,
            'defects': defects,
            'palm_center': palm_center,
            'palm_radius': palm_radius,
            'direction': direction,
            'area': area,
            'position': (norm_x, norm_y),
            'bbox_size': (w, h),
            'confidence': confidence
        }

        return result, confidence

    def analyze_fingers(self, contour, palm_center, palm_radius, epsilon):
        """分析手指"""
        approx = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(approx, returnPoints=False)

        if hull is None or len(hull) < 3:
            return [], [], []

        defects = cv2.convexityDefects(approx, hull)
        fingers = []
        fingertips = []
        defect_points = []

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])

                start_dist = np.linalg.norm(np.array(start) - np.array(palm_center))
                end_dist = np.linalg.norm(np.array(end) - np.array(palm_center))
                far_dist = np.linalg.norm(np.array(far) - np.array(palm_center))

                if (start_dist > palm_radius * 1.2 and
                        end_dist > palm_radius * 1.2 and
                        d > self.config.get('gesture', 'defect_distance_threshold') * 256):

                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                    if b * c != 0:
                        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                        angle_degrees = math.degrees(angle)

                        if angle_degrees < 90:
                            finger = {
                                'start': start,
                                'end': end,
                                'far': far,
                                'depth': d,
                                'angle': angle_degrees
                            }
                            fingers.append(finger)

                            for point in [start, end]:
                                if point not in fingertips:
                                    point_dist = np.linalg.norm(np.array(point) - np.array(palm_center))
                                    if point_dist > palm_radius * 1.5:
                                        fingertips.append(point)

                            defect_points.append((start, end, far, d))

        if len(fingertips) == 0:
            hull_points = cv2.convexHull(approx, returnPoints=True)
            if len(hull_points) > 0:
                hull_points = hull_points.reshape(-1, 2)

                for point in hull_points:
                    point_tuple = tuple(point)
                    point_dist = np.linalg.norm(point - np.array(palm_center))
                    if point_dist > palm_radius * 1.5 and point_tuple not in fingertips:
                        fingertips.append(point_tuple)

        fingertips = fingertips[:5]

        return fingers, fingertips, defect_points

    def calculate_hand_direction(self, contour, cx, cy):
        """计算手部方向"""
        if len(contour) < 5:
            return 0.0

        points = contour.reshape(-1, 2).astype(np.float32)
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean)
        direction = math.degrees(math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]))

        return direction

    def calculate_confidence(self, area, finger_count, contour_length, bbox_area):
        """计算手势置信度"""
        confidence = 0.5
        min_area = self.config.get('gesture', 'min_hand_area') * (self.resize_factor ** 2)
        max_area = self.config.get('gesture', 'max_hand_area') * (self.resize_factor ** 2)

        if min_area < area < max_area:
            area_norm = (area - min_area) / (max_area - min_area)
            confidence += area_norm * 0.2

        if 0 <= finger_count <= 5:
            confidence += 0.2

        if contour_length > 200:
            confidence += 0.1

        if bbox_area > 0:
            fill_ratio = area / bbox_area
            if 0.2 < fill_ratio < 0.8:
                fill_score = 1.0 - abs(fill_ratio - 0.5) / 0.3
                confidence += fill_score * 0.1

        return min(confidence, 1.0)

    def recognize_gesture_improved(self, hand_data):
        """改进的手势识别逻辑"""
        if hand_data is None:
            return "Waiting", 0.3

        finger_count = len(hand_data.get('fingers', []))
        fingertips = hand_data.get('fingertips', [])
        norm_x, norm_y = hand_data['position']
        direction = hand_data.get('direction', 0.0)
        confidence = hand_data['confidence']
        w, h = hand_data['bbox_size']
        aspect_ratio = w / h if h > 0 else 1.0

        # 根据手指数量分类
        if finger_count == 0:
            if len(fingertips) == 0:
                return "Stop", confidence * 0.9
            else:
                return "Stop", confidence * 0.7

        elif finger_count == 1:
            if len(fingertips) >= 1:
                cx, cy = hand_data['center']
                fingertip = fingertips[0]
                dx = fingertip[0] - cx
                dy = fingertip[1] - cy

                if abs(dx) > abs(dy):
                    if dx > 0:
                        return "Right", confidence * 0.8
                    else:
                        return "Left", confidence * 0.8
                else:
                    if dy < 0:
                        return "Up", confidence * 0.8
                    else:
                        return "Forward", confidence * 0.8
            return "Forward", confidence * 0.7

        elif finger_count == 2:
            return "Forward", confidence * 0.7

        elif finger_count == 3:
            if -45 <= direction <= 45:
                if direction > 0:
                    return "Right", confidence * 0.7
                else:
                    return "Left", confidence * 0.7
            else:
                if direction > 0:
                    return "Down", confidence * 0.7
                else:
                    return "Up", confidence * 0.7

        elif finger_count >= 4:
            if norm_x < 0.4:
                return "Left", confidence * 0.8
            elif norm_x > 0.6:
                return "Right", confidence * 0.8
            elif norm_y < 0.4:
                return "Up", confidence * 0.8
            elif norm_y > 0.6:
                return "Down", confidence * 0.8
            else:
                if -45 <= direction <= 45:
                    return "Forward", confidence * 0.7
                else:
                    return "Stop", confidence * 0.7

        return "Waiting", confidence * 0.5

    def smooth_gesture_enhanced(self, new_gesture, new_confidence, hand_data):
        """增强的手势平滑处理 - 使用加权平均"""
        current_time = time.time()

        # 检查手势冷却时间
        if current_time - self.last_gesture_change_time < self.config.get('gesture', 'gesture_cooldown'):
            return self.current_gesture, self.current_confidence

        # 计算当前手势的权重（基于置信度和时间）
        # 权重 = 置信度 * 时间衰减因子
        if len(self.timestamp_history) > 0:
            time_since_last = current_time - self.timestamp_history[-1]
            time_weight = max(0.1, 1.0 - time_since_last / 5.0)  # 时间衰减因子
        else:
            time_weight = 1.0

        weight = new_confidence * time_weight

        # 添加到历史
        self.gesture_history.append(new_gesture)
        self.confidence_history.append(new_confidence)
        self.timestamp_history.append(current_time)
        self.weight_history.append(weight)

        if hand_data is not None:
            self.position_history.append(hand_data['position'])

        # 如果历史数据不足，返回当前手势
        if len(self.gesture_history) < 3:
            self.current_gesture = new_gesture
            self.current_confidence = new_confidence
            return self.current_gesture, self.current_confidence

        # 使用加权投票算法确定最佳手势
        gesture_scores = {}
        total_weight = 0

        # 为历史中的每个手势计算加权分数
        for i in range(len(self.gesture_history)):
            gesture = self.gesture_history[i]
            conf = self.confidence_history[i]
            wgt = self.weight_history[i]

            # 计算时间衰减因子（越近的权重越高）
            time_factor = 1.0
            if i < len(self.timestamp_history):
                time_diff = current_time - self.timestamp_history[i]
                time_factor = max(0.1, 1.0 - time_diff / 3.0)  # 3秒内衰减到0.1

            # 计算综合权重
            final_weight = wgt * time_factor * (i / len(self.gesture_history) + 0.5)  # 越近权重越高

            if gesture not in gesture_scores:
                gesture_scores[gesture] = 0
            gesture_scores[gesture] += final_weight * conf
            total_weight += final_weight

        if total_weight == 0:
            self.current_gesture = new_gesture
            self.current_confidence = new_confidence
            return self.current_gesture, self.current_confidence

        # 找出得分最高的手势
        best_gesture = max(gesture_scores.items(), key=lambda x: x[1])[0]
        best_score = gesture_scores[best_gesture] / total_weight

        # 计算位置稳定性
        position_stability = 1.0
        if len(self.position_history) >= 2 and hand_data is not None:
            current_pos = hand_data['position']
            prev_pos = self.position_history[-2] if len(self.position_history) >= 2 else current_pos
            position_diff = math.sqrt((current_pos[0] - prev_pos[0]) ** 2 + (current_pos[1] - prev_pos[1]) ** 2)
            position_stability = max(0, 1.0 - position_diff * 5.0)

        # 检查手势稳定性阈值
        stability_threshold = self.mode_config['gesture_stability_threshold']
        transition_threshold = self.config.get('gesture', 'transition_threshold')
        position_weight = self.config.get('gesture', 'position_stability_weight')

        # 计算综合稳定性得分
        gesture_stability = best_score
        overall_stability = gesture_stability * (1.0 - position_weight) + position_stability * position_weight

        # 手势状态转换逻辑
        if overall_stability >= transition_threshold:
            if best_gesture != self.last_stable_gesture:
                self.gesture_stability_counter += 1
            else:
                self.gesture_stability_counter = max(0, self.gesture_stability_counter - 1)

            if self.gesture_stability_counter >= stability_threshold:
                self.current_gesture = best_gesture

                # 计算加权平均置信度
                weighted_conf_sum = 0
                weight_sum = 0
                for i in range(len(self.gesture_history)):
                    if self.gesture_history[i] == best_gesture:
                        weighted_conf_sum += self.confidence_history[i] * self.weight_history[i]
                        weight_sum += self.weight_history[i]

                if weight_sum > 0:
                    self.current_confidence = weighted_conf_sum / weight_sum
                else:
                    self.current_confidence = best_score

                self.last_stable_gesture = best_gesture
                self.gesture_stability_counter = 0
                self.last_gesture_change_time = current_time
        else:
            self.gesture_stability_counter = max(0, self.gesture_stability_counter - 2)

        return self.current_gesture, self.current_confidence

    def update_gesture_state(self, hand_data, gesture, confidence):
        """更新手势状态"""
        current_time = time.time()

        if hand_data is None:
            # 手部丢失
            self.hand_lost_frames += 1
            self.hand_detected_frames = max(0, self.hand_detected_frames - 1)

            if self.hand_lost_frames > 10 and self.gesture_state != "none":
                self.gesture_state = "none"
                if (self.speech_manager and
                        self.speech_manager.enabled and
                        current_time - self.last_hand_status_time > 3.0):
                    self.speech_manager.speak('hand_lost', immediate=True)
                    self.last_hand_status_time = current_time
            return

        # 手部检测到
        self.hand_detected_frames += 1
        self.hand_lost_frames = 0

        # 检查手部大小和位置
        hand_area = hand_data['area']
        min_area = self.config.get('gesture', 'min_hand_area') * (self.resize_factor ** 2)
        max_area = self.config.get('gesture', 'max_hand_area') * (self.resize_factor ** 2)

        # 提供手部位置反馈
        if (self.speech_manager and
                self.speech_manager.enabled and
                current_time - self.last_hand_status_time > 5.0):

            if hand_area < min_area * 0.8:
                self.speech_manager.speak('move_closer', immediate=True)
                self.last_hand_status_time = current_time
            elif hand_area > max_area * 1.2:
                self.speech_manager.speak('move_away', immediate=True)
                self.last_hand_status_time = current_time
            elif self.hand_detected_frames == 5:
                self.speech_manager.speak('hand_detected', immediate=True)
                self.last_hand_status_time = current_time

        # 手势状态转换
        if self.gesture_state == "none" and gesture != "Waiting" and confidence > 0.6:
            self.gesture_state = "starting"
            self.gesture_active_frames = 0
            if (self.speech_manager and
                    self.speech_manager.enabled):
                self.speech_manager.speak('gesture_start', immediate=True)

        elif self.gesture_state == "starting":
            self.gesture_active_frames += 1
            if self.gesture_active_frames >= self.config.get('speech', 'gesture_start_threshold'):
                self.gesture_state = "active"
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('gesture_stable', immediate=True)

        elif self.gesture_state == "active":
            if gesture == "Waiting" or confidence < 0.5:
                self.gesture_active_frames = max(0, self.gesture_active_frames - 2)
                if self.gesture_active_frames <= 0:
                    self.gesture_state = "ending"
            else:
                self.gesture_active_frames = min(20, self.gesture_active_frames + 1)

        elif self.gesture_state == "ending":
            self.gesture_active_frames -= 1
            if self.gesture_active_frames <= 0:
                self.gesture_state = "none"
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('gesture_end', immediate=True)

    def visualize_detection(self, frame, hand_data, gesture, confidence):
        """可视化检测结果"""
        if hand_data is None:
            return frame

        show_contours = self.config.get('display', 'show_contours')
        show_bbox = self.config.get('display', 'show_bbox')
        show_fingertips = self.config.get('display', 'show_fingertips')
        show_palm_center = self.config.get('display', 'show_palm_center')
        show_hand_direction = self.config.get('display', 'show_hand_direction')
        show_debug_info = self.config.get('display', 'show_debug_info')
        show_gesture_history = self.config.get('display', 'show_gesture_history')
        show_stability_indicator = self.config.get('display', 'show_stability_indicator')

        # 绘制轮廓
        if show_contours and 'contour' in hand_data:
            cv2.drawContours(frame, [hand_data['contour']], -1, (0, 255, 0), 2)

        # 绘制边界框
        if show_bbox and 'bbox' in hand_data:
            x1, y1, x2, y2 = hand_data['bbox']
            color = self.gesture_colors.get(gesture, (255, 255, 255))

            # 根据手势状态调整边界框颜色
            state_color = self.state_colors.get(self.gesture_state, color)
            if self.gesture_state != "none":
                color = state_color

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 显示手势标签
            label = f"{gesture}"
            if self.config.get('display', 'show_confidence'):
                # 使用UI渲染器绘制文本，避免字体问题
                if self.ui_renderer:
                    # 首先绘制手势名称
                    gesture_text = f"{gesture}"

                    # 计算文本位置
                    (text_width, text_height), baseline = cv2.getTextSize(
                        gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )

                    # 绘制文本背景
                    cv2.rectangle(frame,
                                  (x1, y1 - text_height - 10),
                                  (x1 + text_width, y1),
                                  color, -1)

                    # 使用UI渲染器绘制手势名称
                    frame = self.ui_renderer.draw_text(frame, gesture_text,
                                                      (x1, y1 - 5),
                                                      size=16, color=(255, 255, 255))

                    # 绘制置信度（使用UI渲染器确保正确显示）
                    confidence_text = f"{confidence:.0%}"
                    # 计算置信度文本位置
                    frame = self.ui_renderer.draw_text(frame, confidence_text,
                                                      (x1 + text_width + 5, y1 - 5),
                                                      size=16, color=(255, 255, 255))
                else:
                    # 备用方案
                    label += f" ({confidence:.0%})"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )

                    cv2.rectangle(frame,
                                  (x1, y1 - text_height - 10),
                                  (x1 + text_width, y1),
                                  color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # 不使用置信度显示
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )

                cv2.rectangle(frame,
                              (x1, y1 - text_height - 10),
                              (x1 + text_width, y1),
                              color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 显示手势状态
            if self.gesture_state != "none":
                state_text = {"starting": "Starting", "active": "Active", "ending": "Ending"}
                state_label = f" [{state_text.get(self.gesture_state, '')}]"

                # 使用UI渲染器绘制状态文本
                if self.ui_renderer:
                    frame = self.ui_renderer.draw_text(frame, state_label,
                                                      (x1, y1 - 30),
                                                      size=14, color=color)
                else:
                    cv2.putText(frame, state_label, (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 绘制手掌中心
        if show_palm_center and 'palm_center' in hand_data:
            cx, cy = hand_data['palm_center']
            palm_radius = hand_data.get('palm_radius', 20)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), palm_radius, (0, 0, 255), 1)
            # 使用UI渲染器绘制"Palm"文本
            if self.ui_renderer:
                frame = self.ui_renderer.draw_text(frame, "Palm",
                                                  (cx + 10, cy),
                                                  size=14, color=(0, 0, 255))
            else:
                cv2.putText(frame, "Palm", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 绘制指尖
        if show_fingertips and 'fingertips' in hand_data:
            for i, point in enumerate(hand_data['fingertips']):
                cv2.circle(frame, point, 4, (255, 0, 0), -1)
                # 使用UI渲染器绘制指尖编号
                if self.ui_renderer:
                    frame = self.ui_renderer.draw_text(frame, f"F{i+1}",
                                                      (point[0] + 5, point[1]),
                                                      size=14, color=(255, 0, 0))
                else:
                    cv2.putText(frame, f"F{i+1}", (point[0] + 5, point[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 绘制手部方向
        if show_hand_direction and 'direction' in hand_data and 'center' in hand_data:
            cx, cy = hand_data['center']
            direction = hand_data['direction']
            length = 50

            dx = length * math.cos(math.radians(direction))
            dy = length * math.sin(math.radians(direction))

            end_point = (int(cx + dx), int(cy + dy))
            cv2.arrowedLine(frame, (cx, cy), end_point, (255, 255, 0), 2)

            angle_text = f"Dir: {direction:.0f}°"
            # 使用UI渲染器绘制方向文本
            if self.ui_renderer:
                frame = self.ui_renderer.draw_text(frame, angle_text,
                                                  (cx, cy - 20),
                                                  size=14, color=(255, 255, 0))
            else:
                cv2.putText(frame, angle_text, (cx, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 绘制稳定性指示器
        if show_stability_indicator:
            h, w = frame.shape[:2]
            indicator_x = w - 100
            indicator_y = 30

            # 绘制稳定性背景
            cv2.rectangle(frame, (indicator_x, indicator_y),
                          (indicator_x + 80, indicator_y + 15), (50, 50, 50), -1)

            # 计算稳定性指示条长度
            stability_level = min(1.0,
                                  self.gesture_stability_counter / self.mode_config['gesture_stability_threshold'])
            bar_length = int(70 * stability_level)

            # 根据稳定性级别选择颜色
            if stability_level > 0.7:
                bar_color = (0, 255, 0)
            elif stability_level > 0.4:
                bar_color = (255, 165, 0)
            else:
                bar_color = (255, 0, 0)

            # 绘制稳定性指示条
            cv2.rectangle(frame, (indicator_x + 5, indicator_y + 5),
                          (indicator_x + 5 + bar_length, indicator_y + 10), bar_color, -1)

            # 绘制稳定性文本
            if self.ui_renderer:
                frame = self.ui_renderer.draw_text(frame, "稳定度",
                                                  (indicator_x, indicator_y - 5),
                                                  size=14, color=(255, 255, 255))
            else:
                cv2.putText(frame, "稳定度", (indicator_x, indicator_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 绘制手势历史
        if show_gesture_history and len(self.gesture_history) > 0:
            h, w = frame.shape[:2]
            history_y = h - 50

            # 绘制历史背景
            cv2.rectangle(frame, (10, history_y - 20), (200, history_y + 10), (0, 0, 0), -1)

            # 使用UI渲染器绘制历史标题
            if self.ui_renderer:
                frame = self.ui_renderer.draw_text(frame, "手势历史:",
                                                  (15, history_y),
                                                  size=14, color=(255, 255, 255))
            else:
                cv2.putText(frame, "手势历史:", (15, history_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 显示最近几个手势
            recent_gestures = list(self.gesture_history)[-5:] if len(self.gesture_history) >= 5 else list(
                self.gesture_history)
            for i, gest in enumerate(recent_gestures):
                color = self.gesture_colors.get(gest, (255, 255, 255))
                # 使用UI渲染器绘制手势历史
                if self.ui_renderer:
                    frame = self.ui_renderer.draw_text(frame, gest[0],
                                                      (85 + i * 20, history_y),
                                                      size=16, color=color)
                else:
                    cv2.putText(frame, gest[0], (85 + i * 20, history_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示调试信息
        if show_debug_info:
            finger_count = len(hand_data.get('fingers', []))
            finger_text = f"Fingers: {finger_count}"
            # 使用UI渲染器绘制调试信息
            if self.ui_renderer:
                frame = self.ui_renderer.draw_text(frame, finger_text,
                                                  (10, frame.shape[0] - 30),
                                                  size=14, color=(255, 255, 255))
            else:
                cv2.putText(frame, finger_text, (10, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            pos_text = f"Pos: ({hand_data['position'][0]:.2f}, {hand_data['position'][1]:.2f})"
            if self.ui_renderer:
                frame = self.ui_renderer.draw_text(frame, pos_text,
                                                  (10, frame.shape[0] - 10),
                                                  size=14, color=(255, 255, 255))
            else:
                cv2.putText(frame, pos_text, (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            state_text = f"State: {self.gesture_state}"
            if self.ui_renderer:
                frame = self.ui_renderer.draw_text(frame, state_text,
                                                  (150, frame.shape[0] - 30),
                                                  size=14, color=(255, 255, 255))
            else:
                cv2.putText(frame, state_text, (150, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            stability_text = f"Stability: {self.gesture_stability_counter}"
            if self.ui_renderer:
                frame = self.ui_renderer.draw_text(frame, stability_text,
                                                  (150, frame.shape[0] - 10),
                                                  size=14, color=(255, 255, 255))
            else:
                cv2.putText(frame, stability_text, (150, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def recognize(self, frame):
        """识别手势"""
        start_time = time.time()

        try:
            # 预处理帧
            processed_frame = cv2.flip(frame, 1)

            # 根据性能模式调整图像大小
            original_frame = processed_frame.copy()
            if self.resize_factor != 1.0:
                processed_frame = self.preprocess_frame(processed_frame)

            # 每隔几帧检测一次以提高性能
            if self.frame_counter % self.detection_interval != 0:
                self.frame_counter += 1
                return self.current_gesture, self.current_confidence, original_frame

            # 获取肤色掩码
            skin_mask, hsv = self.get_skin_mask(processed_frame)

            # 增强肤色检测
            enhanced_mask = self.enhance_skin_detection(processed_frame, skin_mask)

            # 找到最佳的手部轮廓
            hand_contour, contour_score = self.find_best_hand_contour(enhanced_mask, processed_frame)

            # 分析手部特征
            hand_data, confidence = self.analyze_hand_features(hand_contour, processed_frame.shape)

            # 保存手部数据用于轨迹记录
            self.last_hand_data = hand_data

            # 识别手势
            if hand_data is not None:
                # 校准肤色
                self.config.calibrate_skin_color(processed_frame, enhanced_mask)

                # 校准手部大小
                self.config.calibrate_hand_size(hand_data['area'])

                # 识别手势
                gesture, raw_confidence = self.recognize_gesture_improved(hand_data)
                confidence = max(confidence, raw_confidence)

                # 更新手势状态
                self.update_gesture_state(hand_data, gesture, confidence)

                # 增强的手势平滑
                final_gesture, final_confidence = self.smooth_gesture_enhanced(gesture, confidence, hand_data)
            else:
                gesture, confidence = "Waiting", 0.3
                self.update_gesture_state(None, gesture, confidence)
                final_gesture, final_confidence = gesture, confidence

            # 手势语音播报
            if (self.speech_manager and
                    self.speech_manager.enabled):

                current_time = time.time()

                # 根据置信度提供反馈
                if confidence >= 0.8 and current_time - self.last_hand_status_time > 5.0:
                    self.speech_manager.speak('gesture_good_confidence', immediate=True)
                    self.last_hand_status_time = current_time
                elif confidence < 0.5 and current_time - self.last_hand_status_time > 5.0:
                    self.speech_manager.speak('gesture_low_confidence', immediate=True)
                    self.last_hand_status_time = current_time

                # 手势语音播报
                if (final_gesture != "Waiting" and
                        final_gesture != "Error" and
                        final_gesture != "摄像头错误" and
                        final_confidence >= self.min_confidence and
                        current_time - self.last_announced_time > self.gesture_announce_interval):

                    if final_gesture in self.gesture_speech_map:
                        speech_key = self.gesture_speech_map[final_gesture]
                        self.speech_manager.speak(speech_key)
                    else:
                        self.speech_manager.speak_direct(f"手势{final_gesture}")

                    self.last_announced_gesture = final_gesture
                    self.last_announced_time = current_time

            # 性能报告
            current_time = time.time()
            if current_time - self.last_performance_report > 30.0:
                if (self.speech_manager and
                        self.speech_manager.enabled):

                    if self.avg_process_time < 20:
                        self.speech_manager.speak('performance_good', immediate=True)
                    elif self.avg_process_time > 50:
                        self.speech_manager.speak('performance_warning', immediate=True)

                    self.last_performance_report = current_time

            # 可视化结果
            if hand_data is not None:
                # 需要将坐标转换回原始图像大小
                if self.resize_factor != 1.0:
                    scale_factor = 1.0 / self.resize_factor
                    if 'center' in hand_data:
                        hand_data['center'] = (int(hand_data['center'][0] * scale_factor),
                                               int(hand_data['center'][1] * scale_factor))
                    if 'bbox' in hand_data:
                        x1, y1, x2, y2 = hand_data['bbox']
                        hand_data['bbox'] = (int(x1 * scale_factor), int(y1 * scale_factor),
                                             int(x2 * scale_factor), int(y2 * scale_factor))
                    if 'fingertips' in hand_data:
                        hand_data['fingertips'] = [(int(x * scale_factor), int(y * scale_factor))
                                                   for (x, y) in hand_data['fingertips']]
                    if 'palm_center' in hand_data:
                        hand_data['palm_center'] = (int(hand_data['palm_center'][0] * scale_factor),
                                                    int(hand_data['palm_center'][1] * scale_factor))

                original_frame = self.visualize_detection(
                    original_frame, hand_data, final_gesture, final_confidence
                )
            else:
                original_frame = self.visualize_detection(
                    original_frame, None, final_gesture, final_confidence
                )

            # 更新计数器
            self.frame_counter += 1

            # 计算处理时间
            process_time = (time.time() - start_time) * 1000
            self.process_times.append(process_time)

            # 更新平均处理时间
            if len(self.process_times) > 0:
                self.avg_process_time = np.mean(list(self.process_times))

            # 更新帧率
            current_time = time.time()
            if current_time - self.last_fps_check >= 1.0:
                self.frame_rate = self.frame_counter
                self.frame_counter = 0
                self.last_fps_check = current_time

            return final_gesture, final_confidence, original_frame

        except Exception as e:
            print(f"⚠ 手势识别错误: {e}")
            return "Error", 0.0, frame

    def get_performance_stats(self):
        """获取性能统计"""
        if len(self.process_times) == 0:
            return 0.0, self.frame_rate

        return np.mean(list(self.process_times)), self.frame_rate

    def get_performance_mode_info(self):
        """获取性能模式信息"""
        return {
            'name': self.performance_mode_name,
            'mode': self.performance_mode,
            'color': self.performance_mode_color,
            'detection_interval': self.detection_interval,
            'resize_factor': self.resize_factor,
            'smooth_frames': self.smooth_frames,
            'min_confidence': self.min_confidence
        }

    def set_simulated_gesture(self, gesture):
        """设置模拟的手势"""
        self.current_gesture = gesture
        self.current_confidence = 0.9

        # 模拟手势也触发语音提示
        if (self.speech_manager and
                self.speech_manager.enabled):

            if gesture in self.gesture_speech_map:
                self.speech_manager.speak(self.gesture_speech_map[gesture])
            else:
                self.speech_manager.speak_direct(f"手势{gesture}")

            self.last_announced_gesture = gesture
            self.last_announced_time = time.time()