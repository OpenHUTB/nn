import cv2
import numpy as np
import math

# 尝试导入 MediaPipe（可选）
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    mp = None


class GestureDetector:
    """
    纯 OpenCV 手势检测器
    基于肤色检测和轮廓分析
    """
    
    def __init__(self):
        """初始化手势检测器"""
        # 肤色检测的颜色范围 (HSV)
        self.skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        
        # 手势到控制指令的映射
        self.gesture_commands = {
            "open_palm": "takeoff",    # 张开手掌 - 起飞
            "closed_fist": "land",     # 握拳 - 降落
            "pointing_up": "up",       # 食指上指 - 上升
            "pointing_down": "down",   # 食指向下 - 下降
            "victory": "forward",      # 胜利手势 - 前进
            "thumb_up": "backward",   # 大拇指 - 后退
            "thumb_down": "stop",      # 大拇指向下 - 停止
            "ok_sign": "hover",       # OK手势 - 悬停
            "left_palm": "left",      # 左手掌 - 左移
            "right_palm": "right",    # 右手掌 - 右移
            "three_fingers_left": "turn_left",   # 3指+左手位置 - 左转
            "three_fingers_right": "turn_right"  # 3指+右手位置 - 右转
        }
        
        # 手势序列检测（用于握拳→松开触发起飞）
        self.prev_gesture = None
        self.fist_start_time = None
        self.FIST_TIMEOUT = 1.5  # 握拳后1.5秒内松开才触发起飞
        
        print("[INFO] 使用纯 OpenCV 手势检测器")
    
    def detect_gestures(self, image, simulation_mode=False):
        """
        检测图像中的手势
        
        Args:
            image: 输入图像
            simulation_mode: 是否为仿真模式
            
        Returns:
            processed_image: 处理后的图像
            gesture: 识别到的手势
            confidence: 置信度
            landmarks: 关键点坐标（仿真模式下返回简化数据）
        """
        # 复制图像
        result_image = image.copy()
        height, width = image.shape[:2]
        
        # 肤色检测
        skin_mask = self._detect_skin(image)
        
        # 找轮廓
        contours, hierarchy = cv2.findContours(
            skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        gesture = "no_hand"
        confidence = 0.0
        landmarks_data = None
        
        if contours:
            # 找到最大的轮廓（假设是手）
            max_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(max_contour)
            
            # 过滤太小的轮廓
            min_area = (width * height) * 0.01  # 至少占图像的1%
            if contour_area > min_area:
                # 绘制轮廓
                cv2.drawContours(result_image, [max_contour], -1, (0, 255, 0), 2)
                
                # 分析手势
                gesture, confidence = self._analyze_hand_shape(max_contour, result_image)
                
                # 手势序列检测：握拳→松开在中央区域=起飞
                import time
                current_time = time.time()
                
                if gesture == "open_palm":
                    # 检查是否从握拳状态转换而来
                    if self.prev_gesture == "closed_fist" and self.fist_start_time:
                        time_diff = current_time - self.fist_start_time
                        if time_diff <= self.FIST_TIMEOUT:
                            # 握拳后1.5秒内松开，触发起飞
                            pass  # 保持 gesture = "open_palm"
                        else:
                            # 超时，不触发起飞
                            gesture = "hand_detected"
                            confidence = 0.5
                    else:
                        # 不是从握拳转换而来，不触发起飞
                        gesture = "hand_detected"
                        confidence = 0.5
                
                # 更新状态
                if gesture != "no_hand" and gesture != "hand_detected":
                    if gesture == "closed_fist" and self.prev_gesture != "closed_fist":
                        self.fist_start_time = current_time
                    self.prev_gesture = gesture
                elif gesture == "no_hand":
                    # 手消失，重置状态
                    self.prev_gesture = None
                    self.fist_start_time = None
                
                # 在仿真模式下生成简化关键点
                if simulation_mode:
                    landmarks_data = self._generate_landmarks_from_contour(max_contour)
        
        # 在图像上显示手势信息
        if gesture != "no_hand":
            cv2.putText(result_image, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_image, f"Confidence: {confidence:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示控制指令
            command = self.gesture_commands.get(gesture, "none")
            cv2.putText(result_image, f"Command: {command}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # 调试：显示手掌位置
            if 'cx' in dir() or M["m00"] != 0:
                cv2.circle(result_image, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(result_image, f"Hand X: {cx}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.putText(result_image, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return result_image, gesture, confidence, landmarks_data
    
    def _detect_skin(self, image):
        """检测肤色区域"""
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 肤色掩码
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        
        # 模糊去噪
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        
        return skin_mask
    
    def _analyze_hand_shape(self, contour, image):
        """分析手型，返回手势类型和置信度"""
        # 获取凸包和凸缺陷
        hull = cv2.convexHull(contour, returnPoints=False)
        hull_indices = cv2.convexHull(contour).flatten()
        
        # 计算凸缺陷
        try:
            hull_with_defects = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull_with_defects)
        except:
            defects = None
        
        # 计算手指数量
        finger_count = self._count_fingers(contour, defects)
        
        # 获取图像尺寸和手部中心位置
        height, width = image.shape[:2]
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = width // 2
            cy = height // 2
        
        # 根据手指数量判断手势
        gesture = "hand_detected"
        confidence = 0.5
        
        if finger_count == 0:
            gesture = "closed_fist"
            confidence = 0.85
        elif finger_count == 1:
            gesture = "pointing_up"
            confidence = 0.80
        elif finger_count == 2:
            gesture = "victory"
            confidence = 0.80
        elif finger_count == 3:
            # 3个手指：根据手掌位置判断左转/右转
            if cx < width // 2:
                gesture = "three_fingers_left"
                confidence = 0.75
            else:
                gesture = "three_fingers_right"
                confidence = 0.75
        elif finger_count >= 4:
            # 根据手掌位置区分手势
            # 定义中央区域（图像宽度的中间1/3）
            left_boundary = width // 3
            right_boundary = 2 * width // 3
            
            if cx < left_boundary:
                # 手掌在左侧 → 左移
                gesture = "left_palm"
                confidence = 0.75
            elif cx > right_boundary:
                # 手掌在右侧 → 右移
                gesture = "right_palm"
                confidence = 0.75
            else:
                # 手掌在中央 → 起飞
                gesture = "open_palm"
                confidence = 0.75
        
        # 额外检测拇指
        if gesture == "hand_detected":
            thumb_dir = self._detect_thumb(contour)
            if thumb_dir == "up":
                gesture = "thumb_up"
                confidence = 0.80
            elif thumb_dir == "down":
                gesture = "thumb_down"
                confidence = 0.80
        
        return gesture, confidence
    
    def _count_fingers(self, contour, defects):
        """计算伸出的手指数量"""
        if defects is None:
            return 0
        
        finger_count = 0
        height, width = 480, 640  # 默认尺寸
        
        # 分析凸缺陷
        for i in range(defects.shape[0]):
            s, e, d, _ = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            
            # 计算缺陷点到凸包的距离
            far = tuple(contour[d][0])
            
            # 如果缺陷深度足够大，认为是两个手指之间的空隙
            depth = abs(far[1] - (start[1] + end[1]) / 2)
            if depth > 30:  # 阈值
                finger_count += 1
        
        return max(0, finger_count // 2)
    
    def _detect_thumb(self, contour):
        """检测拇指方向"""
        # 简化实现：检测轮廓最左边的点
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        
        # 获取中心点
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            return None
        
        # 如果最左点在中心左边，认为是拇指
        if leftmost[0] < cx - 20:
            return "up"  # 简化处理
        return None
    
    def _generate_landmarks_from_contour(self, contour):
        """从轮廓生成简化的关键点数据"""
        landmarks = []
        
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 生成5个手指尖的简化位置
        for i in range(5):
            # 简化的手指位置
            finger_x = x + w * (0.2 + i * 0.15)
            finger_y = y
            landmarks.append({
                'x': finger_x / 640,
                'y': finger_y / 480,
                'z': 0
            })
        
        # 添加手掌中心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        # 添加掌指关节位置
        for i in range(5):
            mcp_x = x + w * (0.2 + i * 0.15)
            mcp_y = y + h * 0.3
            landmarks.append({
                'x': mcp_x / 640,
                'y': mcp_y / 480,
                'z': 0
            })
        
        # 填充到21个关键点
        while len(landmarks) < 21:
            landmarks.append({'x': 0, 'y': 0, 'z': 0})
        
        return landmarks[:21]
    
    def get_command(self, gesture):
        """根据手势获取控制指令"""
        return self.gesture_commands.get(gesture, "none")
    
    def get_gesture_intensity(self, landmarks, gesture_type):
        """获取手势强度"""
        return 0.5  # 默认强度
    
    def get_hand_position(self, landmarks):
        """获取手部位置"""
        if not landmarks or len(landmarks) < 21:
            return None
        
        x_coords = [p['x'] for p in landmarks if p['x'] > 0]
        y_coords = [p['y'] for p in landmarks if p['y'] > 0]
        
        if not x_coords or not y_coords:
            return None
        
        return {
            'center_x': sum(x_coords) / len(x_coords),
            'center_y': sum(y_coords) / len(y_coords),
            'width': max(x_coords) - min(x_coords) if x_coords else 0,
            'height': max(y_coords) - min(y_coords) if y_coords else 0,
            'bbox': (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        }
    
    def release(self):
        """释放资源"""
        pass
