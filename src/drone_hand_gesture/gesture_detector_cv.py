"""
基于OpenCV的手势检测器（无需MediaPipe依赖）

该模块使用肤色检测和轮廓分析来实现手势识别，
兼容MediaPipe的关键点格式（21个关键点）。

使用方法:
    from gesture_detector_cv import CVGestureDetector
    detector = CVGestureDetector()
"""

import cv2
import numpy as np
import math
from collections import deque


class HandLandmark:
    """模拟MediaPipe的手部关键点数据结构"""
    def __init__(self):
        self.landmark = []  # 21个关键点


class LandmarkPoint:
    """模拟MediaPipe的单个关键点"""
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = 1.0


class CVGestureDetector:
    """
    基于OpenCV的手势检测器
    
    使用肤色检测和轮廓分析来识别手部手势，
    输出与MediaPipe兼容的关键点格式。
    """

    def __init__(self):
        """初始化检测器"""
        # 肤色检测的HSV范围
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 备用的YCrCb肤色范围
        self.lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)

        # 手指尖和指根的索引（在生成的关键点中）
        # 21个关键点的定义（与MediaPipe兼容）
        self.finger_tips = [4, 8, 12, 16, 20]  # 拇指、食、中、无名、小指尖
        self.finger_pips = [2, 6, 10, 14, 18]  # 食指到小指的近端指间关节
        self.finger_mcps = [5, 9, 13, 17]  # 食指到小指的掌指关节
        
        # 手势到控制指令的映射
        self.gesture_commands = {
            "open_palm": "takeoff",
            "closed_fist": "land",
            "pointing_up": "up",
            "pointing_down": "down",
            "victory": "forward",
            "thumb_up": "backward",
            "thumb_down": "stop",
            "ok_sign": "hover"
        }

        # 用于平滑的历史记录
        self.history = deque(maxlen=5)
        self.last_gesture = "no_hand"
        self.last_confidence = 0.0

        # 检测参数
        self.min_area = 2000  # 最小手部区域面积
        self.max_area = 100000  # 最大手部区域面积

    def detect_hand(self, frame):
        """
        检测手部区域
        
        Args:
            frame: BGR格式的图像
            
        Returns:
            mask: 二值化后的手部掩码
            contour: 最大手部轮廓
        """
        # 转换为灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 模糊处理
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 方法1: HSV肤色检测
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # 方法2: YCrCb肤色检测
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask2 = cv2.inRange(ycrcb, self.lower_ycrcb, self.upper_ycrcb)
        
        # 合并两种方法的掩码
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # 轮廓检测
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return mask, None
        
        # 找最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        
        if area < self.min_area:
            return mask, None
            
        return mask, max_contour

    def generate_landmarks(self, contour, frame_shape):
        """
        根据轮廓生成21个关键点（模拟MediaPipe格式）
        
        Args:
            contour: 手部轮廓
            frame_shape: 图像尺寸
            
        Returns:
            hand_landmarks: HandLandmark对象
        """
        h, w = frame_shape[:2]
        hand_landmarks = HandLandmark()
        
        # 计算轮廓的凸包和凸缺陷
        hull = cv2.convexHull(contour, returnPoints=False)
        hull_indices = []
        
        if len(hull) > 0:
            hull_indices = [int(idx[0]) for idx in hull]
        
        try:
            defects = cv2.convexityDefects(contour, hull)
        except:
            defects = None
        
        # 创建关键点容器
        landmarks = []
        
        # 使用轮廓的特征点来估算关键点位置
        # 这里简化处理，根据手掌中心和轮廓边界来估算
        
        # 计算手掌中心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m00"] / M["m00"])
        else:
            cx, cy = w // 2, h // 2
        
        # 根据轮廓创建简化的21点模型
        # MediaPipe的21点定义：
        # 0: 手腕
        # 1-4: 拇指(基底->指尖)
        # 5-8: 食指
        # 9-12: 中指
        # 13-16: 无名指
        # 17-20: 小指
        
        # 获取轮廓边界
        x, y, pw, ph = cv2.boundingRect(contour)
        palm_width = pw
        palm_height = ph
        
        # 生成手腕点 (0)
        landmarks.append(LandmarkPoint(cx / w, cy / h, 0))
        
        # 生成拇指点 (1-4)
        thumb_base_x = x
        thumb_tip_x = x + pw * 0.1
        thumb_y = y + ph * 0.3
        for i in range(4):
            ratio = i / 3
            lx = thumb_base_x + (thumb_tip_x - thumb_base_x) * ratio
            landmarks.append(LandmarkPoint(lx / w, thumb_y / h, 0))
        
        # 生成四指点 (5-20)
        fingers = [(x + pw * 0.25, y),   # 食指
                   (x + pw * 0.45, y),   # 中指
                   (x + pw * 0.65, y),   # 无名指
                   (x + pw * 0.85, y)]   # 小指
        
        for finger_idx, (fx, fy) in enumerate(fingers):
            base_idx = 5 + finger_idx * 4
            
            # MCP (5, 9, 13, 17)
            mcp_x, mcp_y = fx, fy + ph * 0.15
            landmarks.append(LandmarkPoint(mcp_x / w, mcp_y / h, 0))
            
            # PIP (6, 10, 14, 18)
            pip_x, pip_y = fx, fy + ph * 0.35
            landmarks.append(LandmarkPoint(pip_x / w, pip_y / h, 0))
            
            # DIP (7, 11, 15, 19)
            dip_x, dip_y = fx, fy + ph * 0.55
            landmarks.append(LandmarkPoint(dip_x / w, dip_y / h, 0))
            
            # TIP (8, 12, 16, 20)
            tip_x, tip_y = fx, fy
            landmarks.append(LandmarkPoint(tip_x / w, tip_y / h, 0))
        
        hand_landmarks.landmark = landmarks
        return hand_landmarks

    def classify_gesture(self, hand_landmarks):
        """
        分类手势（基于生成的关键点）
        
        Args:
            hand_landmarks: HandLandmark对象
            
        Returns:
            gesture: 手势名称
            confidence: 置信度
        """
        if len(hand_landmarks.landmark) < 21:
            return "no_hand", 0.0
        
        points = hand_landmarks.landmark
        
        # 计算手指伸展情况
        def is_finger_extended(tip_idx, mcp_idx):
            """判断手指是否伸展"""
            tip = points[tip_idx]
            mcp = points[mcp_idx]
            # y坐标越小表示越靠上（伸展）
            return tip.y < mcp.y + 0.1
        
        # 检查各手指
        fingers_extended = {
            'thumb': is_finger_extended(4, 3),
            'index': is_finger_extended(8, 5),
            'middle': is_finger_extended(12, 9),
            'ring': is_finger_extended(16, 13),
            'pinky': is_finger_extended(20, 17)
        }
        
        extended_count = sum(fingers_extended.values())
        
        # 手势分类
        # 1. 张开手掌: 4或5个手指伸展
        is_open_palm = extended_count >= 4
        
        # 2. 握拳: 所有手指弯曲
        is_closed_fist = extended_count <= 1
        
        # 3. 食指上指: 只有食指伸展
        is_pointing_up = (fingers_extended['index'] and 
                         not fingers_extended['middle'] and 
                         not fingers_extended['ring'])
        
        # 4. 食指向下: 只有食指伸展但朝下
        is_pointing_down = (not fingers_extended['index'] and 
                            points[8].y > points[5].y)
        
        # 5. 胜利手势: 食指和中指伸展
        is_victory = (fingers_extended['index'] and 
                     fingers_extended['middle'] and 
                     not fingers_extended['ring'])
        
        # 6. 大拇指向上
        is_thumb_up = fingers_extended['thumb'] and points[4].y < points[5].y
        
        # 7. 大拇指向下
        is_thumb_down = fingers_extended['thumb'] and points[4].y > points[17].y
        
        # 8. OK手势: 食指和拇指接近
        thumb_tip = points[4]
        index_tip = points[8]
        distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + 
                            (thumb_tip.y - index_tip.y)**2)
        is_ok_sign = distance < 0.15
        
        # 根据检测结果返回手势
        if is_open_palm:
            return "open_palm", 0.80
        elif is_closed_fist:
            return "closed_fist", 0.75
        elif is_thumb_up:
            return "thumb_up", 0.85
        elif is_thumb_down:
            return "thumb_down", 0.85
        elif is_ok_sign:
            return "ok_sign", 0.80
        elif is_victory:
            return "victory", 0.75
        elif is_pointing_up:
            return "pointing_up", 0.70
        elif is_pointing_down:
            return "pointing_down", 0.70
        
        return "hand_detected", 0.5

    def detect_gestures(self, image, simulation_mode=False):
        """
        检测图像中的手势
        
        Args:
            image: 输入图像 (BGR格式)
            simulation_mode: 是否为仿真模式
            
        Returns:
            processed_image: 处理后的图像
            gesture: 识别到的手势
            confidence: 置信度
            landmarks_data: 关键点数据
        """
        result_image = image.copy()
        gesture = "no_hand"
        confidence = 0.0
        landmarks_data = None
        
        # 检测手部
        mask, contour = self.detect_hand(image)
        
        if contour is not None:
            # 生成关键点
            hand_landmarks = self.generate_landmarks(contour, image.shape)
            
            # 绘制轮廓和凸包
            cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
            
            # 简单绘制关键点
            landmarks = hand_landmarks.landmark
            for i, lm in enumerate(landmarks):
                x = int(lm.x * image.shape[1])
                y = int(lm.y * image.shape[0])
                cv2.circle(result_image, (x, y), 3, (0, 0, 255), -1)
            
            # 分类手势
            gesture, confidence = self.classify_gesture(hand_landmarks)
            
            # 平滑处理
            self.history.append((gesture, confidence))
            if len(self.history) >= 3:
                from collections import Counter
                gestures_in_history = [g for g, _ in self.history]
                most_common = Counter(gestures_in_history).most_common(1)[0]
                if most_common[1] >= 2:
                    gesture = most_common[0]
                    confidences = [c for g, c in self.history if g == gesture]
                    confidence = sum(confidences) / len(confidences)
            
            # 获取归一化的关键点数据
            if simulation_mode:
                landmarks_data = [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                    for lm in landmarks
                ]
            
            # 绘制信息
            cv2.putText(result_image, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_image, f"Confidence: {confidence:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            command = self.gesture_commands.get(gesture, "none")
            cv2.putText(result_image, f"Command: {command}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(result_image, "No Hand Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        self.last_gesture = gesture
        self.last_confidence = confidence
        
        return result_image, gesture, confidence, landmarks_data

    def get_command(self, gesture):
        """获取控制指令"""
        return self.gesture_commands.get(gesture, "none")

    def get_gesture_intensity(self, landmarks, gesture_type):
        """获取手势强度"""
        if not landmarks or len(landmarks) < 21:
            return 0.5
        
        # 简化实现
        return self.last_confidence if self.last_confidence > 0 else 0.5

    def release(self):
        """释放资源"""
        pass  # 无需释放资源
