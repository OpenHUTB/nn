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
            "ok_sign": "hover"        # OK手势 - 悬停
        }
        
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
        elif finger_count >= 4:
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
