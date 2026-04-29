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
        """
        初始化手势检测器（支持双手控制）
        左手控制方向，右手控制高度
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 初始化手部检测模型（支持双手）
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # 支持检测2只手
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # 左手手势到控制指令的映射（方向控制）
        self.left_hand_commands = {
            "victory": "forward",      # 胜利手势 - 前进
            "thumb_up": "backward",    # 大拇指向上 - 后退
            "pointing_up": "left",     # 食指向左 - 左转
            "pointing_down": "right",  # 食指向右 - 右转
        }

        # 右手手势到控制指令的映射（高度控制）
        self.right_hand_commands = {
            "pointing_up": "up",       # 食指向下 - 上升
            "pointing_down": "down",   # 食指向下 - 下降
            "ok_sign": "hover",        # OK手势 - 悬停
        }

        # 双手手势指令（特殊命令）
        self.both_hands_commands = {
            "open_palm": "takeoff",    # 张开手掌（任意手）- 起飞
            "closed_fist": "land",     # 握拳（任意手）- 降落
            "thumb_down": "stop",       # 大拇指向下 - 停止
        }

        # 合并所有手势命令（用于显示）
        self.gesture_commands = {
            "open_palm": "takeoff",
            "closed_fist": "land",
            "pointing_up": "up",
            "pointing_down": "down",
            "victory": "forward",
            "thumb_up": "backward",
            "thumb_down": "stop",
            "ok_sign": "hover",
        }
        
        # 手势序列检测（用于握拳→松开触发起飞）
        self.prev_gesture = None
        self.fist_start_time = None
        self.FIST_TIMEOUT = 1.5  # 握拳后1.5秒内松开才触发起飞
        
        print("[INFO] 使用纯 OpenCV 手势检测器")
    
    def detect_gestures(self, image, simulation_mode=False):
        """
        检测图像中的手势（支持双手）

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
        left_hand_data = None
        right_hand_data = None

        if results.multi_hand_landmarks and results.multi_handedness:
            # 遍历检测到的每只手
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # 获取手的类型（左手还是右手）
                hand_type = handedness.classification[0].label  # "Left" 或 "Right"
                hand_confidence = handedness.classification[0].score

                # 绘制手部关键点和连接线
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # 识别具体手势
                detected_gesture, gesture_confidence = self._classify_gesture(hand_landmarks)

                # 提取关键点数据
                if simulation_mode:
                    normalized_landmarks = self._get_normalized_landmarks(hand_landmarks)
                    normalized_landmarks['hand_type'] = hand_type
                    normalized_landmarks['gesture'] = detected_gesture
                    normalized_landmarks['confidence'] = gesture_confidence

                    if hand_type == "Left":
                        left_hand_data = normalized_landmarks
                    else:
                        right_hand_data = normalized_landmarks

                # 在图像上显示手势信息
                y_offset = 30 if idx == 0 else 150
                color = (0, 255, 0) if hand_type == "Left" else (255, 128, 0)

                cv2.putText(image, f"{hand_type}: {detected_gesture}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(image, f"{hand_type} Conf: {gesture_confidence:.2f}", (10, y_offset + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

                # 保存置信度最高的手势
                if gesture_confidence > confidence:
                    gesture = detected_gesture
                    confidence = gesture_confidence

            # 显示双手控制提示
            if left_hand_data and right_hand_data:
                cv2.putText(image, "LEFT: Direction | RIGHT: Altitude", (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            elif left_hand_data:
                cv2.putText(image, "Left hand: Direction control", (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            elif right_hand_data:
                cv2.putText(image, "Right hand: Altitude control", (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 1)

        # 显示控制指令
        command = self.gesture_commands.get(gesture, "none")
        cv2.putText(image, f"Command: {command}", (10, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 返回包含双手数据的landmarks
        if simulation_mode:
            landmarks_data = {
                'left_hand': left_hand_data,
                'right_hand': right_hand_data
            }

        return image, gesture, confidence, landmarks_data

    def _get_normalized_landmarks(self, hand_landmarks):
        """
        获取归一化的关键点坐标（用于仿真模式）

        Args:
            hand_landmarks: 手部关键点

        Returns:
            list: 包含21个关键点的字典列表，每个点有x,y,z坐标
        """
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

    def get_dual_hand_commands(self, left_hand_data, right_hand_data):
        """
        获取双手控制命令

        Args:
            left_hand_data: 左手关键点数据
            right_hand_data: 右手关键点数据

        Returns:
            dict: 包含方向命令和高度命令的字典
                {
                    'direction_command': 命令或None,
                    'direction_intensity': 强度,
                    'altitude_command': 命令或None,
                    'altitude_intensity': 强度,
                    'special_command': 特殊命令（起飞/降落/停止）或None
                }
        """
        result = {
            'direction_command': None,
            'direction_intensity': 0.5,
            'altitude_command': None,
            'altitude_intensity': 0.5,
            'special_command': None,
            'left_gesture': None,
            'right_gesture': None
        }

        # 处理左手 - 方向控制
        if left_hand_data:
            gesture = left_hand_data.get('gesture', 'none')
            intensity = self.get_gesture_intensity(left_hand_data, gesture)
            result['left_gesture'] = gesture

            # 检查是否是方向控制手势
            if gesture in self.left_hand_commands:
                result['direction_command'] = self.left_hand_commands[gesture]
                result['direction_intensity'] = intensity
            # 检查是否是特殊手势
            elif gesture in self.both_hands_commands:
                result['special_command'] = self.both_hands_commands[gesture]

        # 处理右手 - 高度控制
        if right_hand_data:
            gesture = right_hand_data.get('gesture', 'none')
            intensity = self.get_gesture_intensity(right_hand_data, gesture)
            result['right_gesture'] = gesture

            # 检查是否是高度控制手势
            if gesture in self.right_hand_commands:
                result['altitude_command'] = self.right_hand_commands[gesture]
                result['altitude_intensity'] = intensity
            # 检查是否是特殊手势
            elif gesture in self.both_hands_commands:
                result['special_command'] = self.both_hands_commands[gesture]

        return result

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
