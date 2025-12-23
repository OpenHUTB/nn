import cv2
import mediapipe as mp
import numpy as np
import math


class GestureDetector:
    def __init__(self):
        """
        初始化手势检测器
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 初始化手部检测模型
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # 手势到控制指令的映射
        self.gesture_commands = {
            "open_palm": "takeoff",  # 张开手掌 - 起飞
            "closed_fist": "land",  # 握拳 - 降落
            "pointing_up": "up",  # 食指上指 - 上升
            "pointing_down": "down",  # 食指向下 - 下降
            "victory": "forward",  # 胜利手势 - 前进
            "thumb_up": "backward",  # 大拇指 - 后退
            "thumb_down": "stop",  # 大拇指向下 - 停止
            "ok_sign": "hover"  # OK手势 - 悬停
        }

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
            landmarks: 关键点坐标（仅仿真模式返回）
        """
        # 转换颜色空间
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        gesture = "no_hand"
        confidence = 0.0
        landmarks_data = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点和连接线
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # 识别具体手势
                gesture, confidence = self._classify_gesture(hand_landmarks)

                # 在仿真模式下提取关键点数据
                if simulation_mode:
                    landmarks_data = self._get_normalized_landmarks(hand_landmarks)

                # 在图像上显示手势信息
                cv2.putText(image, f"Gesture: {gesture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 显示控制指令
                command = self.gesture_commands.get(gesture, "none")
                cv2.putText(image, f"Command: {command}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

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
        for landmark in hand_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            })
        return landmarks

    def _classify_gesture(self, landmarks):
        """
        分类具体手势

        Args:
            landmarks: 手部关键点

        Returns:
            gesture: 手势名称
            confidence: 置信度
        """
        # 获取关键点坐标
        points = []
        for landmark in landmarks.landmark:
            points.append((landmark.x, landmark.y, landmark.z))

        # 计算各种手势的特征
        is_thumb_up = self._is_thumb_up(points)
        is_thumb_down = self._is_thumb_down(points)
        is_open_palm = self._is_open_palm(points)
        is_closed_fist = self._is_closed_fist(points)
        is_pointing_up = self._is_pointing_up(points)
        is_pointing_down = self._is_pointing_down(points)
        is_victory = self._is_victory_gesture(points)
        is_ok_sign = self._is_ok_sign(points)

        # 调试输出
        print(f"[手势特征] "
              f"拇指上:{is_thumb_up} "
              f"拇指向下:{is_thumb_down} "
              f"张开掌:{is_open_palm} "
              f"握拳:{is_closed_fist} "
              f"指上:{is_pointing_up} "
              f"指下:{is_pointing_down} "
              f"胜利:{is_victory} "
              f"OK:{is_ok_sign}")

        # 根据优先级返回手势
        gestures = [
            (is_thumb_up, "thumb_up", 0.95),
            (is_thumb_down, "thumb_down", 0.95),
            (is_ok_sign, "ok_sign", 0.90),
            (is_victory, "victory", 0.85),
            (is_open_palm, "open_palm", 0.80),  # 提高优先级
            (is_closed_fist, "closed_fist", 0.80),
            (is_pointing_up, "pointing_up", 0.75),
            (is_pointing_down, "pointing_down", 0.75),
        ]

        for condition, gesture_name, conf in gestures:
            if condition:
                return gesture_name, conf

        return "hand_detected", 0.5

    def _is_thumb_up(self, points):
        """检测大拇指向上手势"""
        thumb_tip = points[4]  # 大拇指指尖
        thumb_ip = points[3]  # 大拇指指间关节
        index_mcp = points[5]  # 食指掌指关节

        # 大拇指伸直且向上
        thumb_extended = thumb_tip[1] < thumb_ip[1]  # y坐标更小表示更靠上
        thumb_raised = thumb_tip[1] < index_mcp[1]  # 大拇指高于食指基部

        return thumb_extended and thumb_raised

    def _is_thumb_down(self, points):
        """检测大拇指向下手势"""
        thumb_tip = points[4]
        thumb_ip = points[3]
        pinky_mcp = points[17]  # 小指掌指关节

        # 大拇指伸直且向下
        thumb_extended = thumb_tip[1] > thumb_ip[1]  # y坐标更大表示更靠下
        thumb_lowered = thumb_tip[1] > pinky_mcp[1]  # 大拇指低于小指基部

        return thumb_extended and thumb_lowered

    def _is_open_palm(self, points):
        """检测张开手掌手势"""
        finger_tips = [8, 12, 16, 20]  # 食指、中指、无名指、小指指尖
        finger_mcps = [5, 9, 13, 17]  # 掌指关节

        extended_fingers = 0
        for tip, mcp in zip(finger_tips, finger_mcps):
            # 放宽条件：指尖比掌指关节高（y值小），增加容差0.15
            if points[tip][1] < points[mcp][1] + 0.15:
                extended_fingers += 1

        # 放宽条件：至少2个手指伸直
        return extended_fingers >= 4

    def _is_closed_fist(self, points):
        """检测握拳手势"""
        finger_tips = [8, 12, 16, 20]  # 指尖
        finger_mcps = [5, 9, 13, 17]  # 掌指关节

        bent_fingers = 0
        for tip, mcp in zip(finger_tips, finger_mcps):
            # 放宽条件：指尖在掌指关节下方或接近
            if points[tip][1] > points[mcp][1] - 0.1:
                bent_fingers += 1

        # 放宽条件：至少3个手指弯曲
        return bent_fingers >= 3

    def _is_pointing_up(self, points):
        """检测食指上指手势"""
        index_tip = points[8]  # 食指尖
        index_dip = points[7]  # 食指指间关节
        middle_tip = points[12]  # 中指尖
        middle_mcp = points[9]  # 中指掌指关节

        # 食指伸直且向上，其他手指弯曲
        index_extended = index_tip[1] < index_dip[1]
        middle_bent = middle_tip[1] > middle_mcp[1]

        return index_extended and middle_bent

    def _is_pointing_down(self, points):
        """检测食指向下手势"""
        index_tip = points[8]
        index_dip = points[7]
        middle_tip = points[12]
        middle_mcp = points[9]

        # 食指伸直且向下，其他手指弯曲
        index_extended = index_tip[1] > index_dip[1]
        middle_bent = middle_tip[1] > middle_mcp[1]

        return index_extended and middle_bent

    def _is_victory_gesture(self, points):
        """检测胜利手势（食指和中指伸直）"""
        index_tip, middle_tip = points[8], points[12]
        index_dip, middle_dip = points[7], points[11]
        ring_tip = points[16]
        ring_mcp = points[13]

        # 食指和中指伸直
        index_extended = index_tip[1] < index_dip[1]
        middle_extended = middle_tip[1] < middle_dip[1]
        # 无名指弯曲
        ring_bent = ring_tip[1] > ring_mcp[1]

        return index_extended and middle_extended and ring_bent

    def _is_ok_sign(self, points):
        """检测OK手势（食指和拇指接触）"""
        thumb_tip = points[4]
        index_tip = points[8]

        # 计算食指和拇指之间的距离
        distance = math.sqrt(
            (thumb_tip[0] - index_tip[0]) ** 2 +
            (thumb_tip[1] - index_tip[1]) ** 2
        )

        # 距离很小表示接触
        return distance < 0.05

    def get_command(self, gesture):
        """
        根据手势获取控制指令

        Args:
            gesture: 手势名称

        Returns:
            command: 控制指令
        """
        return self.gesture_commands.get(gesture, "none")

    def get_gesture_intensity(self, landmarks, gesture_type):
        """
        获取手势强度（用于精细控制）

        Args:
            landmarks: 关键点数据
            gesture_type: 手势类型

        Returns:
            float: 手势强度 (0.0-1.0)
        """
        if not landmarks or len(landmarks) < 21:
            return 0.5  # 默认强度

        if gesture_type == "pointing_up":
            # 基于食指的角度计算强度
            index_tip = landmarks[8]
            index_mcp = landmarks[5]
            if index_tip['y'] < index_mcp['y']:
                intensity = (index_mcp['y'] - index_tip['y']) * 2
                return min(max(intensity, 0.1), 1.0)

        elif gesture_type == "pointing_down":
            # 基于食指的角度计算强度
            index_tip = landmarks[8]
            index_mcp = landmarks[5]
            if index_tip['y'] > index_mcp['y']:
                intensity = (index_tip['y'] - index_mcp['y']) * 2
                return min(max(intensity, 0.1), 1.0)

        elif gesture_type in ["open_palm", "closed_fist"]:
            # 基于手掌张开程度计算强度
            thumb_tip = landmarks[4]
            pinky_tip = landmarks[20]
            distance = math.sqrt(
                (thumb_tip['x'] - pinky_tip['x']) ** 2 +
                (thumb_tip['y'] - pinky_tip['y']) ** 2
            )
            if gesture_type == "open_palm":
                return min(max(distance * 3, 0.1), 1.0)
            else:
                return min(max((0.2 - distance) * 5, 0.1), 1.0)

        return 0.5  # 默认强度

    def get_hand_position(self, landmarks):
        """
        获取手部在画面中的位置

        Args:
            landmarks: 关键点数据

        Returns:
            dict: 包含手部中心位置和大小
        """
        if not landmarks or len(landmarks) < 21:
            return None

        # 计算手部边界框
        x_coords = [p['x'] for p in landmarks]
        y_coords = [p['y'] for p in landmarks]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        return {
            'center_x': (min_x + max_x) / 2,
            'center_y': (min_y + max_y) / 2,
            'width': max_x - min_x,
            'height': max_y - min_y,
            'bbox': (min_x, min_y, max_x, max_y)
        }

    def release(self):
        """释放资源"""
        self.hands.close()