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

    def detect_gestures(self, image):
        """
        检测图像中的手势

        Args:
            image: 输入图像

        Returns:
            processed_image: 处理后的图像
            gesture: 识别到的手势
            confidence: 置信度
        """
        # 转换颜色空间
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        gesture = "no_hand"
        confidence = 0.0

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

                # 在图像上显示手势信息
                cv2.putText(image, f"Gesture: {gesture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 显示控制指令
                command = self.gesture_commands.get(gesture, "none")
                cv2.putText(image, f"Command: {command}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return image, gesture, confidence

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

        # 根据优先级返回手势
        gestures = [
            (is_thumb_up, "thumb_up", 0.95),
            (is_thumb_down, "thumb_down", 0.95),
            (is_ok_sign, "ok_sign", 0.90),
            (is_victory, "victory", 0.85),
            (is_open_palm, "open_palm", 0.80),
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
        finger_dips = [6, 10, 14, 18]  # 对应的指间关节

        extended_fingers = 0
        for tip, dip in zip(finger_tips, finger_dips):
            if points[tip][1] < points[dip][1]:  # 指尖在指间关节上方
                extended_fingers += 1

        # 至少3个手指伸直
        return extended_fingers >= 3

    def _is_closed_fist(self, points):
        """检测握拳手势"""
        finger_tips = [8, 12, 16, 20]  # 指尖
        finger_mcps = [5, 9, 13, 17]  # 掌指关节

        bent_fingers = 0
        for tip, mcp in zip(finger_tips, finger_mcps):
            if points[tip][1] > points[mcp][1]:  # 指尖在掌指关节下方
                bent_fingers += 1

        # 所有4个手指都弯曲
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

    def release(self):
        """释放资源"""
        self.hands.close()