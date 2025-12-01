import cv2
import mediapipe as mp
import numpy as np

class GestureDetector:
    """手势检测类"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_gestures(self, frame):
        """检测帧中的手势"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = "未检测到手势"
        landmarks = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 获取关键点坐标
                landmarks = []
                for lm in hand_landmarks.landmark:
                    h, w, c = frame.shape
                    landmarks.append((int(lm.x * w), int(lm.y * h)))
                
                # 识别手势
                gesture = self._classify_gesture(landmarks)
        
        return frame, gesture, landmarks
    
    def _classify_gesture(self, landmarks):
        """根据关键点分类手势"""
        if not landmarks or len(landmarks) < 21:
            return "未检测到手势"
        
        # 获取关键点
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        # 判断手指状态
        fingers = []
        fingers.append(thumb_tip[0] > landmarks[3][0])  # 拇指
        
        # 其他手指
        for tip in [index_tip, middle_tip, ring_tip, pinky_tip]:
            fingers.append(tip[1] < wrist[1])
        
        # 手势分类
        if all(fingers):
            return "张开手掌"
        elif not any(fingers):
            return "握拳"
        elif fingers[1] and not any(fingers[2:]):
            return "食指指向"
        elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            return "胜利手势"
        elif all(fingers[1:5]):
            return "张开五指"
        else:
            return "其他手势"
