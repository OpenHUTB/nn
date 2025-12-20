import cv2
import mediapipe as mp
import numpy as np

class GestureDetector:
    """基于MediaPipe的手势检测类（优化版）"""
    # 手部关键点索引常量
    WRIST = 0
    THUMB_TIP = 4
    THUMB_IP = 3
    INDEX_TIP = 8
    INDEX_PIP = 6
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    RING_TIP = 16
    RING_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18

    def __init__(self, 
                 max_hands: int = 1,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5):
        # 初始化MediaPipe手部检测
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        # 初始化绘图工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.draw_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

    def detect_gestures(self, frame):
        """检测帧中的手势，返回处理后的帧、手势名称、关键点坐标"""
        if frame is None or frame.size == 0:
            return frame, "无效帧", None
        
        # 颜色空间转换+性能优化
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        gesture = "未检测到手势"
        landmarks = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # 绘制关键点
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.draw_spec, self.draw_spec
            )
            # 转换坐标并识别手势
            landmarks = self._convert_landmarks_to_pixels(hand_landmarks, frame.shape)
            gesture = self._classify_gesture(landmarks)
        
        return frame, gesture, landmarks
    
    def _convert_landmarks_to_pixels(self, hand_landmarks, frame_shape):
        """将归一化坐标转换为像素坐标，限制坐标范围"""
        h, w, _ = frame_shape
        landmarks = []
        for lm in hand_landmarks.landmark:
            x = int(np.clip(lm.x * w, 0, w-1))
            y = int(np.clip(lm.y * h, 0, h-1))
            landmarks.append((x, y))
        return landmarks
    
    def _is_finger_open(self, landmarks, tip_idx, pip_idx):
        """判断非拇指手指是否张开"""
        return landmarks[tip_idx][1] < landmarks[pip_idx][1]
    
    def _is_thumb_open(self, landmarks):
        """判断拇指是否张开（适配左右手）"""
        wrist_x = landmarks[self.WRIST][0]
        thumb_tip_x = landmarks[self.THUMB_TIP][0]
        thumb_ip_x = landmarks[self.THUMB_IP][0]
        
        if thumb_tip_x > wrist_x:  # 右手
            return thumb_tip_x > thumb_ip_x + 10
        else:  # 左手
            return thumb_tip_x < thumb_ip_x - 10
    
    def _classify_gesture(self, landmarks):
        """根据关键点坐标分类手势"""
        if not landmarks or len(landmarks) < 21:
            return "未检测到手势"
        
        # 判断各手指状态
        thumb_open = self._is_thumb_open(landmarks)
        index_open = self._is_finger_open(landmarks, self.INDEX_TIP, self.INDEX_PIP)
        middle_open = self._is_finger_open(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_open = self._is_finger_open(landmarks, self.RING_TIP, self.RING_PIP)
        pinky_open = self._is_finger_open(landmarks, self.PINKY_TIP, self.PINKY_PIP)
        
        finger_states = [thumb_open, index_open, middle_open, ring_open, pinky_open]
        
        # 手势分类逻辑
        if not any(finger_states):
            return "握拳"
        elif all(finger_states):
            return "张开手掌"
        elif index_open and not middle_open and not ring_open and not pinky_open:
            return "食指指向"
        elif index_open and middle_open and not ring_open and not pinky_open:
            return "胜利手势"
        else:
            return "其他手势"

# 测试代码
if __name__ == "__main__":
    detector = GestureDetector(max_hands=1, detection_confidence=0.7, tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面，退出...")
                break
            
            frame = cv2.flip(frame, 1)  # 镜像翻转
            frame, gesture, _ = detector.detect_gestures(frame)
            
            # 显示手势结果
            cv2.putText(
                frame, f"当前手势: {gesture}", 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, (0, 255, 0), 3
            )
            
            cv2.imshow("手势检测（按Q退出）", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"运行出错: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
