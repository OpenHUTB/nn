import cv2          # 导入OpenCV库，用于图像处理和视频捕获
import mediapipe as mp  # 导入MediaPipe库，用于手部关键点检测
import numpy as np   # 导入numpy库，用于数值计算

class GestureDetector:
    """手势检测类
    基于MediaPipe实现单只手的关键点检测和基础手势识别
    支持识别：张开手掌、握拳、食指指向、胜利手势、其他手势
    """
    
    def __init__(self):
        """初始化函数：创建MediaPipe手部检测实例和绘图工具"""
        # 初始化MediaPipe手部检测模块
        self.mp_hands = mp.solutions.hands
        # 创建Hands对象，配置检测参数
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,       # 动态视频模式（False），非静态图片模式
            max_num_hands=1,               # 最多检测1只手
            min_detection_confidence=0.7,  # 检测置信度阈值，低于0.7则认为未检测到
            min_tracking_confidence=0.5    # 跟踪置信度阈值，低于0.5则重新检测
        )
        # 初始化MediaPipe绘图工具，用于绘制手部关键点和连接线
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_gestures(self, frame):
        """检测帧中的手势
        Args:
            frame: OpenCV读取的BGR格式视频帧
        Returns:
            frame: 绘制了手部关键点的帧
            gesture: 识别出的手势名称（字符串）
            landmarks: 手部关键点像素坐标列表，格式[(x1,y1), (x2,y2), ...]
        """
        # 将BGR格式（OpenCV默认）转换为RGB格式（MediaPipe要求）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 处理帧，获取手部检测结果
        results = self.hands.process(rgb_frame)
        
        # 初始化返回值
        gesture = "未检测到手势"  # 默认手势状态
        landmarks = None          # 关键点坐标初始化为空
        
        # 如果检测到至少一只手的关键点
        if results.multi_hand_landmarks:
            # 遍历检测到的每只手（此处最多1只）
            for hand_landmarks in results.multi_hand_landmarks:
                # 在帧上绘制手部关键点和连接线
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 提取关键点像素坐标
                landmarks = []
                h, w, c = frame.shape  # 获取帧的高度、宽度、通道数
                for lm in hand_landmarks.landmark:
                    # 将MediaPipe的归一化坐标（0-1）转换为像素坐标
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    landmarks.append((px, py))
                
                # 根据关键点坐标分类手势
                gesture = self._classify_gesture(landmarks)
        
        # 返回处理后的帧、手势名称、关键点坐标
        return frame, gesture, landmarks
    
    def _classify_gesture(self, landmarks):
        """根据关键点坐标分类手势（内部方法）
        Args:
            landmarks: 手部关键点像素坐标列表
        Returns:
            str: 识别出的手势名称
        """
        # 校验关键点数量（正常手部关键点应为21个）
        if not landmarks or len(landmarks) < 21:
            return "未检测到手势"
        
        # 提取关键部位的坐标
        thumb_tip = landmarks[4]    # 拇指指尖
        index_tip = landmarks[8]    # 食指指尖
        middle_tip = landmarks[12]  # 中指指尖
        ring_tip = landmarks[16]    # 无名指指尖
        pinky_tip = landmarks[20]   # 小指指尖
        wrist = landmarks[0]        # 手腕
        
        # 判断每根手指是否张开，存入列表
        fingers = []
        # 拇指判断：指尖x坐标 > 拇指近节指关节x坐标 视为张开（仅适配右手）
        fingers.append(thumb_tip[0] > landmarks[3][0])  
        
        # 其他手指判断：指尖y坐标 < 手腕y坐标 视为张开（逻辑较简单）
        for tip in [index_tip, middle_tip, ring_tip, pinky_tip]:
            fingers.append(tip[1] < wrist[1])
        
        # 根据手指状态分类手势
        if all(fingers):            # 所有手指都张开
            return "张开手掌"
        elif not any(fingers):      # 所有手指都闭合
            return "握拳"
        elif fingers[1] and not any(fingers[2:]):  # 仅食指张开
            return "食指指向"
        elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:  # 食指+中指张开
            return "胜利手势"
        elif all(fingers[1:5]):     # 除拇指外其余四指张开
            return "张开五指"
        else:                       # 其他未匹配的手势
            return "其他手势"

# ------------------- 测试代码（可直接运行） -------------------
if __name__ == "__main__":
    # 创建手势检测器实例
    detector = GestureDetector()
    # 打开摄像头（0为默认摄像头）
    cap = cv2.VideoCapture(0)
    
    # 循环读取视频帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # 读取帧失败则退出
            break
        
        # 检测手势
        frame, gesture, landmarks = detector.detect_gestures(frame)
        
        # 在帧上显示识别结果
        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示处理后的帧
        cv2.imshow("Gesture Detection", frame)
        
        # 按下q键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
