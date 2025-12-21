import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont  

class GestureDetector:
    """基于MediaPipe的手势检测类（修复中文显示+帧率显示+摄像头容错）"""
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

    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        # 初始化MediaPipe手部检测
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.draw_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

    def detect_gestures(self, frame):
        """检测帧中的手势"""
        if frame is None or frame.size == 0:
            return frame, "无效帧", None
        
        # 颜色转换+性能优化
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        gesture = "未检测到手势"
        landmarks = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.draw_spec, self.draw_spec
            )
            landmarks = self._convert_landmarks_to_pixels(hand_landmarks, frame.shape)
            gesture = self._classify_gesture(landmarks)
        
        return frame, gesture, landmarks
    
    def _convert_landmarks_to_pixels(self, hand_landmarks, frame_shape):
        """归一化坐标转像素坐标"""
        h, w, _ = frame_shape
        landmarks = []
        for lm in hand_landmarks.landmark:
            x = int(np.clip(lm.x * w, 0, w-1))
            y = int(np.clip(lm.y * h, 0, h-1))
            landmarks.append((x, y))
        return landmarks
    
    def _is_finger_open(self, landmarks, tip_idx, pip_idx):
        """判断非拇指是否张开"""
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
        """分类手势"""
        if not landmarks or len(landmarks) < 21:
            return "未检测到手势"
        
        # 各手指状态
        thumb_open = self._is_thumb_open(landmarks)
        index_open = self._is_finger_open(landmarks, self.INDEX_TIP, self.INDEX_PIP)
        middle_open = self._is_finger_open(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_open = self._is_finger_open(landmarks, self.RING_TIP, self.RING_PIP)
        pinky_open = self._is_finger_open(landmarks, self.PINKY_TIP, self.PINKY_PIP)
        
        finger_states = [thumb_open, index_open, middle_open, ring_open, pinky_open]
        
        # 手势判断
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


def put_chinese_text(frame, text, position, font_size=32, color=(0, 255, 0)):
    """修复OpenCV中文显示乱码"""
    # OpenCV转PIL
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_frame)
    
    # 加载中文字体
    try:
        font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
    except:
        print("加载中文字体失败，使用默认字体")
        font = ImageFont.load_default()
    
    # 绘制中文
    draw.text(position, text, font=font, fill=color)
    
    # PIL转OpenCV
    return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)


def get_available_camera():
    """自动检测可用摄像头（尝试索引0/1/2）"""
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Windows用CAP_DSHOW避免延迟/占用问题
        if cap.isOpened():
            print(f"成功打开摄像头，索引：{idx}")
            return cap
        cap.release()
    return None


# 主程序入口
if __name__ == "__main__":
    # 创建手势检测器
    detector = GestureDetector()
    
    # 自动检测可用摄像头（核心修复点）
    cap = get_available_camera()
    if cap is None:
        print("错误：未检测到可用摄像头！")
        print("请检查：")
        print("1. 摄像头是否被其他程序占用（如微信、钉钉、系统相机）")
        print("2. 摄像头是否已正确连接并安装驱动")
        print("3. 是否授予程序摄像头权限（Windows设置→隐私和安全性→摄像头）")
        exit(1)
    
    # 配置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)  # 目标帧率60fps
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区，降低延迟
    
    # 帧率计算变量
    prev_time = time.time()
    fps = 0
    fps_counter = 0
    fps_show_interval = 0.1
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            # 增加帧读取失败的容错处理
            if not ret or frame is None:
                print("警告：暂时无法读取摄像头帧，重试中...")
                time.sleep(0.1)
                continue
            
            # 镜像翻转
            frame = cv2.flip(frame, 1)
            
            # 检测手势
            frame, gesture, _ = detector.detect_gestures(frame)
            
            # 计算帧率
            fps_counter += 1
            current_time = time.time()
            if current_time - prev_time >= fps_show_interval:
                fps = fps_counter / (current_time - prev_time)
                fps_counter = 0
                prev_time = current_time
            
            # 显示手势和帧率
            frame = put_chinese_text(frame, f"当前手势: {gesture}", (20, 50), font_size=32)
            cv2.putText(
                frame, f"FPS: {fps:.1f} ", 
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (255, 0, 0), 2
            )
            
            # 显示窗口
            cv2.imshow("手势检测（按Q退出）- 目标帧率60fps", frame)
            
            # 按Q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"程序运行出错：{e}")
    finally:
        # 确保资源释放
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出，资源已释放")
