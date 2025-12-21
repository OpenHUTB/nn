import cv2
import mediapipe as mp
import numpy as np
# 引入Pillow库，用于解决OpenCV中文显示乱码问题
from PIL import Image, ImageDraw, ImageFont  

class GestureDetector:
    """基于MediaPipe的手势检测类（修复中文显示版）
    核心功能：检测单只手关键点，识别常见手势（握拳/张开手掌/食指指向/胜利手势）
    修复点：解决OpenCV显示中文乱码（问号）问题
    """
    # 定义手部关键点索引常量（简化后续代码调用）
    WRIST = 0               # 手腕关键点索引
    THUMB_TIP = 4           # 拇指指尖索引
    THUMB_IP = 3            # 拇指远节指关节索引
    INDEX_TIP = 8           # 食指指尖索引
    INDEX_PIP = 6           # 食指近节指关节索引
    MIDDLE_TIP = 12         # 中指指尖索引
    MIDDLE_PIP = 10         # 中指近节指关节索引
    RING_TIP = 16           # 无名指指尖索引
    RING_PIP = 14           # 无名指近节指关节索引
    PINKY_TIP = 20          # 小指指尖索引
    PINKY_PIP = 18          # 小指近节指关节索引

    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        """初始化手势检测器
        Args:
            max_hands: 最大检测手数（默认1只）
            detection_confidence: 检测置信度阈值（0-1，低于则不识别）
            tracking_confidence: 跟踪置信度阈值（0-1，低于则重新检测）
        """
        # 初始化MediaPipe手部检测模块
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,       # 视频流模式（False），静态图片设为True
            max_num_hands=max_hands,       # 最多检测1只手
            min_detection_confidence=detection_confidence,  # 检测置信度阈值
            min_tracking_confidence=tracking_confidence     # 跟踪置信度阈值
        )
        # 初始化MediaPipe绘图工具（用于绘制手部关键点）
        self.mp_drawing = mp.solutions.drawing_utils
        # 设置绘图样式：绿色、线宽2、关键点半径2
        self.draw_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

    def detect_gestures(self, frame):
        """核心方法：检测帧中的手势
        Args:
            frame: OpenCV读取的BGR格式视频帧
        Returns:
            frame: 绘制了关键点的帧
            gesture: 识别的手势名称（中文）
            landmarks: 手部关键点像素坐标列表
        """
        # 异常处理：校验输入帧是否有效
        if frame is None or frame.size == 0:
            return frame, "无效帧", None
        
        # 颜色空间转换：OpenCV(BGR) → MediaPipe(RGB)
        # 禁用写入权限提升处理性能
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        # 处理帧，获取手部检测结果
        results = self.hands.process(rgb_frame)
        # 恢复写入权限（后续绘制关键点需要）
        rgb_frame.flags.writeable = True

        # 初始化返回值
        gesture = "未检测到手势"  # 默认未检测到
        landmarks = None          # 关键点坐标初始化为空
        
        # 如果检测到至少一只手的关键点
        if results.multi_hand_landmarks:
            # 仅处理第一只手（max_num_hands=1）
            hand_landmarks = results.multi_hand_landmarks[0]
            # 在帧上绘制手部关键点和连接线
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.draw_spec, self.draw_spec  # 关键点和连接线样式
            )
            # 将归一化坐标转换为像素坐标
            landmarks = self._convert_landmarks_to_pixels(hand_landmarks, frame.shape)
            # 根据关键点识别手势
            gesture = self._classify_gesture(landmarks)
        
        return frame, gesture, landmarks
    
    def _convert_landmarks_to_pixels(self, hand_landmarks, frame_shape):
        """辅助方法：将MediaPipe归一化坐标转换为像素坐标
        Args:
            hand_landmarks: MediaPipe手部关键点对象
            frame_shape: 帧的形状 (高度, 宽度, 通道数)
        Returns:
            landmarks: 像素坐标列表 [(x1,y1), (x2,y2), ...]
        """
        h, w, _ = frame_shape  # 获取帧的高、宽
        landmarks = []
        for lm in hand_landmarks.landmark:
            # 归一化坐标转像素坐标，并限制在帧范围内（避免越界）
            x = int(np.clip(lm.x * w, 0, w-1))
            y = int(np.clip(lm.y * h, 0, h-1))
            landmarks.append((x, y))
        return landmarks
    
    def _is_finger_open(self, landmarks, tip_idx, pip_idx):
        """辅助方法：判断非拇指手指是否张开
        原理：指尖y坐标 < 近节指关节y坐标 → 手指张开
        Args:
            landmarks: 关键点像素坐标列表
            tip_idx: 指尖索引
            pip_idx: 近节指关节索引
        Returns:
            bool: 手指是否张开
        """
        return landmarks[tip_idx][1] < landmarks[pip_idx][1]
    
    def _is_thumb_open(self, landmarks):
        """辅助方法：判断拇指是否张开（适配左右手）
        原理：
            右手：拇指指尖x > 拇指远节指关节x + 偏移 → 张开
            左手：拇指指尖x < 拇指远节指关节x - 偏移 → 张开
        Args:
            landmarks: 关键点像素坐标列表
        Returns:
            bool: 拇指是否张开
        """
        wrist_x = landmarks[self.WRIST][0]        # 手腕x坐标
        thumb_tip_x = landmarks[self.THUMB_TIP][0]  # 拇指指尖x坐标
        thumb_ip_x = landmarks[self.THUMB_IP][0]    # 拇指远节指关节x坐标
        
        # 判断左右手：拇指指尖x > 手腕x → 右手；反之→左手
        if thumb_tip_x > wrist_x:  # 右手逻辑
            return thumb_tip_x > thumb_ip_x + 10  # +10减少误判
        else:  # 左手逻辑
            return thumb_tip_x < thumb_ip_x - 10
    
    def _classify_gesture(self, landmarks):
        """核心方法：根据关键点分类手势
        Args:
            landmarks: 关键点像素坐标列表
        Returns:
            str: 手势名称（中文）
        """
        # 校验关键点数量（正常应为21个）
        if not landmarks or len(landmarks) < 21:
            return "未检测到手势"
        
        # 逐个判断手指张开状态
        thumb_open = self._is_thumb_open(landmarks)
        index_open = self._is_finger_open(landmarks, self.INDEX_TIP, self.INDEX_PIP)
        middle_open = self._is_finger_open(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_open = self._is_finger_open(landmarks, self.RING_TIP, self.RING_PIP)
        pinky_open = self._is_finger_open(landmarks, self.PINKY_TIP, self.PINKY_PIP)
        
        # 整理所有手指状态
        finger_states = [thumb_open, index_open, middle_open, ring_open, pinky_open]
        
        # 手势分类逻辑（优先级：精准手势→通用手势）
        if not any(finger_states):  # 所有手指闭合 → 握拳
            return "握拳"
        elif all(finger_states):    # 所有手指张开 → 张开手掌
            return "张开手掌"
        elif index_open and not middle_open and not ring_open and not pinky_open:  # 仅食指张开 → 食指指向
            return "食指指向"
        elif index_open and middle_open and not ring_open and not pinky_open:     # 食指+中指张开 → 胜利手势
            return "胜利手势"
        else:                       # 其他组合 → 其他手势
            return "其他手势"


def put_chinese_text(frame, text, position, font_size=32, color=(0, 255, 0)):
    """修复OpenCV中文显示乱码的核心函数
    原理：OpenCV默认字体不支持中文，通过Pillow绘制中文后转回OpenCV格式
    Args:
        frame: OpenCV的BGR格式帧
        text: 要显示的中文文本
        position: 文本显示位置 (x, y)
        font_size: 字体大小（默认32）
        color: 文本颜色（RGB格式，默认绿色）
    Returns:
        frame: 绘制了中文文本的BGR格式帧
    """
    # 步骤1：OpenCV(BGR) 转换为 Pillow(RGB)
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # 创建Pillow绘图对象
    draw = ImageDraw.Draw(pil_frame)
    
    # 步骤2：加载中文字体（解决中文显示问题）
    try:
        # Windows系统默认黑体路径：C:/Windows/Fonts/simhei.ttf
        # Linux/macOS需替换为对应中文字体路径（如思源黑体、文泉驿等）
        font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
    except Exception as e:
        # 异常处理：若加载字体失败，使用默认字体（可能仍显示问号）
        print(f"加载中文字体失败：{e}，将使用默认字体")
        font = ImageFont.load_default()
    
    # 步骤3：用Pillow绘制中文文本
    draw.text(position, text, font=font, fill=color)
    
    # 步骤4：Pillow(RGB) 转回 OpenCV(BGR)
    frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
    
    return frame


# 测试代码（主程序入口）
if __name__ == "__main__":
    # 创建手势检测器实例
    detector = GestureDetector()
    # 打开摄像头（0为默认摄像头，若无效可尝试1）
    cap = cv2.VideoCapture(0)
    # 设置摄像头分辨率（640x480，平衡性能和清晰度）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        # 循环读取摄像头帧
        while cap.isOpened():
            ret, frame = cap.read()
            # 校验帧是否读取成功
            if not ret:
                print("无法读取摄像头画面，退出...")
                break
            
            # 镜像翻转帧（符合人眼视觉习惯，左右操作对应）
            frame = cv2.flip(frame, 1)
            # 检测手势
            frame, gesture, _ = detector.detect_gestures(frame)
            
            # 关键修改：用自定义函数显示中文（替换原cv2.putText）
            frame = put_chinese_text(frame, f"当前手势: {gesture}", (20, 50), font_size=32)
            
            # 显示处理后的帧
            cv2.imshow("手势检测（按Q退出）", frame)
            # 按Q键退出循环（需小写）
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        # 异常捕获：打印错误信息
        print(f"程序运行出错：{e}")
    finally:
        # 释放摄像头资源
        cap.release()
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
