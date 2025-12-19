import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import sys
from pygame.locals import *

class VirtualDrone:
    """
    虚拟无人机模拟器类
    负责创建无人机可视化界面、管理无人机状态、执行控制命令
    
    属性:
        width/height: 窗口尺寸
        screen: pygame显示窗口
        position: 无人机2D坐标 [x, y]
        altitude: 无人机高度(m)
        battery: 电池电量(%)
        is_flying: 飞行状态标识
        speed: 移动速度
        各类颜色/字体配置
    """
    
    def __init__(self):
        """初始化pygame环境和无人机初始状态"""
        pygame.init()
        
        # 窗口配置
        self.width, self.height = 400, 300
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("虚拟无人机模拟器")
        
        # 无人机初始状态
        self.position = [200, 150]  # 初始位置居中
        self.altitude = 0  # 初始高度为0（地面）
        self.battery = 100  # 初始电量100%
        self.is_flying = False  # 初始未飞行
        self.speed = 3  # 移动速度
        
        # 视觉样式配置
        self._init_visual_config()
        
        print("虚拟无人机模拟器已启动")
    
    def _init_visual_config(self):
        """初始化颜色和字体配置（提取独立函数）"""
        self.bg_color = (30, 30, 50)
        self.drone_color_flying = (0, 255, 0)  # 飞行中绿色
        self.drone_color_ground = (255, 100, 100)  # 地面红色
        self.text_color = (255, 255, 255)  # 白色文字
        self.font = pygame.font.Font(None, 24)  # 默认字体，24号大小
    
    def execute_command(self, command):
        """
        执行无人机控制命令
        
        参数:
            command: 控制命令字符串（起飞/降落/前进/上升/紧急停止）
        
        返回:
            bool: 命令是否成功执行
        """
        result = False
        
        # 命令执行逻辑（功能完全不变）
        if command == "起飞" and not self.is_flying:
            self.is_flying = True
            self.altitude = 10
            print("🛫 无人机起飞")
            result = True
            
        elif command == "降落" and self.is_flying:
            self.is_flying = False
            self.altitude = 0
            print("🛬 无人机降落")
            result = True
            
        elif command == "前进" and self.is_flying:
            self.position[1] = max(50, self.position[1] - self.speed)
            self.altitude = min(50, self.altitude + 0.5)
            print("➡️ 无人机前进")
            result = True
            
        elif command == "上升" and self.is_flying:
            self.altitude = min(100, self.altitude + 10)
            print(f"⬆️ 无人机上升 | 当前高度: {self.altitude}m")
            result = True
            
        elif command == "紧急停止":
            self.is_flying = False
            self.altitude = 0
            print("🚨 紧急停止!")
            result = True
            
        # 模拟电池消耗（仅飞行时消耗）
        if self.is_flying:
            self.battery = max(0, self.battery - 0.05)
            
        return result
    
    def draw(self):
        """绘制无人机界面和状态信息（视觉渲染主函数）"""
        # 清屏
        self.screen.fill(self.bg_color)
        
        # 绘制地面
        pygame.draw.rect(self.screen, (50, 50, 70), (0, self.height - 100, self.width, 100))
        
        # 绘制无人机（根据飞行状态切换颜色）
        drone_color = self.drone_color_flying if self.is_flying else self.drone_color_ground
        drone_y = self.height - 120 - self.altitude * 2
        
        # 绘制无人机主体（圆形）
        pygame.draw.circle(self.screen, drone_color, (self.position[0], drone_y), 15)
        
        # 绘制无人机螺旋桨（四个小圆形）
        self._draw_drone_propellers(drone_y)
        
        # 绘制状态信息和控制说明
        self._draw_status_info()
        self._draw_control_instructions()
        
        # 更新显示
        pygame.display.flip()
    
    def _draw_drone_propellers(self, drone_y):
        """绘制无人机螺旋桨（提取独立函数）"""
        propeller_color = (200, 200, 200)  # 螺旋桨颜色
        propeller_radius = 6  # 螺旋桨半径
        
        # 四个螺旋桨位置
        prop_positions = [
            (self.position[0] - 20, drone_y - 12),
            (self.position[0] + 20, drone_y - 12),
            (self.position[0] - 20, drone_y + 12),
            (self.position[0] + 20, drone_y + 12)
        ]
        
        for pos in prop_positions:
            pygame.draw.circle(self.screen, propeller_color, pos, propeller_radius)
    
    def _draw_status_info(self):
        """绘制无人机状态信息（提取独立函数）"""
        status = "飞行中" if self.is_flying else "在地面"
        texts = [
            f"状态: {status}",
            f"高度: {self.altitude:.1f}m",
            f"电池: {self.battery:.1f}%",
            f"位置: ({self.position[0]}, {self.position[1]})"
        ]
        
        # 逐行绘制状态文本
        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, self.text_color)
            self.screen.blit(text_surface, (10, 10 + i * 25))
    
    def _draw_control_instructions(self):
        """绘制控制说明文本（提取独立函数）"""
        controls = [
            "控制说明:",
            "张开手掌 - 起飞",
            "握拳 - 降落",
            "食指指向 - 前进",
            "胜利手势 - 上升",
            "OK手势 - 紧急停止"
        ]
        
        # 逐行绘制控制说明
        for i, control in enumerate(controls):
            text_surface = self.font.render(control, True, self.text_color)
            self.screen.blit(text_surface, (self.width - 200, 10 + i * 25))
    
    def process_events(self):
        """处理pygame窗口事件（如关闭窗口）"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
        return True

class GestureRecognizer:
    """
    手势识别器类
    基于MediaPipe实现手部关键点检测，识别预设手势并转换为控制命令
    
    属性:
        mp_hands: MediaPipe手部检测模块
        hands: 手部检测处理器
        mp_drawing: 关键点绘制工具
        cap: 摄像头捕获对象
    """
    
    def __init__(self):
        """初始化MediaPipe手部检测和摄像头"""
        self.mp_hands = mp.solutions.hands
        # 配置手部检测参数（优化参数命名，语义不变）
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,        # 视频流模式
            max_num_hands=1,                # 仅检测单只手
            min_detection_confidence=0.6,   # 检测置信度阈值
            min_tracking_confidence=0.5     # 跟踪置信度阈值
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = None  # 摄像头对象初始化为None
        
    def initialize_camera(self):
        """
        初始化摄像头（自动尝试多个索引）
        
        返回:
            bool: 摄像头初始化是否成功
        """
        print("初始化摄像头...")
        self.cap = cv2.VideoCapture(0)
        
        # 如果索引0失败，尝试1-4索引
        if not self.cap.isOpened():
            for cam_index in range(1, 5):
                self.cap = cv2.VideoCapture(cam_index)
                if self.cap.isOpened():
                    print(f"找到摄像头在索引 {cam_index}")
                    break
            else:
                raise Exception("无法找到可用的摄像头")
        
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("摄像头初始化成功")
        return True
    
    def detect_gesture(self, frame):
        """
        检测帧中的手势并转换为控制命令
        
        参数:
            frame: OpenCV视频帧
        
        返回:
            frame: 绘制了关键点的帧
            gesture: 识别到的手势名称
            command: 对应的控制命令
        """
        # 转换颜色空间（BGR -> RGB）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = "未检测到手势"
        command = "等待"
        
        # 如果检测到手部关键点
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点和连接线
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 提取关键点坐标
                landmarks = self._extract_landmarks(hand_landmarks, frame.shape)
                
                # 识别手势并转换为命令
                gesture = self._improved_classify_gesture(landmarks)
                command = self._gesture_to_command(gesture)
        
        return frame, gesture, command
    
    def _extract_landmarks(self, hand_landmarks, frame_shape):
        """
        提取手部关键点的像素坐标
        
        参数:
            hand_landmarks: MediaPipe手部关键点对象
            frame_shape: 视频帧形状 (h, w, c)
        
        返回:
            list: 关键点像素坐标列表 [(x1,y1), (x2,y2), ...]
        """
        h, w, _ = frame_shape
        landmarks = []
        for lm in hand_landmarks.landmark:
            # 将归一化坐标转换为像素坐标
            px = int(lm.x * w)
            py = int(lm.y * h)
            landmarks.append((px, py))
        return landmarks
    
    def _improved_classify_gesture(self, landmarks):
        """
        改进的手势分类算法（支持OK手势检测）
        
        参数:
            landmarks: 手部关键点像素坐标列表
        
        返回:
            str: 识别到的手势名称
        """
        # 校验关键点数量
        if not landmarks or len(landmarks) < 21:
            return "未检测到手势"
        
        # 定义关键点索引常量（提高可读性）
        THUMB_TIP = 4          # 拇指指尖
        INDEX_FINGER_TIP = 8   # 食指指尖
        MIDDLE_FINGER_TIP = 12 # 中指指尖
        RING_FINGER_TIP = 16   # 无名指指尖
        PINKY_TIP = 20         # 小指指尖
        
        # 获取关键点位坐标
        thumb_tip = landmarks[THUMB_TIP]
        index_tip = landmarks[INDEX_FINGER_TIP]
        middle_tip = landmarks[MIDDLE_FINGER_TIP]
        ring_tip = landmarks[RING_FINGER_TIP]
        pinky_tip = landmarks[PINKY_TIP]
        
        # 检测各手指是否伸直
        fingers = self._detect_extended_fingers(landmarks)
        
        # 计算伸直的手指数量
        extended_fingers = sum(fingers)
        
        # 检测OK手势（拇指和食指接触，其他手指弯曲）
        thumb_index_dist = np.sqrt((thumb_tip[0]-index_tip[0])**2 + (thumb_tip[1]-index_tip[1])**2)
        if thumb_index_dist < 30 and extended_fingers <= 3:
            # 检查中指、无名指、小指是否弯曲
            other_fingers_bent = self._check_other_fingers_bent(landmarks)
            if other_fingers_bent:
                return "OK手势"
        
        # 基础手势分类
        if extended_fingers == 5:
            return "张开手掌"
        elif extended_fingers == 0:
            return "握拳"
        elif extended_fingers == 1 and fingers[1]:  # 仅食指伸直
            return "食指指向"
        elif extended_fingers == 2 and fingers[1] and fingers[2]:  # 食指+中指伸直
            return "胜利手势"
        else:
            return "其他手势"
    
    def _detect_extended_fingers(self, landmarks):
        """
        检测每根手指是否伸直
        
        参数:
            landmarks: 手部关键点坐标列表
        
        返回:
            list: 手指伸直状态 [拇指, 食指, 中指, 无名指, 小指]
        """
        fingers = []
        
        # 拇指：比较指尖和IP关节的x坐标
        fingers.append(landmarks[4][0] < landmarks[3][0])
        
        # 其他手指：比较指尖和PIP关节的y坐标
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            fingers.append(landmarks[tip][1] < landmarks[pip][1])
        
        return fingers
    
    def _check_other_fingers_bent(self, landmarks):
        """检查中指、无名指、小指是否弯曲（OK手势辅助检测）"""
        finger_tips = [12, 16, 20]
        finger_pips = [10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            # 如果手指伸直超过20像素，判定为未弯曲
            if landmarks[tip][1] < landmarks[pip][1] - 20:
                return False
        return True
    
    def _gesture_to_command(self, gesture):
        """
        将识别到的手势映射为无人机控制命令
        
        参数:
            gesture: 手势名称
        
        返回:
            str: 控制命令
        """
        command_map = {
            "张开手掌": "起飞",
            "握拳": "降落",
            "食指指向": "前进",
            "胜利手势": "上升",
            "OK手势": "紧急停止",
            "未检测到手势": "等待",
            "其他手势": "等待"
        }
        return command_map.get(gesture, "等待")
    
    def release_camera(self):
        """释放摄像头资源"""
        if self.cap:
            self.cap.release()

class GestureDroneSystem:
    """
    手势控制无人机主系统类
    整合手势识别和无人机模拟器，提供完整的交互流程
    """
    
    def __init__(self):
        """初始化手势识别器和无人机模拟器"""
        self.gesture_recognizer = GestureRecognizer()
        self.drone_simulator = VirtualDrone()
        self.is_running = False  # 系统运行状态
        
    def initialize(self):
        """
        初始化整个系统
        
        返回:
            bool: 初始化是否成功
        """
        print("=" * 50)
        print("手势控制无人机系统")
        print("=" * 50)
        
        # 初始化摄像头
        if not self.gesture_recognizer.initialize_camera():
            return False
            
        # 打印使用说明
        self._print_usage_instructions()
        
        return True
    
    def _print_usage_instructions(self):
        """打印系统使用说明（提取独立函数）"""
        print("\n系统初始化完成!")
        print("\n手势控制说明:")
        print("✋ 张开手掌 - 起飞")
        print("✊ 握拳 - 降落")
        print("👆 食指指向 - 前进")
        print("✌️ 胜利手势 - 上升")
        print("👌 OK手势 - 紧急停止")
        print("\n按 'q' 键退出程序")
        print("=" * 50)
    
    def run(self):
        """运行系统主循环"""
        if not self.initialize():
            return
        
        self.is_running = True
        print("开始手势控制...")
        
        # 性能统计变量
        frame_count = 0
        start_time = time.time()
        last_command_time = 0
        command_interval = 1.0  # 命令执行间隔（防止重复执行）
        
        try:
            while self.is_running:
                # 处理pygame事件
                if not self.drone_simulator.process_events():
                    break
                
                # 读取摄像头帧
                ret, frame = self.gesture_recognizer.cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                # 帧计数+1，水平翻转帧（镜像显示）
                frame_count += 1
                frame = cv2.flip(frame, 1)
                
                # 检测手势
                processed_frame, gesture, command = self.gesture_recognizer.detect_gesture(frame)
                
                # 执行控制命令（带时间间隔限制）
                current_time = time.time()
                if (current_time - last_command_time > command_interval and 
                    command != "等待"):
                    if self.drone_simulator.execute_command(command):
                        print(f"✅ 执行命令: {command}")
                        last_command_time = current_time
                elif command != "等待":
                    print(f"⏳ 识别到: {gesture} -> {command}")
                
                # 显示帧信息、更新无人机界面
                self._display_info(processed_frame, gesture, command, frame_count, start_time)
                cv2.imshow('手势识别摄像头', processed_frame)
                self.drone_simulator.draw()
                
                # 检测退出按键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"运行时错误: {e}")
        finally:
            self.cleanup()
            
        # 显示性能统计
        self._show_performance_stats(start_time, frame_count)
    
    def _display_info(self, frame, gesture, command, frame_count, start_time):
        """在视频帧上绘制状态信息"""
        # 计算FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # 绘制文本信息（优化坐标和样式）
        cv2.putText(frame, f"手势: {gesture}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"命令: {command}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, "按 'q' 退出", (10, 450),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _show_performance_stats(self, start_time, frame_count):
        """显示系统性能统计信息"""
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print("\n" + "=" * 50)
        print("性能统计")
        print("=" * 50)
        print(f"总运行时间: {total_time:.2f} 秒")
        print(f"处理帧数: {frame_count}")
        print(f"平均FPS: {avg_fps:.2f}")
        print("=" * 50)
    
    def cleanup(self):
        """清理系统资源（摄像头、窗口、pygame）"""
        self.is_running = False
        self.gesture_recognizer.release_camera()
        cv2.destroyAllWindows()
        pygame.quit()
        print("系统已关闭")

if __name__ == "__main__":
    # 创建并运行系统
    drone_system = GestureDroneSystem()
    drone_system.run()
