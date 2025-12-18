"""
手势控制虚拟无人机系统
基于MediaPipe的手部关键点检测 + Pygame的可视化模拟
"""

import sys
import time
from typing import List, Tuple, Dict, Optional, Union

import cv2
import numpy as np
import pygame
import mediapipe as mp
from pygame.locals import QUIT, KEYDOWN

# Pygame类型别名
ColorType = Tuple[int, int, int]
PositionType = List[int]
LandmarkType = List[Tuple[int, int]]

class VirtualDrone:
    """
    虚拟无人机模拟器类
    负责创建无人机可视化界面、管理无人机状态、执行控制命令
    """
    # 窗口配置常量
    WINDOW_WIDTH: int = 400
    WINDOW_HEIGHT: int = 300
    
    # 无人机初始状态常量
    INIT_POSITION: PositionType = [200, 150]
    INIT_ALTITUDE: float = 0.0
    INIT_BATTERY: float = 100.0
    SPEED: int = 3
    
    # 颜色常量 (RGB)
    BG_COLOR: ColorType = (30, 30, 50)
    GROUND_COLOR: ColorType = (50, 50, 70)
    DRONE_COLOR_FLYING: ColorType = (0, 255, 0)
    DRONE_COLOR_GROUND: ColorType = (255, 100, 100)
    PROPELLER_COLOR: ColorType = (200, 200, 200)
    TEXT_COLOR: ColorType = (255, 255, 255)
    
    # 渲染常量
    DRONE_RADIUS: int = 15
    PROPELLER_RADIUS: int = 6
    GROUND_HEIGHT: int = 100
    BATTERY_CONSUMPTION_RATE: float = 0.05

    def __init__(self) -> None:
        """初始化pygame环境和无人机初始状态"""
        try:
            pygame.init()
        except pygame.error as e:
            print(f"Pygame初始化失败: {e}")
            raise
        
        # 窗口配置
        self.screen: pygame.Surface = pygame.display.set_mode(
            (self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("虚拟无人机模拟器")
        
        # 无人机状态
        self.position: PositionType = self.INIT_POSITION.copy()
        self.altitude: float = self.INIT_ALTITUDE
        self.battery: float = self.INIT_BATTERY
        self.is_flying: bool = False
        
        # 视觉样式
        self.font: pygame.font.Font = pygame.font.Font(None, 24)
        
        # 预计算的渲染位置（减少重复计算）
        self._prop_positions: List[Tuple[int, int]] = []
        self._update_prop_positions(150)  # 初始位置
        
        print("✅ 虚拟无人机模拟器已启动")
    
    def _update_prop_positions(self, drone_y: int) -> None:
        """更新螺旋桨位置（减少重复计算）"""
        self._prop_positions = [
            (self.position[0] - 20, drone_y - 12),
            (self.position[0] + 20, drone_y - 12),
            (self.position[0] - 20, drone_y + 12),
            (self.position[0] + 20, drone_y + 12)
        ]
    
    def execute_command(self, command: str) -> bool:
        """
        执行无人机控制命令
        
        参数:
            command: 控制命令字符串
        
        返回:
            命令是否成功执行
        """
        result: bool = False
        
        try:
            # 命令执行逻辑
            if command == "起飞" and not self.is_flying:
                self.is_flying = True
                self.altitude = 10.0
                print("🛫 无人机起飞")
                result = True
                
            elif command == "降落" and self.is_flying:
                self.is_flying = False
                self.altitude = 0.0
                print("🛬 无人机降落")
                result = True
                
            elif command == "前进" and self.is_flying:
                self.position[1] = max(50, self.position[1] - self.SPEED)
                self.altitude = min(50.0, self.altitude + 0.5)
                print("➡️ 无人机前进")
                result = True
                
            elif command == "上升" and self.is_flying:
                self.altitude = min(100.0, self.altitude + 10.0)
                print(f"⬆️ 无人机上升 | 当前高度: {self.altitude:.1f}m")
                result = True
                
            elif command == "紧急停止":
                self.is_flying = False
                self.altitude = 0.0
                print("🚨 紧急停止!")
                result = True
                
            # 模拟电池消耗（仅飞行时）
            if self.is_flying:
                self.battery = max(0.0, self.battery - self.BATTERY_CONSUMPTION_RATE)
                
        except Exception as e:
            print(f"❌ 执行命令 '{command}' 时出错: {e}")
            result = False
            
        return result
    
    def draw(self) -> None:
        """绘制无人机界面和状态信息"""
        try:
            # 清屏
            self.screen.fill(self.BG_COLOR)
            
            # 绘制地面
            pygame.draw.rect(
                self.screen, 
                self.GROUND_COLOR, 
                (0, self.WINDOW_HEIGHT - self.GROUND_HEIGHT, self.WINDOW_WIDTH, self.GROUND_HEIGHT)
            )
            
            # 计算无人机Y坐标
            drone_y: int = self.WINDOW_HEIGHT - 120 - int(self.altitude * 2)
            
            # 选择无人机颜色
            drone_color: ColorType = self.DRONE_COLOR_FLYING if self.is_flying else self.DRONE_COLOR_GROUND
            
            # 绘制无人机主体
            pygame.draw.circle(
                self.screen, 
                drone_color, 
                (self.position[0], drone_y), 
                self.DRONE_RADIUS
            )
            
            # 更新并绘制螺旋桨
            self._update_prop_positions(drone_y)
            for pos in self._prop_positions:
                pygame.draw.circle(self.screen, self.PROPELLER_COLOR, pos, self.PROPELLER_RADIUS)
            
            # 绘制状态信息和控制说明
            self._draw_status_info()
            self._draw_control_instructions()
            
            # 更新显示
            pygame.display.flip()
            
        except Exception as e:
            print(f"❌ 绘制界面时出错: {e}")
    
    def _draw_status_info(self) -> None:
        """绘制无人机状态信息"""
        status: str = "飞行中" if self.is_flying else "在地面"
        texts: List[str] = [
            f"状态: {status}",
            f"高度: {self.altitude:.1f}m",
            f"电池: {self.battery:.1f}%",
            f"位置: ({self.position[0]}, {self.position[1]})"
        ]
        
        # 批量渲染文本
        y_offset: int = 10
        for text in texts:
            text_surface: pygame.Surface = self.font.render(text, True, self.TEXT_COLOR)
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25
    
    def _draw_control_instructions(self) -> None:
        """绘制控制说明文本"""
        controls: List[str] = [
            "控制说明:",
            "张开手掌 - 起飞",
            "握拳 - 降落",
            "食指指向 - 前进",
            "胜利手势 - 上升",
            "OK手势 - 紧急停止"
        ]
        
        # 批量渲染文本
        y_offset: int = 10
        x_pos: int = self.WINDOW_WIDTH - 200
        for text in controls:
            text_surface: pygame.Surface = self.font.render(text, True, self.TEXT_COLOR)
            self.screen.blit(text_surface, (x_pos, y_offset))
            y_offset += 25
    
    def process_events(self) -> bool:
        """处理pygame窗口事件"""
        try:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return False
                elif event.type == KEYDOWN:
                    # 提前处理退出按键（可选）
                    pass
            return True
        except Exception as e:
            print(f"❌ 处理窗口事件时出错: {e}")
            return False

class GestureRecognizer:
    """
    手势识别器类
    基于MediaPipe实现手部关键点检测，识别预设手势并转换为控制命令
    """
    # 摄像头配置
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_INDICES_TO_TRY: List[int] = [0, 1, 2, 3, 4]
    
    # 手势检测参数
    HAND_DETECTION_CONFIDENCE: float = 0.6
    HAND_TRACKING_CONFIDENCE: float = 0.5
    MAX_HANDS: int = 1
    OK_GESTURE_DISTANCE_THRESHOLD: int = 30
    FINGER_BENT_THRESHOLD: int = 20
    
    # 关键点索引
    THUMB_TIP: int = 4
    INDEX_FINGER_TIP: int = 8
    MIDDLE_FINGER_TIP: int = 12
    RING_FINGER_TIP: int = 16
    PINKY_TIP: int = 20

    def __init__(self) -> None:
        """初始化MediaPipe手部检测和摄像头"""
        self.mp_hands: mp.solutions.hands.Hands = mp.solutions.hands
        self.mp_drawing: mp.solutions.drawing_utils = mp.solutions.drawing_utils
        self.cap: Optional[cv2.VideoCapture] = None
        
        # 初始化手部检测器
        try:
            self.hands: mp.solutions.hands.Hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.MAX_HANDS,
                min_detection_confidence=self.HAND_DETECTION_CONFIDENCE,
                min_tracking_confidence=self.HAND_TRACKING_CONFIDENCE
            )
        except Exception as e:
            print(f"❌ MediaPipe手部检测器初始化失败: {e}")
            raise
        
    def initialize_camera(self) -> bool:
        """初始化摄像头"""
        print("🔍 初始化摄像头...")
        
        for cam_index in self.CAMERA_INDICES_TO_TRY:
            try:
                self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # Windows优化
                if self.cap.isOpened():
                    # 设置摄像头参数（一次性设置）
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少延迟
                    
                    print(f"✅ 找到摄像头在索引 {cam_index}")
                    print("✅ 摄像头初始化成功")
                    return True
            except Exception as e:
                print(f"⚠️  摄像头索引 {cam_index} 初始化失败: {e}")
                continue
        
        raise Exception("❌ 无法找到可用的摄像头")
    
    def detect_gesture(self, frame: np.ndarray) -> Tuple[np.ndarray, str, str]:
        """
        检测帧中的手势
        
        返回:
            处理后的帧, 识别到的手势, 对应的命令
        """
        gesture: str = "未检测到手势"
        command: str = "等待"
        
        try:
            # 转换颜色空间（一次性转换）
            rgb_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 处理帧（禁用写操作以优化性能）
            rgb_frame.flags.writeable = False
            results: mp.solutions.hands.Hands.process = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            # 检测到手部
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制关键点
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # 提取关键点
                    landmarks: LandmarkType = self._extract_landmarks(hand_landmarks, frame.shape)
                    
                    # 识别手势
                    gesture = self._improved_classify_gesture(landmarks)
                    command = self._gesture_to_command(gesture)
                    
        except Exception as e:
            print(f"❌ 手势检测时出错: {e}")
        
        return frame, gesture, command
    
    def _extract_landmarks(self, hand_landmarks: mp.solutions.hands.HandLandmark, 
                          frame_shape: Tuple[int, int, int]) -> LandmarkType:
        """提取手部关键点的像素坐标"""
        h, w, _ = frame_shape
        landmarks: LandmarkType = []
        
        # 批量转换坐标
        for lm in hand_landmarks.landmark:
            px: int = int(lm.x * w)
            py: int = int(lm.y * h)
            landmarks.append((px, py))
            
        return landmarks
    
    def _improved_classify_gesture(self, landmarks: LandmarkType) -> str:
        """改进的手势分类算法"""
        # 校验关键点数量
        if not landmarks or len(landmarks) < 21:
            return "未检测到手势"
        
        # 获取关键点位
        thumb_tip: Tuple[int, int] = landmarks[self.THUMB_TIP]
        index_tip: Tuple[int, int] = landmarks[self.INDEX_FINGER_TIP]
        
        # 检测手指状态
        fingers: List[bool] = self._detect_extended_fingers(landmarks)
        extended_fingers: int = sum(fingers)
        
        # 检测OK手势
        thumb_index_dist: float = np.hypot(
            thumb_tip[0] - index_tip[0], 
            thumb_tip[1] - index_tip[1]
        )
        
        if thumb_index_dist < self.OK_GESTURE_DISTANCE_THRESHOLD and extended_fingers <= 3:
            if self._check_other_fingers_bent(landmarks):
                return "OK手势"
        
        # 基础手势分类
        if extended_fingers == 5:
            return "张开手掌"
        elif extended_fingers == 0:
            return "握拳"
        elif extended_fingers == 1 and fingers[1]:
            return "食指指向"
        elif extended_fingers == 2 and fingers[1] and fingers[2]:
            return "胜利手势"
        else:
            return "其他手势"
    
    def _detect_extended_fingers(self, landmarks: LandmarkType) -> List[bool]:
        """检测每根手指是否伸直"""
        fingers: List[bool] = []
        
        # 拇指检测
        fingers.append(landmarks[self.THUMB_TIP][0] < landmarks[self.THUMB_TIP-1][0])
        
        # 其他手指检测（批量处理）
        finger_indices: List[Tuple[int, int]] = [
            (self.INDEX_FINGER_TIP, self.INDEX_FINGER_TIP-2),
            (self.MIDDLE_FINGER_TIP, self.MIDDLE_FINGER_TIP-2),
            (self.RING_FINGER_TIP, self.RING_FINGER_TIP-2),
            (self.PINKY_TIP, self.PINKY_TIP-2)
        ]
        
        for tip, pip in finger_indices:
            fingers.append(landmarks[tip][1] < landmarks[pip][1])
        
        return fingers
    
    def _check_other_fingers_bent(self, landmarks: LandmarkType) -> bool:
        """检查中指、无名指、小指是否弯曲"""
        finger_checks: List[Tuple[int, int]] = [
            (self.MIDDLE_FINGER_TIP, self.MIDDLE_FINGER_TIP-2),
            (self.RING_FINGER_TIP, self.RING_FINGER_TIP-2),
            (self.PINKY_TIP, self.PINKY_TIP-2)
        ]
        
        for tip, pip in finger_checks:
            if landmarks[tip][1] < landmarks[pip][1] - self.FINGER_BENT_THRESHOLD:
                return False
        return True
    
    def _gesture_to_command(self, gesture: str) -> str:
        """手势到命令的映射"""
        command_map: Dict[str, str] = {
            "张开手掌": "起飞",
            "握拳": "降落",
            "食指指向": "前进",
            "胜利手势": "上升",
            "OK手势": "紧急停止",
            "未检测到手势": "等待",
            "其他手势": "等待"
        }
        return command_map.get(gesture, "等待")
    
    def release_camera(self) -> None:
        """释放摄像头资源"""
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                print("✅ 摄像头资源已释放")
        except Exception as e:
            print(f"⚠️  释放摄像头时出错: {e}")

class GestureDroneSystem:
    """
    手势控制无人机主系统类
    整合手势识别和无人机模拟器
    """
    # 系统配置
    COMMAND_INTERVAL: float = 1.0
    EXIT_KEY: int = ord('q')
    WINDOW_NAME: str = '📷 手势识别摄像头'

    def __init__(self) -> None:
        """初始化系统组件"""
        self.gesture_recognizer: GestureRecognizer = GestureRecognizer()
        self.drone_simulator: VirtualDrone = VirtualDrone()
        self.is_running: bool = False
        
    def initialize(self) -> bool:
        """初始化系统"""
        print("=" * 50)
        print("🤖 手势控制无人机系统")
        print("=" * 50)
        
        try:
            if not self.gesture_recognizer.initialize_camera():
                return False
                
            self._print_usage_instructions()
            return True
            
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            self.cleanup()
            return False
    
    def _print_usage_instructions(self) -> None:
        """打印使用说明"""
        print("\n✅ 系统初始化完成!")
        print("\n📋 手势控制说明:")
        print("✋ 张开手掌 - 起飞")
        print("✊ 握拳 - 降落")
        print("👆 食指指向 - 前进")
        print("✌️ 胜利手势 - 上升")
        print("👌 OK手势 - 紧急停止")
        print(f"\n⌨️  按 '{chr(self.EXIT_KEY)}' 键退出程序")
        print("=" * 50)
    
    def run(self) -> None:
        """运行系统主循环"""
        if not self.initialize():
            return
        
        self.is_running = True
        print("▶️  开始手势控制...")
        
        # 性能统计
        frame_count: int = 0
        start_time: float = time.time()
        last_command_time: float = 0.0
        
        try:
            while self.is_running:
                # 处理窗口事件
                if not self.drone_simulator.process_events():
                    break
                
                # 读取摄像头帧
                ret: bool
                frame: np.ndarray
                ret, frame = self.gesture_recognizer.cap.read()
                
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # 帧处理
                frame_count += 1
                frame = cv2.flip(frame, 1)
                
                # 手势检测
                processed_frame, gesture, command = self.gesture_recognizer.detect_gesture(frame)
                
                # 命令执行控制
                current_time: float = time.time()
                if (current_time - last_command_time > self.COMMAND_INTERVAL and 
                    command != "等待"):
                    if self.drone_simulator.execute_command(command):
                        print(f"✅ 执行命令: {command}")
                        last_command_time = current_time
                elif command != "等待":
                    print(f"⏳ 识别到: {gesture} -> {command} (冷却中)")
                
                # 显示更新
                self._display_info(processed_frame, gesture, command, frame_count, start_time)
                cv2.imshow(self.WINDOW_NAME, processed_frame)
                self.drone_simulator.draw()
                
                # 退出检测
                if cv2.waitKey(1) & 0xFF == self.EXIT_KEY:
                    print("\n🛑 用户请求退出程序")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 用户中断程序")
        except Exception as e:
            print(f"❌ 运行时错误: {e}")
        finally:
            self.cleanup()
            
        # 显示性能统计
        self._show_performance_stats(start_time, frame_count)
    
    def _display_info(self, frame: np.ndarray, gesture: str, command: str, 
                     frame_count: int, start_time: float) -> None:
        """在视频帧上绘制信息"""
        # 计算FPS
        elapsed_time: float = time.time() - start_time
        fps: float = frame_count / elapsed_time if elapsed_time > 0 else 0.0
        
        # 文本配置（批量处理）
        text_configs: List[Tuple[str, Tuple[int, int], ColorType, float, int]] = [
            (f"🤘 手势: {gesture}", (10, 30), (0, 255, 0), 0.7, 2),
            (f"🎮 命令: {command}", (10, 60), (0, 255, 255), 0.7, 2),
            (f"⚡ FPS: {fps:.1f}", (10, 90), (255, 255, 255), 0.6, 2),
            (f"按 '{chr(self.EXIT_KEY)}' 退出", (10, 450), (255, 255, 255), 0.5, 1)
        ]
        
        # 批量绘制文本
        for text, pos, color, scale, thickness in text_configs:
            cv2.putText(
                frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness
            )
    
    def _show_performance_stats(self, start_time: float, frame_count: int) -> None:
        """显示性能统计"""
        total_time: float = time.time() - start_time
        avg_fps: float = frame_count / total_time if total_time > 0 else 0.0
        
        print("\n" + "=" * 50)
        print("📊 性能统计")
        print("=" * 50)
