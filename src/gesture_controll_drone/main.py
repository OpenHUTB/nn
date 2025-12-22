import sys
import time
from typing import List, Tuple, Dict, Optional, Union

import cv2
import numpy as np
import pygame
import mediapipe as mp
from pygame.locals import QUIT, KEYDOWN
from PIL import Image, ImageDraw, ImageFont  # 引入Pillow处理中文显示


# Pygame类型别名
ColorType = Tuple[int, int, int]
PositionType = List[int]
LandmarkType = List[Tuple[int, int]]


# 全局工具函数：修复OpenCV/Pygame的中文显示
def get_chinese_font(font_size: int = 24) -> ImageFont.FreeTypeFont:
    """获取中文字体（兼容多系统）"""
    # 优先尝试Windows系统字体
    font_paths = [
        "simhei.ttf",  # Windows默认黑体
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",  # Linux
        "/Library/Fonts/PingFang.ttc"  # macOS
    ]
    
    for path in font_paths:
        try:
            return ImageFont.truetype(path, font_size, encoding="utf-8")
        except:
            continue
    #  fallback到默认字体（可能不支持中文）
    print("⚠️  未找到中文字体，使用默认字体")
    return ImageFont.load_default()


def put_chinese_on_opencv(frame: np.ndarray, text: str, position: Tuple[int, int], 
                          font_size: int = 24, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """在OpenCV帧上绘制中文"""
    # BGR转RGB
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_frame)
    # 绘制中文
    draw.text(position, text, font=get_chinese_font(font_size), fill=(color[2], color[1], color[0]))
    # RGB转BGR
    return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)


def put_chinese_on_pygame(surface: pygame.Surface, text: str, position: Tuple[int, int], 
                          font_size: int = 24, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    """在Pygame表面上绘制中文"""
    # Pygame表面转PIL图像
    pil_surface = Image.fromarray(pygame.surfarray.array3d(surface).swapaxes(0, 1))
    draw = ImageDraw.Draw(pil_surface)
    # 绘制中文
    draw.text(position, text, font=get_chinese_font(font_size), fill=color)
    # PIL图像转Pygame表面
    pygame.surfarray.blit_array(surface, np.array(pil_surface).swapaxes(0, 1))


class VirtualDrone:
    """虚拟无人机模拟器类（修复Pygame中文显示）"""
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
        
        # 预计算的渲染位置
        self._prop_positions: List[Tuple[int, int]] = []
        self._update_prop_positions(150)  # 初始位置
        
        print("✅ 虚拟无人机模拟器已启动")
    
    def _update_prop_positions(self, drone_y: int) -> None:
        """更新螺旋桨位置"""
        self._prop_positions = [
            (self.position[0] - 20, drone_y - 12),
            (self.position[0] + 20, drone_y - 12),
            (self.position[0] - 20, drone_y + 12),
            (self.position[0] + 20, drone_y + 12)
        ]
    
    def execute_command(self, command: str) -> bool:
        """执行无人机控制命令"""
        result: bool = False
        
        try:
            if command == "起飞" and not self.is_flying:
                self.is_flying = True
                self.altitude = 10.0
                result = True
            elif command == "降落" and self.is_flying:
                self.is_flying = False
                self.altitude = 0.0
                result = True
            elif command == "前进" and self.is_flying:
                self.position[1] = max(50, self.position[1] - self.SPEED)
                self.altitude = min(50.0, self.altitude + 0.5)
                result = True
            elif command == "上升" and self.is_flying:
                self.altitude = min(100.0, self.altitude + 10.0)
                result = True
            elif command == "紧急停止":
                self.is_flying = False
                self.altitude = 0.0
                result = True
                
            # 模拟电池消耗
            if self.is_flying:
                self.battery = max(0.0, self.battery - self.BATTERY_CONSUMPTION_RATE)
                
        except Exception as e:
            print(f"❌ 执行命令 '{command}' 时出错: {e}")
            
        return result
    
    def draw(self) -> None:
        """绘制无人机界面（修复中文显示）"""
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
            
            # 绘制无人机主体
            drone_color = self.DRONE_COLOR_FLYING if self.is_flying else self.DRONE_COLOR_GROUND
            pygame.draw.circle(self.screen, drone_color, (self.position[0], drone_y), self.DRONE_RADIUS)
            
            # 绘制螺旋桨
            self._update_prop_positions(drone_y)
            for pos in self._prop_positions:
                pygame.draw.circle(self.screen, self.PROPELLER_COLOR, pos, self.PROPELLER_RADIUS)
            
            # 绘制状态信息（用修复后的中文方法）
            self._draw_status_info()
            self._draw_control_instructions()
            
            # 更新显示
            pygame.display.flip()
            
        except Exception as e:
            print(f"❌ 绘制界面时出错: {e}")
    
    def _draw_status_info(self) -> None:
        """绘制无人机状态信息（中文）"""
        status = "飞行中" if self.is_flying else "在地面"
        texts = [
            f"状态: {status}",
            f"高度: {self.altitude:.1f}m",
            f"电池: {self.battery:.1f}%",
            f"位置: ({self.position[0]}, {self.position[1]})"
        ]
        
        # 用Pillow绘制中文
        y_offset = 10
        for text in texts:
            put_chinese_on_pygame(self.screen, text, (10, y_offset), font_size=24, color=self.TEXT_COLOR)
            y_offset += 25
    
    def _draw_control_instructions(self) -> None:
        """绘制控制说明（中文）"""
        controls = [
            "控制说明:",
            "张开手掌 - 起飞",
            "握拳 - 降落",
            "食指指向 - 前进",
            "胜利手势 - 上升",
            "OK手势 - 紧急停止"
        ]
        
        # 用Pillow绘制中文
        y_offset = 10
        x_pos = self.WINDOW_WIDTH - 200
        for text in controls:
            put_chinese_on_pygame(self.screen, text, (x_pos, y_offset), font_size=24, color=self.TEXT_COLOR)
            y_offset += 25
    
    def process_events(self) -> bool:
        """处理pygame窗口事件"""
        try:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return False
            return True
        except Exception as e:
            print(f"❌ 处理窗口事件时出错: {e}")
            return False


class GestureRecognizer:
    """手势识别器类"""
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
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap: Optional[cv2.VideoCapture] = None
        
        # 初始化手部检测器
        try:
            self.hands = self.mp_hands.Hands(
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
                if sys.platform == "win32":
                    self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(cam_index)
                    
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print(f"✅ 找到摄像头在索引 {cam_index}")
                    return True
            except Exception as e:
                print(f"⚠️  摄像头索引 {cam_index} 初始化失败: {e}")
                continue
        
        raise Exception("❌ 无法找到可用的摄像头")
    
    def detect_gesture(self, frame: np.ndarray) -> Tuple[np.ndarray, str, str]:
        """检测帧中的手势"""
        gesture: str = "未检测到手势"
        command: str = "等待"
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    landmarks = self._extract_landmarks(hand_landmarks, frame.shape)
                    gesture = self._improved_classify_gesture(landmarks)
                    command = self._gesture_to_command(gesture)
                    
        except Exception as e:
            print(f"❌ 手势检测时出错: {e}")
        
        return frame, gesture, command
    
    def _extract_landmarks(self, hand_landmarks, frame_shape: Tuple[int, int, int]) -> LandmarkType:
        """提取手部关键点的像素坐标"""
        h, w, _ = frame_shape
        landmarks: LandmarkType = []
        for lm in hand_landmarks.landmark:
            px = int(lm.x * w)
            py = int(lm.y * h)
            landmarks.append((px, py))
        return landmarks
    
    def _improved_classify_gesture(self, landmarks: LandmarkType) -> str:
        """改进的手势分类算法"""
        if not landmarks or len(landmarks) < 21:
            return "未检测到手势"
        
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_FINGER_TIP]
        fingers = self._detect_extended_fingers(landmarks)
        extended_fingers = sum(fingers)
        
        # 检测OK手势
        thumb_index_dist = np.hypot(thumb_tip[0]-index_tip[0], thumb_tip[1]-index_tip[1])
        if thumb_index_dist < self.OK_GESTURE_DISTANCE_THRESHOLD and extended_fingers <=3 and self._check_other_fingers_bent(landmarks):
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
        fingers = [landmarks[self.THUMB_TIP][0] < landmarks[self.THUMB_TIP-1][0]]
        finger_indices = [
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
        finger_checks = [
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
    
    def release_camera(self) -> None:
        """释放摄像头资源"""
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception as e:
            print(f"⚠️  释放摄像头时出错: {e}")


class GestureDroneSystem:
    """手势控制无人机主系统类"""
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
        
        frame_count = 0
        start_time = time.time()
        last_command_time = 0.0
        
        try:
            while self.is_running:
                if not self.drone_simulator.process_events():
                    break
                
                ret, frame = self.gesture_recognizer.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                frame = cv2.flip(frame, 1)
                processed_frame, gesture, command = self.gesture_recognizer.detect_gesture(frame)
                
                # 执行命令
                current_time = time.time()
                if current_time - last_command_time > self.COMMAND_INTERVAL and command != "等待":
                    if self.drone_simulator.execute_command(command):
                        print(f"✅ 执行命令: {command}")
                        last_command_time = current_time
                elif command != "等待":
                    print(f"⏳ 识别到: {gesture} -> {command} (冷却中)")
                
                # 显示信息（修复OpenCV中文）
                self._display_info(
                    processed_frame, gesture, command, frame_count, start_time,
                    drone_status="飞行中" if self.drone_simulator.is_flying else "在地面",
                    drone_altitude=self.drone_simulator.altitude
                )
                cv2.imshow(self.WINDOW_NAME, processed_frame)
                self.drone_simulator.draw()
                
                # 退出检测
                if cv2.waitKey(1) & 0xFF == self.EXIT_KEY:
                    break
        except Exception as e:
            print(f"❌ 运行时错误: {e}")
        finally:
            self.cleanup()
        self._show_performance_stats(start_time, frame_count)
    
    def _display_info(self, frame: np.ndarray, gesture: str, command: str, 
                     frame_count: int, start_time: float,
                     drone_status: str, drone_altitude: float) -> None:
        """在视频帧上绘制信息（修复中文）"""
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
        
        # 绘制中文信息
        frame = put_chinese_on_opencv(frame, f"🤘 手势: {gesture}", (10, 30), font_size=24, color=(0, 255, 0))
        frame = put_chinese_on_opencv(frame, f"🎮 命令: {command}", (10, 60), font_size=24, color=(0, 255, 255))
        frame = put_chinese_on_opencv(frame, f"✈️  无人机状态: {drone_status}", (10, 90), font_size=24, color=(255, 255, 0))
        frame = put_chinese_on_opencv(frame, f"📏 无人机高度: {drone_altitude:.1f}m", (10, 120), font_size=24, color=(255, 255, 0))
        
        # 绘制FPS和退出提示
        cv2.putText(frame, f"⚡ FPS: {fps:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"按 '{chr(self.EXIT_KEY)}' 退出", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _show_performance_stats(self, start_time: float, frame_count: int) -> None:
        """显示性能统计"""
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0.0
        print("\n" + "=" * 50)
        print("📊 性能统计")
        print(f"⏱️  总运行时间: {total_time:.2f} 秒")
        print(f"🖼️  处理帧数: {frame_count}")
        print(f"⚡ 平均FPS: {avg_fps:.2f}")
        print("=" * 50)
    
    def cleanup(self) -> None:
        """清理系统资源"""
        self.is_running = False
        print("\n🧹 正在清理系统资源...")
        self.gesture_recognizer.release_camera()
        cv2.destroyAllWindows()
        pygame.quit()
        print("✅ 系统已安全关闭")


if __name__ == "__main__":
    try:
        drone_system = GestureDroneSystem()
        drone_system.run()
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        sys.exit(1)
