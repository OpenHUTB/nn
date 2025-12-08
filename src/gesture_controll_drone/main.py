import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import sys
from pygame.locals import *

class VirtualDrone:
    """虚拟无人机模拟器"""
    
    def __init__(self):
        pygame.init()
        self.width, self.height = 400, 300
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("虚拟无人机模拟器")
        
        # 无人机状态
        self.position = [200, 150]
        self.altitude = 0
        self.battery = 100
        self.is_flying = False
        self.speed = 3
        
        # 颜色和字体
        self.bg_color = (30, 30, 50)
        self.drone_color_flying = (0, 255, 0)
        self.drone_color_ground = (255, 100, 100)
        self.text_color = (255, 255, 255)
        self.font = pygame.font.Font(None, 24)
        
        print("虚拟无人机模拟器已启动")
    
    def execute_command(self, command):
        """执行无人机命令"""
        result = False
        
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
            
        # 模拟电池消耗
        if self.is_flying:
            self.battery = max(0, self.battery - 0.05)
            
        return result
    
    def draw(self):
        """绘制无人机和状态信息"""
        # 清屏
        self.screen.fill(self.bg_color)
        
        # 绘制地面
        pygame.draw.rect(self.screen, (50, 50, 70), (0, self.height - 100, self.width, 100))
        
        # 绘制无人机
        drone_color = self.drone_color_flying if self.is_flying else self.drone_color_ground
        drone_y = self.height - 120 - self.altitude * 2
        
        # 无人机主体
        pygame.draw.circle(self.screen, drone_color, (self.position[0], drone_y), 15)
        
        # 无人机螺旋桨
        pygame.draw.circle(self.screen, (200, 200, 200), (self.position[0] - 20, drone_y - 12), 6)
        pygame.draw.circle(self.screen, (200, 200, 200), (self.position[0] + 20, drone_y - 12), 6)
        pygame.draw.circle(self.screen, (200, 200, 200), (self.position[0] - 20, drone_y + 12), 6)
        pygame.draw.circle(self.screen, (200, 200, 200), (self.position[0] + 20, drone_y + 12), 6)
        
        # 绘制状态信息
        status = "飞行中" if self.is_flying else "在地面"
        texts = [
            f"状态: {status}",
            f"高度: {self.altitude:.1f}m",
            f"电池: {self.battery:.1f}%",
            f"位置: ({self.position[0]}, {self.position[1]})"
        ]
        
        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, self.text_color)
            self.screen.blit(text_surface, (10, 10 + i * 25))
        
        # 绘制控制说明
        controls = [
            "控制说明:",
            "张开手掌 - 起飞",
            "握拳 - 降落",
            "食指指向 - 前进",
            "胜利手势 - 上升",
            "OK手势 - 紧急停止"  # 修改这里
        ]
        
        for i, control in enumerate(controls):
            text_surface = self.font.render(control, True, self.text_color)
            self.screen.blit(text_surface, (self.width - 200, 10 + i * 25))
        
        # 更新显示
        pygame.display.flip()
    
    def process_events(self):
        """处理Pygame事件"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
        return True

class GestureRecognizer:
    """改进的手势识别器 - 添加OK手势"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = None
        
    def initialize_camera(self):
        """初始化摄像头"""
        print("初始化摄像头...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            for i in range(1, 5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"找到摄像头在索引 {i}")
                    break
            else:
                raise Exception("无法找到可用的摄像头")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("摄像头初始化成功")
        return True
    
    def detect_gesture(self, frame):
        """检测手势并返回命令"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = "未检测到手势"
        command = "等待"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                landmarks = []
                for lm in hand_landmarks.landmark:
                    h, w, c = frame.shape
                    landmarks.append((int(lm.x * w), int(lm.y * h)))
                
                # 识别手势
                gesture = self._improved_classify_gesture(landmarks)
                command = self._gesture_to_command(gesture)
        
        return frame, gesture, command
    
    def _improved_classify_gesture(self, landmarks):
        """改进的手势分类算法 - 添加OK手势检测"""
        if not landmarks or len(landmarks) < 21:
            return "未检测到手势"
        
        # 关键点索引
        THUMB_TIP = 4
        INDEX_FINGER_TIP = 8
        MIDDLE_FINGER_TIP = 12
        RING_FINGER_TIP = 16
        PINKY_TIP = 20
        
        thumb_tip = landmarks[THUMB_TIP]
        index_tip = landmarks[INDEX_FINGER_TIP]
        middle_tip = landmarks[MIDDLE_FINGER_TIP]
        ring_tip = landmarks[RING_FINGER_TIP]
        pinky_tip = landmarks[PINKY_TIP]
        wrist = landmarks[0]
        
        # 改进的手指状态检测
        fingers = []
        
        # 拇指：比较指尖和IP关节的x坐标
        fingers.append(thumb_tip[0] < landmarks[THUMB_TIP-1][0])
        
        # 其他手指：比较指尖和PIP关节的y坐标
        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        finger_pips = [INDEX_FINGER_TIP-2, MIDDLE_FINGER_TIP-2, RING_FINGER_TIP-2, PINKY_TIP-2]
        
        for tip, pip_index in zip(finger_tips, finger_pips):
            fingers.append(tip[1] < landmarks[pip_index][1])
        
        # 计算手指伸直数量
        extended_fingers = sum(fingers)
        
        # 检测OK手势 - 拇指和食指接触，其他手指伸直或微弯
        # 计算拇指和食指之间的距离
        thumb_index_dist = np.sqrt((thumb_tip[0]-index_tip[0])**2 + (thumb_tip[1]-index_tip[1])**2)
        
        # 如果拇指和食指距离很近，且其他手指没有完全伸直
        if thumb_index_dist < 30 and extended_fingers <= 3:
            # 检查其他手指是否弯曲
            other_fingers_bent = True
            for i in range(2, 5):  # 检查中指、无名指和小指
                if fingers[i] and landmarks[finger_tips[i][1]] < landmarks[finger_pips[i]][1] - 20:
                    other_fingers_bent = False
                    break
            
            if other_fingers_bent:
                return "OK手势"
        
        # 改进的手势分类逻辑
        if extended_fingers == 5:
            return "张开手掌"
        elif extended_fingers == 0:
            return "握拳"
        elif extended_fingers == 1 and fingers[1]:  # 只有食指伸直
            return "食指指向"
        elif extended_fingers == 2 and fingers[1] and fingers[2]:  # 食指和中指伸直
            return "胜利手势"
        else:
            return "其他手势"
    
    def _gesture_to_command(self, gesture):
        """将手势转换为控制命令 - 修改为OK手势作为紧急停止"""
        command_map = {
            "张开手掌": "起飞",
            "握拳": "降落",
            "食指指向": "前进",
            "胜利手势": "上升",
            "OK手势": "紧急停止",  # 修改这里
            "未检测到手势": "等待",
            "其他手势": "等待"
        }
        return command_map.get(gesture, "等待")
    
    def release_camera(self):
        """释放摄像头资源"""
        if self.cap:
            self.cap.release()

class GestureDroneSystem:
    """手势控制无人机主系统"""
    
    def __init__(self):
        self.gesture_recognizer = GestureRecognizer()
        self.drone_simulator = VirtualDrone()
        self.is_running = False
        
    def initialize(self):
        """初始化系统"""
        print("=" * 50)
        print("手势控制无人机系统")
        print("=" * 50)
        
        if not self.gesture_recognizer.initialize_camera():
            return False
            
        print("\n系统初始化完成!")
        print("\n手势控制说明:")
        print("✋ 张开手掌 - 起飞")
        print("✊ 握拳 - 降落")
        print("👆 食指指向 - 前进")
        print("✌️ 胜利手势 - 上升")
        print("👌 OK手势 - 紧急停止")  # 修改这里
        print("\n按 'q' 键退出程序")
        print("=" * 50)
        
        return True
    
    def run(self):
        """运行主循环"""
        if not self.initialize():
            return
        
        self.is_running = True
        print("开始手势控制...")
        
        frame_count = 0
        start_time = time.time()
        last_command_time = 0
        command_interval = 1.0
        
        try:
            while self.is_running:
                if not self.drone_simulator.process_events():
                    break
                
                ret, frame = self.gesture_recognizer.cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                frame_count += 1
                frame = cv2.flip(frame, 1)
                
                processed_frame, gesture, command = self.gesture_recognizer.detect_gesture(frame)
                
                current_time = time.time()
                if (current_time - last_command_time > command_interval and 
                    command != "等待"):
                    if self.drone_simulator.execute_command(command):
                        print(f"✅ 执行命令: {command}")
                        last_command_time = current_time
                elif command != "等待":
                    print(f"⏳ 识别到: {gesture} -> {command}")
                
                self._display_info(processed_frame, gesture, command, frame_count, start_time)
                
                cv2.imshow('手势识别摄像头', processed_frame)
                self.drone_simulator.draw()
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except Exception as e:
            print(f"运行时错误: {e}")
        finally:
            self.cleanup()
            
        self._show_performance_stats(start_time, frame_count)
    
    def _display_info(self, frame, gesture, command, frame_count, start_time):
        """在视频帧上显示信息"""
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        cv2.putText(frame, f"手势: {gesture}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"命令: {command}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, "按 'q' 退出", (10, 450),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _show_performance_stats(self, start_time, frame_count):
        """显示性能统计"""
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
        """清理资源"""
        self.is_running = False
        self.gesture_recognizer.release_camera()
        cv2.destroyAllWindows()
        pygame.quit()
        print("系统已关闭")

if __name__ == "__main__":
    system = GestureDroneSystem()
    system.run()