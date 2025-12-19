# -*- coding: utf-8 -*-
"""
手势控制AirSim无人机 - 深度优化版
优化了手势识别准确性和系统稳定性
作者: xiaoshiyuan888
"""

import sys
import os
import time
import traceback
import subprocess
import math
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from collections import deque

print("=" * 60)
print("Gesture Controlled Drone - Advanced Recognition v2.0")
print("=" * 60)

# ========== 修复导入路径 ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


# ========== 核心模块导入 ==========
def safe_import():
    """安全导入所有模块"""
    modules_status = {}

    try:
        from PIL import Image, ImageDraw, ImageFont
        modules_status['PIL'] = True
        print("[PIL] ✓ Image processing library ready")
    except Exception as e:
        modules_status['PIL'] = False
        print(f"[PIL] ✗ Import failed: {e}")
        return None, modules_status

    try:
        import cv2
        import numpy as np
        modules_status['OpenCV'] = True
        print("[OpenCV] ✓ Computer vision library ready")
    except Exception as e:
        modules_status['OpenCV'] = False
        print(f"[OpenCV] ✗ Import failed: {e}")
        return None, modules_status

    # 尝试导入mediapipe，如果失败则使用OpenCV方案
    mp = None
    mp_hands = None
    mp_drawing = None
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        modules_status['MediaPipe'] = True
        print("[MediaPipe] ✓ Gesture recognition library ready")
    except Exception as e:
        modules_status['MediaPipe'] = False
        print(f"[MediaPipe] ✗ Import failed, using OpenCV fallback: {e}")

    airsim_module = None
    try:
        airsim_module = __import__('airsim')
        modules_status['AirSim'] = True
        print(f"[AirSim] ✓ Successfully imported")
    except ImportError:
        print("\n" + "!" * 60)
        print("⚠ AirSim library NOT FOUND!")
        print("!" * 60)
        print("To install AirSim, run:")
        print("1. First install: pip install msgpack-rpc-python")
        print("2. Then install: pip install airsim")
        print("\nOr from source:")
        print("  pip install git+https://github.com/microsoft/AirSim.git")
        print("!" * 60)

        print("\nContinue without AirSim? (y/n)")
        choice = input().strip().lower()
        if choice != 'y':
            sys.exit(1)

    return {
        'cv2': cv2,
        'np': np,
        'mp': mp,
        'mp_hands': mp_hands,
        'mp_drawing': mp_drawing,
        'PIL': {'Image': Image, 'ImageDraw': ImageDraw, 'ImageFont': ImageFont},
        'airsim': airsim_module
    }, modules_status


# 执行导入
libs, status = safe_import()
if not status.get('OpenCV', False) or not status.get('PIL', False):
    print("\n❌ Core libraries missing, cannot start.")
    input("Press Enter to exit...")
    sys.exit(1)

print("-" * 60)
print("✅ Environment check passed, initializing...")
print("-" * 60)

# 解包库
cv2, np = libs['cv2'], libs['np']
Image, ImageDraw, ImageFont = libs['PIL']['Image'], libs['PIL']['ImageDraw'], libs['PIL']['ImageFont']


# ========== 中文显示模块 ==========
class ChineseTextRenderer:
    """中文文本渲染器"""

    def __init__(self):
        self.fonts = {}
        self.load_fonts()

    def load_fonts(self):
        """加载字体"""
        font_paths = [
            'simhei.ttf',
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttc',
            '/System/Library/Fonts/PingFang.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        ]
        for path in font_paths:
            try:
                self.fonts[16] = ImageFont.truetype(path, 16)
                self.fonts[20] = ImageFont.truetype(path, 20)
                self.fonts[24] = ImageFont.truetype(path, 24)
                self.fonts[28] = ImageFont.truetype(path, 28)
                self.fonts[32] = ImageFont.truetype(path, 32)
                print(f"✓ Chinese fonts loaded: {path}")
                return
            except:
                continue
        print("⚠ No Chinese fonts found, using default")

    def put_text(self, frame, text, pos, size=20, color=(255, 255, 255), bg_color=None):
        """在图像上绘制中文文本"""
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)

            font = self.fonts.get(size, self.fonts.get(20))

            # 如果有背景色，先绘制背景
            if bg_color:
                bbox = draw.textbbox(pos, text, font=font)
                padding = 2
                draw.rectangle(
                    [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
                    fill=bg_color
                )

            # 绘制阴影
            shadow_color = (0, 0, 0)
            shadow_pos = (pos[0] + 1, pos[1] + 1)
            draw.text(shadow_pos, text, font=font, fill=shadow_color)

            # 绘制文字
            rgb_color = color[::-1]  # BGR to RGB
            draw.text(pos, text, font=font, fill=rgb_color)

            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            # 备用方案：使用OpenCV绘制英文
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        size / 25, color, 2)
            return frame


chinese_renderer = ChineseTextRenderer()


# ========== 高级手势识别模块 ==========
class AdvancedGestureRecognizer:
    """高级手势识别器，结合多种检测方法"""

    def __init__(self):
        self.use_mediapipe = status.get('MediaPipe', False)

        # 初始化MediaPipe（如果可用）
        if self.use_mediapipe and libs['mp'] is not None:
            try:
                self.hands = libs['mp_hands'].Hands(
                    max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5,
                    static_image_mode=False,
                    model_complexity=1
                )
                print("✓ Using MediaPipe for gesture recognition")
            except:
                self.use_mediapipe = False
                print("⚠ MediaPipe initialization failed, using OpenCV fallback")
        else:
            self.use_mediapipe = False

        # OpenCV手势识别参数
        self.history_size = 10
        self.gesture_history = deque(maxlen=self.history_size)
        self.finger_history = deque(maxlen=self.history_size)
        self.current_gesture = "等待手势"
        self.confidence = 0.0
        self.last_gesture_time = time.time()

        # 肤色检测参数（自适应）
        self.skin_lower = np.array([0, 30, 60], dtype=np.uint8)
        self.skin_upper = np.array([25, 255, 255], dtype=np.uint8)

        # 背景减除
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

        # 手部跟踪
        self.track_window = None
        self.track_box = None
        self.tracking = False

        print("✓ Advanced gesture recognizer initialized")

    def adaptive_skin_detection(self, frame):
        """自适应肤色检测"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 自适应调整肤色范围
        mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        return mask

    def find_hand_contours(self, frame):
        """查找手部轮廓"""
        # 使用背景减除
        fg_mask = self.bg_subtractor.apply(frame)

        # 肤色检测
        skin_mask = self.adaptive_skin_detection(frame)

        # 结合两种掩码
        combined_mask = cv2.bitwise_and(fg_mask, skin_mask)

        # 形态学操作优化
        kernel = np.ones((7, 7), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, combined_mask

        # 找到最大的轮廓（假设是手）
        largest_contour = max(contours, key=cv2.contourArea)

        # 过滤掉太小的轮廓
        if cv2.contourArea(largest_contour) < 5000:
            return None, combined_mask

        return largest_contour, combined_mask

    def analyze_hand_contour(self, contour):
        """分析手部轮廓特征"""
        # 计算轮廓特征
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # 计算凸包和凸缺陷
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) < 3:
            return 0, None, None

        defects = cv2.convexityDefects(contour, hull)

        if defects is None:
            return 0, None, None

        # 计算手指数量
        finger_count = 0
        finger_tips = []

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # 计算角度
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

            # 如果角度小于90度，可能是指缝
            if angle < math.pi / 2:
                finger_count += 1
                finger_tips.append(end)

        # 手指数量加1（大拇指）
        finger_count = min(finger_count + 1, 5)

        # 计算轮廓中心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
        else:
            center = None

        return finger_count, center, finger_tips

    def detect_fingertips_with_contours(self, contour):
        """使用轮廓检测指尖"""
        # 简化轮廓
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 查找凸点（可能是指尖）
        hull = cv2.convexHull(approx, returnPoints=True)

        fingertips = []
        for point in hull:
            fingertips.append(tuple(point[0]))

        return fingertips[:5]  # 最多5个指尖

    def recognize_gesture_by_fingers(self, finger_count, center, frame_shape):
        """根据手指数量识别手势"""
        if center is None:
            return "等待手势", 0.5

        h, w = frame_shape[:2]
        cx, cy = center

        # 计算手部位置（归一化）
        norm_x = cx / w
        norm_y = cy / h

        if finger_count == 0:
            return "停止", 0.9
        elif finger_count == 1:
            return "向前", 0.8
        elif finger_count == 2:
            # V字手势，可能是向前或特殊命令
            return "向前", 0.7
        elif finger_count == 3:
            # 三指手势，根据位置判断
            if norm_y < 0.4:
                return "向上", 0.7
            elif norm_y > 0.6:
                return "向下", 0.7
            else:
                return "停止", 0.6
        elif finger_count >= 4:
            # 手掌张开，根据位置判断方向
            if norm_x < 0.3:
                return "向左", 0.8
            elif norm_x > 0.7:
                return "向右", 0.8
            elif norm_y < 0.4:
                return "向上", 0.8
            elif norm_y > 0.6:
                return "向下", 0.8
            else:
                return "停止", 0.7

        return "等待手势", 0.5

    def recognize_with_mediapipe(self, frame):
        """使用MediaPipe进行手势识别"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gesture = "等待手势"
        confidence = 0.5

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                libs['mp_drawing'].draw_landmarks(
                    frame, hand_landmarks, libs['mp_hands'].HAND_CONNECTIONS,
                    libs['mp_drawing'].DrawingSpec(color=(0, 255, 0), thickness=2),
                    libs['mp_drawing'].DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                # 计算手指状态
                landmarks = hand_landmarks.landmark

                # 简单的手指状态检测
                finger_tips = [4, 8, 12, 16, 20]
                finger_pips = [2, 6, 10, 14, 18]

                extended_fingers = 0
                for tip, pip in zip(finger_tips, finger_pips):
                    if landmarks[tip].y < landmarks[pip].y:
                        extended_fingers += 1

                # 根据伸直的手指数量判断手势
                if extended_fingers == 0:
                    gesture = "停止"
                    confidence = 0.9
                elif extended_fingers == 1:
                    gesture = "向前"
                    confidence = 0.8
                elif extended_fingers >= 4:
                    # 根据手掌方向判断
                    wrist = landmarks[0]
                    middle_mcp = landmarks[9]

                    dx = middle_mcp.x - wrist.x
                    dy = middle_mcp.y - wrist.y

                    if abs(dx) > abs(dy):
                        if dx > 0:
                            gesture = "向右"
                        else:
                            gesture = "向左"
                    else:
                        if dy > 0:
                            gesture = "向下"
                        else:
                            gesture = "向上"
                    confidence = 0.7

        return gesture, confidence, frame

    def smooth_gesture(self, gesture, confidence):
        """平滑手势输出"""
        self.gesture_history.append((gesture, confidence))

        if len(self.gesture_history) < 3:
            return gesture, confidence

        # 统计最近的手势
        gesture_counts = {}
        confidence_sum = {}

        for g, c in self.gesture_history:
            if g not in gesture_counts:
                gesture_counts[g] = 0
                confidence_sum[g] = 0
            gesture_counts[g] += 1
            confidence_sum[g] += c

        # 找到最频繁的手势
        max_count = 0
        best_gesture = gesture
        best_confidence = confidence

        for g, count in gesture_counts.items():
            if count > max_count and g != "等待手势":
                max_count = count
                best_gesture = g
                best_confidence = confidence_sum[g] / count

        # 如果手势稳定，更新当前手势
        if max_count >= len(self.gesture_history) // 2:
            self.current_gesture = best_gesture
            self.confidence = best_confidence

        return self.current_gesture, self.confidence

    def visualize_hand(self, frame, contour, center, finger_count, fingertips):
        """可视化手部检测结果"""
        if contour is not None:
            # 绘制轮廓
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

            # 绘制边界框
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 绘制中心点
            if center:
                cx, cy = center
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

            # 绘制指尖
            for tip in fingertips[:finger_count]:
                cv2.circle(frame, tip, 6, (255, 255, 0), -1)

            # 显示手指数量
            cv2.putText(frame, f'Fingers: {finger_count}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def recognize(self, frame):
        """识别手势"""
        start_time = time.time()

        try:
            # 镜像图像以便更直观的控制
            frame = cv2.flip(frame, 1)

            # 方法1：使用MediaPipe（如果可用）
            if self.use_mediapipe:
                gesture, confidence, processed_frame = self.recognize_with_mediapipe(frame)
                frame = processed_frame
            else:
                # 方法2：使用OpenCV
                contour, mask = self.find_hand_contours(frame)

                if contour is not None:
                    finger_count, center, finger_tips = self.analyze_hand_contour(contour)

                    # 如果凸缺陷检测失败，使用轮廓检测指尖
                    if finger_tips is None or len(finger_tips) < finger_count:
                        finger_tips = self.detect_fingertips_with_contours(contour)

                    self.finger_history.append(finger_count)

                    # 计算平均手指数量（平滑）
                    if len(self.finger_history) > 0:
                        avg_fingers = sum(self.finger_history) / len(self.finger_history)
                        finger_count = int(round(avg_fingers))

                    # 识别手势
                    gesture, confidence = self.recognize_gesture_by_fingers(
                        finger_count, center, frame.shape
                    )

                    # 可视化结果
                    frame = self.visualize_hand(frame, contour, center, finger_count, finger_tips)
                else:
                    gesture, confidence = "等待手势", 0.3

            # 手势平滑处理
            final_gesture, final_confidence = self.smooth_gesture(gesture, confidence)

            # 更新最后检测时间
            self.last_gesture_time = time.time()

            # 显示处理时间
            process_time = (time.time() - start_time) * 1000
            cv2.putText(frame, f'Process: {process_time:.1f}ms', (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            return final_gesture, final_confidence, frame

        except Exception as e:
            print(f"Gesture recognition error: {e}")
            return "识别异常", 0.0, frame

    def set_simulated_gesture(self, gesture):
        """设置模拟的手势"""
        self.current_gesture = gesture
        self.confidence = 0.9


# ========== 增强的无人机控制模块 ==========
class EnhancedDroneController:
    """增强的无人机控制器"""

    def __init__(self, airsim_module):
        self.airsim = airsim_module
        self.client = None
        self.connected = False
        self.flying = False
        self.connection_attempted = False

        # 控制参数
        self.velocity = 3.0
        self.duration = 0.2
        self.altitude = -10.0  # 初始高度（负值表示向上）

        # 控制历史
        self.control_history = deque(maxlen=10)
        self.last_control_time = 0

    def connect(self):
        """连接AirSim无人机"""
        if self.connection_attempted:
            return self.connected

        self.connection_attempted = True

        if self.airsim is None:
            print("⚠ AirSim not available, using simulation mode")
            self.connected = True
            return True

        print("Connecting to AirSim...")

        try:
            self.client = self.airsim.MultirotorClient()
            self.client.confirmConnection()
            print("✅ Connected to AirSim!")

            self.client.enableApiControl(True)
            print("✅ API control enabled")

            self.client.armDisarm(True)
            print("✅ Drone armed")

            self.connected = True
            return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("\n请确保：")
            print("1. AirSim模拟器正在运行")
            print("2. 选择环境（如：Landscape Mountains）")
            print("3. 无人机已在世界中生成")
            print("4. 如果需要，按模拟器中的'R'键重置无人机")

            print("\n使用模拟模式继续？(y/n)")
            choice = input().strip().lower()
            if choice == 'y':
                self.connected = True
                print("✅ Using simulation mode")
                return True

            return False

    def takeoff(self):
        """起飞"""
        if not self.connected:
            print("❌ Drone not connected")
            return False

        try:
            if self.airsim is None or self.client is None:
                print("✅ Simulated takeoff")
                self.flying = True
                return True

            print("Taking off...")
            self.client.takeoffAsync().join()
            time.sleep(1)

            # 上升到指定高度
            self.client.moveToZAsync(self.altitude, 3).join()

            self.flying = True
            print("✅ Drone took off successfully")
            return True
        except Exception as e:
            print(f"❌ Takeoff failed: {e}")
            self.flying = True  # 模拟模式
            return True

    def land(self):
        """降落"""
        if not self.connected:
            return False

        try:
            if self.airsim is None or self.client is None:
                print("✅ Simulated landing")
                self.flying = False
                return True

            print("Landing...")
            self.client.landAsync().join()
            self.flying = False
            print("✅ Drone landed")
            return True
        except Exception as e:
            print(f"Landing failed: {e}")
            self.flying = False
            return False

    def move_by_gesture(self, gesture, confidence=0.7):
        """根据手势移动"""
        if not self.connected or not self.flying:
            return False

        # 检查控制间隔
        current_time = time.time()
        if current_time - self.last_control_time < 0.1:  # 最小控制间隔
            return False

        # 如果置信度太低，不执行动作
        if confidence < 0.5:
            return False

        try:
            if self.airsim is None or self.client is None:
                print(f"Simulated move: {gesture}")
                self.control_history.append((gesture, current_time))
                self.last_control_time = current_time
                return True

            success = False

            if gesture == "向上":
                self.client.moveByVelocityZAsync(0, 0, -self.velocity, self.duration)
                success = True
            elif gesture == "向下":
                self.client.moveByVelocityZAsync(0, 0, self.velocity, self.duration)
                success = True
            elif gesture == "向左":
                self.client.moveByVelocityAsync(-self.velocity, 0, 0, self.duration)
                success = True
            elif gesture == "向右":
                self.client.moveByVelocityAsync(self.velocity, 0, 0, self.duration)
                success = True
            elif gesture == "向前":
                self.client.moveByVelocityAsync(0, -self.velocity, 0, self.duration)
                success = True
            elif gesture == "停止":
                self.client.hoverAsync()
                success = True

            if success:
                self.control_history.append((gesture, current_time))
                self.last_control_time = current_time

            return success
        except Exception as e:
            print(f"Control command failed: {e}")
            return False

    def emergency_stop(self):
        """紧急停止"""
        if self.connected:
            try:
                if self.flying and self.client is not None:
                    print("Emergency landing...")
                    self.land()
                if self.client is not None:
                    self.client.armDisarm(False)
                    self.client.enableApiControl(False)
                    print("✅ Emergency stop complete")
            except:
                pass
        self.connected = False
        self.flying = False


# ========== 性能监控模块 ==========
class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.frame_times = deque(maxlen=60)
        self.fps = 0
        self.last_update = time.time()

    def update(self):
        """更新性能数据"""
        current_time = time.time()
        self.frame_times.append(current_time)

        # 每秒更新一次FPS
        if current_time - self.last_update >= 1.0:
            if len(self.frame_times) > 1:
                self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
            self.last_update = current_time

    def get_stats(self):
        """获取性能统计"""
        return {
            'fps': self.fps,
            'frame_count': len(self.frame_times)
        }


# ========== 主程序 ==========
def main():
    """主函数"""
    # 初始化组件
    gesture_recognizer = AdvancedGestureRecognizer()
    drone_controller = EnhancedDroneController(libs['airsim'])
    performance_monitor = PerformanceMonitor()

    # 初始化摄像头
    cap = None
    available_cameras = []

    # 检测可用摄像头
    for i in range(0, 4):
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            available_cameras.append(i)
            temp_cap.release()

    if available_cameras:
        print(f"✓ Available cameras: {available_cameras}")
        cap = cv2.VideoCapture(available_cameras[0])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"✓ Using camera {available_cameras[0]}")
    else:
        print("❌ No cameras available, using keyboard control only")
        cap = None

    # 显示说明
    print("\n" + "=" * 60)
    print("手势控制无人机系统 - 高级版")
    print("=" * 60)
    print("操作说明：")
    print("1. 首先启动AirSim模拟器（可选）")
    print("2. 按 [C] 连接无人机")
    print("3. 按 [空格键] 起飞/降落")
    if cap:
        print("4. 手势控制：")
        print("   - 握拳（0指）：停止")
        print("   - 食指（1指）：向前")
        print("   - 手掌张开（4-5指）：根据手的位置控制方向")
        print("   * 手势识别置信度 > 50% 时才会执行")
    else:
        print("4. 键盘控制：")
        print("   [W]向上 [S]向下 [A]向左 [D]向右 [F]向前 [X]停止")
    print("5. 按 [ESC] 安全退出")
    print("6. 按 [R] 重置手势识别")
    print("=" * 60)

    # 键盘手势映射
    key_to_gesture = {
        ord('w'): "向上", ord('W'): "向上",
        ord('s'): "向下", ord('S'): "向下",
        ord('a'): "向左", ord('A'): "向左",
        ord('d'): "向右", ord('D'): "向右",
        ord('f'): "向前", ord('F'): "向前",
        ord('x'): "停止", ord('X'): "停止",
    }

    # 主循环
    last_gesture_time = time.time()
    gesture_timeout = 2.0  # 手势超时时间

    while True:
        performance_monitor.update()

        # 读取摄像头帧
        if cap:
            ret, frame = cap.read()
            if not ret:
                # 如果读取失败，尝试重新初始化摄像头
                print("⚠ Camera read failed, trying to reinitialize...")
                cap.release()
                if available_cameras:
                    cap = cv2.VideoCapture(available_cameras[0])
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    ret, frame = cap.read()

                if not ret:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    current_gesture = "摄像头错误"
                    confidence = 0.0
                else:
                    current_gesture, confidence, frame = gesture_recognizer.recognize(frame)
            else:
                current_gesture, confidence, frame = gesture_recognizer.recognize(frame)
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            current_gesture = gesture_recognizer.current_gesture
            confidence = gesture_recognizer.confidence

        # ========== 绘制界面 ==========
        # 创建信息覆盖层
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # 标题和版本
        title = "手势控制无人机系统 - 高级版"
        frame = chinese_renderer.put_text(frame, title, (10, 10), size=28, color=(0, 255, 255))

        # 状态信息
        status_color = (0, 255, 0) if drone_controller.connected else (0, 0, 255)
        status_text = f"状态: {'已连接' if drone_controller.connected else '未连接'}"
        frame = chinese_renderer.put_text(frame, status_text, (10, 50), size=24, color=status_color)

        flight_color = (0, 255, 0) if drone_controller.flying else (255, 165, 0)
        flight_text = f"飞行: {'是' if drone_controller.flying else '否'}"
        frame = chinese_renderer.put_text(frame, flight_text, (10, 80), size=24, color=flight_color)

        # 手势信息
        gesture_color = (0, 255, 0) if confidence > 0.7 else (255, 165, 0) if confidence > 0.5 else (200, 200, 200)
        gesture_text = f"手势: {current_gesture} ({confidence:.1%})"
        frame = chinese_renderer.put_text(frame, gesture_text, (10, 110), size=24, color=gesture_color)

        # 性能信息
        stats = performance_monitor.get_stats()
        perf_text = f"FPS: {stats['fps']:.1f} | 延迟: {1000 / stats['fps']:.1f}ms" if stats['fps'] > 0 else "FPS: 0"
        frame = chinese_renderer.put_text(frame, perf_text, (440, 50), size=20, color=(255, 255, 255))

        # 操作提示
        help_text = "C:连接  空格:起飞/降落  R:重置  ESC:退出"
        if not cap:
            help_text += "  W/A/S/D/F/X:控制"

        # 绘制底部提示栏
        cv2.rectangle(frame, (0, 440), (640, 480), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 1.0, frame, 0.0, 0)
        frame = chinese_renderer.put_text(frame, help_text, (10, 450), size=18, color=(255, 255, 255))

        # 手势提示
        if cap:
            hint_text = "提示: 保持手势稳定，确保良好光线"
            frame = chinese_renderer.put_text(frame, hint_text, (10, 470), size=16, color=(255, 200, 100))

        # 显示连接提示
        if not drone_controller.connected:
            warning_text = "⚠ 按C键连接无人机，或继续使用模拟模式"
            frame = chinese_renderer.put_text(frame, warning_text, (10, 400),
                                              size=18, color=(0, 165, 255), bg_color=(0, 0, 0))

        # 显示图像
        cv2.imshow('Gesture Controlled Drone - Advanced Version', frame)

        # ========== 键盘控制 ==========
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC键
            print("\n退出程序...")
            break

        elif key == ord('c') or key == ord('C'):
            if not drone_controller.connected:
                drone_controller.connect()

        elif key == 32:  # 空格键
            if drone_controller.connected:
                if drone_controller.flying:
                    drone_controller.land()
                else:
                    drone_controller.takeoff()
                time.sleep(0.5)

        elif key == ord('r') or key == ord('R'):
            print("重置手势识别...")
            gesture_recognizer = AdvancedGestureRecognizer()

        elif key in key_to_gesture:
            gesture = key_to_gesture[key]
            gesture_recognizer.set_simulated_gesture(gesture)
            current_gesture = gesture
            confidence = 0.9
            if drone_controller.connected and drone_controller.flying:
                drone_controller.move_by_gesture(gesture, confidence)

        # 真实手势控制
        current_time = time.time()
        if (current_gesture and current_gesture != "等待手势" and
                current_gesture != "摄像头错误" and current_gesture != "识别异常" and
                drone_controller.connected and drone_controller.flying and
                confidence > 0.5 and  # 只有置信度足够高才执行
                current_time - last_gesture_time > 0.2):  # 控制频率限制

            success = drone_controller.move_by_gesture(current_gesture, confidence)
            if success:
                last_gesture_time = current_time

        # 检查手势超时
        if current_time - gesture_recognizer.last_gesture_time > gesture_timeout:
            gesture_recognizer.current_gesture = "等待手势"

    # 清理资源
    print("\n清理资源...")
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    drone_controller.emergency_stop()
    print("程序安全退出")
    print("=" * 60)


# ========== 程序入口 ==========
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序错误: {e}")
        traceback.print_exc()
    finally:
        print("\n按回车键退出...")
        input()