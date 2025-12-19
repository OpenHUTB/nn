# -*- coding: utf-8 -*-
"""
手势控制AirSim无人机 - 修复手势识别版
优化了手势识别算法，增加更多手势判断逻辑
作者: xiaoshiyuan888
"""

import sys
import os
import time
import traceback
import subprocess
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

print("=" * 60)
print("Gesture Controlled Drone - Enhanced Gesture Recognition")
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
        print(f"[MediaPipe] ✗ Import failed: {e}")
        print("  Please install: pip install mediapipe")

    airsim_module = None
    import_methods = [
        lambda: __import__('airsim'),
        lambda: __import__('AirSim'),
    ]

    for method in import_methods:
        try:
            airsim_module = method()
            modules_status['AirSim'] = True
            print(f"[AirSim] ✓ Successfully imported")
            break
        except ImportError:
            continue
        except Exception as e:
            print(f"[AirSim] Import error: {e}")
            continue

    if not modules_status.get('AirSim', False):
        print("\n" + "!" * 60)
        print("⚠ AirSim library NOT FOUND!")
        print("!" * 60)
        print("To install AirSim, run:")
        print("1. First install: pip install msgpack-rpc-python")
        print("2. Then install: pip install airsim")
        print("\nOr from source:")
        print("  pip install git+https://github.com/microsoft/AirSim.git")
        print("!" * 60)

        print("\nDo you want to try automatic installation? (y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            try:
                print("Installing msgpack-rpc-python...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "msgpack-rpc-python"])
                print("Installing AirSim...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "airsim"])
                print("\n✅ Installation complete! Please restart the program.")
            except Exception as e:
                print(f"Installation failed: {e}")
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
cv2, np, mp = libs['cv2'], libs['np'], libs['mp']
Image, ImageDraw, ImageFont = libs['PIL']['Image'], libs['PIL']['ImageDraw'], libs['PIL']['ImageFont']
mp_hands, mp_drawing = libs['mp_hands'], libs['mp_drawing']


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
                self.fonts[20] = ImageFont.truetype(path, 20)
                self.fonts[24] = ImageFont.truetype(path, 24)
                self.fonts[30] = ImageFont.truetype(path, 30)
                print(f"✓ Chinese fonts loaded: {path}")
                return
            except:
                continue
        print("⚠ No Chinese fonts found, using default")

    def put_text(self, frame, text, pos, size=20, color=(255, 255, 255)):
        """在图像上绘制中文文本"""
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)

            font = self.fonts.get(size, self.fonts.get(20))

            # 绘制阴影
            shadow_color = (0, 0, 0)
            shadow_pos = (pos[0] + 2, pos[1] + 2)
            draw.text(shadow_pos, text, font=font, fill=shadow_color)

            # 绘制文字
            rgb_color = color[::-1]
            draw.text(pos, text, font=font, fill=rgb_color)

            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except:
            # 备用方案：使用OpenCV绘制英文
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        size / 30, color, 2)
            return frame


chinese_renderer = ChineseTextRenderer()


# ========== 简化的手势识别模块 ==========
class SimpleGestureRecognizer:
    """简化的手势识别器，不依赖MediaPipe"""

    def __init__(self):
        self.current_gesture = "等待手势"
        self.last_gesture_time = time.time()
        self.gesture_history = []
        self.max_history = 5
        print("✓ Simple gesture recognizer initialized (不使用MediaPipe)")

        # 加载OpenCV的预训练模型用于手部检测
        try:
            # 尝试加载Haar级联分类器
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            # 我们使用人脸检测器作为替代，实际上应该用手部检测器
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("✓ OpenCV cascade classifier loaded")
        except:
            print("⚠ OpenCV cascade classifier not available")
            self.face_cascade = None

    def detect_hand_contour(self, frame):
        """使用轮廓检测手部"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 定义肤色范围
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # 创建肤色掩码
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 找到最大的轮廓（假设是手）
            largest_contour = max(contours, key=cv2.contourArea)

            # 计算轮廓面积
            area = cv2.contourArea(largest_contour)

            if area > 1000:  # 最小面积阈值
                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(largest_contour)

                # 计算轮廓的中心点
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # 计算凸包和凸缺陷
                    hull = cv2.convexHull(largest_contour, returnPoints=False)
                    if len(hull) > 3:
                        defects = cv2.convexityDefects(largest_contour, hull)

                        if defects is not None:
                            # 计算手指数量
                            finger_count = 0
                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                start = tuple(largest_contour[s][0])
                                end = tuple(largest_contour[e][0])
                                far = tuple(largest_contour[f][0])

                                # 计算角度
                                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

                                if angle <= np.pi / 2:
                                    finger_count += 1

                            finger_count = min(finger_count + 1, 5)

                            # 绘制轮廓和中心点
                            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                            cv2.circle(frame, (cx, cy), 7, (255, 0, 0), -1)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            # 在边界框上方显示手指数量
                            cv2.putText(frame, f'Fingers: {finger_count}', (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            return cx, cy, w, h, finger_count, frame

        return None, None, None, None, 0, frame

    def recognize_simple_gesture(self, frame):
        """简化的手势识别"""
        # 检测手部轮廓
        cx, cy, w, h, finger_count, frame = self.detect_hand_contour(frame)

        if cx is None or cy is None:
            return "等待手势", frame

        # 获取图像尺寸
        height, width = frame.shape[:2]

        # 定义区域
        center_x = width // 2
        center_y = height // 2

        # 根据手指数量和位置判断手势
        if finger_count == 0:
            return "停止", frame
        elif finger_count == 1:
            return "向前", frame
        elif finger_count >= 4:
            # 手掌张开，根据位置判断方向
            if cy < center_y - 50:
                return "向上", frame
            elif cy > center_y + 50:
                return "向下", frame
            elif cx < center_x - 50:
                return "向左", frame
            elif cx > center_x + 50:
                return "向右", frame
            else:
                return "停止", frame
        else:
            return "等待手势", frame

    def recognize(self, frame):
        """识别手势"""
        try:
            # 使用简化的手势识别
            gesture, processed_frame = self.recognize_simple_gesture(frame)

            # 手势历史记录和平滑处理
            self.gesture_history.append(gesture)
            if len(self.gesture_history) > self.max_history:
                self.gesture_history.pop(0)

            # 使用多数表决决定最终手势
            if self.gesture_history:
                gesture_counts = {}
                for g in self.gesture_history:
                    gesture_counts[g] = gesture_counts.get(g, 0) + 1

                # 找到出现次数最多的手势
                max_gesture = max(gesture_counts, key=gesture_counts.get)
                if gesture_counts[max_gesture] > len(self.gesture_history) // 2:
                    self.current_gesture = max_gesture
                else:
                    self.current_gesture = gesture

            return self.current_gesture
        except Exception as e:
            print(f"Gesture recognition error: {e}")
            return "识别异常"

    def set_simulated_gesture(self, gesture):
        """设置模拟的手势"""
        self.current_gesture = gesture


# ========== 无人机控制模块 ==========
class DroneController:
    """无人机控制器"""

    def __init__(self, airsim_module):
        self.airsim = airsim_module
        self.client = None
        self.connected = False
        self.flying = False
        self.connection_attempted = False

    def connect(self):
        """连接AirSim无人机"""
        if self.connection_attempted:
            return self.connected

        self.connection_attempted = True
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
            return False

    def takeoff(self):
        """起飞"""
        if not self.connected:
            print("❌ Drone not connected")
            return False

        try:
            print("Taking off...")
            self.client.takeoffAsync().join()
            time.sleep(1)
            self.flying = True
            print("✅ Drone took off successfully")
            return True
        except Exception as e:
            print(f"❌ Takeoff failed: {e}")
            return False

    def land(self):
        """降落"""
        if not self.connected:
            return False

        try:
            print("Landing...")
            self.client.landAsync().join()
            self.flying = False
            print("✅ Drone landed")
            return True
        except Exception as e:
            print(f"Landing failed: {e}")
            return False

    def move_by_gesture(self, gesture):
        """根据手势移动"""
        if not self.connected or not self.flying:
            return

        try:
            velocity = 2.0  # 降低速度以获得更平滑的控制
            duration = 0.3  # 减少持续时间

            if gesture == "向上":
                self.client.moveByVelocityZAsync(0, 0, -velocity, duration).join()
            elif gesture == "向下":
                self.client.moveByVelocityZAsync(0, 0, velocity, duration).join()
            elif gesture == "向左":
                self.client.moveByVelocityAsync(-velocity, 0, 0, duration).join()
            elif gesture == "向右":
                self.client.moveByVelocityAsync(velocity, 0, 0, duration).join()
            elif gesture == "向前":
                self.client.moveByVelocityAsync(0, -velocity, 0, duration).join()
            elif gesture == "停止":
                self.client.hoverAsync().join()
        except Exception as e:
            print(f"Control command failed: {e}")

    def emergency_stop(self):
        """紧急停止"""
        if self.connected:
            try:
                if self.flying:
                    print("Emergency landing...")
                    self.land()
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
                print("✅ Emergency stop complete")
            except:
                pass
        self.connected = False
        self.flying = False


# ========== 主程序 ==========
def main():
    """主函数"""
    # 使用简化的手势识别器
    gesture_recognizer = SimpleGestureRecognizer()
    drone_controller = DroneController(libs['airsim'])

    # 初始化摄像头
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            # 尝试其他摄像头索引
            for i in range(1, 5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"✓ Camera {i} initialized")
                    break
            else:
                print("❌ Camera not available, using keyboard control only")
                cap = None
        else:
            print("✓ Camera 0 initialized")

        if cap:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception as e:
        print(f"⚠ Camera init failed: {e}")
        cap = None

    # 显示说明
    print("\n" + "=" * 60)
    print("操作说明")
    print("=" * 60)
    print("1. 首先启动AirSim模拟器")
    print("2. 选择环境（例如：Landscape Mountains）")
    print("3. 按 [C] 连接无人机")
    print("4. 按 [空格键] 起飞/降落")
    if cap:
        print("5. 手势控制：")
        print("   - 握拳（0个手指）：停止")
        print("   - 伸出1个手指：向前")
        print("   - 张开手掌（4-5个手指）：根据手的位置控制方向")
        print("   * 确保手部在摄像头视野内，光线充足，背景简单")
    else:
        print("5. 键盘控制：")
        print("   [W]向上 [S]向下 [A]向左 [D]向右 [F]向前 [X]停止")
    print("6. 按 [ESC] 安全退出")
    print("=" * 60)
    print("程序启动成功！")
    print("-" * 60)

    # 键盘手势映射
    key_to_gesture = {
        ord('w'): "向上", ord('W'): "向上",
        ord('s'): "向下", ord('S'): "向下",
        ord('a'): "向左", ord('A'): "向左",
        ord('d'): "向右", ord('D'): "向右",
        ord('f'): "向前", ord('F'): "向前",
        ord('x'): "停止", ord('X'): "停止",
    }

    last_control_time = time.time()
    control_interval = 0.5  # 控制间隔，避免过于频繁

    # 主循环
    while True:
        frame = None

        if cap:
            ret, frame = cap.read()
            if ret:
                # 镜像图像以便更直观的控制
                frame = cv2.flip(frame, 1)
                current_gesture = gesture_recognizer.recognize(frame)
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                current_gesture = "摄像头错误"
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            current_gesture = gesture_recognizer.current_gesture

        # ========== 绘制界面 ==========
        # 创建半透明覆盖层
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # 绘制底部操作提示区域
        cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 1.0, frame, 0.0, 0)

        # 标题
        frame = chinese_renderer.put_text(frame, "手势控制无人机系统", (10, 10),
                                          size=30, color=(0, 255, 255))

        # 状态信息
        status_color = (0, 255, 0) if drone_controller.connected else (0, 0, 255)
        status_text = f"无人机: {'已连接' if drone_controller.connected else '未连接'}"
        frame = chinese_renderer.put_text(frame, status_text, (10, 50),
                                          size=24, color=status_color)

        flight_color = (0, 255, 0) if drone_controller.flying else (255, 255, 0)
        flight_status = f"飞行: {'是' if drone_controller.flying else '否'}"
        frame = chinese_renderer.put_text(frame, flight_status, (10, 80),
                                          size=24, color=flight_color)

        # 手势信息
        if current_gesture != "等待手势" and current_gesture != "摄像头错误":
            gesture_color = (0, 255, 0)  # 绿色
            # 添加手势识别置信度指示
            cv2.circle(frame, (600, 60), 10, gesture_color, -1)
        else:
            gesture_color = (200, 200, 200)  # 灰色

        gesture_text = f"手势: {current_gesture}"
        frame = chinese_renderer.put_text(frame, gesture_text, (10, 110),
                                          size=24, color=gesture_color)

        # 操作提示
        help_text = "C:连接  空格:起飞/降落  ESC:退出"
        if not cap:
            help_text += "  W/A/S/D/F/X:控制"

        frame = chinese_renderer.put_text(frame, help_text, (10, 420),
                                          size=20, color=(255, 255, 255))

        # 添加手势提示
        if cap:
            gesture_hint = "提示: 握拳停止，食指向前，手掌张开移动"
            frame = chinese_renderer.put_text(frame, gesture_hint, (10, 450),
                                              size=18, color=(255, 200, 100))

        if not drone_controller.connected:
            warning_text = "⚠ 请先启动AirSim模拟器，然后按C连接"
            frame = chinese_renderer.put_text(frame, warning_text, (10, 360),
                                              size=18, color=(0, 165, 255))

        cv2.imshow('Gesture Controlled Drone - Press ESC to exit', frame)

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
                time.sleep(0.3)

        elif key in key_to_gesture:
            gesture = key_to_gesture[key]
            gesture_recognizer.set_simulated_gesture(gesture)
            current_gesture = gesture
            if drone_controller.connected and drone_controller.flying:
                drone_controller.move_by_gesture(gesture)

        # 真实手势控制（添加间隔控制避免过于频繁）
        current_time = time.time()
        if (current_gesture and current_gesture != "等待手势" and
                current_gesture != "摄像头错误" and current_gesture != "识别异常" and
                drone_controller.connected and drone_controller.flying and
                current_time - last_control_time > control_interval):
            drone_controller.move_by_gesture(current_gesture)
            last_control_time = current_time

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
        input("\n按回车键退出...")