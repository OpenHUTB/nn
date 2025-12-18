# -*- coding: utf-8 -*-
"""
手势控制AirSim无人机 - 修复连接问题版
确保正确连接AirSim，移除模拟模式
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
print("Gesture Controlled Drone - Fixed Connection Version")
print("=" * 60)

# ========== 修复导入路径 ==========
# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


# ========== 核心模块导入与异常处理 ==========
def safe_import():
    """安全导入所有模块"""
    modules_status = {}

    # 1. 导入PIL
    try:
        from PIL import Image, ImageDraw, ImageFont
        modules_status['PIL'] = True
        print("[PIL] ✓ Image processing library ready")
    except Exception as e:
        modules_status['PIL'] = False
        print(f"[PIL] ✗ Import failed: {e}")
        return None, modules_status

    # 2. 导入OpenCV
    try:
        import cv2
        import numpy as np
        modules_status['OpenCV'] = True
        print("[OpenCV] ✓ Computer vision library ready")
    except Exception as e:
        modules_status['OpenCV'] = False
        print(f"[OpenCV] ✗ Import failed: {e}")
        return None, modules_status

    # 3. 导入MediaPipe
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

    # 4. 尝试导入AirSim - 关键修复部分
    airsim_module = None
    airsim_client = None

    # 尝试多种导入方式
    import_methods = [
        lambda: __import__('airsim'),
        lambda: __import__('AirSim'),
    ]

    for method in import_methods:
        try:
            airsim_module = method()
            modules_status['AirSim'] = True
            print(f"[AirSim] ✓ Successfully imported airsim from {airsim_module.__file__}")
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

        # 询问是否自动安装
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
            'C:/Windows/Fonts/msyh.ttc'
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
            shadow_pos = (pos[0] + 1, pos[1] + 1)
            draw.text(shadow_pos, text, font=font, fill=shadow_color)

            # 绘制文字
            rgb_color = color[::-1]
            draw.text(pos, text, font=font, fill=rgb_color)

            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        size / 30, color, 2)
            return frame


# 初始化中文渲染器
chinese_renderer = ChineseTextRenderer()


# ========== 手势识别模块 ==========
class GestureRecognizer:
    """手势识别器"""

    def __init__(self):
        try:
            self.hands = mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            print("✓ MediaPipe gesture detector initialized")
        except Exception as e:
            print(f"⚠ MediaPipe init failed: {e}")
            self.hands = None

        self.current_gesture = "等待手势"

    def recognize(self, frame):
        """识别手势"""
        if self.hands is None:
            return self.current_gesture

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制手部关键点
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

                    # 简化的手势判断
                    landmarks = hand_landmarks.landmark
                    if len(landmarks) >= 9:
                        wrist = landmarks[0]
                        index_tip = landmarks[8]

                        # 手势判断逻辑
                        if index_tip.y < wrist.y - 0.2:
                            self.current_gesture = "向上"
                        elif index_tip.y > wrist.y + 0.2:
                            self.current_gesture = "向下"
                        elif index_tip.x < wrist.x - 0.2:
                            self.current_gesture = "向左"
                        elif index_tip.x > wrist.x + 0.2:
                            self.current_gesture = "向右"
                        elif abs(index_tip.y - wrist.y) < 0.1:
                            self.current_gesture = "向前"
                        else:
                            self.current_gesture = "停止"

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
            # 创建客户端
            self.client = self.airsim.MultirotorClient()
            self.client.confirmConnection()
            print("✅ Connected to AirSim!")

            # 启用API控制
            self.client.enableApiControl(True)
            print("✅ API control enabled")

            # 解锁无人机
            self.client.armDisarm(True)
            print("✅ Drone armed")

            self.connected = True
            return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("\nPlease ensure:")
            print("1. AirSim simulator is running")
            print("2. In simulator: Settings → Computer Vision mode is OFF")
            print("3. Drone is spawned in the world")
            print("4. Press 'R' in simulator to reset drone if needed")
            return False

    def takeoff(self):
        """起飞"""
        if not self.connected:
            print("❌ Drone not connected")
            return False

        try:
            print("Taking off...")
            # 起飞到5米高度
            self.client.takeoffAsync().join()
            # 悬停1秒稳定
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
            # 设置速度参数（单位：米/秒）
            velocity = 3
            duration = 0.5  # 持续0.5秒

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
    # 初始化组件
    gesture_recognizer = GestureRecognizer()
    drone_controller = DroneController(libs['airsim'])

    # 初始化摄像头
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not available")
            cap = None
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("✓ Camera initialized")
    except:
        print("⚠ Camera init failed")
        cap = None

    # 显示说明
    print("\n" + "=" * 60)
    print("INSTRUCTIONS")
    print("=" * 60)
    print("1. Start AirSim simulator first")
    print("2. Choose environment (e.g., Landscape Mountains)")
    print("3. Press [C] in this window to connect drone")
    print("4. Press [SPACE] to takeoff/land")
    if cap:
        print("5. Gesture control with hand in front of camera")
    else:
        print("5. Keyboard control:")
        print("   [W]Up [S]Down [A]Left [D]Right [F]Forward [X]Stop")
    print("6. Press [ESC] to exit safely")
    print("=" * 60)
    print("Program started successfully!")
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

    # 主循环
    while True:
        frame = None

        # 读取摄像头画面
        if cap:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                current_gesture = gesture_recognizer.recognize(frame)
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                current_gesture = "摄像头错误"
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            current_gesture = gesture_recognizer.current_gesture

        # ========== 绘制界面 ==========
        # 标题栏
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # 主标题
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
        gesture_color = (0, 255, 0) if current_gesture != "等待手势" else (200, 200, 200)
        gesture_text = f"手势: {current_gesture}"
        frame = chinese_renderer.put_text(frame, gesture_text, (10, 120),
                                          size=24, color=gesture_color)

        # 操作提示
        help_text = "C:连接  空格:起飞/降落  ESC:退出"
        if not cap:
            help_text += "  W/A/S/D/F/X:控制"

        frame = chinese_renderer.put_text(frame, help_text, (10, 450),
                                          size=20, color=(255, 255, 255))

        # AirSim连接状态
        if not drone_controller.connected:
            warning_text = "⚠ 请先启动AirSim模拟器，然后按C连接"
            frame = chinese_renderer.put_text(frame, warning_text, (10, 400),
                                              size=18, color=(0, 165, 255))

        # 显示窗口
        cv2.imshow('Gesture Controlled Drone - Press ESC to exit', frame)

        # ========== 键盘控制 ==========
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC键
            print("\nExiting...")
            break

        elif key == ord('c') or key == ord('C'):  # 连接无人机
            if not drone_controller.connected:
                drone_controller.connect()

        elif key == 32:  # 空格键
            if drone_controller.connected:
                if drone_controller.flying:
                    drone_controller.land()
                else:
                    drone_controller.takeoff()
                time.sleep(0.3)

        elif key in key_to_gesture:  # 键盘控制
            gesture = key_to_gesture[key]
            gesture_recognizer.set_simulated_gesture(gesture)
            if drone_controller.connected and drone_controller.flying:
                drone_controller.move_by_gesture(gesture)

        # 真实手势控制
        if (current_gesture and current_gesture != "等待手势" and
                current_gesture != "摄像头错误" and
                drone_controller.connected and drone_controller.flying):
            drone_controller.move_by_gesture(current_gesture)

    # ========== 清理资源 ==========
    print("\nCleaning up resources...")
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    drone_controller.emergency_stop()
    print("Program safely exited")
    print("=" * 60)


# ========== 程序入口 ==========
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nProgram error: {e}")
        traceback.print_exc()
    finally:
        input("\nPress Enter to exit...")