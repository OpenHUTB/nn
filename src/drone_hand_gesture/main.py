import cv2
import numpy as np
import time
import threading
import sys
import os
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
# 优先使用MediaPipe，失败则使用OpenCV版本的检测器
USE_CV_DETECTOR = False
HAS_ENHANCED_DETECTOR = False

try:
    import mediapipe as mp
    mp.solutions.hands  # 测试是否可用
    from gesture_detector_enhanced import EnhancedGestureDetector
    print("[OK] 使用 MediaPipe 手势检测器")
    HAS_ENHANCED_DETECTOR = True
except (ImportError, AttributeError) as e:
    print(f"[WARNING] MediaPipe 不可用: {e}")
    try:
        from gesture_detector_cv import CVGestureDetector
        print("[OK] 使用 OpenCV 手势检测器 (无MediaPipe依赖)")
        USE_CV_DETECTOR = True
        HAS_ENHANCED_DETECTOR = True
    except ImportError:
        print("[WARNING] OpenCV检测器也不可用")
        try:
            from gesture_detector import GestureDetector
            HAS_ENHANCED_DETECTOR = False
        except ImportError:
            print("[ERROR] 没有任何手势检测器可用!")
            sys.exit(1)

from drone_controller import DroneController
from simulation_3d import Drone3DViewer

# 注意：physics_engine.py 是可选的，如果没有可以先注释掉
try:
    from physics_engine import PhysicsEngine

    HAS_PHYSICS_ENGINE = True
except ImportError:
    print("警告：未找到 physics_engine.py，使用简化的物理模拟")
    HAS_PHYSICS_ENGINE = False


class IntegratedDroneSimulation:
    """集成的无人机仿真系统"""

    def __init__(self, config=None):
        # 配置
        self.config = config or {}

        # 系统状态
        self.running = True
        self.paused = False

        # 初始化模块
        print("正在初始化手势检测器...")

        # 检查可用的模型文件（按优先级排序）
        model_candidates = [
            ("dataset/models/gesture_svm.pkl", "SVM模型"),
            ("dataset/models/gesture_random_forest.pkl", "随机森林模型"),
            ("dataset/models/gesture_mlp.pkl", "神经网络模型"),
        ]

        selected_model = None
        selected_model_name = None

        for model_path, model_name in model_candidates:
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                print(f"[INFO] 找到 {model_name}: {file_size / 1024:.1f} KB")

                # 检查文件大小是否合理
                if file_size > 10 * 1024:
                    selected_model = model_path
                    selected_model_name = model_name
                    print(f"[OK] 选择: {model_name}")
                    break

        # 初始化手势检测器（根据可用情况选择）
        if USE_CV_DETECTOR:
            # 使用OpenCV版本的检测器（无MediaPipe依赖）
            print("[OK] 初始化 OpenCV 手势检测器")
            self.gesture_detector = CVGestureDetector()
            self.gesture_detector.use_ml = False
        elif HAS_ENHANCED_DETECTOR:
            # 优先使用增强版检测器（MediaPipe）
            try:
                from gesture_detector_enhanced import EnhancedGestureDetector
                print("[OK] 导入增强版手势检测器")

                if selected_model:
                    # 使用机器学习模型
                    self.gesture_detector = EnhancedGestureDetector(
                        ml_model_path=selected_model,
                        use_ml=True
                    )
                    print(f"[INFO] 使用模型: {selected_model_name}")
                    
                    # 验证模型是否真正加载成功
                    if hasattr(self.gesture_detector, 'ml_classifier') and self.gesture_detector.ml_classifier:
                        print(f"[OK] 机器学习模型加载成功")
                        print(f"   可识别手势: {self.gesture_detector.ml_classifier.gesture_classes}")
                    else:
                        print("[WARNING] 机器学习模型未加载，使用规则检测")
                        self.gesture_detector = EnhancedGestureDetector(use_ml=False)
                else:
                    # 使用规则检测（不需要模型）
                    print("[INFO] 未找到模型文件，使用规则检测")
                    self.gesture_detector = EnhancedGestureDetector(use_ml=False)

            except (ImportError, Exception) as e:
                print(f"[WARNING] 无法导入增强版检测器: {e}")
                print("[OK] 使用原始手势检测器")
                from gesture_detector import GestureDetector
                self.gesture_detector = GestureDetector()
        else:
            print("[WARNING] 增强版检测器不可用")
            print("[OK] 使用原始手势检测器")
            from gesture_detector import GestureDetector
            self.gesture_detector = GestureDetector()

        print("正在初始化无人机控制器...")
        self.drone_controller = DroneController(simulation_mode=True)

        print("正在初始化3D仿真显示...")
        self.viewer = Drone3DViewer(
            width=self.config.get('window_width', 1024),
            height=self.config.get('window_height', 768)
        )

        # 初始化物理引擎（可选）
        if HAS_PHYSICS_ENGINE:
            print("正在初始化物理引擎...")
            self.physics_engine = PhysicsEngine(
                mass=self.config.get('drone_mass', 1.0),
                gravity=self.config.get('gravity', 9.81)
            )
        else:
            self.physics_engine = None

        # 线程
        self.gesture_thread = None
        self.simulation_thread = None

        # 数据共享
        self.current_frame = None
        self.current_gesture = None
        self.gesture_confidence = 0.0
        self.hand_landmarks = None
        self.current_intensity = 0.5  # 当前手势强度

        # 连续控制参数
        self.continuous_control_enabled = True  # 启用连续控制
        self.continuous_control_interval = 0.3  # 连续控制间隔（秒）

        # 控制参数（降低阈值以提高识别率）
        self.control_intensity = 1.0
        self.last_command_time = time.time()
        self.command_cooldown = 1.5  # 命令冷却时间（秒），从2.0降低到1.5

        # 双手控制模式
        self.dual_hand_mode = True  # 启用双手控制模式
        self.last_direction_command = None
        self.last_altitude_command = None
        self.dual_control_cooldown = 0.3  # 双手控制更新间隔（秒）

        # 手势识别阈值（降低以提高灵敏度）
        # 如果是机器学习模式，阈值可以进一步降低
        if USE_CV_DETECTOR:
            print("[OK] OpenCV模式，置信度阈值为0.55")
            base_threshold = 0.55
        elif HAS_ENHANCED_DETECTOR and hasattr(self.gesture_detector, 'use_ml') and self.gesture_detector.use_ml:
            print("[OK] 使用机器学习模式，置信度阈值更低")
            base_threshold = 0.55  # 机器学习可以更低
        else:
            base_threshold = 0.6  # 规则检测需要高一点

        self.gesture_thresholds = {
            'open_palm': base_threshold,
            'closed_fist': base_threshold + 0.05,
            'victory': base_threshold + 0.05,
            'thumb_up': base_threshold + 0.05,
            'thumb_down': base_threshold + 0.05,
            'pointing_up': base_threshold,
            'pointing_down': base_threshold,
            'ok_sign': base_threshold + 0.1,
            'default': base_threshold
        }

        # 初始化摄像头
        self.cap = self._initialize_camera()

        # 镜像设置
        self.mirror_mode = True  # 默认开启镜像

        # 数据记录
        self.data_log = []
        self.log_file = "flight_log.json"

        print("无人机初始化完成，等待手势指令...")

        print("无人机仿真系统初始化完成 [OK]")

        # 显示初始灵敏度设置
        if hasattr(self.gesture_detector, 'get_sensitivity_info'):
            sens_info = self.gesture_detector.get_sensitivity_info()
            print(f"[INFO] 初始灵敏度: {sens_info['name']} (等级 {sens_info['level']})")
            print(f"       手势阈值: {sens_info['gesture_threshold']:.2f} | 滑动阈值: {sens_info['swipe_threshold']:.2f}")

        if USE_CV_DETECTOR:
            print("[INFO] 当前模式: OpenCV手势识别 (无需MediaPipe)")
        elif HAS_ENHANCED_DETECTOR and hasattr(self.gesture_detector, 'use_ml'):
            if self.gesture_detector.use_ml:
                print("[INFO] 当前模式: 机器学习手势识别")
            else:
                print("[INFO] 当前模式: 规则手势识别")

    def _initialize_camera(self):
        """初始化摄像头（恢复正常分辨率）"""
        # 尝试多个摄像头ID，优先使用1，如果失败则尝试0
        camera_ids = [1, 0]

        for camera_id in camera_ids:
            print(f"尝试打开摄像头 {camera_id}...")
            cap = cv2.VideoCapture(camera_id)

            if cap.isOpened():
                # 正常分辨率
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # 尝试读取一帧测试
                ret, test_frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"[OK] 摄像头 {camera_id} 初始化成功: {width}x{height} @ {fps:.1f}fps")
                    return cap
                else:
                    cap.release()
                    print(f"摄像头 {camera_id} 能打开但无法读取帧")
            else:
                print(f"摄像头 {camera_id} 无法打开")

        print("[ERROR] 所有摄像头尝试失败，使用虚拟模式")
        return None

    def _gesture_recognition_loop(self):
        """手势识别循环（恢复每帧检测）"""
        print("手势识别线程启动...")

        # 显示当前检测模式
        if USE_CV_DETECTOR:
            print("[INFO] 当前模式: OpenCV手势识别 (无需MediaPipe)")
        elif HAS_ENHANCED_DETECTOR and hasattr(self.gesture_detector, 'use_ml'):
            if self.gesture_detector.use_ml:
                mode_text = "机器学习模式"
            else:
                mode_text = "规则检测模式"
        else:
            mode_text = "规则检测模式"

        # 显示虚拟模式提示（如果摄像头未连接）
        if self.cap is None:
            print("[WARNING] 使用虚拟摄像头模式，请连接摄像头进行真实手势识别")

        # 确保mode_text在所有分支都有定义
        mode_text = "OpenCV模式"

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            # 获取图像帧
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # 镜像模式切换
                    if self.mirror_mode:
                        frame = cv2.flip(frame, 1)  # 左右镜像
                else:
                    # 创建虚拟帧
                    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
                    cv2.putText(frame, "Camera Error - Virtual Mode", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f"Connect camera for real gesture detection", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    cv2.putText(frame, f"Mode: {mode_text}", (50, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)
            else:
                # 虚拟模式
                frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
                cv2.putText(frame, "虚拟摄像头模式", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 每帧检测手势（流畅响应）
            try:
                processed_frame, gesture, confidence, landmarks = \
                    self.gesture_detector.detect_gestures(frame, simulation_mode=True)

                # 更新共享数据
                self.current_frame = processed_frame
                self.current_gesture = gesture
                self.gesture_confidence = confidence
                self.hand_landmarks = landmarks

                # 处理手势命令
                self._process_gesture_command(gesture, confidence)

                enhanced_frame = self._enhance_interface(processed_frame, gesture, confidence)
                cv2.imshow('Gesture Control', enhanced_frame)

            except Exception as e:
                print(f"手势检测错误: {e}")
                self.current_frame = frame
                self.current_gesture = None
                enhanced_frame = self._enhance_interface(frame, "error", 0.0)
                cv2.imshow('Gesture Control', enhanced_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("收到退出指令...")
                self.running = False
                break
            elif key == ord('c'):
                self._switch_camera()
            elif key == ord('d'):
                self._debug_gesture_detection()
            elif key == ord('m'):  # 切换单手/双手控制模式
                self._toggle_dual_hand_mode()
            elif key == ord('h'):  # 显示帮助
                self._show_help()
            elif key == ord('f'):  # 切换全屏
                self._toggle_fullscreen()
            elif key == ord('['):  # 降低灵敏度
                self._adjust_sensitivity(-1)
            elif key == ord(']'):  # 提高灵敏度
                self._adjust_sensitivity(1)
            elif key == ord('='):  # 重置灵敏度为默认值
                self._reset_sensitivity()

        print("手势识别线程结束")

    def _adjust_sensitivity(self, delta):
        """调整手势识别灵敏度
        
        Args:
            delta: 灵敏度变化量 (+1增加, -1降低)
        """
        if delta > 0:
            result = self.gesture_detector.increase_sensitivity()
            if result:
                print(f"[灵敏度] 已提高灵敏度到: {self.gesture_detector.get_sensitivity_info()['name']}")
            else:
                print("[灵敏度] 已达到最高灵敏度")
        else:
            result = self.gesture_detector.decrease_sensitivity()
            if result:
                print(f"[灵敏度] 已降低灵敏度到: {self.gesture_detector.get_sensitivity_info()['name']}")
            else:
                print("[灵敏度] 已达到最低灵敏度")
                
    def _reset_sensitivity(self):
        """重置灵敏度为默认值（中）"""
        self.gesture_detector.set_sensitivity(2)
        print("[灵敏度] 已重置为默认灵敏度: MEDIUM")

    def _enhance_interface(self, frame, gesture, confidence):
        """增强界面显示（支持双手控制模式）"""
        # 创建一个更大的画布，包含摄像头画面和信息面板
        height, width = frame.shape[:2]
        panel_width = 320
        total_width = width + panel_width
        enhanced_frame = np.ones((height, total_width, 3), dtype=np.uint8) * 20  # 深灰色背景
        
        # 复制摄像头画面
        enhanced_frame[:, :width] = frame
        
        # 绘制信息面板边框
        cv2.rectangle(enhanced_frame, (width, 0), (total_width, height), (50, 50, 50), 2)
        
        # 显示标题
        cv2.putText(enhanced_frame, "DRONE CONTROL", (width + 20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 显示控制模式
        mode_color = (0, 255, 255) if self.dual_hand_mode else (150, 150, 150)
        mode_text = "DUAL-HAND MODE" if self.dual_hand_mode else "SINGLE-HAND MODE"
        cv2.putText(enhanced_frame, mode_text, (width + 20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        
        # 显示灵敏度信息
        if hasattr(self.gesture_detector, 'get_sensitivity_info'):
            sens_info = self.gesture_detector.get_sensitivity_info()
            sens_level = sens_info['level']
            # 根据灵敏度级别选择颜色
            if sens_level == 1:
                sens_color = (0, 0, 255)    # 红色 - 低
            elif sens_level == 2:
                sens_color = (0, 255, 255)  # 黄色 - 中
            else:
                sens_color = (0, 255, 0)    # 绿色 - 高
            cv2.putText(enhanced_frame, f"SENSITIVITY: {sens_info['name']}", (width + 20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, sens_color, 1)
        
        # 显示手势信息
        y_offset = 105
        
        # 如果是双手模式，显示左右手分别的信息
        if self.dual_hand_mode and isinstance(self.hand_landmarks, dict):
            left_hand = self.hand_landmarks.get('left_hand')
            right_hand = self.hand_landmarks.get('right_hand')
            
            # 左手信息
            if left_hand:
                cv2.putText(enhanced_frame, "LEFT HAND (Direction)", (width + 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20
                cv2.putText(enhanced_frame, f"Gesture: {left_hand.get('gesture', 'none')}", 
                            (width + 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
                y_offset += 18
                
                # 显示方向命令
                direction_cmd = ""
                direction_source = ""  # 手势或滑动
                gesture_name = left_hand.get('gesture', 'none')
                
                if gesture_name == 'victory':
                    direction_cmd = "FORWARD"
                    direction_source = "(手势)"
                elif gesture_name == 'thumb_up':
                    direction_cmd = "BACKWARD"
                    direction_source = "(手势)"
                elif gesture_name == 'pointing_up':
                    direction_cmd = "TURN LEFT"
                    direction_source = "(手势)"
                elif gesture_name == 'pointing_down':
                    direction_cmd = "TURN RIGHT"
                    direction_source = "(手势)"
                # 滑动手势
                elif gesture_name == 'swipe_left':
                    direction_cmd = "LEFT"
                    direction_source = "(滑动)"
                elif gesture_name == 'swipe_right':
                    direction_cmd = "RIGHT"
                    direction_source = "(滑动)"
                elif gesture_name == 'swipe_up':
                    direction_cmd = "FORWARD"
                    direction_source = "(滑动)"
                elif gesture_name == 'swipe_down':
                    direction_cmd = "BACKWARD"
                    direction_source = "(滑动)"
                    
                if direction_cmd:
                    cv2.putText(enhanced_frame, f"Direction: {direction_cmd} {direction_source}", 
                                (width + 20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    y_offset += 18
            else:
                cv2.putText(enhanced_frame, "LEFT HAND: Not detected", (width + 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                y_offset += 38
            
            y_offset += 10
            
            # 右手信息
            if right_hand:
                cv2.putText(enhanced_frame, "RIGHT HAND (Altitude)", (width + 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
                y_offset += 20
                cv2.putText(enhanced_frame, f"Gesture: {right_hand.get('gesture', 'none')}", 
                            (width + 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 100, 0), 1)
                y_offset += 18
                
                # 显示高度命令
                altitude_cmd = ""
                if right_hand.get('gesture') == 'pointing_up':
                    altitude_cmd = "UP"
                elif right_hand.get('gesture') == 'pointing_down':
                    altitude_cmd = "DOWN"
                elif right_hand.get('gesture') == 'ok_sign':
                    altitude_cmd = "HOVER"
                    
                if altitude_cmd:
                    cv2.putText(enhanced_frame, f"Altitude: {altitude_cmd}", 
                                (width + 20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 128, 0), 1)
                    y_offset += 18
            else:
                cv2.putText(enhanced_frame, "RIGHT HAND: Not detected", (width + 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                y_offset += 38
            
            y_offset += 10
            
        else:
            # 单手模式显示
            if gesture and gesture != "no_hand":
                # 手势名称
                cv2.putText(enhanced_frame, f"GESTURE: {gesture.upper()}", 
                            (width + 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                y_offset += 30
                
                # 置信度
                confidence_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
                cv2.putText(enhanced_frame, f"CONFIDENCE: {confidence:.2f}", 
                            (width + 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, confidence_color, 1)
                y_offset += 40
            else:
                cv2.putText(enhanced_frame, "GESTURE: NO HAND", 
                            (width + 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                y_offset += 40
        
        # 显示无人机状态
        cv2.putText(enhanced_frame, "DRONE STATUS", 
                    (width + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        # 获取无人机状态
        drone_state = self.drone_controller.get_state()
        
        # 状态信息
        status_info = [
            f"MODE: {drone_state['mode'].upper()}",
            f"ARMED: {'YES' if drone_state['armed'] else 'NO'}",
            f"BATTERY: {drone_state['battery']:.1f}%",
            f"ALTITUDE: {abs(drone_state['position'][1]):.2f}m"
        ]
        
        for info in status_info:
            cv2.putText(enhanced_frame, info, 
                        (width + 20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
            y_offset += 20
        
        y_offset += 10
        
        # 显示位置信息
        cv2.putText(enhanced_frame, "POSITION", 
                    (width + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        pos = drone_state['position']
        cv2.putText(enhanced_frame, f"X: {pos[0]:.2f}m", 
                    (width + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        y_offset += 15
        cv2.putText(enhanced_frame, f"Y: {pos[1]:.2f}m", 
                    (width + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        y_offset += 15
        cv2.putText(enhanced_frame, f"Z: {pos[2]:.2f}m", 
                    (width + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        y_offset += 30
        
        # 显示控制提示
        cv2.putText(enhanced_frame, "CONTROLS", 
                    (width + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        controls = [
            "Q/ESC: Exit",
            "C: Switch Camera",
            "I: Mirror On/Off",
            "P: Record Trajectory",
            "O: Save Recording",
            "J: Replay Trajectory",
            "D: Debug Info",
            "H: Help",
            "F: Fullscreen",
            "M: Toggle Mode",
            "[ : Lower Sensitivity",
            "] : Raise Sensitivity",
            "= : Reset Sensitivity"
        ]
        
        for control in controls:
            cv2.putText(enhanced_frame, control, 
                        (width + 20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_offset += 15
        
        # 显示帧率
        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            fps = 1.0 / (current_time - self.last_frame_time)
            cv2.putText(enhanced_frame, f"FPS: {fps:.1f}",
                        (width + 20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.last_frame_time = current_time

        # 显示镜像模式状态
        mirror_text = "MIRROR: ON" if self.mirror_mode else "MIRROR: OFF"
        mirror_color = (0, 255, 0) if self.mirror_mode else (0, 0, 255)
        cv2.putText(enhanced_frame, mirror_text,
                    (width + 20, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mirror_color, 1)

        # 显示录制/回放状态
        if self.drone_controller.is_recording:
            record_count = len(self.drone_controller.recorded_trajectory)
            cv2.putText(enhanced_frame, f"REC: {record_count} pts",
                        (width + 150, height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif self.drone_controller.is_replaying:
            replay_progress = self.drone_controller.replay_index
            replay_total = len(self.drone_controller.replay_trajectory)
            speed = self.drone_controller.replay_speed
            cv2.putText(enhanced_frame, f"REPLAY: {replay_progress}/{replay_total} @{speed}x",
                        (width + 150, height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return enhanced_frame

    def _show_help(self):
        """显示帮助信息"""
        print("=" * 60)
        print("手势控制无人机 - 帮助信息")
        print("=" * 60)
        
        mode_text = "双手控制模式" if self.dual_hand_mode else "单手控制模式"
        print(f"【当前模式: {mode_text}】")
        print()
        
        if self.dual_hand_mode:
            print("【双手控制模式】")
            print("-" * 40)
            print("左手控制方向:")
            print("  胜利手势 - 前进")
            print("  大拇指向上 - 后退")
            print("  食指向左倾斜 - 左转")
            print("  食指向右倾斜 - 右转")
            print("  向左滑动 - 左移 (滑动手势)")
            print("  向右滑动 - 右移 (滑动手势)")
            print("  向上滑动 - 前进 (滑动手势)")
            print("  向下滑动 - 后退 (滑动手势)")
            print()
            print("右手控制高度:")
            print("  食指向上 - 上升")
            print("  食指向下 - 下降")
            print("  OK手势 - 悬停")
            print()
            print("双手通用:")
            print("  张开手掌 - 起飞")
            print("  握拳 - 降落")
            print("  大拇指向下 - 停止")
        else:
            print("【单手控制模式】")
            print("-" * 40)
            print("  张开手掌 - 起飞")
            print("  握拳 - 降落")
            print("  胜利手势 - 前进")
            print("  大拇指 - 后退")
            print("  食指上指 - 上升")
            print("  食指向下 - 下降")
            print("  OK手势 - 悬停")
            print("  大拇指向下 - 停止")
        
        print("=" * 60)
        print("键盘控制:")
        print("  Q/ESC - 退出")
        print("  C - 切换摄像头")
        print("  I - 切换镜像模式")
        print("  P - 开始录制轨迹")
        print("  O - 停止录制并保存")
        print("  J - 加载并回放轨迹")
        print("  +/- - 调整回放速度")
        print("  D - 显示调试信息")
        print("  H - 显示帮助")
        print("  F - 切换全屏")
        print("  M - 切换单手/双手模式")
        print("  [ - 降低灵敏度")
        print("  ] - 提高灵敏度")
        print("  = - 重置灵敏度为默认")
        print("  R - 重置无人机位置")
        print("  T - 手动起飞")
        print("  L - 手动降落")
        print("  S - 停止")
        print("=" * 60)
        print()
        print("灵敏度说明:")
        print("  LOW (1)   - 严格模式，误识别少但可能漏检")
        print("  MEDIUM (2) - 平衡模式，推荐默认设置")
        print("  HIGH (3)   - 宽松模式，灵敏度高但可能误识别")
        print("=" * 60)

    def _toggle_fullscreen(self):
        """切换全屏模式"""
        # 简化实现，实际需要更复杂的窗口管理
        print("全屏模式切换功能已触发")

    def _toggle_dual_hand_mode(self):
        """切换单手/双手控制模式"""
        self.dual_hand_mode = not self.dual_hand_mode
        mode_text = "双手控制模式" if self.dual_hand_mode else "单手控制模式"
        print(f"\n{'='*60}")
        print(f"已切换到: {mode_text}")
        print(f"{'='*60}")
        
        if self.dual_hand_mode:
            print("\n【双手控制模式说明】")
            print("  左手: 胜利=前进 | 拇指向上=后退 | 食指向左=左转 | 食指向右=右转")
            print("  右手: 食指向上=上升 | 食指向下=下降 | OK手势=悬停")
            print("  通用: 张开手掌=起飞 | 握拳=降落 | 拇指向下=停止")
        else:
            print("\n【单手控制模式说明】")
            print("  张开手掌=起飞 | 握拳=降落 | 胜利=前进 | 拇指=后退")
            print("  食指向上=上升 | 食指向下=下降 | OK=悬停 | 拇指向下=停止")

    def _switch_detection_mode(self):
        """切换检测模式（如果有多个可用模型）"""
        if USE_CV_DETECTOR:
            print("当前使用OpenCV检测器，不支持切换模式")
            return
        if not HAS_ENHANCED_DETECTOR:
            print("当前只有规则检测器可用")
            return

        # 检查可用的模型
        model_files = [
            ("dataset/models/gesture_ensemble.pkl", "集成模型"),
            ("dataset/models/gesture_svm.pkl", "SVM模型"),
            ("dataset/models/gesture_random_forest.pkl", "随机森林模型"),
            ("dataset/models/gesture_mlp.pkl", "神经网络模型"),
        ]

        available_models = []
        for path, name in model_files:
            if os.path.exists(path):
                available_models.append((path, name))

        if len(available_models) == 0:
            print("未找到任何机器学习模型")
            return
        elif len(available_models) == 1:
            print(f"只有 {available_models[0][1]} 可用")
            return

        # 显示可用模型
        print("\n可用的手势识别模型:")
        for i, (path, name) in enumerate(available_models, 1):
            print(f"  {i}. {name}")

        print("按数字键选择模型，或按其他键取消")

        # 这里简化处理，实际需要更复杂的交互
        # 暂时只记录一下
        print("注意: 需要重启程序切换模型")

    def _switch_camera(self):
        """切换摄像头"""
        if self.cap:
            self.cap.release()
            print("释放当前摄像头...")

        # 获取当前摄像头ID
        current_id = 1 if self.cap is None else 0

        print(f"切换到摄像头 {current_id}...")
        self.cap = cv2.VideoCapture(current_id)

        if self.cap.isOpened():
            print(f"✅ 切换到摄像头 {current_id} 成功")
        else:
            print(f"❌ 切换到摄像头 {current_id} 失败")
            self.cap = None

    def _debug_gesture_detection(self):
        """调试手势检测"""
        print("\n[手势调试信息]")
        print(f"当前手势: {self.current_gesture}")
        print(f"置信度: {self.gesture_confidence:.2f}")
        
        # 显示灵敏度信息
        if hasattr(self.gesture_detector, 'get_sensitivity_info'):
            sens_info = self.gesture_detector.get_sensitivity_info()
            print(f"灵敏度: {sens_info['name']} (等级: {sens_info['level']})")
            print(f"  - 检测置信度: {sens_info['detection_confidence']:.2f}")
            print(f"  - 跟踪置信度: {sens_info['tracking_confidence']:.2f}")
            print(f"  - 手势阈值: {sens_info['gesture_threshold']:.2f}")
            print(f"  - 滑动阈值: {sens_info['swipe_threshold']:.2f}")
        
        print(f"冷却时间: {time.time() - self.last_command_time:.1f}s")
        print(f"无人机解锁: {self.drone_controller.state['armed']}")
        print(f"无人机模式: {self.drone_controller.state['mode']}")
        print(f"无人机位置: ({self.drone_controller.state['position'][0]:.1f}, "
              f"{self.drone_controller.state['position'][1]:.1f}, "
              f"{self.drone_controller.state['position'][2]:.1f})")

    def _process_gesture_command(self, gesture, confidence):
        """处理手势命令（支持双手控制）"""
        current_time = time.time()

        # 获取该手势的阈值（降低以提高识别率）
        threshold = self.gesture_thresholds.get(gesture, self.gesture_thresholds['default'])

        # 检查是否是双手控制模式
        if self.dual_hand_mode and self.hand_landmarks:
            # 检查是否有双手数据
            if isinstance(self.hand_landmarks, dict):
                left_hand = self.hand_landmarks.get('left_hand')
                right_hand = self.hand_landmarks.get('right_hand')

                if left_hand or right_hand:
                    # 使用双手控制逻辑
                    dual_commands = self.gesture_detector.get_dual_hand_commands(
                        left_hand, right_hand
                    )

                    # 处理特殊命令（起飞/降落/停止）
                    if dual_commands['special_command']:
                        special_cmd = dual_commands['special_command']
                        if current_time - self.last_command_time >= self.command_cooldown:
                            print(f"[双手模式] 特殊命令: {special_cmd}")
                            self.drone_controller.send_command(special_cmd, 1.0)
                            self.last_command_time = current_time
                            self.last_processed_gesture = dual_commands.get(
                                'left_gesture') or dual_commands.get('right_gesture')
                            self.last_processed_time = current_time
                            self.current_command = special_cmd
                        return

                    # 处理方向命令（左手）
                    if dual_commands['direction_command']:
                        dir_cmd = dual_commands['direction_command']
                        dir_intensity = dual_commands['direction_intensity']

                        # 检查是否需要发送新命令
                        if (dir_cmd != self.last_direction_command or
                            current_time - getattr(self, 'last_direction_time', 0) >= self.dual_control_cooldown):

                            print(f"[左手-方向] {dual_commands['left_gesture']} -> {dir_cmd} (强度:{dir_intensity:.0%})")
                            self.drone_controller.send_command(dir_cmd, dir_intensity)
                            self.last_direction_command = dir_cmd
                            self.last_direction_time = current_time
                            self.last_command_time = current_time
                            self.current_command = dir_cmd

                    # 处理高度命令（右手）
                    if dual_commands['altitude_command']:
                        alt_cmd = dual_commands['altitude_command']
                        alt_intensity = dual_commands['altitude_intensity']

                        # 检查是否需要发送新命令
                        if (alt_cmd != self.last_altitude_command or
                            current_time - getattr(self, 'last_altitude_time', 0) >= self.dual_control_cooldown):

                            print(f"[右手-高度] {dual_commands['right_gesture']} -> {alt_cmd} (强度:{alt_intensity:.0%})")
                            self.drone_controller.send_command(alt_cmd, alt_intensity)
                            self.last_altitude_command = alt_cmd
                            self.last_altitude_time = current_time
                            self.last_command_time = current_time
                            self.current_command = alt_cmd

                    # 双手同时检测时显示状态
                    if left_hand and right_hand:
                        pass  # 信息已在detect_gestures中显示

                    return

        # 如果不是双手模式或没有双手数据，使用原有单手逻辑
        # 计算手势强度（如果有手部关键点）
        intensity = 0.5
        intensity_info = None
        if self.hand_landmarks and not isinstance(self.hand_landmarks, dict):
            # 检查landmarks是否是有效的数据结构
            if hasattr(self.hand_landmarks, '__len__') or isinstance(self.hand_landmarks, (list, tuple)):
                if len(self.hand_landmarks) > 21 and 'intensity_info' in self.hand_landmarks[-1]:
                    intensity_info = self.hand_landmarks[-1]['intensity_info']
            intensity = self.gesture_detector.get_gesture_intensity(
                self.hand_landmarks, gesture
            )
            self.current_intensity = intensity

        # 检查是否在冷却期内（仅针对新手势触发）
        in_cooldown = current_time - self.last_command_time <= self.command_cooldown

        # 检查是否是重复手势（避免频繁处理同一个手势）
        same_gesture = (gesture == self.current_gesture and
                        hasattr(self, 'last_processed_gesture') and
                        gesture == self.last_processed_gesture and
                        current_time - getattr(self, 'last_processed_time', 0) < 2.0)

        # 只处理置信度高于阈值的手势且不在冷却期
        if (gesture not in ["no_hand", "hand_detected"] and
                confidence > threshold and
                not in_cooldown and
                not same_gesture):

            # 获取控制命令
            command = self.gesture_detector.get_command(gesture)

            if command != "none":
                # 判断是否是滑动手势
                is_swipe = gesture.startswith("swipe_")
                swipe_info = ""
                if is_swipe:
                    swipe_info = f" [滑动手势]"
                    # 滑动手势使用更高的强度
                    intensity = max(self.gesture_confidence, 0.7)
                
                # 添加调试信息
                palm_info = ""
                if intensity_info:
                    palm_info = f" 手掌:{intensity_info['palm_openness']:.0%}"
                print(
                    f"[INFO] 检测到手势: {gesture} (置信度:{confidence:.2f}) -> 执行:{command} (速度:{intensity:.0%}){palm_info}{swipe_info}")

                # 发送命令到控制器
                self.drone_controller.send_command(command, intensity)

                # 记录命令
                self._log_command(gesture, command, confidence, intensity)

                # 更新最后命令时间和手势状态
                self.last_command_time = current_time
                self.last_processed_gesture = gesture
                self.last_processed_time = current_time

                # 存储当前命令用于连续控制
                self.current_command = command

        # 连续控制：持续手势时动态调整速度（滑动手势不进行连续控制）
        elif (gesture not in ["no_hand", "hand_detected"] and
              same_gesture and
              self.continuous_control_enabled and
              current_time - getattr(self, 'last_continuous_time', 0) >= self.continuous_control_interval and
              not gesture.startswith("swipe_")):  # 滑动手势不进行连续控制

            command = self.gesture_detector.get_command(gesture)

            if command != "none" and command in ["forward", "backward", "up", "down", "left", "right"]:
                # 根据强度动态调整速度
                self.drone_controller.send_command(command, intensity)
                self.last_continuous_time = current_time

        # 无手势时停止移动
        elif gesture in ["no_hand", "hand_detected"]:
            self.current_command = None

        elif gesture not in ["no_hand", "hand_detected"] and confidence > 0.3:
            # 只在调试模式下显示检测到但未触发的情况
            debug_mode = False  # 可以设为True启用详细调试
            if debug_mode:
                if in_cooldown:
                    print(
                        f"  [冷却中] {gesture} 冷却时间剩余: {self.command_cooldown - (current_time - self.last_command_time):.1f}s")
                elif same_gesture:
                    print(f"  [重复手势] {gesture} 已处理过，冷却中")
                elif confidence < threshold:
                    print(f"  [置信度不足] {gesture} 置信度: {confidence:.2f} < 阈值: {threshold}")

    def _simulation_loop(self):
        """仿真主循环"""
        print("3D仿真线程启动...")

        last_time = time.time()
        frame_count = 0
        last_status_print = time.time()

        # 帧率控制
        target_fps = 60
        frame_delay = 1.0 / target_fps

        print("\n[INFO] 键盘提示：按 'R' 键重置无人机位置到原点")
        print("           按 'T' 键手动起飞")
        print("           按 'L' 键手动降落")
        print("           按 'H' 键悬停")
        print("           按 'P' 键开始录制轨迹")
        print("           按 'O' 键停止录制并保存")
        print("           按 'J' 键加载并回放轨迹")

        # 按键防抖记录
        self._last_key_press = {}

        while self.running:
            start_time = time.time()
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            if dt <= 0:
                dt = frame_delay
            elif dt > 0.1:
                dt = 0.1

            # 每3秒打印一次状态
            if current_time - last_status_print > 3:
                status = self.drone_controller.get_status_string()
                print(f"[状态监控] {status}")
                if self.current_gesture:
                    print(f"[状态监控] 当前手势: {self.current_gesture} (置信度: {self.gesture_confidence:.2f})")
                last_status_print = current_time

            if self.paused:
                if not self.viewer.handle_events():
                    self.running = False
                time.sleep(0.01)
                continue

            keys = pygame.key.get_pressed()

            # 检查重置键 R
            if keys[pygame.K_r]:
                if ('r' not in self._last_key_press or
                        current_time - self._last_key_press['r'] > 1.0):
                    print("[INFO] 键盘：重置无人机位置")
                    self.drone_controller.reset()
                    print("  无人机已重置到原点位置")
                    self._last_key_press['r'] = current_time

            # 检查起飞键 T
            if keys[pygame.K_t]:
                if ('t' not in self._last_key_press or
                        current_time - self._last_key_press['t'] > 1.0):
                    print("[INFO] 键盘：起飞")
                    self.drone_controller.send_command("takeoff", 0.8)
                    self._last_key_press['t'] = current_time

            # 检查降落键 L
            if keys[pygame.K_l]:
                if ('l' not in self._last_key_press or
                        current_time - self._last_key_press['l'] > 1.0):
                    print("[INFO] 键盘：降落")
                    self.drone_controller.send_command("land", 0.5)
                    self._last_key_press['l'] = current_time

            # 检查悬停键 H
            if keys[pygame.K_h]:
                if ('h' not in self._last_key_press or
                        current_time - self._last_key_press['h'] > 1.0):
                    print("[INFO] 键盘：悬停")
                    self.drone_controller.send_command("hover")
                    self._last_key_press['h'] = current_time

            # 检查停止键 S
            if keys[pygame.K_s]:
                if ('s' not in self._last_key_press or
                        current_time - self._last_key_press['s'] > 1.0):
                    print("[INFO] 键盘：停止")
                    self.drone_controller.send_command("stop")
                    self._last_key_press['s'] = current_time

            # ========== 轨迹录制/回放控制 ==========
            # 检查录制键 P
            if keys[pygame.K_p]:
                if ('p' not in self._last_key_press or
                        current_time - self._last_key_press['p'] > 1.0):
                    if not self.drone_controller.is_replaying:
                        self.drone_controller.start_recording()
                    self._last_key_press['p'] = current_time

            # 检查停止录制键 O
            if keys[pygame.K_o]:
                if ('o' not in self._last_key_press or
                        current_time - self._last_key_press['o'] > 1.0):
                    if self.drone_controller.is_recording:
                        self.drone_controller.stop_recording()
                        self.drone_controller.save_trajectory_to_file()
                    self._last_key_press['o'] = current_time

            # 检查回放键 J
            if keys[pygame.K_j]:
                if ('j' not in self._last_key_press or
                        current_time - self._last_key_press['j'] > 1.0):
                    if not self.drone_controller.is_recording:
                        saved_files = self.drone_controller.list_saved_trajectories()
                        if saved_files:
                            print("\n已保存的轨迹文件：")
                            for i, f in enumerate(saved_files[:5]):
                                print(f"  {i + 1}. {f}")
                            print(f"\n最新轨迹: {saved_files[0]}")
                            # 自动加载最新的轨迹并回放
                            loaded = self.drone_controller.load_trajectory_from_file(saved_files[0])
                            if loaded:
                                self.drone_controller.start_replay(speed=1.0)
                        else:
                            print("[INFO] 没有找到已保存的轨迹文件")
                            print("   请先录制轨迹（按P开始录制，按O停止并保存）")
                    self._last_key_press['j'] = current_time

            # 检查回放速度调整
            if self.drone_controller.is_replaying:
                if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
                    if ('=' not in self._last_key_press or
                            current_time - self._last_key_press['='] > 0.3):
                        self.drone_controller.replay_speed = min(5.0, self.drone_controller.replay_speed + 0.5)
                        print(f"[INFO] 回放速度: {self.drone_controller.replay_speed}x")
                        self._last_key_press['='] = current_time
                if keys[pygame.K_MINUS]:
                    if ('-' not in self._last_key_press or
                            current_time - self._last_key_press['-'] > 0.3):
                        self.drone_controller.replay_speed = max(0.1, self.drone_controller.replay_speed - 0.5)
                        print(f"[INFO] 回放速度: {self.drone_controller.replay_speed}x")
                        self._last_key_press['-'] = current_time

            if not self.viewer.handle_events():
                self.running = False
                break

            if not self.running:
                break

            drone_state = self.drone_controller.get_state()

            # 处理轨迹回放
            if self.drone_controller.is_replaying:
                replay_result = self.drone_controller.update_replay()
                if replay_result:
                    position, mode = replay_result
                    self.drone_controller.state['position'] = position
                    self.drone_controller.state['mode'] = mode
                elif replay_result is None and not self.drone_controller.is_replaying:
                    # 回放结束，停止更新物理
                    self.drone_controller.update_physics(dt)
            else:
                self.drone_controller.update_physics(dt)

            if self.physics_engine and self.drone_controller.state['armed']:
                control_input = self._get_control_input_from_state(drone_state)
                physics_state = self.physics_engine.update(dt, control_input)

            trajectory = self.drone_controller.get_trajectory()

            drone_state_with_gesture = drone_state.copy()
            if self.current_gesture:
                drone_state_with_gesture['current_gesture'] = self.current_gesture
                drone_state_with_gesture['gesture_confidence'] = self.gesture_confidence

            self.viewer.render(drone_state_with_gesture, trajectory)

            # 控制帧率，避免CPU占用过高
            elapsed = time.time() - start_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            frame_count += 1
            if frame_count % 120 == 0:
                fps = 1.0 / (time.time() - start_time) if start_time > 0 else 0
                print(f"3D仿真帧率: {fps:.1f} FPS")

        print("3D仿真线程结束")

    def _get_control_input_from_state(self, drone_state):
        """从无人机状态生成控制输入"""
        control_input = {
            'throttle': 0.5,  # 默认油门
            'roll': 0.0,
            'pitch': 0.0,
            'yaw_rate': 0.0
        }

        # 如果检测到手部关键点，可以用于精细控制
        if self.hand_landmarks and self.current_gesture:
            # 简单示例：根据手势调整控制
            if self.current_gesture == "pointing_up":
                control_input['throttle'] = 0.8
            elif self.current_gesture == "pointing_down":
                control_input['throttle'] = 0.2
            elif self.current_gesture == "victory":
                control_input['pitch'] = 0.3  # 轻微前倾
            elif self.current_gesture == "thumb_up":
                control_input['pitch'] = -0.3  # 轻微后倾

        return control_input

    def _log_command(self, gesture, command, confidence, intensity):
        """记录命令到日志"""
        log_entry = {
            'timestamp': time.time(),
            'gesture': gesture,
            'command': command,
            'confidence': confidence,
            'intensity': intensity,
            'position': self.drone_controller.state['position'].tolist(),
            'battery': self.drone_controller.state['battery'],
            'armed': self.drone_controller.state['armed'],
            'mode': self.drone_controller.state['mode']
        }
        self.data_log.append(log_entry)

        # 实时显示
        pos = self.drone_controller.state['position']
        print(f"  位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | "
              f"电池: {self.drone_controller.state['battery']:.1f}%")

    def _save_log(self):
        """保存日志到文件"""
        if self.data_log:
            try:
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    json.dump(self.data_log, f, indent=2, ensure_ascii=False)
                print(f"飞行日志已保存到: {self.log_file} ({len(self.data_log)}条记录)")
            except Exception as e:
                print(f"保存日志失败: {e}")
        else:
            print("没有飞行记录需要保存")

    def run(self):
        """运行主程序"""
        print("=" * 60)
        print("     手势控制无人机仿真系统（机器学习增强版）")
        print("=" * 60)

        # 显示当前检测模式
        if USE_CV_DETECTOR:
            print("[INFO] 当前模式: OpenCV手势识别 (无需MediaPipe)")
        elif HAS_ENHANCED_DETECTOR and hasattr(self.gesture_detector, 'use_ml'):
            if self.gesture_detector.use_ml:
                mode_info = "机器学习模式 (更高精度)"
            else:
                mode_info = "规则检测模式 (基础)"
        else:
            mode_info = "规则检测模式"

        if USE_CV_DETECTOR:
            mode_info = "OpenCV手势识别 (无需MediaPipe)"

        print(f"检测模式: {mode_info}")

        print("系统功能:")
        print("  1. 实时手势识别 (双手检测)")
        print("  2. 双手控制模式（左手方向+右手高度）")
        print("  3. 动态灵敏度调节（3档可调）")
        print("  4. 无人机控制仿真")
        print("  5. 3D可视化 (OpenGL渲染)")
        print("  6. 飞行数据记录")
        print("=" * 60)
        print("【双手控制模式】(默认)")
        print("-" * 40)
        print("左手控制方向:")
        print("  胜利手势 - 前进")
        print("  大拇指向上 - 后退")
        print("  食指向左 - 左转")
        print("  食指向右 - 右转")
        print("  向左滑动 - 无人机左移")
        print("  向右滑动 - 无人机右移")
        print("  向上滑动 - 无人机前进")
        print("  向下滑动 - 无人机后退")
        print()
        print("右手控制高度:")
        print("  食指向上 - 上升")
        print("  食指向下 - 下降")
        print("  OK手势 - 悬停")
        print()
        print("双手通用:")
        print("  张开手掌 - 起飞")
        print("  握拳 - 降落")
        print("  大拇指向下 - 停止")
        print("=" * 60)
        print("使用说明:")
        print("  手势控制窗口: 按 'q' 退出")
        print("  手势控制窗口: 按 'c' 切换摄像头")
        print("  手势控制窗口: 按 'i' 切换镜像模式")
        print("  手势控制窗口: 按 'd' 显示调试信息")
        print("  手势控制窗口: 按 'm' 切换单手/双手模式")
        print("  3D仿真窗口: 按 'ESC' 退出")
        print("  3D窗口按键控制:")
        print("    G - 切换网格显示")
        print("    T - 切换轨迹显示")
        print("    A - 切换坐标轴显示")
        print("    ↑↓←→ - 旋转视角")
        print("    +/- - 缩放视角")
        print("    空格 - 重置视角")
        print("  轨迹录制回放:")
        print("    P - 开始录制轨迹")
        print("    O - 停止录制并保存")
        print("    J - 加载并回放轨迹")
        print("    +/- - 调整回放速度")
        print("=" * 60)
        print("提示:")
        print("  1. 无人机初始在地面，等待手势指令")
        print("  2. 使用双手控制时，同时伸出左右手")
        print("  3. 左手在屏幕左侧控制方向，右手在右侧控制高度")
        print("  4. 按 'm' 键可切换回单手控制模式")
        print("  5. 按 '[' 或 ']' 键调节手势识别灵敏度")
        print("=" * 60)
        print("系统启动中...")

        try:
            # 启动手势识别线程
            self.gesture_thread = threading.Thread(
                target=self._gesture_recognition_loop,
                name="GestureThread",
                daemon=True
            )
            self.gesture_thread.start()

            print("手势识别线程已启动")
            print("3D仿真窗口即将打开...")
            time.sleep(1)  # 给手势窗口一点时间显示

            # 主线程运行仿真
            self._simulation_loop()

            # 等待手势线程结束
            if self.gesture_thread.is_alive():
                self.gesture_thread.join(timeout=2.0)

        except KeyboardInterrupt:
            print("\n系统被用户中断")
        except Exception as e:
            print(f"系统运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理资源
            self.running = False

            if self.cap:
                self.cap.release()
                print("摄像头已释放")

            cv2.destroyAllWindows()
            print("OpenCV窗口已关闭")

            # 保存日志
            self._save_log()

            print("无人机仿真系统已安全关闭 [OK]")


def load_config():
    """加载配置文件"""
    config = {
        'camera_id': 1,  # 默认使用摄像头1
        'window_width': 1024,
        'window_height': 768,
        'drone_mass': 1.0,
        'gravity': 9.81,
        'simulation_fps': 60,
        'gesture_threshold': 0.6  # 降低默认阈值
    }
    return config


if __name__ == "__main__":
    print("手势控制无人机仿真系统")
    print("=" * 60)

    # 检查必要的模块
    try:
        import pygame

        print("[OK] Pygame 已安装")
    except ImportError:
        print("[ERROR] 错误: Pygame 未安装!")
        print("请运行: pip install pygame")
        sys.exit(1)

    try:
        import OpenGL

        print("[OK] PyOpenGL 已安装")
    except ImportError:
        print("[ERROR] 错误: PyOpenGL 未安装!")
        print("请运行: pip install PyOpenGL PyOpenGL-accelerate")
        sys.exit(1)

    # 加载配置
    config = load_config()

    # 创建并运行仿真系统
    try:
        simulation = IntegratedDroneSimulation(config)
        simulation.run()
    except Exception as e:
        print(f"系统启动失败: {e}")
        import traceback

        traceback.print_exc()