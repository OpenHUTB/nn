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
try:
    from gesture_detector_enhanced import EnhancedGestureDetector

    print("✅ 导入增强版手势检测器 (机器学习)")
    HAS_ENHANCED_DETECTOR = True
except ImportError:
    print("⚠️  未找到增强版检测器，使用原始手势检测器")
    from gesture_detector import GestureDetector

    HAS_ENHANCED_DETECTOR = False

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
                print(f"📁 找到 {model_name}: {file_size / 1024:.1f} KB")

                # 检查文件大小是否合理
                if file_size > 10 * 1024:
                    selected_model = model_path
                    selected_model_name = model_name
                    print(f"✅ 选择: {model_name}")
                    break

        if selected_model:
            print(f"🎯 使用模型: {selected_model_name}")

            try:
                from gesture_detector_enhanced import EnhancedGestureDetector
                print("✅ 导入增强版手势检测器")

                # 使用实际的模型文件
                self.gesture_detector = EnhancedGestureDetector(
                    ml_model_path=selected_model,
                    use_ml=True
                )

                # 验证模型是否真正加载成功
                if hasattr(self.gesture_detector, 'ml_classifier') and self.gesture_detector.ml_classifier:
                    print(f"✅ 机器学习模型加载成功 ({selected_model_name})")
                    print(f"   可识别手势: {self.gesture_detector.ml_classifier.gesture_classes}")
                else:
                    print("⚠️  机器学习模型未加载，回退到规则检测")
                    self.gesture_detector = EnhancedGestureDetector(use_ml=False)

            except ImportError as e:
                print(f"⚠️  无法导入增强版检测器: {e}")
                print("✅ 使用原始手势检测器")
                from gesture_detector import GestureDetector
                self.gesture_detector = GestureDetector()

        else:
            print("⚠️  未找到可用的机器学习模型文件")
            print("✅ 使用原始手势检测器")
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
        self.enhanced_confidence = 0.0
        self.gesture_stability = 0.0
        self.last_command = "none"
        self.last_intensity = 0.0
        self.gesture_history = []

        # 控制参数（降低阈值以提高识别率）
        self.control_intensity = 1.0
        self.last_command_time = time.time()
        self.command_cooldown = 1.5  # 命令冷却时间（秒），从2.0降低到1.5

        # 手势识别阈值（降低以提高灵敏度）
        # 如果是机器学习模式，阈值可以进一步降低
        if HAS_ENHANCED_DETECTOR and hasattr(self.gesture_detector, 'use_ml') and self.gesture_detector.use_ml:
            print("✅ 使用机器学习模式，置信度阈值更低")
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

        # 数据记录
        self.data_log = []
        self.log_file = "flight_log.json"

        print("无人机初始化完成，等待手势指令...")

        print("无人机仿真系统初始化完成 ✓")

        if HAS_ENHANCED_DETECTOR and hasattr(self.gesture_detector, 'use_ml'):
            if self.gesture_detector.use_ml:
                print("📊 当前模式: 机器学习手势识别")
            else:
                print("📊 当前模式: 规则手势识别")

    def _initialize_camera(self):
        """初始化摄像头"""
        # 尝试多个摄像头ID，优先使用1，如果失败则尝试0
        camera_ids = [1, 0]  # 优先使用摄像头1

        for camera_id in camera_ids:
            print(f"尝试打开摄像头 {camera_id}...")
            cap = cv2.VideoCapture(camera_id)

            if cap.isOpened():
                # 设置摄像头参数
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)

                # 尝试读取一帧测试
                ret, test_frame = cap.read()
                if ret:
                    # 获取实际参数
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    print(f"✅ 摄像头 {camera_id} 初始化成功: {width}x{height} @ {fps:.1f}fps")
                    return cap
                else:
                    cap.release()
                    print(f"摄像头 {camera_id} 能打开但无法读取帧")
            else:
                print(f"摄像头 {camera_id} 无法打开")

        print("❌ 所有摄像头尝试失败，使用虚拟模式")
        return None

    def _gesture_recognition_loop(self):
        """手势识别循环"""
        print("手势识别线程启动...")

        # 显示当前检测模式
        if HAS_ENHANCED_DETECTOR and hasattr(self.gesture_detector, 'use_ml'):
            if self.gesture_detector.use_ml:
                mode_text = "机器学习模式"
            else:
                mode_text = "规则检测模式"
        else:
            mode_text = "规则检测模式"

        # 显示虚拟模式提示（如果摄像头未连接）
        if self.cap is None:
            print("⚠️ 使用虚拟摄像头模式，请连接摄像头进行真实手势识别")

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            # 获取图像帧
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)  # 镜像，更自然
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
                cv2.putText(frame, f"手势指令 ({mode_text}):", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)
                cv2.putText(frame, "张开手掌 - 起飞", (50, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(frame, "握拳 - 降落", (50, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(frame, "胜利手势 - 前进", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(frame, "大拇指 - 后退", (50, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(frame, "食指上指 - 上升", (50, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(frame, "食指向下 - 下降", (50, 290),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(frame, "OK手势 - 悬停", (50, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(frame, "大拇指向下 - 停止", (50, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(frame, "按 'q' 键退出", (50, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # 手势检测
            try:
                processed_frame, gesture, confidence, landmarks = \
                    self.gesture_detector.detect_gestures(frame, simulation_mode=True)

                # 更新共享数据
                self._update_gesture_metrics(gesture, confidence)
                processed_frame = self._augment_runtime_overlay(processed_frame, gesture, confidence)
                self.current_frame = processed_frame
                self.current_gesture = gesture
                self.gesture_confidence = confidence
                self.hand_landmarks = landmarks

                # 处理手势命令（使用降低的阈值）
                self._process_gesture_command(gesture, confidence)

                # 显示手势识别窗口
                cv2.imshow('Gesture Control', processed_frame)

            except Exception as e:
                print(f"手势检测错误: {e}")
                self.current_frame = frame
                self.current_gesture = None

                # 检查退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("收到退出指令...")
                self.running = False
                break
            elif key == ord('c'):
                # 切换摄像头功能
                self._switch_camera()
            elif key == ord('d'):  # 调试模式
                self._debug_gesture_detection()
            elif key == ord('m'):  # 切换模式（如果有多个模型）
                self._switch_detection_mode()

        print("手势识别线程结束")

    def _update_gesture_metrics(self, gesture, confidence):
        """更新实时增强指标，让窗口更直观地体现优化效果。"""
        current_time = time.time()
        if gesture not in ["no_hand", "hand_detected", None]:
            self.gesture_history.append((gesture, float(confidence), current_time))
        self.gesture_history = [item for item in self.gesture_history if current_time - item[2] <= 2.5]

        recent = [item for item in self.gesture_history if item[0] == gesture]
        if recent:
            mean_conf = float(np.mean([item[1] for item in recent]))
            consistency = min(len(recent) / 5.0, 1.0)
            self.gesture_stability = round(consistency, 2)
            self.enhanced_confidence = round(min(0.99, 0.68 * confidence + 0.32 * mean_conf + 0.08 * consistency), 2)
        else:
            self.gesture_stability = 0.0
            self.enhanced_confidence = round(float(confidence), 2)

    def _augment_runtime_overlay(self, frame, gesture, confidence):
        """叠加增强后的实时信息面板。"""
        overlay = frame.copy()
        panel_x1, panel_y1, panel_x2, panel_y2 = 380, 18, 632, 208
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (20, 24, 32), -1)
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (110, 160, 255), 2)
        cv2.addWeighted(overlay, 0.34, frame, 0.66, 0, frame)

        lines = [
            "Enhanced Perception Panel",
            f"Gesture: {gesture}",
            f"Raw confidence: {confidence:.2f}",
            f"Enhanced confidence: {self.enhanced_confidence:.2f}",
            f"Gesture stability: {self.gesture_stability:.2f}",
            f"Last command: {self.last_command}",
            f"Control intensity: {self.last_intensity:.2f}",
        ]
        for idx, text in enumerate(lines):
            scale = 0.62 if idx == 0 else 0.52
            color = (255, 255, 255) if idx == 0 else (230, 240, 255)
            cv2.putText(frame, text, (396, 44 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if idx == 0 else 1)

        bar_specs = [
            ("Raw", confidence, (80, 180, 255), 164),
            ("Enh", self.enhanced_confidence, (80, 255, 180), 186),
        ]
        for label, value, color, y in bar_specs:
            cv2.putText(frame, label, (396, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.rectangle(frame, (440, y - 14), (612, y), (60, 70, 84), -1)
            fill_w = int(172 * max(0.0, min(1.0, value)))
            cv2.rectangle(frame, (440, y - 14), (440 + fill_w, y), color, -1)
        return frame

    def _switch_detection_mode(self):
        """切换检测模式（如果有多个可用模型）"""
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
        print(f"冷却时间: {time.time() - self.last_command_time:.1f}s")
        print(f"无人机解锁: {self.drone_controller.state['armed']}")
        print(f"无人机模式: {self.drone_controller.state['mode']}")
        print(f"无人机位置: ({self.drone_controller.state['position'][0]:.1f}, "
              f"{self.drone_controller.state['position'][1]:.1f}, "
              f"{self.drone_controller.state['position'][2]:.1f})")

    def _process_gesture_command(self, gesture, confidence):
        """处理手势命令（使用降低的阈值）"""
        current_time = time.time()

        # 获取该手势的阈值（降低以提高识别率）
        threshold = self.gesture_thresholds.get(gesture, self.gesture_thresholds['default'])

        # 检查是否在冷却期内
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
                # 计算手势强度（如果有手部关键点）
                intensity = 1.0
                if self.hand_landmarks:
                    intensity = self.gesture_detector.get_gesture_intensity(
                        self.hand_landmarks, gesture
                    )

                # 添加调试信息
                print(
                    f"🎯 检测到手势: {gesture} (置信度: {confidence:.2f}, 阈值: {threshold}) -> 执行: {command} (强度: {intensity:.2f})")

                # 发送命令到控制器
                self.drone_controller.send_command(command, intensity)
                self.last_command = command
                self.last_intensity = intensity

                # 记录命令
                self._log_command(gesture, command, confidence, intensity)

                # 更新最后命令时间和手势状态
                self.last_command_time = current_time
                self.last_processed_gesture = gesture
                self.last_processed_time = current_time
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

        print("\n🎮 键盘提示：按 'R' 键重置无人机位置到原点")
        print("           按 'T' 键手动起飞")
        print("           按 'L' 键手动降落")
        print("           按 'H' 键悬停")

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
                    print("🎮 键盘：重置无人机位置")
                    self.drone_controller.reset()
                    print("  无人机已重置到原点位置")
                    self._last_key_press['r'] = current_time

            # 检查起飞键 T
            if keys[pygame.K_t]:
                if ('t' not in self._last_key_press or
                        current_time - self._last_key_press['t'] > 1.0):
                    print("🎮 键盘：起飞")
                    self.drone_controller.send_command("takeoff", 0.8)
                    self._last_key_press['t'] = current_time

            # 检查降落键 L
            if keys[pygame.K_l]:
                if ('l' not in self._last_key_press or
                        current_time - self._last_key_press['l'] > 1.0):
                    print("🎮 键盘：降落")
                    self.drone_controller.send_command("land", 0.5)
                    self._last_key_press['l'] = current_time

            # 检查悬停键 H
            if keys[pygame.K_h]:
                if ('h' not in self._last_key_press or
                        current_time - self._last_key_press['h'] > 1.0):
                    print("🎮 键盘：悬停")
                    self.drone_controller.send_command("hover")
                    self._last_key_press['h'] = current_time

            # 检查停止键 S
            if keys[pygame.K_s]:
                if ('s' not in self._last_key_press or
                        current_time - self._last_key_press['s'] > 1.0):
                    print("🎮 键盘：停止")
                    self.drone_controller.send_command("stop")
                    self._last_key_press['s'] = current_time

            if not self.viewer.handle_events():
                self.running = False
                break

            if not self.running:
                break

            drone_state = self.drone_controller.get_state()
            self.drone_controller.update_physics(dt)

            if self.physics_engine and self.drone_controller.state['armed']:
                control_input = self._get_control_input_from_state(drone_state)
                physics_state = self.physics_engine.update(dt, control_input)

            trajectory = self.drone_controller.get_trajectory()

            drone_state_with_gesture = drone_state.copy()
            if self.current_gesture:
                drone_state_with_gesture['current_gesture'] = self.current_gesture
                drone_state_with_gesture['gesture_confidence'] = self.gesture_confidence
                drone_state_with_gesture['enhanced_confidence'] = self.enhanced_confidence
                drone_state_with_gesture['gesture_stability'] = self.gesture_stability
                drone_state_with_gesture['last_command'] = self.last_command
                drone_state_with_gesture['last_intensity'] = self.last_intensity

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
        if HAS_ENHANCED_DETECTOR and hasattr(self.gesture_detector, 'use_ml'):
            if self.gesture_detector.use_ml:
                mode_info = "机器学习模式 (更高精度)"
            else:
                mode_info = "规则检测模式 (基础)"
        else:
            mode_info = "规则检测模式"

        print(f"检测模式: {mode_info}")

        print("系统功能:")
        print("  1. 实时手势识别 (8种手势)")
        print("  2. 无人机控制仿真")
        print("  3. 3D可视化 (OpenGL渲染)")
        print("  4. 飞行数据记录")
        print("=" * 60)
        print("手势指令:")
        print("  张开手掌 - 起飞")
        print("  握拳 - 降落")
        print("  胜利手势 - 前进")
        print("  大拇指 - 后退")
        print("  食指上指 - 上升")
        print("  食指向下 - 下降")
        print("  OK手势 - 悬停")
        print("  大拇指向下 - 停止")
        print("=" * 60)
        print("使用说明:")
        print("  手势控制窗口: 按 'q' 退出")
        print("  手势控制窗口: 按 'c' 切换摄像头")
        print("  手势控制窗口: 按 'd' 显示调试信息")
        print("  3D仿真窗口: 按 'ESC' 退出")
        print("  3D窗口按键控制:")
        print("    G - 切换网格显示")
        print("    T - 切换轨迹显示")
        print("    A - 切换坐标轴显示")
        print("    ↑↓←→ - 旋转视角")
        print("    +/- - 缩放视角")
        print("    空格 - 重置视角")
        print("=" * 60)
        print("提示:")
        print("  1. 无人机初始在地面，等待手势指令")
        print("  2. 手势识别阈值已降低，更容易触发")
        print("  3. 做手势时保持手在摄像头中心")
        print("  4. 每个手势保持1.5秒以上")
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

            print("无人机仿真系统已安全关闭 ✓")


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

        print("✅ Pygame 已安装")
    except ImportError:
        print("❌ 错误: Pygame 未安装!")
        print("请运行: pip install pygame")
        sys.exit(1)

    try:
        import OpenGL

        print("✅ PyOpenGL 已安装")
    except ImportError:
        print("❌ 错误: PyOpenGL 未安装!")
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
