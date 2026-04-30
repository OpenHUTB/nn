# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import threading
import sys
import os
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入新架构的模块
from core import ConfigManager, Logger
from drone_controller import DroneController
from simulation_3d import Drone3DViewer

# 注意：physics_engine.py 是可选的，如果没有可以先注释掉
try:
    from physics_engine import PhysicsEngine
    HAS_PHYSICS_ENGINE = True
except ImportError:
    print("警告：未找到 physics_engine.py，使用简化的物理模拟")
    HAS_PHYSICS_ENGINE = False

# 导入手势检测器
try:
    from gesture_detector_enhanced import EnhancedGestureDetector
    print("[OK] 导入增强版手势检测器 (机器学习)")
    HAS_ENHANCED_DETECTOR = True
except ImportError:
    print("[WARNING] 未找到增强版检测器，使用原始手势检测器")
    from gesture_detector import GestureDetector
    HAS_ENHANCED_DETECTOR = False


class IntegratedDroneSimulationV2:
    """集成的无人机仿真系统 - 新版架构"""

    def __init__(self):
        # 初始化配置和日志
        self.config = ConfigManager()
        self.logger = Logger()

        # 系统状态
        self.running = True
        self.paused = False

        # 初始化模块
        self.logger.info("正在初始化手势检测器...")

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
                self.logger.info(f"找到 {model_name}: {file_size / 1024:.1f} KB")

                if file_size > 10 * 1024:
                    selected_model = model_path
                    selected_model_name = model_name
                    self.logger.info(f"选择: {model_name}")
                    break

        if HAS_ENHANCED_DETECTOR:
            # 优先使用增强版检测器（MediaPipe）
            try:
                from gesture_detector_enhanced import EnhancedGestureDetector
                self.logger.info("导入增强版手势检测器")

                if selected_model:
                    self.logger.info(f"使用模型: {selected_model_name}")
                    self.gesture_detector = EnhancedGestureDetector(
                        ml_model_path=selected_model,
                        use_ml=True
                    )

                    if hasattr(self.gesture_detector, 'ml_classifier') and self.gesture_detector.ml_classifier:
                        self.logger.info("机器学习模型加载成功")
                    else:
                        self.logger.warning("机器学习模型未加载，使用规则检测")
                        self.gesture_detector = EnhancedGestureDetector(use_ml=False)
                else:
                    # 使用规则检测（不需要模型）
                    self.logger.info("未找到模型文件，使用规则检测")
                    self.gesture_detector = EnhancedGestureDetector(use_ml=False)

            except ImportError as e:
                self.logger.warning(f"无法导入增强版检测器: {e}")
                self.logger.info("使用原始手势检测器")
                from gesture_detector import GestureDetector
                self.gesture_detector = GestureDetector()
        else:
            self.logger.warning("增强版检测器不可用")
            self.logger.info("使用原始手势检测器")
            from gesture_detector import GestureDetector
            self.gesture_detector = GestureDetector()

        self.logger.info("正在初始化无人机控制器...")
        self.drone_controller = DroneController(self.config, simulation_mode=True)

        self.logger.info("正在初始化3D仿真显示...")
        window_width = self.config.get("simulation.window_width", 1024)
        window_height = self.config.get("simulation.window_height", 768)
        self.viewer = Drone3DViewer(width=window_width, height=window_height)

        if HAS_PHYSICS_ENGINE:
            self.logger.info("正在初始化物理引擎...")
            mass = self.config.get("drone.mass", 1.0)
            gravity = self.config.get("drone.gravity", 9.81)
            self.physics_engine = PhysicsEngine(mass=mass, gravity=gravity)
        else:
            self.physics_engine = None

        self.gesture_thread = None
        self.simulation_thread = None

        self.current_frame = None
        self.current_gesture = None
        self.gesture_confidence = 0.0
        self.hand_landmarks = None

        self.control_intensity = 1.0
        self.last_command_time = time.time()
        self.command_cooldown = self.config.get("gesture.command_cooldown", 1.5)

        if HAS_ENHANCED_DETECTOR and hasattr(self.gesture_detector, 'use_ml') and self.gesture_detector.use_ml:
            self.logger.info("使用机器学习模式，置信度阈值更低")
            base_threshold = 0.55
        else:
            base_threshold = self.config.get("gesture.threshold", 0.6)

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

        self.logger.info("正在初始化摄像头...")
        camera_id = self.config.get("camera.default_id", 1)
        self.cap = self._initialize_camera(camera_id)

        self.data_log = []

        self.logger.info("无人机初始化完成，等待手势指令...")

    def _initialize_camera(self, camera_id):
        for cid in [camera_id, 0, 1]:
            self.logger.info(f"尝试打开摄像头 {cid}...")
            cap = cv2.VideoCapture(cid)

            if cap.isOpened():
                width = 640
                height = 480
                fps = 30

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ret, test_frame = cap.read()
                if ret:
                    self.logger.info(f"摄像头 {cid} 初始化成功: {width}x{height} @ {fps}fps")
                    return cap
                else:
                    cap.release()

        self.logger.error("所有摄像头尝试失败，使用虚拟模式")
        return None

    def _gesture_recognition_loop(self):
        self.logger.info("手势识别线程启动...")

        if HAS_ENHANCED_DETECTOR and hasattr(self.gesture_detector, 'use_ml'):
            mode_text = "机器学习模式" if self.gesture_detector.use_ml else "规则检测模式"
        else:
            mode_text = "规则检测模式"

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                else:
                    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
                    cv2.putText(frame, "Camera Error - Virtual Mode", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
                cv2.putText(frame, "虚拟摄像头模式", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 每帧检测手势（流畅响应）
            try:
                processed_frame, gesture, confidence, landmarks = \
                    self.gesture_detector.detect_gestures(frame, simulation_mode=True)

                self.current_frame = processed_frame
                self.current_gesture = gesture
                self.gesture_confidence = confidence
                self.hand_landmarks = landmarks

                self._process_gesture_command(gesture, confidence)

                enhanced_frame = self._enhance_interface(processed_frame, gesture, confidence)
                cv2.imshow('Gesture Control', enhanced_frame)

            except Exception as e:
                self.logger.error(f"手势检测错误: {e}")
                self.current_frame = frame
                self.current_gesture = None
                enhanced_frame = self._enhance_interface(frame, "error", 0.0)
                cv2.imshow('Gesture Control', enhanced_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.logger.info("收到退出指令...")
                self.running = False
                break
            elif key == ord('c'):
                self._switch_camera()
            elif key == ord('d'):
                self._debug_gesture_detection()

        self.logger.info("手势识别线程结束")

    def _enhance_interface(self, frame, gesture, confidence):
        height, width = frame.shape[:2]
        panel_width = 300
        total_width = width + panel_width
        enhanced_frame = np.ones((height, total_width, 3), dtype=np.uint8) * 20

        enhanced_frame[:, :width] = frame

        cv2.rectangle(enhanced_frame, (width, 0), (total_width, height), (50, 50, 50), 2)

        cv2.putText(enhanced_frame, "DRONE CONTROL", (width + 20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        y_offset = 80
        if gesture and gesture != "no_hand":
            cv2.putText(enhanced_frame, f"GESTURE: {gesture.upper()}",
                        (width + 20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_offset += 30

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

        cv2.putText(enhanced_frame, "DRONE STATUS",
                    (width + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25

        drone_state = self.drone_controller.get_state()

        status_info = [
            f"MODE: {drone_state['mode'].upper()}",
            f"ARMED: {'YES' if drone_state['armed'] else 'NO'}",
            f"BATTERY: {drone_state['battery']:.1f}%",
            f"ALTITUDE: {abs(drone_state['position'][2]):.2f}m"
        ]

        for info in status_info:
            cv2.putText(enhanced_frame, info,
                        (width + 20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
            y_offset += 20

        y_offset += 10

        cv2.putText(enhanced_frame, "CONTROLS",
                    (width + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25

        controls = [
            "Q/ESC - Exit",
            "C - Switch Camera",
            "D - Debug Info",
        ]

        for control in controls:
            cv2.putText(enhanced_frame, control,
                        (width + 20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_offset += 15

        return enhanced_frame

    def _switch_camera(self):
        if self.cap:
            self.cap.release()

        current_id = 1 if self.cap is None else 0
        self.logger.info(f"切换到摄像头 {current_id}...")
        self.cap = cv2.VideoCapture(current_id)

    def _debug_gesture_detection(self):
        self.logger.info(f"当前手势: {self.current_gesture}")
        self.logger.info(f"置信度: {self.gesture_confidence:.2f}")
        self.logger.info(f"无人机位置: {self.drone_controller.state['position']}")

    def _process_gesture_command(self, gesture, confidence):
        current_time = time.time()

        threshold = self.gesture_thresholds.get(gesture, self.gesture_thresholds['default'])

        in_cooldown = current_time - self.last_command_time <= self.command_cooldown

        same_gesture = (gesture == self.current_gesture and
                        hasattr(self, 'last_processed_gesture') and
                        gesture == self.last_processed_gesture and
                        current_time - getattr(self, 'last_processed_time', 0) < 2.0)

        if (gesture not in ["no_hand", "hand_detected"] and
                confidence > threshold and
                not in_cooldown and
                not same_gesture):

            command = self.gesture_detector.get_command(gesture)

            if command != "none":
                intensity = 1.0
                if self.hand_landmarks:
                    intensity = self.gesture_detector.get_gesture_intensity(
                        self.hand_landmarks, gesture
                    )

                self.logger.info(f"检测到手势: {gesture} (置信度: {confidence:.2f}) -> 执行: {command}")

                self.drone_controller.send_command(command, intensity)

                self._log_command(gesture, command, confidence, intensity)

                self.last_command_time = current_time
                self.last_processed_gesture = gesture
                self.last_processed_time = current_time

    def _simulation_loop(self):
        self.logger.info("3D仿真线程启动...")

        import pygame

        last_time = time.time()
        last_status_print = time.time()

        target_fps = self.config.get("simulation.target_fps", 60)
        frame_delay = 1.0 / target_fps

        self.logger.info("键盘提示：按 'R' 键重置无人机位置到原点")

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

            if current_time - last_status_print > 3:
                status = self.drone_controller.get_status_string()
                self.logger.info(status)
                if self.current_gesture:
                    self.logger.info(f"当前手势: {self.current_gesture} (置信度: {self.gesture_confidence:.2f})")
                last_status_print = current_time

            if self.paused:
                if not self.viewer.handle_events():
                    self.running = False
                time.sleep(0.01)
                continue

            keys = pygame.key.get_pressed()

            if keys[pygame.K_r]:
                if ('r' not in self._last_key_press or
                        current_time - self._last_key_press['r'] > 1.0):
                    self.logger.info("键盘：重置无人机位置")
                    self.drone_controller.reset()
                    self._last_key_press['r'] = current_time

            if keys[pygame.K_t]:
                if ('t' not in self._last_key_press or
                        current_time - self._last_key_press['t'] > 1.0):
                    self.logger.info("键盘：起飞")
                    self.drone_controller.send_command("takeoff", 0.8)
                    self._last_key_press['t'] = current_time

            if keys[pygame.K_l]:
                if ('l' not in self._last_key_press or
                        current_time - self._last_key_press['l'] > 1.0):
                    self.logger.info("键盘：降落")
                    self.drone_controller.send_command("land", 0.5)
                    self._last_key_press['l'] = current_time

            if keys[pygame.K_h]:
                if ('h' not in self._last_key_press or
                        current_time - self._last_key_press['h'] > 1.0):
                    self.logger.info("键盘：悬停")
                    self.drone_controller.send_command("hover")
                    self._last_key_press['h'] = current_time

            if not self.viewer.handle_events():
                self.running = False
                break

            if not self.running:
                break

            drone_state = self.drone_controller.get_state()
            if hasattr(self.drone_controller, 'update_physics'):
                self.drone_controller.update_physics(dt)

            trajectory = self.drone_controller.get_trajectory()

            drone_state_with_gesture = drone_state.copy()
            if self.current_gesture:
                drone_state_with_gesture['current_gesture'] = self.current_gesture
                drone_state_with_gesture['gesture_confidence'] = self.gesture_confidence

            self.viewer.render(drone_state_with_gesture, trajectory)

            elapsed = time.time() - start_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.logger.info("3D仿真线程结束")

    def _get_control_input_from_state(self, drone_state):
        control_input = {
            'throttle': 0.5,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw_rate': 0.0
        }

        if self.hand_landmarks and self.current_gesture:
            if self.current_gesture == "pointing_up":
                control_input['throttle'] = 0.8
            elif self.current_gesture == "pointing_down":
                control_input['throttle'] = 0.2
            elif self.current_gesture == "victory":
                control_input['pitch'] = 0.3
            elif self.current_gesture == "thumb_up":
                control_input['pitch'] = -0.3

        return control_input

    def _log_command(self, gesture, command, confidence, intensity):
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

    def _save_log(self):
        if self.data_log:
            try:
                log_file = "flight_log.json"
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(self.data_log, f, indent=2, ensure_ascii=False)
                self.logger.info(f"飞行日志已保存到: {log_file}")
            except Exception as e:
                self.logger.error(f"保存日志失败: {e}")

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("手势控制无人机仿真系统 - 新架构版")
        self.logger.info("=" * 60)

        if HAS_ENHANCED_DETECTOR and hasattr(self.gesture_detector, 'use_ml'):
            mode_info = "机器学习模式 (更高精度)" if self.gesture_detector.use_ml else "规则检测模式 (基础)"
        else:
            mode_info = "规则检测模式"

        self.logger.info(f"检测模式: {mode_info}")

        try:
            self.drone_controller.connect()

            self.gesture_thread = threading.Thread(
                target=self._gesture_recognition_loop,
                name="GestureThread",
                daemon=True
            )
            self.gesture_thread.start()

            self.logger.info("手势识别线程已启动")
            time.sleep(1)

            self._simulation_loop()

            if self.gesture_thread.is_alive():
                self.gesture_thread.join(timeout=2.0)

        except KeyboardInterrupt:
            self.logger.info("系统被用户中断")
        except Exception as e:
            self.logger.error(f"系统运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False

            if self.cap:
                self.cap.release()
                self.logger.info("摄像头已释放")

            cv2.destroyAllWindows()
            self.logger.info("OpenCV窗口已关闭")

            self._save_log()

            self.drone_controller.disconnect()

            self.logger.info("无人机仿真系统已安全关闭")


def main():
    print("=" * 60)
    print("手势控制无人机仿真系统 - 新架构版")
    print("=" * 60)

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

    simulation = IntegratedDroneSimulationV2()
    simulation.run()


if __name__ == "__main__":
    main()
