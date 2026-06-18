import cv2
<<<<<<< HEAD
import mediapipe as mp
import numpy as np
import math


class GestureDetector:
    def __init__(self):
        """
        初始化手势检测器
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 初始化手部检测模型
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # 手势到控制指令的映射
        self.gesture_commands = {
            "open_palm": "takeoff",  # 张开手掌 - 起飞
            "closed_fist": "land",  # 握拳 - 降落
            "pointing_up": "up",  # 食指上指 - 上升
            "pointing_down": "down",  # 食指向下 - 下降
            "victory": "forward",  # 胜利手势 - 前进
            "thumb_up": "backward",  # 大拇指 - 后退
            "thumb_down": "stop",  # 大拇指向下 - 停止
            "ok_sign": "hover"  # OK手势 - 悬停
=======
import numpy as np
import math
import time

# 尝试导入 MediaPipe（可选）
HAS_MEDIAPIPE = False
mp = None

try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
        HAS_MEDIAPIPE = True
        print("[OK] MediaPipe 可用")
    else:
        print("[WARNING] MediaPipe 版本不支持手势检测，将使用 OpenCV 模式")
        mp = None
except ImportError:
    print("[WARNING] MediaPipe 未安装，将使用 OpenCV 模式")
    mp = None


class GestureDetector:
    """
    手势检测器
    支持 OpenCV 和 MediaPipe 两种模式
    支持滑动手势控制和灵敏度动态调节
    """
    
    def __init__(self):
        """
        初始化手势检测器（支持双手控制）
        左手控制方向，右手控制高度
        """
        # ============ 灵敏度配置 ============
        # 灵敏度级别: 1=低(严格), 2=中, 3=高(宽松)
        self.sensitivity_level = 2
        self.sensitivity_presets = {
            1: {  # 低灵敏度 - 严格模式
                'min_detection_confidence': 0.8,
                'min_tracking_confidence': 0.7,
                'gesture_threshold': 0.7,
                'swipe_threshold': 0.20,
                'swipe_min_velocity': 0.4,
                # OpenCV 模式参数
                'skin_threshold1': 0,
                'skin_threshold2': 30,
                'contour_area_min': 3000,
            },
            2: {  # 中灵敏度 - 平衡模式
                'min_detection_confidence': 0.6,
                'min_tracking_confidence': 0.5,
                'gesture_threshold': 0.55,
                'swipe_threshold': 0.15,
                'swipe_min_velocity': 0.3,
                # OpenCV 模式参数
                'skin_threshold1': 0,
                'skin_threshold2': 20,
                'contour_area_min': 2000,
            },
            3: {  # 高灵敏度 - 宽松模式
                'min_detection_confidence': 0.4,
                'min_tracking_confidence': 0.3,
                'gesture_threshold': 0.4,
                'swipe_threshold': 0.10,
                'swipe_min_velocity': 0.2,
                # OpenCV 模式参数
                'skin_threshold1': 0,
                'skin_threshold2': 15,
                'contour_area_min': 1000,
            }
        }
        self.sensitivity_names = {
            1: "LOW",
            2: "MEDIUM", 
            3: "HIGH"
        }
        
        self.hands = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.mode = "none"
        
        # 尝试初始化 MediaPipe
        if HAS_MEDIAPIPE:
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                
                # 初始化手部检测模型（支持双手）
                self._update_hands_model()
                self.mode = "mediapipe"
                print("[INFO] 使用 MediaPipe 手势检测模式")
            except Exception as e:
                print(f"[WARNING] MediaPipe 初始化失败: {e}")
                self.mode = "opencv"
        else:
            self.mode = "opencv"
            self._init_opencv_detector()
        
        # 更新滑动参数
        self._update_swipe_params()
        
        # 左手手势到控制指令的映射（方向控制）
        self.left_hand_commands = {
            "victory": "forward",      # 胜利手势 - 前进
            "thumb_up": "backward",    # 大拇指向上 - 后退
            "pointing_up": "left",     # 食指向左 - 左转
            "pointing_down": "right",  # 食指向右 - 右转
        }

        # 右手手势到控制指令的映射（高度控制）
        self.right_hand_commands = {
            "pointing_up": "up",       # 食指向下 - 上升
            "pointing_down": "down",   # 食指向下 - 下降
            "ok_sign": "hover",        # OK手势 - 悬停
        }

        # 双手手势指令（特殊命令）
        self.both_hands_commands = {
            "open_palm": "takeoff",    # 张开手掌（任意手）- 起飞
            "closed_fist": "land",     # 握拳（任意手）- 降落
            "thumb_down": "stop",      # 大拇指向下 - 停止
        }

        # ============ 多无人机控制配置 ============
        # 控制模式: "dual" - 左手控制Drone1，右手控制Drone2
        #           "single" - 只控制选中的无人机
        #           "sync" - 同步控制所有无人机
        self.multi_drone_mode = False  # 默认不启用多无人机模式
        self.multi_drone_commands = {
            # 左手手势 - 控制 Drone 1
            "left_victory": "forward",        # 左手胜利 - 前进
            "left_thumb_up": "backward",       # 左手大拇指向上 - 后退
            "left_pointing_left": "left",      # 左手食指向左 - 左转
            "left_pointing_right": "right",    # 左手食指向右 - 右转
            "left_swipe_left": "drone1_left",  # 左手左滑 - Drone1左移
            "left_swipe_right": "drone1_right",# 左手右滑 - Drone1右移
            # 右手手势 - 控制 Drone 2
            "right_pointing_up": "up",          # 右手食指向上 - 上升
            "right_pointing_down": "down",      # 右手食指向下 - 下降
            "right_ok": "hover",                # 右手OK - 悬停
            "right_swipe_left": "drone2_left",  # 右手左滑 - Drone2左移
            "right_swipe_right": "drone2_right",# 右手右滑 - Drone2右移
            # 双手手势 - 全体命令
            "both_open_palm": "takeoff_all",   # 双手张开 - 全体起飞
            "both_closed_fist": "land_all",    # 双手握拳 - 全体降落
            "both_thumbs_up": "formation_line", # 双手拇指向上 - 切换直线编队
            "both_thumbs_down": "formation_v",  # 双手拇指向下 - 切换V字编队
        }

        # 合并所有手势命令（用于显示）
        self.gesture_commands = {
            "open_palm": "takeoff",
            "closed_fist": "land",
            "pointing_up": "up",
            "pointing_down": "down",
            "victory": "forward",
            "thumb_up": "backward",
            "thumb_down": "stop",
            "ok_sign": "hover",
            # 滑动手势命令
            "swipe_left": "left",
            "swipe_right": "right",
            "swipe_up": "forward",
            "swipe_down": "backward",
        }
        
        # ============ 滑动手势控制相关 ============
        # 手掌位置历史记录（用于检测滑动）
        self.palm_history = {
            'left': [],   # 左手历史位置 [(x, y, timestamp), ...]
            'right': []  # 右手历史位置
        }
        self.max_history_length = 10  # 最大历史记录数量
        
        # 滑动手势命令映射
        self.swipe_commands = {
            "swipe_left": "left",       # 向左滑动 - 无人机左移
            "swipe_right": "right",     # 向右滑动 - 无人机右移
            "swipe_up": "forward",      # 向上滑动 - 无人机前进
            "swipe_down": "backward",   # 向下滑动 - 无人机后退
        }
        
        # 滑动检测参数
        self.swipe_threshold = 0.15
        self.swipe_min_velocity = 0.3
        self.swipe_cooldown = 0.5
        self.last_swipe_time = 0
        
        # 当前检测到的滑动手势
        self.current_swipe = None
        self.swipe_intensity = 0.5
        
        # OpenCV 模式参数
        self._update_opencv_params()
        
        print(f"[INFO] 手势检测器初始化完成: {self.mode} 模式 + 滑动手势支持 + 灵敏度调节")

    def _init_opencv_detector(self):
        """初始化 OpenCV 检测器参数"""
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.prev_hand_pos = None
        self.prev_time = time.time()

    def _update_hands_model(self):
        """根据当前灵敏度更新MediaPipe模型参数"""
        if not HAS_MEDIAPIPE or not hasattr(self, 'mp_hands'):
            return
            
        preset = self.sensitivity_presets[self.sensitivity_level]
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=preset['min_detection_confidence'],
            min_tracking_confidence=preset['min_tracking_confidence']
        )

    def _update_swipe_params(self):
        """根据当前灵敏度更新滑动检测参数"""
        preset = self.sensitivity_presets[self.sensitivity_level]
        self.swipe_threshold = preset['swipe_threshold']
        self.swipe_min_velocity = preset['swipe_min_velocity']

    def _update_opencv_params(self):
        """根据当前灵敏度更新 OpenCV 检测参数"""
        preset = self.sensitivity_presets[self.sensitivity_level]
        self.skin_threshold1 = preset['skin_threshold1']
        self.skin_threshold2 = preset['skin_threshold2']
        self.contour_area_min = preset['contour_area_min']

    def set_sensitivity(self, level):
        """
        设置灵敏度级别
        
        Returns:
            dict: 包含灵敏度级别、名称、阈值等信息的字典
        """
        preset = self.sensitivity_presets[self.sensitivity_level]
        return {
            'level': self.sensitivity_level,
            'name': self.sensitivity_names[self.sensitivity_level],
            'detection_confidence': preset['min_detection_confidence'],
            'tracking_confidence': preset['min_tracking_confidence'],
            'gesture_threshold': preset['gesture_threshold'],
            'swipe_threshold': preset['swipe_threshold'],
            'swipe_velocity': preset['swipe_min_velocity'],
            'contour_area_min': preset['contour_area_min'],
            'mode': self.mode
>>>>>>> upstream/main
        }

    def detect_gestures(self, image, simulation_mode=False):
        """
<<<<<<< HEAD
        检测图像中的手势
=======
        检测图像中的手势（支持双手和滑动手势）

        Args:
            level: 1=低(严格), 2=中, 3=高(宽松)
            
        Returns:
            bool: 是否设置成功
        """
        if level in self.sensitivity_presets:
            old_level = self.sensitivity_level
            self.sensitivity_level = level
            self._update_hands_model()
            self._update_swipe_params()
            self._update_opencv_params()
            print(f"[灵敏度调整] {self.sensitivity_names[old_level]} -> {self.sensitivity_names[level]}")
            return True
        return False

    def increase_sensitivity(self):
        """增加灵敏度（向更宽松方向调整）"""
        if self.sensitivity_level < 3:
            return self.set_sensitivity(self.sensitivity_level + 1)
        return False

    def decrease_sensitivity(self):
        """降低灵敏度（向更严格方向调整）"""
        if self.sensitivity_level > 1:
            return self.set_sensitivity(self.sensitivity_level - 1)
        return False

    def get_sensitivity_info(self):
        """
        获取当前灵敏度信息
        
        Returns:
            dict: 包含灵敏度级别、名称、阈值等信息的字典
        """
        preset = self.sensitivity_presets[self.sensitivity_level]
        return {
            'level': self.sensitivity_level,
            'name': self.sensitivity_names[self.sensitivity_level],
            'detection_confidence': preset['min_detection_confidence'],
            'tracking_confidence': preset['min_tracking_confidence'],
            'gesture_threshold': preset['gesture_threshold'],
            'swipe_threshold': preset['swipe_threshold'],
            'swipe_velocity': preset['swipe_min_velocity'],
            'contour_area_min': preset['contour_area_min'],
            'mode': self.mode
        }

    def detect_gestures(self, image, simulation_mode=False):
        """
        检测图像中的手势（支持双手和滑动手势）
>>>>>>> upstream/main

        Args:
            image: 输入图像
            simulation_mode: 是否为仿真模式
<<<<<<< HEAD

=======
            
>>>>>>> upstream/main
        Returns:
            processed_image: 处理后的图像
            gesture: 识别到的手势
            confidence: 置信度
<<<<<<< HEAD
            landmarks: 关键点坐标（仅仿真模式返回）
        """
        # 转换颜色空间
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        gesture = "no_hand"
        confidence = 0.0
        landmarks_data = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点和连接线
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
=======
            landmarks: 关键点坐标（仿真模式下返回简化数据）
        """
        if self.mode == "mediapipe":
            return self._detect_mediapipe(image, simulation_mode)
        else:
            return self._detect_opencv(image, simulation_mode)

    def _detect_mediapipe(self, image, simulation_mode=False):
        """MediaPipe 模式检测"""
        result_image = image.copy()
        height, width = image.shape[:2]
        
        self.current_swipe = None
        self.swipe_intensity = 0.5
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        gesture = "no_hand"
        confidence = 0.0
        landmarks_data = None
        left_hand_data = None
        right_hand_data = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                hand_type = handedness.classification[0].label
                hand_confidence = handedness.classification[0].score

                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
>>>>>>> upstream/main
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

<<<<<<< HEAD
                # 识别具体手势
                gesture, confidence = self._classify_gesture(hand_landmarks)

                # 在仿真模式下提取关键点数据
                if simulation_mode:
                    landmarks_data = self._get_normalized_landmarks(hand_landmarks)

                # 在图像上显示手势信息
                cv2.putText(image, f"Gesture: {gesture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 显示控制指令
                command = self.gesture_commands.get(gesture, "none")
                cv2.putText(image, f"Command: {command}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return image, gesture, confidence, landmarks_data

    def _get_normalized_landmarks(self, hand_landmarks):
        """
        获取归一化的关键点坐标（用于仿真模式）

        Args:
            hand_landmarks: 手部关键点

        Returns:
            list: 包含21个关键点的字典列表，每个点有x,y,z坐标
        """
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            })
        return landmarks

    def _classify_gesture(self, landmarks):
        """
        分类具体手势

        Args:
            landmarks: 手部关键点

        Returns:
            gesture: 手势名称
            confidence: 置信度
        """
        # 获取关键点坐标
        points = []
        for landmark in landmarks.landmark:
            points.append((landmark.x, landmark.y, landmark.z))

        # 计算各种手势的特征
        is_thumb_up = self._is_thumb_up(points)
        is_thumb_down = self._is_thumb_down(points)
        is_open_palm = self._is_open_palm(points)
        is_closed_fist = self._is_closed_fist(points)
        is_pointing_up = self._is_pointing_up(points)
        is_pointing_down = self._is_pointing_down(points)
        is_victory = self._is_victory_gesture(points)
        is_ok_sign = self._is_ok_sign(points)

        # 调试输出
        print(f"[手势特征] "
              f"拇指上:{is_thumb_up} "
              f"拇指向下:{is_thumb_down} "
              f"张开掌:{is_open_palm} "
              f"握拳:{is_closed_fist} "
              f"指上:{is_pointing_up} "
              f"指下:{is_pointing_down} "
              f"胜利:{is_victory} "
              f"OK:{is_ok_sign}")

        # 根据优先级返回手势
        gestures = [
            (is_thumb_up, "thumb_up", 0.95),
            (is_thumb_down, "thumb_down", 0.95),
            (is_ok_sign, "ok_sign", 0.90),
            (is_victory, "victory", 0.85),
            (is_open_palm, "open_palm", 0.80),  # 提高优先级
            (is_closed_fist, "closed_fist", 0.80),
            (is_pointing_up, "pointing_up", 0.75),
            (is_pointing_down, "pointing_down", 0.75),
        ]

        for condition, gesture_name, conf in gestures:
            if condition:
                return gesture_name, conf

        return "hand_detected", 0.5

    def _is_thumb_up(self, points):
        """检测大拇指向上手势"""
        thumb_tip = points[4]  # 大拇指指尖
        thumb_ip = points[3]  # 大拇指指间关节
        index_mcp = points[5]  # 食指掌指关节

        # 大拇指伸直且向上
        thumb_extended = thumb_tip[1] < thumb_ip[1]  # y坐标更小表示更靠上
        thumb_raised = thumb_tip[1] < index_mcp[1]  # 大拇指高于食指基部

        return thumb_extended and thumb_raised

    def _is_thumb_down(self, points):
        """检测大拇指向下手势"""
        thumb_tip = points[4]
        thumb_ip = points[3]
        pinky_mcp = points[17]  # 小指掌指关节

        # 大拇指伸直且向下
        thumb_extended = thumb_tip[1] > thumb_ip[1]  # y坐标更大表示更靠下
        thumb_lowered = thumb_tip[1] > pinky_mcp[1]  # 大拇指低于小指基部

        return thumb_extended and thumb_lowered

    def _is_open_palm(self, points):
        """检测张开手掌手势"""
        finger_tips = [8, 12, 16, 20]  # 食指、中指、无名指、小指指尖
        finger_mcps = [5, 9, 13, 17]  # 掌指关节

        extended_fingers = 0
        for tip, mcp in zip(finger_tips, finger_mcps):
            # 放宽条件：指尖比掌指关节高（y值小），增加容差0.15
            if points[tip][1] < points[mcp][1] + 0.15:
                extended_fingers += 1

        # 放宽条件：至少2个手指伸直
        return extended_fingers >= 4

    def _is_closed_fist(self, points):
        """检测握拳手势"""
        finger_tips = [8, 12, 16, 20]  # 指尖
        finger_mcps = [5, 9, 13, 17]  # 掌指关节

        bent_fingers = 0
        for tip, mcp in zip(finger_tips, finger_mcps):
            # 放宽条件：指尖在掌指关节下方或接近
            if points[tip][1] > points[mcp][1] - 0.1:
                bent_fingers += 1

        # 放宽条件：至少3个手指弯曲
        return bent_fingers >= 3

    def _is_pointing_up(self, points):
        """检测食指上指手势"""
        index_tip = points[8]  # 食指尖
        index_dip = points[7]  # 食指指间关节
        middle_tip = points[12]  # 中指尖
        middle_mcp = points[9]  # 中指掌指关节

        # 食指伸直且向上，其他手指弯曲
        index_extended = index_tip[1] < index_dip[1]
        middle_bent = middle_tip[1] > middle_mcp[1]

        return index_extended and middle_bent

    def _is_pointing_down(self, points):
        """检测食指向下手势"""
        index_tip = points[8]
        index_dip = points[7]
        middle_tip = points[12]
        middle_mcp = points[9]

        # 食指伸直且向下，其他手指弯曲
        index_extended = index_tip[1] > index_dip[1]
        middle_bent = middle_tip[1] > middle_mcp[1]

        return index_extended and middle_bent

    def _is_victory_gesture(self, points):
        """检测胜利手势（食指和中指伸直）"""
        index_tip, middle_tip = points[8], points[12]
        index_dip, middle_dip = points[7], points[11]
        ring_tip = points[16]
        ring_mcp = points[13]

        # 食指和中指伸直
        index_extended = index_tip[1] < index_dip[1]
        middle_extended = middle_tip[1] < middle_dip[1]
        # 无名指弯曲
        ring_bent = ring_tip[1] > ring_mcp[1]

        return index_extended and middle_extended and ring_bent

    def _is_ok_sign(self, points):
        """检测OK手势（食指和拇指接触）"""
        thumb_tip = points[4]
        index_tip = points[8]

        # 计算食指和拇指之间的距离
        distance = math.sqrt(
            (thumb_tip[0] - index_tip[0]) ** 2 +
            (thumb_tip[1] - index_tip[1]) ** 2
        )

        # 距离很小表示接触
        return distance < 0.05

    def get_command(self, gesture):
        """
        根据手势获取控制指令

        Args:
            gesture: 手势名称

        Returns:
            command: 控制指令
        """
        return self.gesture_commands.get(gesture, "none")

    def get_gesture_intensity(self, landmarks, gesture_type):
        """
        获取手势强度（用于精细控制）

        Args:
            landmarks: 关键点数据
            gesture_type: 手势类型

        Returns:
            float: 手势强度 (0.0-1.0)
        """
        if not landmarks or len(landmarks) < 21:
            return 0.5  # 默认强度

        if gesture_type == "pointing_up":
            # 基于食指的角度计算强度
            index_tip = landmarks[8]
            index_mcp = landmarks[5]
            if index_tip['y'] < index_mcp['y']:
                intensity = (index_mcp['y'] - index_tip['y']) * 2
                return min(max(intensity, 0.1), 1.0)

        elif gesture_type == "pointing_down":
            # 基于食指的角度计算强度
            index_tip = landmarks[8]
            index_mcp = landmarks[5]
            if index_tip['y'] > index_mcp['y']:
                intensity = (index_tip['y'] - index_mcp['y']) * 2
                return min(max(intensity, 0.1), 1.0)

        elif gesture_type in ["open_palm", "closed_fist"]:
            # 基于手掌张开程度计算强度
            thumb_tip = landmarks[4]
            pinky_tip = landmarks[20]
            distance = math.sqrt(
                (thumb_tip['x'] - pinky_tip['x']) ** 2 +
                (thumb_tip['y'] - pinky_tip['y']) ** 2
            )
            if gesture_type == "open_palm":
                return min(max(distance * 3, 0.1), 1.0)
            else:
                return min(max((0.2 - distance) * 5, 0.1), 1.0)

        return 0.5  # 默认强度

    def get_hand_position(self, landmarks):
        """
        获取手部在画面中的位置

        Args:
            landmarks: 关键点数据

        Returns:
            dict: 包含手部中心位置和大小
        """
        if not landmarks or len(landmarks) < 21:
            return None

        # 计算手部边界框
        x_coords = [p['x'] for p in landmarks]
        y_coords = [p['y'] for p in landmarks]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        return {
            'center_x': (min_x + max_x) / 2,
            'center_y': (min_y + max_y) / 2,
            'width': max_x - min_x,
            'height': max_y - min_y,
            'bbox': (min_x, min_y, max_x, max_y)
        }

    def release(self):
        """释放资源"""
        self.hands.close()
=======
                detected_gesture, gesture_confidence = self._classify_gesture(hand_landmarks)
                palm_position = self._get_palm_center(hand_landmarks)
                
                if palm_position:
                    current_time = time.time()
                    palm_key = 'left' if hand_type == "Left" else 'right'
                    self.palm_history[palm_key].append({
                        'position': (palm_position['x'], palm_position['y']),
                        'timestamp': current_time
                    })
                    if len(self.palm_history[palm_key]) > self.max_history_length:
                        self.palm_history[palm_key].pop(0)
                    
                    swipe_result = self._detect_swipe_gesture(palm_key, width, height, current_time)
                    if swipe_result:
                        self.current_swipe = swipe_result['direction']
                        self.swipe_intensity = swipe_result['intensity']
                        detected_gesture = swipe_result['gesture_name']
                        gesture_confidence = swipe_result['confidence']

                if simulation_mode:
                    normalized_landmarks = self._get_normalized_landmarks(hand_landmarks)
                    normalized_landmarks['hand_type'] = hand_type
                    normalized_landmarks['gesture'] = detected_gesture
                    normalized_landmarks['confidence'] = gesture_confidence

                    if hand_type == "Left":
                        left_hand_data = normalized_landmarks
                    else:
                        right_hand_data = normalized_landmarks

                y_offset = 30 if idx == 0 else 150
                color = (0, 255, 0) if hand_type == "Left" else (255, 128, 0)
                cv2.putText(image, f"{hand_type}: {detected_gesture}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(image, f"{hand_type} Conf: {gesture_confidence:.2f}", (10, y_offset + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

                if gesture_confidence > confidence:
                    gesture = detected_gesture
                    confidence = gesture_confidence

            if left_hand_data and right_hand_data:
                cv2.putText(image, "LEFT: Direction | RIGHT: Altitude", (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            elif left_hand_data:
                cv2.putText(image, "Left hand: Direction control", (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            elif right_hand_data:
                cv2.putText(image, "Right hand: Altitude control", (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 1)

        command = self.gesture_commands.get(gesture, "none")
        cv2.putText(image, f"Command: {command}", (10, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if simulation_mode:
            landmarks_data = {
                'left_hand': left_hand_data,
                'right_hand': right_hand_data
            }

        return image, gesture, confidence, landmarks_data

    def _detect_opencv(self, image, simulation_mode=False):
        """OpenCV 模式检测（基于肤色和轮廓）"""
        result_image = image.copy()
        height, width = image.shape[:2]
        
        self.current_swipe = None
        self.swipe_intensity = 0.5
        
        # 肤色检测
        skin_mask = self._detect_skin(image)
        
        # 找轮廓
        contours, _ = cv2.findContours(
            skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        gesture = "no_hand"
        confidence = 0.0
        landmarks_data = None
        left_hand_data = None
        right_hand_data = None
        hand_centers = []
        
        # 过滤小的轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.contour_area_min:
                hand_centers.append(self._get_contour_center(contour))
        
        if len(hand_centers) > 0:
            # 简单手势识别
            detected_gesture, gesture_confidence = self._detect_simple_gesture(
                image, contours, skin_mask
            )
            gesture = detected_gesture
            confidence = gesture_confidence
            
            # 绘制检测结果
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
            cv2.putText(image, f"Gesture: {detected_gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f"Conf: {gesture_confidence:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # 检测滑动（基于手部位置变化）
            current_time = time.time()
            if self.prev_hand_pos and hand_centers:
                center = hand_centers[0]
                delta_x = center[0] / width - self.prev_hand_pos[0]
                delta_y = center[1] / height - self.prev_hand_pos[1]
                time_delta = current_time - self.prev_time
                
                if time_delta > 0:
                    velocity_x = abs(delta_x) / time_delta
                    velocity_y = abs(delta_y) / time_delta
                    
                    if (abs(delta_x) > self.swipe_threshold and 
                        velocity_x > self.swipe_min_velocity):
                        direction = "swipe_right" if delta_x > 0 else "swipe_left"
                        gesture = direction
                        confidence = 0.75
                        self.current_swipe = direction
                        self.swipe_intensity = min(abs(delta_x) * 2, 1.0)
                    elif (abs(delta_y) > self.swipe_threshold and 
                          velocity_y > self.swipe_min_velocity):
                        direction = "swipe_down" if delta_y > 0 else "swipe_up"
                        gesture = direction
                        confidence = 0.75
                        self.current_swipe = direction
                        self.swipe_intensity = min(abs(delta_y) * 2, 1.0)
            
            self.prev_hand_pos = hand_centers[0] if hand_centers else None
            self.prev_time = current_time
        
        command = self.gesture_commands.get(gesture, "none")
        cv2.putText(image, f"Command: {command}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # OpenCV模式提示
        cv2.putText(image, f"[OpenCV Mode] Sensitivity: {self.sensitivity_names[self.sensitivity_level]}", 
                    (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        return image, gesture, confidence, landmarks_data

    def _detect_skin(self, image):
        """肤色检测"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 调整肤色阈值（基于灵敏度）
        lower_skin = np.array([0, self.skin_threshold1, max(0, 30 - self.skin_threshold2 * 2)])
        upper_skin = np.array([20, self.skin_threshold2 * 3, 255])
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask

    def _get_contour_center(self, contour):
        """获取轮廓中心"""
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        return (0, 0)

    def _detect_simple_gesture(self, image, contours, mask):
        """简单手势检测"""
        if len(contours) == 0:
            return "no_hand", 0.0
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # 近似轮廓
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 根据灵敏度和轮廓特征判断手势
        finger_count = self._count_fingers(approx, mask)
        
        if finger_count >= 4:
            return "open_palm", 0.8
        elif finger_count == 0:
            return "closed_fist", 0.8
        elif finger_count == 2:
            return "victory", 0.75
        elif finger_count == 1:
            return "pointing_up", 0.7
        
        return "unknown", 0.3

    def _count_fingers(self, contour, mask):
        """简单手指计数"""
        return 2  # 默认返回2（简化实现）

    def _classify_gesture(self, hand_landmarks):
        """MediaPipe 模式分类手势"""
        if not hand_landmarks:
            return "no_hand", 0.0
            
        landmarks = hand_landmarks.landmark
        finger_states = self._get_finger_states(landmarks)
        
        if all(finger_states[:4]):
            return "open_palm", 0.85
        
        if not any(finger_states[:4]):
            return "closed_fist", 0.85
        
        if self._is_ok_sign(landmarks):
            return "ok_sign", 0.80
        
        if self._is_thumb_up(landmarks):
            return "thumb_up", 0.80
        
        if self._is_thumb_down(landmarks):
            return "thumb_down", 0.80
        
        if finger_states[1] and not any(finger_states[2:4]):
            return "pointing_up", 0.75
        
        if finger_states[1] and finger_states[2] and not finger_states[3] and not finger_states[4]:
            return "victory", 0.75
        
        return "unknown", 0.3

    def _get_finger_states(self, landmarks):
        """获取手指伸展状态"""
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        states = []
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y - 0.02:
                states.append(True)
            else:
                states.append(False)
        
        return states

    def _is_ok_sign(self, landmarks):
        """检测OK手势"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        thumb_index_dist = ((thumb_tip.x - index_tip.x)**2 + 
                           (thumb_tip.y - index_tip.y)**2)**0.5
        return thumb_index_dist < 0.08

    def _is_thumb_up(self, landmarks):
        """检测大拇指向上"""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        return (thumb_tip.y < thumb_ip.y and 
                (thumb_tip.x < 0.3 or thumb_tip.x > 0.7))

    def _is_thumb_down(self, landmarks):
        """检测大拇指向下"""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        return (thumb_tip.y > thumb_ip.y and 
                (thumb_tip.x < 0.3 or thumb_tip.x > 0.7))

    def _get_palm_center(self, hand_landmarks):
        """获取手掌中心位置"""
        if not hand_landmarks:
            return None
        palm_landmark = hand_landmarks.landmark[9]
        return {
            'x': palm_landmark.x,
            'y': palm_landmark.y,
            'z': palm_landmark.z if hasattr(palm_landmark, 'z') else 0
        }

    def _get_normalized_landmarks(self, hand_landmarks):
        """获取归一化的关键点坐标"""
        landmarks = []
        
        if hasattr(hand_landmarks, 'landmark'):
            for landmark in hand_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z if hasattr(landmark, 'z') else 0
                })
        elif isinstance(hand_landmarks, list):
            landmarks = hand_landmarks
        
        while len(landmarks) < 21:
            landmarks.append({'x': 0, 'y': 0, 'z': 0})
        
        return landmarks[:21]
    
    def get_command(self, gesture):
        """根据手势获取控制指令"""
        return self.gesture_commands.get(gesture, "none")

    def get_dual_hand_commands(self, left_hand_data, right_hand_data):
        """获取双手控制命令"""
        result = {
            'direction_command': None,
            'direction_intensity': 0.5,
            'altitude_command': None,
            'altitude_intensity': 0.5,
            'special_command': None,
            'left_gesture': None,
            'right_gesture': None
        }

        if left_hand_data:
            gesture = left_hand_data.get('gesture', 'none')
            intensity = self.get_gesture_intensity(left_hand_data, gesture)
            result['left_gesture'] = gesture

            if gesture in self.swipe_commands:
                result['direction_command'] = self.swipe_commands[gesture]
                result['direction_intensity'] = self.swipe_intensity
            elif gesture in self.left_hand_commands:
                result['direction_command'] = self.left_hand_commands[gesture]
                result['direction_intensity'] = intensity
            elif gesture in self.both_hands_commands:
                result['special_command'] = self.both_hands_commands[gesture]

        if right_hand_data:
            gesture = right_hand_data.get('gesture', 'none')
            intensity = self.get_gesture_intensity(right_hand_data, gesture)
            result['right_gesture'] = gesture

            if gesture in self.swipe_commands:
                result['direction_command'] = self.swipe_commands[gesture]
                result['direction_intensity'] = self.swipe_intensity
            elif gesture in self.right_hand_commands:
                result['altitude_command'] = self.right_hand_commands[gesture]
                result['altitude_intensity'] = intensity
            elif gesture in self.both_hands_commands:
                result['special_command'] = self.both_hands_commands[gesture]

        return result

    def get_multi_drone_commands(self, left_hand_data, right_hand_data):
        """
        获取多无人机控制命令（双手分离控制）
        左手控制 Drone 1，右手控制 Drone 2

        Args:
            left_hand_data: 左手数据
            right_hand_data: 右手数据

        Returns:
            dict: 包含两个无人机的控制命令
        """
        result = {
            # Drone 1 命令（左手控制）
            'drone1_command': None,
            'drone1_intensity': 0.5,
            'drone1_gesture': None,
            # Drone 2 命令（右手控制）
            'drone2_command': None,
            'drone2_intensity': 0.5,
            'drone2_gesture': None,
            # 全体命令（双手手势触发）
            'all_command': None,
            'formation_command': None,
            # 手势原始数据
            'left_gesture': None,
            'right_gesture': None
        }

        # 处理左手 - 控制 Drone 1（方向）
        if left_hand_data:
            gesture = left_hand_data.get('gesture', 'none')
            intensity = self.get_gesture_intensity(left_hand_data, gesture)
            result['left_gesture'] = gesture
            result['drone1_gesture'] = gesture
            result['drone1_intensity'] = intensity

            # 滑动手势 - Drone 1 侧向移动
            if gesture == 'swipe_left':
                result['drone1_command'] = 'left'
                result['drone1_intensity'] = self.swipe_intensity
            elif gesture == 'swipe_right':
                result['drone1_command'] = 'right'
                result['drone1_intensity'] = self.swipe_intensity
            elif gesture == 'swipe_up':
                result['drone1_command'] = 'forward'
                result['drone1_intensity'] = self.swipe_intensity
            elif gesture == 'swipe_down':
                result['drone1_command'] = 'backward'
                result['drone1_intensity'] = self.swipe_intensity
            # 普通手势 - Drone 1 方向控制
            elif gesture in self.left_hand_commands:
                result['drone1_command'] = self.left_hand_commands[gesture]
            # 双手手势检测
            elif gesture in ['open_palm', 'closed_fist', 'thumb_up', 'thumb_down']:
                if right_hand_data and right_hand_data.get('gesture') == gesture:
                    # 双手同时做相同手势 - 触发全体命令
                    if gesture == 'open_palm':
                        result['all_command'] = 'takeoff'
                    elif gesture == 'closed_fist':
                        result['all_command'] = 'land'
                    elif gesture == 'thumb_up':
                        result['formation_command'] = 'line'
                    elif gesture == 'thumb_down':
                        result['formation_command'] = 'v'

        # 处理右手 - 控制 Drone 2（高度）
        if right_hand_data:
            gesture = right_hand_data.get('gesture', 'none')
            intensity = self.get_gesture_intensity(right_hand_data, gesture)
            result['right_gesture'] = gesture
            result['drone2_intensity'] = intensity

            # 滑动手势 - Drone 2 侧向移动
            if gesture == 'swipe_left':
                result['drone2_command'] = 'left'
                result['drone2_intensity'] = self.swipe_intensity
            elif gesture == 'swipe_right':
                result['drone2_command'] = 'right'
                result['drone2_intensity'] = self.swipe_intensity
            elif gesture == 'swipe_up':
                result['drone2_command'] = 'forward'
                result['drone2_intensity'] = self.swipe_intensity
            elif gesture == 'swipe_down':
                result['drone2_command'] = 'backward'
                result['drone2_intensity'] = self.swipe_intensity
            # 普通手势 - Drone 2 高度控制
            elif gesture in self.right_hand_commands:
                result['drone2_command'] = self.right_hand_commands[gesture]
            # 双手手势已在上方处理

        return result

    def set_multi_drone_mode(self, enabled):
        """启用/禁用多无人机模式"""
        self.multi_drone_mode = enabled
        if enabled:
            print("[手势检测器] 多无人机模式已启用")
            print("  左手 -> 控制 Drone 1 (方向)")
            print("  右手 -> 控制 Drone 2 (高度)")
        else:
            print("[手势检测器] 多无人机模式已禁用")

    def is_multi_drone_mode(self):
        """检查是否启用多无人机模式"""
        return self.multi_drone_mode

    def get_gesture_intensity(self, landmarks, gesture_type):
        """获取手势强度"""
        return 0.5

    def get_hand_position(self, landmarks):
        """获取手部位置"""
        if not landmarks or len(landmarks) < 21:
            return None
        
        x_coords = [p['x'] for p in landmarks if p['x'] > 0]
        y_coords = [p['y'] for p in landmarks if p['y'] > 0]
        
        if not x_coords or not y_coords:
            return None
        
        return {
            'center_x': sum(x_coords) / len(x_coords),
            'center_y': sum(y_coords) / len(y_coords),
            'width': max(x_coords) - min(x_coords) if x_coords else 0,
            'height': max(y_coords) - min(y_coords) if y_coords else 0,
            'bbox': (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        }
    
    def release(self):
        """释放资源"""
        if HAS_MEDIAPIPE and hasattr(self, 'hands'):
            self.hands.close()

    def _detect_swipe_gesture(self, hand_key, frame_width, frame_height, current_time):
        """检测滑动手势"""
        if current_time - self.last_swipe_time < self.swipe_cooldown:
            return None
        
        history = self.palm_history.get(hand_key, [])
        
        if len(history) < 3:
            return None
        
        recent_points = history[-3:]
        
        start_point = recent_points[0]['position']
        end_point = recent_points[-1]['position']
        
        delta_x = end_point[0] - start_point[0]
        delta_y = end_point[1] - start_point[1]
        
        time_delta = recent_points[-1]['timestamp'] - recent_points[0]['timestamp']
        if time_delta <= 0:
            return None
        
        velocity_x = abs(delta_x) / time_delta
        velocity_y = abs(delta_y) / time_delta
        
        swipe_threshold = self.swipe_threshold
        velocity_threshold = self.swipe_min_velocity
        
        direction = None
        gesture_name = None
        
        if abs(delta_x) > swipe_threshold and velocity_x > velocity_threshold:
            if delta_x > 0:
                direction = "swipe_right"
                gesture_name = "swipe_right"
            else:
                direction = "swipe_left"
                gesture_name = "swipe_left"
            intensity = min(abs(delta_x) * 2, 1.0)
            confidence = min(velocity_x / 2.0, 1.0)
            
        elif abs(delta_y) > swipe_threshold and velocity_y > velocity_threshold:
            if delta_y < 0:
                direction = "swipe_up"
                gesture_name = "swipe_up"
            else:
                direction = "swipe_down"
                gesture_name = "swipe_down"
            intensity = min(abs(delta_y) * 2, 1.0)
            confidence = min(velocity_y / 2.0, 1.0)
        
        if direction:
            self.last_swipe_time = current_time
            self.palm_history[hand_key] = []
            
            return {
                'direction': direction,
                'intensity': intensity,
                'gesture_name': gesture_name,
                'confidence': confidence,
                'delta_x': delta_x,
                'delta_y': delta_y,
                'velocity_x': velocity_x,
                'velocity_y': velocity_y
            }
        
        return None
    
    def get_swipe_command(self, swipe_gesture):
        """获取滑动手势对应的控制指令"""
        return self.swipe_commands.get(swipe_gesture, "none")
    
    def get_current_swipe(self):
        """获取当前检测到的滑动手势"""
        return (self.current_swipe, self.swipe_intensity)
    
    def reset_swipe_history(self):
        """重置滑动历史记录"""
        self.palm_history = {'left': [], 'right': []}
        self.last_swipe_time = 0
        self.current_swipe = None
>>>>>>> upstream/main
