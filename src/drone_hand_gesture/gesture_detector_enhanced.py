import cv2
import os
import numpy as np
import time

# 尝试导入 MediaPipe（可选）
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    mp = None

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARNING] PIL未安装，中文显示不可用")


class EnhancedGestureDetector:
    """增强版手势检测器（支持机器学习或纯OpenCV + 滑动手势）"""

    def __init__(self, ml_model_path=None, use_ml=True):
        self.use_ml = use_ml and HAS_MEDIAPIPE
        self.ml_classifier = None
        
        if HAS_MEDIAPIPE:
            # MediaPipe 模式
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.mode = "mediapipe"
                print("[INFO] 使用 MediaPipe 手势检测模式")
            except Exception as e:
                print(f"[WARNING] MediaPipe 初始化失败: {e}")
                self.mode = "opencv"
                self.use_ml = False
                self._init_opencv_detector()
        else:
            # OpenCV 模式
            self.mode = "opencv"
            self.use_ml = False
            self._init_opencv_detector()
            print("[INFO] 使用纯 OpenCV 手势检测模式")

        # 加载机器学习模型
        if self.use_ml and ml_model_path and os.path.exists(ml_model_path):
            try:
                from gesture_classifier import GestureClassifier
                self.ml_classifier = GestureClassifier(model_path=ml_model_path)
                print(f"[INFO] 机器学习模型已加载: {ml_model_path}")
            except Exception as e:
                print(f"[WARNING] 加载ML模型失败: {e}")
                self.use_ml = False

        # 手势命令映射
        self.gesture_commands = {
            "open_palm": "takeoff",
            "closed_fist": "land",
            "pointing_up": "up",
            "pointing_down": "down",
            "victory": "forward",
            "thumb_up": "backward",
            "thumb_down": "stop",
            "ok_sign": "hover",
            "hand_detected": "hover",
            # 滑动手势
            "swipe_left": "left",
            "swipe_right": "right",
            "swipe_up": "forward",
            "swipe_down": "backward",
        }

        # 滑动手势相关
        self.palm_history = {'left': [], 'right': []}
        self.max_history_length = 10
        self.swipe_commands = {
            "swipe_left": "left",
            "swipe_right": "right",
            "swipe_up": "forward",
            "swipe_down": "backward",
        }
        self.swipe_threshold = 0.15
        self.swipe_min_velocity = 0.3
        self.swipe_cooldown = 0.5
        self.last_swipe_time = 0
        self.current_swipe = None
        self.swipe_intensity = 0.5

        # 历史记录
        self.prediction_history = []
        self.max_history = 5
        
        # 中文字体
        self.chinese_font = None
        if HAS_PIL:
            self._init_chinese_font()

    def _init_opencv_detector(self):
        """初始化 OpenCV 检测器"""
        self.skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        
        # 加载 Haar Cascade 作为备选
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def _init_chinese_font(self):
        """初始化中文字体"""
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    self.chinese_font = ImageFont.truetype(font_path, 30)
                    print(f"[OK] 加载中文字体: {os.path.basename(font_path)}")
                    break
                except:
                    continue

    def detect_gestures(self, image, simulation_mode=False):
        """检测手势"""
        if self.mode == "mediapipe":
            return self._detect_with_mediapipe(image, simulation_mode)
        else:
            return self._detect_with_opencv(image, simulation_mode)

    def _detect_with_mediapipe(self, image, simulation_mode):
        """使用 MediaPipe 检测（支持滑动手势）"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        gesture = "no_hand"
        confidence = 0.0
        landmarks_data = None
        height, width = image.shape[:2]
        current_time = time.time()

        # 重置滑动手势
        self.current_swipe = None
        self.swipe_intensity = 0.5

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # 获取手类型
                hand_type = handedness.classification[0].label  # "Left" 或 "Right"
                
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                landmarks = self._extract_landmarks(hand_landmarks)
                
                if self.use_ml and self.ml_classifier:
                    gesture, confidence = self.ml_classifier.predict(landmarks)
                    self._smooth_prediction(gesture, confidence)
                else:
                    gesture, confidence = self._classify_by_rules(hand_landmarks)
                
                # 获取手掌中心位置用于滑动检测
                palm_position = self._get_palm_center(hand_landmarks)
                
                # 检测滑动手势
                if palm_position:
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
                        gesture = swipe_result['gesture_name']
                        confidence = swipe_result['confidence']
                
                landmarks_data = landmarks

        return image, gesture, confidence, landmarks_data

    def _detect_with_opencv(self, image, simulation_mode):
        """使用纯 OpenCV 检测"""
        result_image = image.copy()
        height, width = image.shape[:2]
        
        # 肤色检测
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        # 去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        
        # 找轮廓
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        gesture = "no_hand"
        confidence = 0.0
        landmarks_data = None
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(max_contour)
            min_area = (width * height) * 0.01
            
            if contour_area > min_area:
                cv2.drawContours(result_image, [max_contour], -1, (0, 255, 0), 2)
                gesture, confidence = self._analyze_hand_opencv(max_contour)
                
                if simulation_mode:
                    landmarks_data = self._generate_landmarks(max_contour)
        
        return result_image, gesture, confidence, landmarks_data

    def _analyze_hand_opencv(self, contour):
        """分析手型（OpenCV模式）"""
        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
        except:
            defects = None
        
        finger_count = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, d, _ = defects[i, 0]
                far = tuple(contour[d][0])
                if abs(far[1]) > 20:
                    finger_count += 1
            finger_count = max(0, finger_count // 2)
        
        if finger_count == 0:
            return "closed_fist", 0.85
        elif finger_count == 1:
            return "pointing_up", 0.80
        elif finger_count == 2:
            return "victory", 0.80
        elif finger_count >= 4:
            return "open_palm", 0.75
        
        return "hand_detected", 0.5

    def _generate_landmarks(self, contour):
        """生成简化关键点"""
        x, y, w, h = cv2.boundingRect(contour)
        landmarks = []
        
        for i in range(5):
            landmarks.extend([
                (x + w * (0.2 + i * 0.15)) / 640,
                (y) / 480,
                0
            ])
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        for i in range(5):
            landmarks.extend([
                (x + w * (0.2 + i * 0.15)) / 640,
                (y + h * 0.3) / 480,
                0
            ])
        
        while len(landmarks) < 63:
            landmarks.extend([0.0])
        
        return landmarks[:63]

    def _extract_landmarks(self, hand_landmarks):
        """提取关键点"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        if len(landmarks) < 63:
            landmarks.extend([0.0] * (63 - len(landmarks)))
        return landmarks[:63]

    def _classify_by_rules(self, hand_landmarks):
        """规则分类"""
        return "open_palm", 0.5

        left_commands = {
            "victory": "forward",
            "thumb_up": "backward",
            "pointing_up": "left",
            "pointing_down": "right",
        }

        right_commands = {
            "pointing_up": "up",
            "pointing_down": "down",
            "ok_sign": "hover",
        }

        both_commands = {
            "open_palm": "takeoff",
            "closed_fist": "land",
            "thumb_down": "stop",
        }

        if left_hand_data:
            gesture = left_hand_data.get('gesture', 'none')
            intensity = self.get_gesture_intensity(left_hand_data, gesture)
            result['left_gesture'] = gesture

            if gesture in self.swipe_commands:
                result['direction_command'] = self.swipe_commands[gesture]
                result['direction_intensity'] = self.swipe_intensity
            elif gesture in left_commands:
                result['direction_command'] = left_commands[gesture]
                result['direction_intensity'] = intensity
            elif gesture in both_commands:
                result['special_command'] = both_commands[gesture]

        if right_hand_data:
            gesture = right_hand_data.get('gesture', 'none')
            intensity = self.get_gesture_intensity(right_hand_data, gesture)
            result['right_gesture'] = gesture

            if gesture in self.swipe_commands:
                result['direction_command'] = self.swipe_commands[gesture]
                result['direction_intensity'] = self.swipe_intensity
            elif gesture in right_commands:
                result['altitude_command'] = right_commands[gesture]
                result['altitude_intensity'] = intensity
            elif gesture in both_commands:
                result['special_command'] = both_commands[gesture]

        return result

    def release(self):
        """释放资源"""
        if hasattr(self, 'hands'):
            self.hands.close()
