import cv2
import os
import numpy as np

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
    """增强版手势检测器（支持机器学习或纯OpenCV）"""

    def __init__(self, ml_model_path=None, use_ml=True):
        self.use_ml = use_ml and HAS_MEDIAPIPE
        self.ml_classifier = None
        
        if HAS_MEDIAPIPE:
            # MediaPipe 模式
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                
                # 使用更轻量的配置，提高帧率
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=0,  # 使用最轻量的模型
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.3
                )
                self.mode = "mediapipe"
                print("[INFO] 使用 MediaPipe 手势检测模式 (轻量配置)")
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
            "hand_detected": "hover"
        }

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
        """使用 MediaPipe 检测"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        gesture = "no_hand"
        confidence = 0.0
        landmarks_data = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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
        """规则分类（基于MediaPipe关键点）"""
        if len(hand_landmarks.landmark) < 21:
            return "no_hand", 0.0
        
        points = hand_landmarks.landmark
        
        # 计算手指伸展情况
        def is_finger_extended(tip_idx, pip_idx):
            """判断手指是否伸展"""
            tip = points[tip_idx]
            pip = points[pip_idx]
            mcp = points[pip_idx - 1]
            return tip.y < pip.y
        
        # 检查各手指
        fingers_extended = {
            'thumb': self._is_thumb_extended(points),
            'index': is_finger_extended(8, 6),
            'middle': is_finger_extended(12, 10),
            'ring': is_finger_extended(16, 14),
            'pinky': is_finger_extended(20, 18)
        }
        
        extended_count = sum(fingers_extended.values())
        
        # 手势分类
        # 1. 张开手掌: 4或5个手指伸展
        if extended_count >= 4:
            return "open_palm", 0.85
        
        # 2. 握拳: 所有手指弯曲
        if extended_count <= 1:
            return "closed_fist", 0.85
        
        # 3. 食指上指: 只有食指伸展
        if (fingers_extended['index'] and 
            not fingers_extended['middle'] and 
            not fingers_extended['ring'] and 
            not fingers_extended['pinky']):
            return "pointing_up", 0.80
        
        # 4. 食指向下: 只有食指伸展
        if (fingers_extended['index'] and 
            not fingers_extended['middle'] and 
            not fingers_extended['ring'] and 
            not fingers_extended['pinky']):
            return "pointing_up", 0.80
        
        # 5. 胜利手势: 食指和中指伸展
        if (fingers_extended['index'] and 
            fingers_extended['middle'] and 
            not fingers_extended['ring'] and 
            not fingers_extended['pinky']):
            return "victory", 0.80
        
        # 6. 大拇指向上/向下
        if fingers_extended['thumb']:
            thumb_tip = points[4]
            wrist = points[0]
            if thumb_tip.y < wrist.y:
                return "thumb_up", 0.80
            else:
                return "thumb_down", 0.80
        
        # 7. OK手势: 食指和拇指接近
        thumb_tip = points[4]
        index_tip = points[8]
        distance = ((thumb_tip.x - index_tip.x)**2 + 
                   (thumb_tip.y - index_tip.y)**2)**0.5
        if distance < 0.15 and not fingers_extended['index']:
            return "ok_sign", 0.80
        
        return "hand_detected", 0.5
    
    def _is_thumb_extended(self, points):
        """判断拇指是否伸展"""
        thumb_tip = points[4]
        thumb_mcp = points[2]
        wrist = points[0]
        
        # 检查拇指是否在手掌外侧
        if thumb_tip.x < thumb_mcp.x:
            return True
        return False

    def _smooth_prediction(self, gesture, confidence):
        """平滑预测"""
        self.prediction_history.append((gesture, confidence))
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)

    def get_command(self, gesture):
        """获取控制指令"""
        return self.gesture_commands.get(gesture, "none")

    def get_gesture_intensity(self, landmarks, gesture_type):
        """获取手势强度"""
        return 0.5

    def release(self):
        """释放资源"""
        if hasattr(self, 'hands'):
            self.hands.close()
