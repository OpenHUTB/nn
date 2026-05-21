import cv2
import os
import numpy as np

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

class DeepGestureDetector:
    """深度学习手势检测器"""
    
    def __init__(self, model_path=None, model_type='deep_cnn', use_deep_learning=True):
        self.use_deep_learning = use_deep_learning and HAS_MEDIAPIPE
        self.model_type = model_type
        self.deep_classifier = None
        
        if HAS_MEDIAPIPE:
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.3
                )
                self.mode = "mediapipe"
                print("[INFO] 使用 MediaPipe 手势检测模式")
            except Exception as e:
                print(f"[WARNING] MediaPipe 初始化失败: {e}")
                self.mode = "opencv"
                self.use_deep_learning = False
                self._init_opencv_detector()
        else:
            self.mode = "opencv"
            self.use_deep_learning = False
            self._init_opencv_detector()
            print("[INFO] 使用纯 OpenCV 手势检测模式")
        
        if self.use_deep_learning and model_path and os.path.exists(model_path):
            try:
                from deep_gesture_classifier import DeepGestureClassifier
                model_type = 'cnn' if model_path.endswith('.pth') else model_type
                self.deep_classifier = DeepGestureClassifier(model_type=model_type, model_path=model_path)
                print(f"[INFO] 深度学习模型已加载: {model_path}")
            except Exception as e:
                print(f"[WARNING] 加载深度学习模型失败: {e}")
                self.use_deep_learning = False
        
        self.gesture_commands = {
            "open_palm": "takeoff",
            "closed_fist": "land",
            "pointing_up": "up",
            "pointing_down": "down",
            "victory": "forward",
            "thumb_up": "backward",
            "thumb_down": "stop",
            "ok_sign": "hover",
            "rock": "left",
            "peace": "right",
            "hand_detected": "hover"
        }
        
        self.prediction_history = []
        self.max_history = 5
        self.chinese_font = None
        if HAS_PIL:
            self._init_chinese_font()
        
        # 手势颜色映射
        self.gesture_colors = {
            "open_palm": (0, 255, 0),      # 绿色
            "closed_fist": (0, 0, 255),    # 红色
            "pointing_up": (255, 0, 0),    # 蓝色
            "pointing_down": (255, 165, 0),# 橙色
            "victory": (255, 0, 255),      # 紫色
            "thumb_up": (0, 255, 255),     # 青色
            "thumb_down": (128, 0, 128),   # 深紫
            "ok_sign": (255, 255, 0),      # 黄色
            "rock": (192, 192, 192),       # 银色
            "peace": (255, 192, 203),      # 粉色
            "hand_detected": (128, 128, 128),# 灰色
            "no_hand": (0, 0, 0)           # 黑色
        }
    
    def _init_opencv_detector(self):
        self.skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def _init_chinese_font(self):
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    self.chinese_font = ImageFont.truetype(font_path, 30)
                    break
                except:
                    continue
    
    def detect_gestures(self, image, draw_landmarks=True):
        if self.mode == "mediapipe":
            return self._detect_with_mediapipe(image, draw_landmarks)
        else:
            return self._detect_with_opencv(image)
    
    def _detect_with_mediapipe(self, image, draw_landmarks):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        gestures = []
        all_landmarks = []
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if draw_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                landmarks = self._extract_landmarks(hand_landmarks)
                
                if self.use_deep_learning and self.deep_classifier:
                    gesture, confidence = self.deep_classifier.predict(landmarks)
                else:
                    gesture, confidence = self._classify_by_rules(hand_landmarks)
                
                self._smooth_prediction(gesture, confidence)
                
                gestures.append({
                    'gesture': gesture,
                    'confidence': confidence,
                    'hand_idx': hand_idx,
                    'landmarks': landmarks
                })
                all_landmarks.extend(landmarks)
        
        # 获取最终预测（基于历史平滑）
        final_gesture = self._get_smoothed_gesture()
        
        return image, gestures, final_gesture
    
    def _detect_with_opencv(self, image):
        result_image = image.copy()
        height, width = image.shape[:2]
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        gestures = []
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(max_contour)
            min_area = (width * height) * 0.01
            
            if contour_area > min_area:
                cv2.drawContours(result_image, [max_contour], -1, (0, 255, 0), 2)
                gesture, confidence = self._analyze_hand_opencv(max_contour)
                
                gestures.append({
                    'gesture': gesture,
                    'confidence': confidence,
                    'hand_idx': 0,
                    'landmarks': None
                })
        
        final_gesture = gestures[0]['gesture'] if gestures else 'no_hand'
        
        return result_image, gestures, final_gesture
    
    def _analyze_hand_opencv(self, contour):
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
    
    def _extract_landmarks(self, hand_landmarks):
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        if len(landmarks) < 63:
            landmarks.extend([0.0] * (63 - len(landmarks)))
        return landmarks[:63]
    
    def _classify_by_rules(self, hand_landmarks):
        if len(hand_landmarks.landmark) < 21:
            return "no_hand", 0.0
        
        points = hand_landmarks.landmark
        
        def is_finger_extended(tip_idx, pip_idx):
            tip = points[tip_idx]
            pip = points[pip_idx]
            return tip.y < pip.y
        
        fingers_extended = {
            'thumb': self._is_thumb_extended(points),
            'index': is_finger_extended(8, 6),
            'middle': is_finger_extended(12, 10),
            'ring': is_finger_extended(16, 14),
            'pinky': is_finger_extended(20, 18)
        }
        
        extended_count = sum(fingers_extended.values())
        
        if extended_count >= 4:
            return "open_palm", 0.85
        if extended_count <= 1:
            return "closed_fist", 0.85
        if (fingers_extended['index'] and not fingers_extended['middle'] and 
            not fingers_extended['ring'] and not fingers_extended['pinky']):
            return "pointing_up", 0.80
        if (fingers_extended['index'] and fingers_extended['middle'] and 
            not fingers_extended['ring'] and not fingers_extended['pinky']):
            return "victory", 0.80
        if fingers_extended['thumb']:
            thumb_tip = points[4]
            wrist = points[0]
            if thumb_tip.y < wrist.y:
                return "thumb_up", 0.80
            else:
                return "thumb_down", 0.80
        
        thumb_tip = points[4]
        index_tip = points[8]
        distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
        if distance < 0.15 and not fingers_extended['index']:
            return "ok_sign", 0.80
        
        return "hand_detected", 0.5
    
    def _is_thumb_extended(self, points):
        thumb_tip = points[4]
        thumb_mcp = points[2]
        return thumb_tip.x < thumb_mcp.x
    
    def _smooth_prediction(self, gesture, confidence):
        self.prediction_history.append((gesture, confidence))
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)
    
    def _get_smoothed_gesture(self):
        if not self.prediction_history:
            return 'no_hand'
        
        gesture_counts = {}
        for gesture, confidence in self.prediction_history:
            if gesture not in gesture_counts:
                gesture_counts[gesture] = 0
            gesture_counts[gesture] += confidence
        
        if gesture_counts:
            return max(gesture_counts.items(), key=lambda x: x[1])[0]
        return 'no_hand'
    
    def get_command(self, gesture):
        return self.gesture_commands.get(gesture, "none")
    
    def draw_gesture_info(self, image, gestures):
        """在图像上绘制手势信息"""
        for i, gesture_info in enumerate(gestures):
            gesture = gesture_info['gesture']
            confidence = gesture_info['confidence']
            color = self.gesture_colors.get(gesture, (128, 128, 128))
            
            text = f"Hand {i}: {gesture} ({confidence:.2f})"
            cv2.putText(image, text, (10, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        return image
    
    def release(self):
        if hasattr(self, 'hands'):
            self.hands.close()