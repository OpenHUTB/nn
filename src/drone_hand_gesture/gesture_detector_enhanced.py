import cv2
import os
import numpy as np
<<<<<<< HEAD
import mediapipe as mp
from gesture_classifier import GestureClassifier, EnsembleGestureClassifier
import PIL

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️  PIL未安装，中文显示不可用")

class EnhancedGestureDetector:
    """手势检测器（集成机器学习）"""

    def __init__(self, ml_model_path=None, use_ml=True):
        # 基础检测器
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 机器学习模型
        self.use_ml = use_ml
        self.ml_classifier = None

        if use_ml and ml_model_path:
            # 检查是否是 ensemble 信息文件（很小的文件）
            if os.path.exists(ml_model_path):
                file_size = os.path.getsize(ml_model_path)
                if file_size < 1024:  # 小于1KB，可能是信息文件
                    print(f"⚠️  检测到小文件 ({file_size}字节)，可能是集成模型信息文件")
                    print(f"   尝试加载单个模型文件...")

                    # 尝试加载对应的单个模型
                    base_name = os.path.basename(ml_model_path)
                    if "ensemble" in base_name:
                        # 尝试加载 svm 模型
                        svm_path = ml_model_path.replace("ensemble", "svm")
                        if os.path.exists(svm_path):
                            print(f"   尝试加载: {svm_path}")
                            success = self.load_ml_model(svm_path)
                        else:
                            print(f"   未找到替代模型，使用规则检测")
                            self.use_ml = False
                else:
                    # 正常加载
                    success = self.load_ml_model(ml_model_path)
        else:
            self.use_ml = False

        # 手势到控制指令的映射（8个手势）
=======
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
>>>>>>> upstream/main
        self.gesture_commands = {
            "open_palm": "takeoff",
            "closed_fist": "land",
            "pointing_up": "up",
            "pointing_down": "down",
            "victory": "forward",
            "thumb_up": "backward",
            "thumb_down": "stop",
            "ok_sign": "hover",
<<<<<<< HEAD
        }

        # 中文手势名称映射
        self.chinese_gesture_names = {
            "open_palm": "张开手掌",
            "closed_fist": "握拳",
            "victory": "胜利手势",
            "thumb_up": "大拇指",
            "thumb_down": "大拇指向下",
            "pointing_up": "食指上指",
            "pointing_down": "食指向下",
            "ok_sign": "OK手势",
            "none": "未检测到手"
        }

        # 中文指令名称映射
        self.chinese_command_names = {
            "takeoff": "起飞",
            "land": "降落",
            "up": "上升",
            "down": "下降",
            "forward": "前进",
            "backward": "后退",
            "stop": "停止",
            "hover": "悬停",
            "none": "无指令"
        }

        # 中文字体
        self.chinese_font = None
        if HAS_PIL:
            try:
                # Windows字体路径
                font_paths = [
                    "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
                    "C:/Windows/Fonts/simhei.ttf",  # 黑体
                    "C:/Windows/Fonts/simsun.ttc",  # 宋体
                ]

                for font_path in font_paths:
                    try:
                        if os.path.exists(font_path):
                            self.chinese_font = ImageFont.truetype(font_path, 30)
                            print(f"✅ 加载中文字体: {os.path.basename(font_path)}")
                            break
                    except:
                        continue

                if self.chinese_font is None:
                    print("⚠️  无法加载系统字体，将使用英文显示")
            except Exception as e:
                print(f"⚠️  字体初始化失败: {e}")


        # 历史记录（用于平滑预测）
        self.prediction_history = []
        self.max_history = 5

    def load_ml_model(self, model_path):
        """加载机器学习模型"""
        try:
            self.ml_classifier = GestureClassifier(model_path=model_path)
            print(f"机器学习模型已加载: {model_path}")
            return True
        except Exception as e:
            print(f"加载机器学习模型失败: {e}")
            print("将使用规则检测器")
            self.use_ml = False
            return False

    def extract_landmarks_for_ml(self, hand_landmarks):
        """提取关键点用于机器学习"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        # 确保是63维
        if len(landmarks) < 63:
            landmarks.extend([0.0] * (63 - len(landmarks)))
        elif len(landmarks) > 63:
            landmarks = landmarks[:63]

        return landmarks

    def detect_gestures(self, image, simulation_mode=False):
        """检测手势（支持中文显示）"""
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        gesture = "none"
        confidence = 0.0
        landmarks_data = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点（使用OpenCV绘制）
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # 提取关键点用于机器学习
                landmarks = self.extract_landmarks_for_ml(hand_landmarks)

                # 手势识别逻辑
                if self.use_ml and self.ml_classifier and len(landmarks) == 63:
                    # 机器学习预测
                    gesture, confidence = self.ml_classifier.predict(landmarks)

                    # 应用平滑滤波
                    self.prediction_history.append((gesture, confidence))
                    if len(self.prediction_history) > self.max_history:
                        self.prediction_history.pop(0)

                    # 使用历史中的多数投票
                    if len(self.prediction_history) >= 3:
                        from collections import Counter
                        recent_gestures = [g for g, _ in self.prediction_history[-3:]]
                        most_common = Counter(recent_gestures).most_common(1)[0]
                        if most_common[1] >= 2:  # 至少2/3一致
                            gesture = most_common[0]
                            # 计算平均置信度
                            matching_confs = [c for g, c in self.prediction_history if g == gesture]
                            if matching_confs:
                                confidence = np.mean(matching_confs)
                else:
                    # 备用：规则检测
                    gesture, confidence = self._classify_by_rules(hand_landmarks)

                # 使用PIL绘制中文文本
                try:
                    # 将OpenCV图像转换为PIL图像
                    image_rgb_for_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image_rgb_for_pil)
                    draw = ImageDraw.Draw(image_pil)

                    # 设置字体
                    if self.chinese_font:
                        font = self.chinese_font
                    else:
                        # 如果中文字体不可用，使用默认字体
                        font = ImageFont.load_default()
                        print("⚠️ 使用默认字体，中文可能显示为方块")

                    # 绘制手势信息
                    chinese_gesture = self.chinese_gesture_names.get(gesture, gesture)
                    ml_info = " (训练模型)" if self.use_ml else " (规则)"
                    text = f"手势: {chinese_gesture}{ml_info}"
                    draw.text((10, 30), text, fill=(0, 255, 0), font=font)

                    # 绘制置信度
                    draw.text((10, 70), f"置信度: {confidence:.2f}", fill=(0, 255, 0), font=font)

                    # 绘制指令
                    command = self.gesture_commands.get(gesture, "none")
                    chinese_command = self.chinese_command_names.get(command, command)
                    draw.text((10, 110), f"指令: {chinese_command}", fill=(255, 0, 0), font=font)

                    # 将PIL图像转换回OpenCV格式
                    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                except Exception as e:
                    print(f"PIL绘制失败，回退到OpenCV英文显示: {e}")
                    # 回退到英文显示
                    ml_info = " (ML)" if self.use_ml else " (Rule)"
                    cv2.putText(image, f"Gesture: {gesture}{ml_info}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Command: {command}", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        else:
            # 没有检测到手部时也使用PIL绘制
            try:
                image_rgb_for_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb_for_pil)
                draw = ImageDraw.Draw(image_pil)

                if self.chinese_font:
                    font = self.chinese_font
                else:
                    font = ImageFont.load_default()

                draw.text((10, 30), "未检测到手部", fill=(0, 0, 255), font=font)

                # 转换回OpenCV格式
                image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            except Exception as e:
                print(f"PIL绘制失败: {e}")
                cv2.putText(image, "No Hand Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return image, gesture, confidence, landmarks_data

    def _classify_by_rules(self, hand_landmarks):
        """简单的规则分类器（备用）"""
        # 获取关键点
        landmarks = hand_landmarks.landmark

        # 这里实现简单的规则检测逻辑
        # 例如：计算手指是否伸直等

        # 暂时返回一个默认值
        return "open_palm", 0.5

    def get_command(self, gesture):
        """获取控制指令"""
        return self.gesture_commands.get(gesture, "none")

    def release(self):
        """释放资源"""
        self.hands.close()
=======
            "hand_detected": "hover",
            # 滑动手势
            "swipe_left": "left",
            "swipe_right": "right",
            "swipe_up": "forward",
            "swipe_down": "backward",
            # 紧急返航手势
            "rock": "return_home",        # 摇滚手势 (食指+小指) -> 一键返航
            "peace": "return_home",       # 和平手势 (食指+中指) -> 一键返航（备用）
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

    # ============ 滑动手势检测方法 ============
    
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
        
        direction = None
        gesture_name = None
        
        if abs(delta_x) > self.swipe_threshold and velocity_x > self.swipe_min_velocity:
            if delta_x > 0:
                direction = "swipe_right"
                gesture_name = "swipe_right"
            else:
                direction = "swipe_left"
                gesture_name = "swipe_left"
            intensity = min(abs(delta_x) * 2, 1.0)
            confidence = min(velocity_x / 2.0, 1.0)
            
        elif abs(delta_y) > self.swipe_threshold and velocity_y > self.swipe_min_velocity:
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
            }
        
        return None
    
    def get_swipe_command(self, swipe_gesture):
        """获取滑动手势对应的控制指令"""
        return self.swipe_commands.get(swipe_gesture, "none")
    
    def get_current_swipe(self):
        """获取当前检测到的滑动手势"""
        return (self.current_swipe, self.swipe_intensity)
    
    def get_dual_hand_commands(self, left_hand_data, right_hand_data):
        """获取双手控制命令（支持滑动手势）"""
        result = {
            'direction_command': None,
            'direction_intensity': 0.5,
            'altitude_command': None,
            'altitude_intensity': 0.5,
            'special_command': None,
            'left_gesture': None,
            'right_gesture': None
        }

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

    # ============ 滑动手势检测方法 ============
    
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
        
        direction = None
        gesture_name = None
        
        if abs(delta_x) > self.swipe_threshold and velocity_x > self.swipe_min_velocity:
            if delta_x > 0:
                direction = "swipe_right"
                gesture_name = "swipe_right"
            else:
                direction = "swipe_left"
                gesture_name = "swipe_left"
            intensity = min(abs(delta_x) * 2, 1.0)
            confidence = min(velocity_x / 2.0, 1.0)
            
        elif abs(delta_y) > self.swipe_threshold and velocity_y > self.swipe_min_velocity:
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
            }
        
        return None
    
    def get_swipe_command(self, swipe_gesture):
        """获取滑动手势对应的控制指令"""
        return self.swipe_commands.get(swipe_gesture, "none")
    
    def get_current_swipe(self):
        """获取当前检测到的滑动手势"""
        return (self.current_swipe, self.swipe_intensity)
    
    def get_dual_hand_commands(self, left_hand_data, right_hand_data):
        """获取双手控制命令（支持滑动手势）"""
        result = {
            'direction_command': None,
            'direction_intensity': 0.5,
            'altitude_command': None,
            'altitude_intensity': 0.5,
            'special_command': None,
            'left_gesture': None,
            'right_gesture': None
        }

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
>>>>>>> upstream/main
