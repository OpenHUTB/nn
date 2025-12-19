import cv2
import os
import numpy as np
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
        self.gesture_commands = {
            "open_palm": "takeoff",
            "closed_fist": "land",
            "pointing_up": "up",
            "pointing_down": "down",
            "victory": "forward",
            "thumb_up": "backward",
            "thumb_down": "stop",
            "ok_sign": "hover",
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