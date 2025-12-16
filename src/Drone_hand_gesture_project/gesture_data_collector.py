import cv2
import numpy as np
import os
import time
import json
from datetime import datetime
import mediapipe as mp
import pickle
from PIL import Image, ImageDraw, ImageFont
import sys


class GestureDataCollector:
    """手势数据收集器"""

    def __init__(self, data_dir="dataset"):
        self.data_dir = data_dir
        self.create_directories()

        # 初始化MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 定义手势类别
        self.gesture_classes = [
            "张开手掌",  # open_palm
            "握拳",  # closed_fist
            "胜利手势",  # victory
            "大拇指向上",  # thumb_up
            "大拇指向下",  # thumb_down
            "食指上指",  # pointing_up
            "食指向下",  # pointing_down
            "OK手势",  # ok_sign
            "无手"  # no_hand
        ]

        # 手势类别的英文标识（用于保存文件）
        self.gesture_english_names = [
            "open_palm",
            "closed_fist",
            "victory",
            "thumb_up",
            "thumb_down",
            "pointing_up",
            "pointing_down",
            "ok_sign",
            "no_hand"
        ]

        # 数据存储
        self.collected_data = []
        self.current_gesture = None
        self.collection_active = False

        # 尝试加载中文字体
        self.font = None
        try:
            # Windows系统字体路径
            if sys.platform == 'win32':
                font_path = 'C:/Windows/Fonts/simhei.ttf'  # 黑体
            # Linux系统字体路径
            elif sys.platform == 'linux':
                font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # 文泉驿微米黑
            # macOS系统字体路径
            elif sys.platform == 'darwin':
                font_path = '/System/Library/Fonts/PingFang.ttc'  # 苹方字体
            else:
                font_path = None

            if font_path and os.path.exists(font_path):
                self.font = ImageFont.truetype(font_path, 30)
                print(f"已加载中文字体: {font_path}")
            else:
                print("未找到中文字体，将使用默认字体")
                self.font = ImageFont.load_default()
        except Exception as e:
            print(f"加载字体失败: {e}")
            self.font = ImageFont.load_default()

    def create_directories(self):
        """创建必要的目录"""
        directories = [
            os.path.join(self.data_dir, "raw"),
            os.path.join(self.data_dir, "processed"),
            os.path.join(self.data_dir, "models"),
            os.path.join(self.data_dir, "raw", "images"),
            os.path.join(self.data_dir, "raw", "landmarks")
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def extract_landmarks(self, image):
        """从图像中提取手部关键点"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        landmarks_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_data.append(landmarks)

        return landmarks_data, results.multi_hand_landmarks

    def collect_sample(self, gesture_class, image, landmarks):
        """收集一个样本"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # 获取对应的英文手势名称用于保存文件
        if gesture_class in self.gesture_classes:
            english_name = self.gesture_english_names[self.gesture_classes.index(gesture_class)]
        else:
            english_name = gesture_class

        # 保存图像
        img_filename = f"{english_name}_{timestamp}.jpg"
        img_path = os.path.join(self.data_dir, "raw", "images", img_filename)
        cv2.imwrite(img_path, image)

        # 保存关键点数据
        data_entry = {
            'timestamp': timestamp,
            'gesture_class': gesture_class,
            'gesture_english': english_name,
            'image_path': img_path,
            'landmarks': landmarks[0] if landmarks else [],  # 取第一个手
            'num_landmarks': 21 if landmarks else 0,
            'has_hand': len(landmarks) > 0
        }

        self.collected_data.append(data_entry)
        return data_entry

    def start_collection(self, gesture_class):
        """开始收集指定手势的数据"""
        self.current_gesture = gesture_class
        self.collection_active = True
        print(f"开始收集手势: {gesture_class}")
        print("按 'c' 收集样本，按 's' 停止")

    def save_dataset(self, filename="gesture_dataset.pkl"):
        """保存收集的数据集"""
        dataset_path = os.path.join(self.data_dir, "processed", filename)

        # 转换为特征和标签
        X = []
        y = []

        for data in self.collected_data:
            if data['has_hand'] and len(data['landmarks']) == 63:  # 21个点 * 3坐标
                X.append(data['landmarks'])
                y.append(self.gesture_classes.index(data['gesture_class']))

        dataset = {
            'features': np.array(X),
            'labels': np.array(y),
            'gesture_classes': self.gesture_classes,
            'gesture_english_names': self.gesture_english_names,
            'num_samples': len(X)
        }

        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"数据集已保存到 {dataset_path}")
        print(f"总样本数: {len(X)}")
        print(f"各类样本数:")
        for i, class_name in enumerate(self.gesture_classes):
            count = np.sum(y == i)
            print(f"  {class_name}: {count}")

        return dataset

    def put_chinese_text(self, img, text, position, font_color=(255, 255, 255)):
        """在图像上绘制中文文本"""
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 绘制文本
        draw.text(position, text, font=self.font, fill=font_color)

        # 将PIL图像转换回OpenCV图像
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def run_collection_app(self):
        """运行数据收集应用程序"""
        cap = cv2.VideoCapture(1)
        collecting_for = None
        sample_count = 0

        print("手势数据收集工具")
        print("=" * 50)
        print("命令:")
        print("  1-9: 选择手势类别")
        print("  c: 收集当前样本")
        print("  s: 停止收集当前手势")
        print("  w: 保存数据集")
        print("  q: 退出")
        print("=" * 50)
        print("手势类别:")
        for i, gesture in enumerate(self.gesture_classes):
            print(f"  {i + 1}: {gesture}")
        print("=" * 50)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break

            frame = cv2.flip(frame, 1)

            # 创建显示帧的副本用于绘制
            display_frame = frame.copy()

            # 提取关键点
            landmarks, hand_landmarks = self.extract_landmarks(frame)

            # 绘制手部关键点
            if hand_landmarks:
                for hand_landmarks in hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # 在图像上添加中文文本
            try:
                # 第一行：当前手势
                if collecting_for:
                    status_text = f"当前手势: {collecting_for}"
                else:
                    status_text = "当前手势: 无"
                display_frame = self.put_chinese_text(display_frame, status_text, (10, 30), (0, 255, 0))

                # 第二行：当前手势收集样本数
                display_frame = self.put_chinese_text(display_frame, f"当前收集样本数: {sample_count}",
                                                      (10, 70), (0, 255, 0))

                # 第三行：总样本数
                display_frame = self.put_chinese_text(display_frame, f"总样本数: {len(self.collected_data)}",
                                                      (10, 110), (0, 255, 0))

                # 第四行：操作提示
                display_frame = self.put_chinese_text(display_frame, "按1-9选择手势，c收集样本", (10, 150),
                                                      (255, 255, 255))

                # 第五行：操作提示
                display_frame = self.put_chinese_text(display_frame, "按w保存，q退出", (10, 190),
                                                      (255, 255, 255))

            except Exception as e:
                print(f"绘制文字时出错: {e}")
                # 如果无法绘制中文，使用英文
                status_text = f"Gesture: {collecting_for if collecting_for else 'None'}"
                cv2.putText(display_frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('手势数据收集', display_frame)

            key = cv2.waitKey(1) & 0xFF

            # 处理按键
            if key == ord('q'):
                break
            elif ord('1') <= key <= ord('9'):
                gesture_idx = key - ord('1')
                if gesture_idx < len(self.gesture_classes):
                    collecting_for = self.gesture_classes[gesture_idx]
                    sample_count = 0
                    print(f"开始收集手势: {collecting_for}")
            elif key == ord('c') and collecting_for:
                if landmarks:
                    self.collect_sample(collecting_for, frame, landmarks)
                    sample_count += 1
                    print(f"已收集 {collecting_for} 样本 #{sample_count}")
                else:
                    print("未检测到手部！")
            elif key == ord('s'):
                collecting_for = None
                sample_count = 0
                print("停止收集")
            elif key == ord('w'):
                if self.collected_data:
                    dataset = self.save_dataset()
                    print(f"数据集已保存，总样本数: {dataset['num_samples']}")
                else:
                    print("没有数据可保存")

        cap.release()
        cv2.destroyAllWindows()

        # 保存最终数据集
        if self.collected_data:
            self.save_dataset()


if __name__ == "__main__":
    collector = GestureDataCollector()
    collector.run_collection_app()