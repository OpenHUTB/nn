import cv2
import numpy as np
import os
import time
import json
from datetime import datetime
import mediapipe as mp
import pickle
from PIL import Image, ImageDraw, ImageFont

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

        # 定义手势类别（8个手势，去掉no_hand）
        self.gesture_classes = [
            "open_palm",  # 张开手掌
            "closed_fist",  # 握拳
            "victory",  # 胜利手势
            "thumb_up",  # 大拇指
            "thumb_down",  # 大拇指向下
            "pointing_up",  # 食指上指
            "pointing_down",  # 食指向下
            "ok_sign",  # OK手势
        ]

        # 数据存储
        self.collected_data = []
        self.current_gesture = None
        self.collection_active = False

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

        # 保存图像
        img_filename = f"{gesture_class}_{timestamp}.jpg"
        img_path = os.path.join(self.data_dir, "raw", "images", img_filename)
        cv2.imwrite(img_path, image)

        # 只保存有手部关键点的数据
        if landmarks and len(landmarks) > 0:
            landmarks_data = landmarks[0]  # 取第一个手
            # 确保是63维
            if len(landmarks_data) < 63:
                landmarks_data.extend([0.0] * (63 - len(landmarks_data)))
            elif len(landmarks_data) > 63:
                landmarks_data = landmarks_data[:63]
        else:
            # 如果没有检测到手，跳过这个样本（不收集无手数据）
            print(f"警告: {gesture_class} 未检测到手部，跳过此样本")
            return None

        data_entry = {
            'timestamp': timestamp,
            'gesture_class': gesture_class,
            'image_path': img_path,
            'landmarks': landmarks_data,
            'num_landmarks': 21,
            'has_hand': True
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
            if data['has_hand'] and len(data['landmarks']) == 63:
                X.append(data['landmarks'])
                y.append(self.gesture_classes.index(data['gesture_class']))

        if len(X) == 0:
            print("错误: 没有有效的训练数据！")
            print("请确保每个手势都收集了足够的样本（每个手势至少10个）")
            return None

        dataset = {
            'features': np.array(X),
            'labels': np.array(y),
            'gesture_classes': self.gesture_classes,
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

    def run_collection_app(self):
        """运行数据收集应用程序"""
        cap = cv2.VideoCapture(1)
        collecting_for = None
        sample_count = 0

        print("手势数据收集工具")
        print("=" * 50)
        print("命令:")
        print("  1-8: 选择手势类别（共8个手势）")
        print("  c: 收集当前样本")
        print("  s: 停止收集当前手势")
        print("  w: 保存数据集")
        print("  q: 退出")
        print("=" * 50)
        print("手势类别:")
        for i, gesture in enumerate(self.gesture_classes):
            print(f"  {i + 1}: {gesture}")
        print("=" * 50)
        print("提示: 每个手势建议收集20-30个样本")
        print("确保手在摄像头中清晰可见")
        print("=" * 50)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # 提取关键点
            landmarks, hand_landmarks = self.extract_landmarks(frame)

            # 绘制手部关键点
            if hand_landmarks:
                for hand_landmarks in hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # 将OpenCV图像转换为PIL图像以支持中文
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            try:
                # 尝试加载中文字体（Windows系统路径）
                font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
                font = ImageFont.truetype(font_path, 30)
            except:
                # 如果找不到字体，使用默认字体
                font = ImageFont.load_default()
                print("使用默认字体，中文可能显示为方框")

            # 显示中文
            if collecting_for:
                status_text = f"当前手势: {collecting_for}"
                color = (0, 255, 0)
            else:
                status_text = "请选择手势 (1-8)"
                color = (0, 255, 255)

            # 绘制文字
            draw.text((10, 10), status_text, font=font, fill=color)
            draw.text((10, 50), f"已收集样本: {sample_count}", font=font, fill=(0, 255, 0))
            draw.text((10, 90), "按1-8选择手势，c收集样本", font=font, fill=(255, 255, 255))

            # 显示手势类别提示
            y_pos = 130
            for i, gesture in enumerate(self.gesture_classes):
                text = f"{i + 1}: {gesture}"
                draw.text((10, y_pos), text, font=font, fill=(200, 200, 0))
                y_pos += 35

            # 将PIL图像转回OpenCV格式
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            cv2.imshow('手势数据收集', frame)

            key = cv2.waitKey(1) & 0xFF

            # 处理按键
            if key == ord('q'):
                break
            elif ord('1') <= key <= ord('8'):  # 只有1-8
                gesture_idx = key - ord('1')
                if gesture_idx < len(self.gesture_classes):
                    collecting_for = self.gesture_classes[gesture_idx]
                    print(f"开始收集手势: {collecting_for}")
            elif key == ord('c') and collecting_for:
                if landmarks:
                    result = self.collect_sample(collecting_for, frame, landmarks)
                    if result:  # 只有成功收集才计数
                        sample_count += 1
                        print(f"已收集 {collecting_for} 样本 #{sample_count}")
                else:
                    print("未检测到手部！请确保手在摄像头中清晰可见")
            elif key == ord('s'):
                collecting_for = None
                print("停止收集")
            elif key == ord('w'):
                dataset = self.save_dataset()
                if dataset:
                    sample_count = dataset['num_samples']
                    print(f"数据集已保存，总样本数: {sample_count}")

        cap.release()
        cv2.destroyAllWindows()

        # 保存最终数据集
        if self.collected_data:
            self.save_dataset()


if __name__ == "__main__":
    collector = GestureDataCollector()
    collector.run_collection_app()