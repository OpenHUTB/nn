import cv2
import numpy as np
import os


class FaceRecognizer:
    def __init__(self, face_db_path="faces"):
        self.face_db_path = face_db_path
        print(f"✅ 人脸识别器初始化，数据库路径: {face_db_path}")

    def recognize_face(self, frame, threshold=0.6):
        """识别人脸"""
        # 模拟识别
        import random

        # 模拟检测一些人脸
        h, w = frame.shape[:2]
        results = []

        # 随机生成1-2个识别结果
        num_faces = random.randint(0, 2)

        for i in range(num_faces):
            size = random.randint(60, 100)
            x = random.randint(0, w - size)
            y = random.randint(0, h - size)

            # 随机选择是否识别成功
            if random.random() > 0.5:
                names = ["张三", "李四", "王五", "赵六"]
                name = random.choice(names)
                confidence = random.uniform(0.7, 0.95)
            else:
                name = "Unknown"
                confidence = random.uniform(0.3, 0.5)

            results.append({
                "bbox": (x, y, size, size),
                "name": name,
                "confidence": confidence
            })

        return results

    def draw_recognition(self, frame, results):
        """绘制识别结果"""
        for res in results:
            x, y, w, h = res["bbox"]
            name = res["name"]
            conf = res["confidence"]

            if name != "Unknown":
                color = (0, 255, 255)  # 黄色表示识别成功
                label = f"{name} ({conf:.2f})"
            else:
                color = (0, 0, 255)  # 红色表示未知
                label = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame