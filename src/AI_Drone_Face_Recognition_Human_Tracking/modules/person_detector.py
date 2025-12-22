import cv2
import numpy as np
import random


class PersonDetector:
    def __init__(self):
        print("✅ 人物检测器初始化")
        # 这里可以加载YOLO模型，但为了简化先使用模拟

    def detect_persons(self, frame):
        """检测人物"""
        h, w = frame.shape[:2]
        persons = []

        # 模拟检测1-4个人物
        num_persons = random.randint(0, 4)

        for i in range(num_persons):
            width = random.randint(40, 120)
            height = random.randint(80, 200)
            x = random.randint(0, w - width)
            y = random.randint(0, h - height)

            persons.append({
                "bbox": (x, y, x + width, y + height),
                "confidence": random.uniform(0.6, 0.95)
            })

        return persons

    def draw_persons(self, frame, persons):
        """绘制人物框"""
        for person in persons:
            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"Person {person['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return frame