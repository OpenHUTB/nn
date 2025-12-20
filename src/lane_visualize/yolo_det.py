from ultralytics import YOLO
import cv2
import numpy as np


class ObjectDetector:
    def __init__(self, model_name='yolov8n.pt'):
        """
        初始化 YOLO 模型
        :param model_name: 模型名称，'yolov8n.pt' 是最快最小的模型，适合CPU
        """
        print(f"正在加载 YOLO 模型: {model_name} (首次运行会自动下载)...")
        self.model = YOLO(model_name)

        # 定义我们关心的类别ID (COCO数据集)
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = [2, 3, 5, 7]

    def detect(self, frame):
        """
        输入一帧图像，返回画好框的图像
        """
        # 运行推理，stream=True 节省内存，verbose=False 关闭打印日志
        results = self.model(frame, stream=True, verbose=False)

        # 在原图的副本上绘图
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取类别ID
                cls_id = int(box.cls[0])

                # 只处理车辆相关的类别
                if cls_id in self.target_classes:
                    # 获取坐标 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # 获取置信度
                    conf = float(box.conf[0])
                    if conf < 0.3: continue  # 过滤掉置信度低的

                    # 获取类别名称
                    cls_name = self.model.names[cls_id]

                    # 绘制矩形框 (红色)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # 绘制标签背景
                    label = f"{cls_name} {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 0, 255), -1)

                    # 绘制文字
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated_frame