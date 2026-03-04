# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import sys

# 导入项目配置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class YOLOv3Detector:
    def __init__(self):
        # 1. 加载类别标签
        self.classes = []
        with open("models/coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # 2. 加载 YOLOv3 网络
        # 注意：确保运行了 model_loader.py 下载了权重文件
        self.net = cv2.dnn.readNet(config.MODEL_WEIGHTS, config.MODEL_CFG)

        # 3. 获取输出层名称
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        print("感知层状态: YOLOv3 模型加载成功。")

    def detect(self, frame):
        """
        对输入的图像帧进行目标检测
        """
        if frame is None:
            return []

        height, width, _ = frame.shape
        # 将图像转换为网络输入的 blob 格式
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # 解析网络输出
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # 过滤低置信度的检测结果
                if confidence > config.CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 应用非极大值抑制 (NMS) 去除重复框
        indices = cv2.dnn.NMSBoxes(boxes, confidences, config.CONFIDENCE_THRESHOLD, config.NMS_THRESHOLD)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    "class": self.classes[class_ids[i]],
                    "confidence": confidences[i],
                    "box": boxes[i]
                })
        return results

    def draw_labels(self, frame, detections):
        """
        在图像上绘制检测框和标签
        """
        for det in detections:
            x, y, w, h = det["box"]
            label = f"{det['class']} {det['confidence']:.2f}"
            color = (0, 255, 0)  # 绿色检测框
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame