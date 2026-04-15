# -*- coding: utf-8 -*-
"""
Project: CARLA AutoVision Navigator
Module: Perception - Object Detection
Version: v1.0.0
Description: 目标检测模块。基于 YOLOv3 深度学习模型，利用 OpenCV DNN 模块实现交通参与者的实时识别。
Author: wangadsa
License: MIT License
"""
import cv2
import numpy as np
import os
import sys

# 导入项目配置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class YOLOv3Detector:
    def __init__(self):
        # 动态获取项目根目录，解决跨环境运行的路径问题
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        coco_path = os.path.join(self.root_dir, "models", "coco.names")
        cfg_path = os.path.join(self.root_dir, "models", "yolov3.cfg")
        weights_path = os.path.join(self.root_dir, "models", "yolov3.weights")

        # 1. 加载类别标签
        with open(coco_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # 2. 加载 YOLOv3 网络（采用更稳定的 Darknet 解析器）
        try:
            self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as e:
            print(f"致命错误: 无法解析模型文件，请检查 cfg 内容。详情: {e}")
            sys.exit(1)

        # 3. 获取输出层名称
        layer_names = self.net.getLayerNames()
        out_layers = self.net.getUnconnectedOutLayers()
        if isinstance(out_layers, np.ndarray):
            out_layers = out_layers.flatten()

        self.output_layers = [layer_names[i - 1] for i in out_layers]
        print(f"感知层状态: 目标检测模块已就绪。")

    def detect(self, frame):
        if frame is None: return []
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > config.CONFIDENCE_THRESHOLD:
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    boxes.append([int(center_x - w / 2), int(center_y - h / 2), w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, config.CONFIDENCE_THRESHOLD, config.NMS_THRESHOLD)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({"class": self.classes[class_ids[i]], "confidence": confidences[i], "box": boxes[i]})
        return results

    def draw_labels(self, frame, detections):
        for det in detections:
            x, y, w, h = det["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, det["class"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame