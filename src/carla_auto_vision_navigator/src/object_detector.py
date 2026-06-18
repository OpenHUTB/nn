<<<<<<< HEAD
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
=======
import torch
import torch.nn as nn
import cv2
import numpy as np
import open3d as o3d
from config import ANOMALY_DETECTOR_CONFIG, UNSTRUCTURED_SCENES
from utils.model_loader import load_pretrained_model


class MultimodalAnomalyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = ANOMALY_DETECTOR_CONFIG["device"]
        self.confidence_thresh = ANOMALY_DETECTOR_CONFIG["confidence_threshold"]
        self.fusion_method = ANOMALY_DETECTOR_CONFIG["fusion_method"]
        self.anomaly_types = self._resolve_anomaly_types()
        self.rgb_feature_dim = 2048
        self.lidar_feature_dim = 1024
        self.fusion_feature_dim = 512

        # 加载单模态检测模型
        self.rgb_model = load_pretrained_model("yolov7", ANOMALY_DETECTOR_CONFIG["model_path"])
        self.lidar_model = self._build_lidar_model()
        self.rgb_projection = nn.Sequential(
            nn.Linear(self.rgb_feature_dim, self.fusion_feature_dim),
            nn.ReLU()
        )
        self.lidar_projection = nn.Sequential(
            nn.Linear(self.lidar_feature_dim, self.fusion_feature_dim),
            nn.ReLU()
        )

        # 多模态融合层
        if self.fusion_method == "weighted":
            self.rgb_weight = nn.Parameter(torch.tensor(0.7, device=self.device))
            self.lidar_weight = nn.Parameter(torch.tensor(0.3, device=self.device))
        elif self.fusion_method == "concat":
            self.fusion_layer = nn.Linear(self.fusion_feature_dim * 2, self.fusion_feature_dim)
        elif self.fusion_method == "attention":
            self.attention_layer = nn.MultiheadAttention(self.fusion_feature_dim, 8, batch_first=True)
        else:
            raise ValueError(f"不支持的融合方式：{self.fusion_method}")

        # 异常分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, len(self.anomaly_types) + 1)  # 异常类型+正常
        )
        self.to(self.device)

    @staticmethod
    def _resolve_anomaly_types():
        configured_types = ANOMALY_DETECTOR_CONFIG.get("anomaly_types")
        if configured_types:
            return configured_types

        merged_types = []
        for scene_config in UNSTRUCTURED_SCENES.values():
            for anomaly_type in scene_config.get("anomaly_types", []):
                if anomaly_type not in merged_types:
                    merged_types.append(anomaly_type)
        return merged_types or ["unknown_anomaly"]

    def _build_lidar_model(self):
        """构建LiDAR点云特征提取模型"""
        model = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 125000, 1024),  # 适配LiDAR点云维度
            nn.ReLU()
        )
        return model

    def preprocess_rgb(self, rgb_data):
        """预处理RGB图像"""
        img = cv2.resize(rgb_data, (640, 640))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)
        return img

    def preprocess_lidar(self, lidar_data):
        """预处理LiDAR点云"""
        points = np.asarray(lidar_data.points).T  # (3, N)
        points = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
        return points

    def forward(self, rgb_data, lidar_data):
        """前向传播：多模态融合+异常检测"""
        # 1. 单模态特征提取
        rgb_feat = self.rgb_projection(self.rgb_model(self.preprocess_rgb(rgb_data)))
        lidar_feat = self.lidar_projection(self.lidar_model(self.preprocess_lidar(lidar_data)))

        # 2. 多模态融合
        if self.fusion_method == "weighted":
            fusion_feat = self.rgb_weight * rgb_feat + self.lidar_weight * lidar_feat
        elif self.fusion_method == "concat":
            fusion_feat = torch.cat([rgb_feat, lidar_feat], dim=1)
            fusion_feat = self.fusion_layer(fusion_feat)
        elif self.fusion_method == "attention":
            fusion_inputs = torch.stack([rgb_feat, lidar_feat], dim=1)
            fusion_feat, _ = self.attention_layer(fusion_inputs, fusion_inputs, fusion_inputs)
            fusion_feat = fusion_feat.mean(dim=1)

        # 3. 异常分类
        logits = self.classifier(fusion_feat)
        probs = torch.softmax(logits, dim=1)

        # 4. 筛选高置信度结果
        max_prob, pred_idx = torch.max(probs, dim=1)
        if max_prob.item() > self.confidence_thresh:
            return {
                "anomaly_type": self.anomaly_types[pred_idx.item() - 1] if pred_idx.item() > 0 else "normal",
                "confidence": max_prob.item(),
                "bbox": None  # 可拓展：输出异常目标框
            }
        else:
            return {"anomaly_type": "unknown", "confidence": 0.0, "bbox": None}

    def detect_anomaly(self, sensor_data):
        """检测非结构化场景异常（对外接口）"""
        if sensor_data["rgb"] is None or sensor_data["lidar"] is None:
            return {"anomaly_type": "no_data", "confidence": 0.0, "bbox": None}

        with torch.no_grad():
            result = self.forward(sensor_data["rgb"], sensor_data["lidar"])
        return result


if __name__ == "__main__":
    # 测试检测器
    detector = MultimodalAnomalyDetector()
    # 模拟传感器数据
    mock_rgb = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    mock_lidar = o3d.geometry.PointCloud()
    mock_lidar.points = o3d.utility.Vector3dVector(np.random.rand(500000, 3))

    result = detector.detect_anomaly({"rgb": mock_rgb, "lidar": mock_lidar})
    print(f"异常检测结果：{result}")
>>>>>>> upstream/main
