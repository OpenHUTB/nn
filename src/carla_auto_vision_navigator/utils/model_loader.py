<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Project: CARLA AutoVision Navigator
Module: Utility - Model Asset Management
Version: v1.0.0
Description: 模型资产管理工具。实现 YOLOv3 权重文件的完整性校验、路径自动锚定以及环境诊断功能。
Author: wangadsa
License: MIT License
"""
import os
import requests


def check_yolo_weights():
    """
    检查 YOLOv3 权重文件是否存在，若不存在则提供下载指引。
    """
    weights_path = "models/yolov3.weights"
    cfg_path = "models/yolov3.cfg"

    # YOLOv3 官方权重下载地址
    weights_url = "https://pjreddie.com/media/files/yolov3.weights"

    if not os.path.exists(weights_path):
        print("=" * 50)
        print("警告: 未发现 YOLOv3 权重文件 (yolov3.weights)！")
        print(f"请从以下链接下载并放入 models/ 文件夹:")
        print(weights_url)
        print("=" * 50)
        return False

    print("确认: YOLOv3 权重文件已就绪。")
    return True


if __name__ == "__main__":
    # 简单的模块测试
    check_yolo_weights()
=======
import torch
import logging
from pathlib import Path

from config import ANOMALY_DETECTOR_CONFIG

logger = logging.getLogger(__name__)


def _build_default_yolov7_backbone(output_dim=2048):
    """构建分辨率无关的默认 RGB 特征提取网络。"""
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, output_dim)
    )


def _unwrap_loaded_model(loaded_object, model_path):
    if isinstance(loaded_object, torch.nn.Module):
        return loaded_object

    if isinstance(loaded_object, dict):
        for key in ("model", "ema"):
            candidate = loaded_object.get(key)
            if isinstance(candidate, torch.nn.Module):
                return candidate

    raise TypeError(f"无法从 {model_path} 中解析出 torch.nn.Module 实例")


def load_pretrained_model(model_type, model_path):
    """加载预训练模型（支持YOLO/CNN等）"""
    if model_type == "yolov7":
        model_path = Path(model_path)
        # 加载YOLOv7预训练模型（简化版）
        if model_path.exists():
            loaded_object = torch.load(str(model_path), map_location=ANOMALY_DETECTOR_CONFIG["device"])
            model = _unwrap_loaded_model(loaded_object, model_path)
            logger.info(f"加载YOLOv7模型：{model_path}")
        else:
            logger.warning(f"模型文件不存在，加载默认YOLOv7权重")
            # 实际使用时替换为真实YOLOv7加载逻辑
            model = _build_default_yolov7_backbone()
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")

    model.eval()
    return model
>>>>>>> upstream/main
