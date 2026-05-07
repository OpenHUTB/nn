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
