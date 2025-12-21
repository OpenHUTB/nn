"""
数据模块（data）
负责自动驾驶感知任务的数据集加载、图像预处理、数据增强等功能
支持KITTI、Waymo、COCO等自动驾驶数据集
"""

# 导出核心类和函数
from .data_loader import AutoDriveDataset, build_dataloader
from .preprocess_utils import load_image_rgb

__all__ = [
    "AutoDriveDataset",
    "build_dataloader",
    "load_image_rgb"
]