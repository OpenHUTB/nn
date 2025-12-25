"""数据预处理工具：归一化、标准化"""
import numpy as np


def normalize_image(img: np.ndarray, target_range: tuple = (0, 1)) -> np.ndarray:
    """
    图像像素值归一化到指定范围
    :param img: 输入图像（np.ndarray, uint8/float32）
    :param target_range: 目标范围，默认(0,1)
    :return: 归一化后的图像（float32）
    """
    img_float = img.astype(np.float32)
    min_val = img_float.min()
    max_val = img_float.max()
    # 避免除零
    if max_val - min_val < 1e-6:
        return np.zeros_like(img_float) + target_range[0]
    # 归一化计算
    normalized = (img_float - min_val) / (max_val - min_val)
    normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
    return normalized


def standardize_image(img: np.ndarray, mean: list = None, std: list = None) -> np.ndarray:
    """
    图像标准化（减均值、除标准差）
    :param img: 输入RGB图像（np.ndarray, (h,w,3)）
    :param mean: 通道均值，默认ImageNet均值
    :param std: 通道标准差，默认ImageNet标准差
    :return: 标准化后的图像
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    img_float = img.astype(np.float32) / 255.0  # 先归一化到0-1
    for i in range(3):
        img_float[..., i] = (img_float[..., i] - mean[i]) / std[i]
    return img_float