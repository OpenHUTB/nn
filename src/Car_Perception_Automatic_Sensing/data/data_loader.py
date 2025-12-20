"""
数据加载器模块
负责自动驾驶感知任务的数据集加载、数据预处理分发等功能
支持图像类数据集（如KITTI、COCO自动驾驶子集）的加载
"""

import os
import numpy as np
import cv2  # 新增：导入opencv用于图像读取
import torch
from torch.utils.data import Dataset, DataLoader


class AutoDriveDataset(Dataset):
    """
    自动驾驶感知数据集基类
    所有自定义数据集需继承此类并实现抽象方法
    """
    def __init__(self, data_root: str, split: str = "train", transform=None):
        """
        初始化数据集
        :param data_root: 数据集根目录路径
        :param split: 数据集划分（train/val/test）
        :param transform: 数据增强/预处理变换
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.sample_list = []  # 存储样本路径/索引的列表

        # 后续将实现：加载样本列表
        self._load_sample_list()

    def _load_sample_list(self):
        """
        加载数据集样本列表（抽象方法，需子类实现）
        """
        raise NotImplementedError("子类必须实现 _load_sample_list 方法")

    def _load_image(self, img_path: str) -> np.ndarray:
        """
        封装图像读取函数
        :param img_path: 图像文件路径
        :return: 读取后的RGB格式图像（np.ndarray）
        """
        # 检查图像路径是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在：{img_path}")

        # 读取图像（cv2默认读取为BGR格式）
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像文件：{img_path}（可能是文件损坏或格式不支持）")

        # 转换为RGB格式（符合大多数深度学习框架的输入要求）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

    def __len__(self):
        """
        返回数据集样本总数
        """
        return len(self.sample_list)

    def __getitem__(self, idx: int):
        """
        根据索引获取单个样本（抽象方法，需子类实现）
        :param idx: 样本索引
        :return: 处理后的图像和标注数据
        """
        raise NotImplementedError("子类必须实现 __getitem__ 方法")


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 4):
    """
    构建数据加载器
    :param dataset: 实例化的 Dataset 对象
    :param batch_size: 批次大小
    :param shuffle: 是否打乱样本顺序
    :param num_workers: 数据加载进程数
    :return: DataLoader 对象
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 加速GPU数据传输
    )
    return dataloader


if __name__ == "__main__":
    # 测试代码框架（后续可完善）
    pass