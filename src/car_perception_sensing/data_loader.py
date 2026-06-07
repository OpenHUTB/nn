"""
数据加载器模块
负责自动驾驶感知任务的数据集加载、数据预处理分发等功能
支持图像类数据集（如KITTI、COCO自动驾驶子集）的加载
"""
import numpy as np
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