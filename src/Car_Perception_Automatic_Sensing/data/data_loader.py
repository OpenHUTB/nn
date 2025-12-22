"""
数据加载器核心模块
支持自动驾驶数据集的加载、样本管理与批次分发
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List


class AutoDriveDataset(Dataset):
    """自动驾驶感知数据集基类（所有自定义数据集需继承此类）"""
    def __init__(self, data_root: str, split: str = "train", transform: Optional[object] = None):
        self.data_root = os.path.abspath(data_root)
        self.split = split
        self.transform = transform
        self.sample_paths: List[str] = []  # 存储所有样本路径
        self._load_sample_paths()  # 加载样本列表

    def _load_sample_paths(self):
        """加载样本路径（抽象方法，子类实现）"""
        raise NotImplementedError("请在子类中实现 _load_sample_paths 方法")

    def __len__(self) -> int:
        """返回样本总数"""
        return len(self.sample_paths)

    def __getitem__(self, idx: int):
        """获取单个样本（抽象方法，子类实现）"""
        raise NotImplementedError("请在子类中实现 __getitem__ 方法")


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = False
) -> DataLoader:
    """构建PyTorch DataLoader（跨系统兼容）"""
    # Windows系统默认关闭多进程
    if os.name == "nt":
        num_workers = 0

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )


if __name__ == "__main__":
    # 框架测试
    pass