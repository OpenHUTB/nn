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
# 新增导入
import cv2
from typing import Tuple

# 在 AutoDriveDataset 类中新增方法
def _load_image(self, img_rel_path: str) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    读取图像并转换为RGB格式
    :param img_rel_path: 图像相对路径（基于data_root）
    :return: (RGB图像数组, 图像形状(h, w, c))
    """
    img_abs_path = os.path.join(self.data_root, img_rel_path)
    # 路径校验
    if not os.path.exists(img_abs_path):
        raise FileNotFoundError(f"图像不存在：{img_abs_path}")
    # 读取图像（忽略透明通道）
    img_bgr = cv2.imread(img_abs_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"无法读取图像（损坏/格式不支持）：{img_abs_path}")
    # BGR转RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, img_rgb.shape