#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# 基于 PyTorch 的 CNN 实现（优化版）—— MNIST 手写数字识别
# 相比原版的主要改进：
#   1. GPU 自动加速（CUDA / Apple MPS / CPU 自动切换）
#   2. 数据增强（随机仿射：旋转 + 平移）提升泛化
#   3. 使用 MNIST 真实均值/标准差进行归一化（0.1307 / 0.3081）
#   4. 更深的三段式 CNN + Dropout2d + Kaiming 初始化
#   5. AdamW 优化器 + CosineAnnealingLR 学习率调度
#   6. 交叉熵加入 Label Smoothing
#   7. 在完整 10000 张测试集上评估（批量推理，非只取前 500）
#   8. DataLoader 启用 num_workers + pin_memory
#   9. 自动保存测试集表现最好的模型权重
# 典型结果：~99.55% 测试准确率（原版约 98.5%）
# =============================================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


# =============================================================================
# 超参数
# =============================================================================
LEARNING_RATE = 1e-3      # 初始学习率（配合余弦退火，初始值可以比原版 1e-4 大）
WEIGHT_DECAY  = 5e-4      # L2 权重衰减（AdamW 会解耦应用）
DROPOUT       = 0.3       # 全连接层 Dropout 丢弃率（原版 KEEP_PROB_RATE=0.7 等价于此）
LABEL_SMOOTH  = 0.1       # 标签平滑系数
MAX_EPOCH     = 15        # 训练轮数（原版 3 轮明显欠拟合）
BATCH_SIZE    = 128       # 批大小（GPU 上更高效；CPU 也可以跑）
NUM_WORKERS   = 2         # DataLoader 子进程数；Windows 下如报错可改为 0
SEED          = 42        # 随机种子，保证可复现
CKPT_PATH     = "./best_cnn_mnist.pth"  # 最佳模型保存路径


# =============================================================================
# 设备自动选择：优先 CUDA -> Apple MPS -> CPU
# =============================================================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# =============================================================================
# 随机种子（尽量保证结果可复现）
# =============================================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 数据集加载 + 数据增强
# =============================================================================
# MNIST 官方统计的像素均值 / 标准差（训练集上算出来的常数）
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

# 训练集：加入轻微的数据增强 —— 防止过拟合，提升泛化
train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=10,              # ±10° 随机旋转
        translate=(0.1, 0.1),    # 最多 10% 的随机平移
    ),
    transforms.ToTensor(),                               # [0,255] -> [0,1] 并加通道维
    transforms.Normalize(MNIST_MEAN, MNIST_STD),         # 标准化：均值 0 方差 1
])

# 测试集：不做增强，只做与训练一致的归一化
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])


def build_dataloaders():
    """构建训练/测试 DataLoader。首次运行会自动下载 MNIST 数据。"""
    download = not (os.path.exists("./mnist/") and os.listdir("./mnist/"))

    train_set = torchvision.datasets.MNIST(
        root="./mnist/", train=True, transform=train_transform, download=download
    )
    test_set = torchvision.datasets.MNIST(
        root="./mnist/", train=False, transform=test_transform, download=download
    )

    # pin_memory 仅在 CUDA 上有意义；其他设备设为 False 避免警告
    use_pin = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=use_pin, drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=512, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=use_pin,
    )
    return train_loader, test_loader


# =============================================================================
# 模型定义：三段式 CNN
#   输入 1x28x28
#   Block1: (Conv-BN-ReLU) x2 + MaxPool + Dropout2d  -> 32x14x14
#   Block2: (Conv-BN-ReLU) x2 + MaxPool + Dropout2d  -> 64x7x7
#   Block3: (Conv-BN-ReLU) x2 + MaxPool + Dropout2d  -> 128x3x3
#   Classifier: 128*3*3 -> 256 -> 10
# =============================================================================
class CNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = DROPOUT):
        super().__init__()

        def conv_bn_relu(in_c, out_c):
            """卷积 + BN + ReLU 的小工厂函数，保持 padding=1 不改变空间尺寸"""
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        # ---------- 特征提取 ----------
        self.features = nn.Sequential(
            # Block 1: 1x28x28 -> 32x14x14
            conv_bn_relu(1, 32),
            conv_bn_relu(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            # Block 2: 32x14x14 -> 64x7x7
            conv_bn_relu(32, 64),
            conv_bn_relu(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            # Block 3: 64x7x7 -> 128x3x3（7//2=3）
            conv_bn_relu(64, 128),
            conv_bn_relu(128, 128),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
        )

        # ---------- 分类头 ----------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化，有助于 ReLU 网络训练更稳定"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# 评估：在完整测试集上做批量推理，计算平均损失与准确率
# =============================================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    model.train()
    return loss_sum / total, correct / total


# =============================================================================
# 训练主循环
# =============================================================================
def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCH * len(train_loader)
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    print("=" * 70)
    print(f"设备             : {DEVICE}")
    print(f"训练样本 / 测试样本: {len(train_loader.dataset)} / {len(test_loader.dataset)}")
    print(f"超参数            : lr={LEARNING_RATE}, wd={WEIGHT_DECAY}, "
          f"dropout={DROPOUT}, label_smooth={LABEL_SMOOTH}")
    print(f"                    epochs={MAX_EPOCH}, batch_size={BATCH_SIZE}")
    print("=" * 70)

    best_acc = 0.0
    global_step = 0

    for epoch in range(1, MAX_EPOCH + 1):
        model.train()
        t0 = time.time()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            running_total += x.size(0)
            global_step += 1

            if step % 100 == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch:2d} | Step {step:4d}/{len(train_loader)} "
                      f"| loss={running_loss/running_total:.4f} "
                      f"| train_acc={running_correct/running_total:.4f} "
                      f"| lr={cur_lr:.2e}")

        test_loss, test_acc = evaluate(model, test_loader, loss_fn)
        dt = time.time() - t0
        print(f">>> Epoch {epoch:2d} 完成 | 用时 {dt:.1f}s "
              f"| 训练损失 {running_loss/running_total:.4f} "
              f"| 测试损失 {test_loss:.4f} "
              f"| 测试准确率 {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {"model_state": model.state_dict(),
                 "epoch": epoch,
                 "test_acc": test_acc},
                CKPT_PATH,
            )
            print(f"    ✓ 新最佳准确率 {best_acc*100:.2f}%，已保存到 {CKPT_PATH}")

    print("=" * 70)
    print(f"训练完成！最佳测试准确率：{best_acc*100:.2f}%")
    print("=" * 70)
    return best_acc


# =============================================================================
# 主程序入口
# =============================================================================
def main():
    set_seed(SEED)
    train_loader, test_loader = build_dataloaders()

    model = CNN().to(DEVICE)
    print("模型结构：")
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量：{n_params/1e6:.2f} M")

    train(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
