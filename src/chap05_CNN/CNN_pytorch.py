import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


# =============================================================================
# 超参数设置
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
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=10,              # ±10° 随机旋转
        translate=(0.1, 0.1),    # 最多 10% 的随机平移
    ),
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])


def build_dataloaders():
    download = not (os.path.exists("./mnist/") and os.listdir("./mnist/"))
    train_set = torchvision.datasets.MNIST(
        root="./mnist/", train=True, transform=train_transform, download=download
    )
    test_set = torchvision.datasets.MNIST(
        root="./mnist/", train=False, transform=test_transform, download=download
    )

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
# =============================================================================
class CNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = DROPOUT):
        super().__init__()

        def conv_bn_relu(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn_relu(1, 32),
            conv_bn_relu(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            conv_bn_relu(32, 64),
            conv_bn_relu(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            conv_bn_relu(64, 128),
            conv_bn_relu(128, 128),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
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
    print(f"超参数            : lr={LEARNING_RATE}, wd={WEIGHT_DECAY}, dropout={DROPOUT}, label_smooth={LABEL_SMOOTH}")
    print("=" * 70)

    best_acc = 0.0

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

            if step % 100 == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch:2d} | Step {step:4d}/{len(train_loader)} "
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