#!/usr/bin/env python
# coding: utf-8
"""
FNN 消融实验 —— MNIST 手写数字识别（PyTorch 实现）
====================================================
对比以下 5 种模型配置，验证每个改进组件的贡献：

  Config-A  Baseline      : 784 -> 128 -> 10  (复现原版逻辑，改为 mini-batch)
  Config-B  +Dropout      : Baseline + Dropout(0.3)
  Config-C  +BN           : Baseline + BatchNormalization
  Config-D  +BN+Dropout   : Baseline + BN + Dropout(0.3)
  Config-E  Enhanced      : 784 -> 256 -> 128 -> 64 -> 10 + BN + Dropout + LR调度

输出（保存在脚本同目录）：
  fnn_ablation_train_acc.png  训练集准确率曲线
  fnn_ablation_test_acc.png   测试集准确率曲线
  fnn_ablation_loss.png       训练 Loss 曲线
  fnn_ablation_bar.png        最终测试准确率柱状图
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

EPOCHS     = 20
BATCH_SIZE = 256
LR         = 1e-3
SEED       = 42
SAVE_DIR   = os.path.dirname(os.path.abspath(__file__))

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_root = os.path.join(SAVE_DIR, "mnist_data")
    train_set = datasets.MNIST(data_root, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


def build_model(config):
    if config == "A":
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    elif config == "B":
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10),
        )
    elif config == "C":
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    elif config == "D":
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10),
        )
    elif config == "E":
        layers_list = [nn.Flatten()]
        dims = [784, 256, 128, 64]
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i+1], bias=False)
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            layers_list += [lin, nn.BatchNorm1d(dims[i+1]), nn.ReLU(), nn.Dropout(0.3)]
        layers_list.append(nn.Linear(64, 10))
        model = nn.Sequential(*layers_list)
    else:
        raise ValueError(f"Unknown config: {config}")
    return model.to(DEVICE)


def train_model(config, train_loader, test_loader):
    model     = build_model(config)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    if config == "E":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*55}")
    print(f"  训练 Config-{config}  |  参数量: {total_params:,}")
    print(f"{'='*55}")

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y_batch.size(0)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            total      += y_batch.size(0)
        tr_loss = total_loss / total
        tr_acc  = correct    / total

        model.eval()
        t_loss, t_correct, t_total = 0.0, 0, 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                logits     = model(x_batch)
                t_loss    += criterion(logits, y_batch).item() * y_batch.size(0)
                t_correct += (logits.argmax(1) == y_batch).sum().item()
                t_total   += y_batch.size(0)
        te_loss = t_loss    / t_total
        te_acc  = t_correct / t_total

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        print(f"  Epoch {epoch+1:>2}/{EPOCHS} | train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f} | test_loss={te_loss:.4f}  test_acc={te_acc:.4f}")
        if scheduler is not None:
            scheduler.step(te_acc)

    return history


CONFIGS = ["A", "B", "C", "D", "E"]
LABELS  = {
    "A": "A: Baseline (784->128->10)",
    "B": "B: +Dropout(0.3)",
    "C": "C: +BatchNorm",
    "D": "D: +BN+Dropout",
    "E": "E: Enhanced (deeper+BN+Dropout+LR)",
}
COLORS  = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6"]
MARKERS = ["o", "s", "^", "D", "P"]


def plot_curves(all_history):
    epochs = list(range(1, EPOCHS + 1))

    for metric, title, ylabel, ylim, fname in [
        ("train_acc", "FNN Ablation Study -- Training Accuracy", "Accuracy", (0.85, 1.01), "fnn_ablation_train_acc.png"),
        ("test_acc",  "FNN Ablation Study -- Test Accuracy",     "Accuracy", (0.93, 1.005),"fnn_ablation_test_acc.png"),
        ("train_loss","FNN Ablation Study -- Training Loss",     "Cross-Entropy Loss", None, "fnn_ablation_loss.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, cfg in enumerate(CONFIGS):
            ax.plot(epochs, all_history[cfg][metric],
                    label=LABELS[cfg], color=COLORS[i],
                    marker=MARKERS[i], markevery=4, linewidth=1.8)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        loc = "lower right" if "acc" in metric else "upper right"
        ax.legend(fontsize=8, loc=loc)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(SAVE_DIR, fname)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[保存] {p}")

    best_accs    = [max(all_history[cfg]["test_acc"]) for cfg in CONFIGS]
    short_labels = ["A\nBaseline", "B\n+Dropout", "C\n+BN", "D\n+BN\n+Dropout", "E\nEnhanced"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(short_labels, best_accs, color=COLORS, width=0.5, edgecolor="white")
    for bar, acc in zip(bars, best_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.0003,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Best Test Accuracy", fontsize=12)
    ax.set_title("FNN Ablation Study -- Best Test Accuracy per Config", fontsize=14)
    ax.set_ylim(min(best_accs)-0.005, 1.005)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(SAVE_DIR, "fnn_ablation_bar.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[保存] {p}")


def print_summary(all_history):
    baseline = max(all_history["A"]["test_acc"])
    descs = {
        "A": "Baseline (原版逻辑 + mini-batch)",
        "B": "Baseline + Dropout(0.3)",
        "C": "Baseline + BatchNorm",
        "D": "Baseline + BN + Dropout",
        "E": "Enhanced (深层+BN+Dropout+LR调度)",
    }
    print("\n" + "="*68)
    print(f"  {'配置':<6}  {'描述':<30}  {'最佳测试准确率':>12}  {'相对提升':>8}")
    print("-"*68)
    for cfg in CONFIGS:
        best  = max(all_history[cfg]["test_acc"])
        delta = (best - baseline) * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {cfg:<6}  {descs[cfg]:<30}  {best:>12.4f}  {sign+f'{delta:.2f}%':>8}")
    print("="*68)


if __name__ == "__main__":
    print("加载 MNIST 数据集 ...")
    train_loader, test_loader = load_mnist()
    all_history = {}
    for cfg in CONFIGS:
        all_history[cfg] = train_model(cfg, train_loader, test_loader)
    print("\n绘制对比图 ...")
    plot_curves(all_history)
    print_summary(all_history)
    print("\n完成！4 张图片已保存在脚本同目录下。")
