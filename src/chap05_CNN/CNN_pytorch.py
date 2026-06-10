#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版 PyTorch CNN —— MNIST 手写数字识别

优化点:
  - 配置集中化，训练/评估函数去全局依赖，便于复用和调参
  - 更稳健的 CUDA / MPS / CPU 设备选择
  - DataLoader 参数按 num_workers 自动适配，避免 Windows / num_workers=0 报错
  - 训练集增强、验证/测试集标准预处理，固定划分随机种子
  - 更轻量的 CNN Head: AdaptiveAvgPool2d + Linear，减少参数量
  - AdamW + CosineAnnealingLR + Label Smoothing + 梯度裁剪
  - CUDA AMP 自动混合精度，兼容不同 PyTorch 版本
  - 早停、最佳 checkpoint 保存、最终测试
  - 单张 Tensor / 图片文件推理接口
"""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


# =============================================================================
# 配置
# =============================================================================
@dataclass
class Config:
    # 路径
    data_dir: str = "./mnist"
    save_dir: str = "./checkpoints"
    ckpt_name: str = "best_cnn_mnist.pth"

    # 训练
    epochs: int = 15
    batch_size: int = 128
    test_batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    grad_clip: Optional[float] = 5.0

    # 模型
    num_classes: int = 10
    channels: Tuple[int, int, int] = (32, 64, 128)
    dropout: float = 0.25

    # 验证 / 早停
    val_ratio: float = 0.1
    early_stop_patience: Optional[int] = 5

    # DataLoader
    # Windows 建议 0；Linux/macOS 可设 2/4/8
    num_workers: int = 0

    # 其他
    seed: int = 42
    log_interval: int = 100
    test_best_after_train: bool = True
    deterministic: bool = False
    compile_model: bool = False

    # MNIST 训练集统计量
    mean: Tuple[float, ...] = (0.1307,)
    std: Tuple[float, ...] = (0.3081,)


# =============================================================================
# 工具
# =============================================================================
def get_device() -> torch.device:
    """自动选择训练设备: CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed: int, deterministic: bool = False) -> None:
    """固定随机种子。deterministic=True 会牺牲部分速度以提高可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def seed_worker(worker_id: int) -> None:
    """DataLoader 多进程 worker 的随机种子初始化。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_checkpoint_path(cfg: Config) -> Path:
    path = Path(cfg.save_dir) / cfg.ckpt_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def autocast_context(device: torch.device):
    """只在 CUDA 上启用 AMP；MPS/CPU 使用普通精度。"""
    if device.type != "cuda":
        return nullcontext()

    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(device: torch.device):
    enabled = device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# =============================================================================
# 数据
# =============================================================================
def build_transforms(cfg: Config):
    train_tf = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.10, 0.10),
            scale=(0.95, 1.05),
            shear=5,
        ),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])

    return train_tf, eval_tf


def build_dataloaders(
    cfg: Config,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """构建 MNIST 训练 / 验证 / 测试 DataLoader。"""
    if not 0.0 < cfg.val_ratio < 1.0:
        raise ValueError(f"val_ratio 必须在 (0, 1) 内，当前为: {cfg.val_ratio}")

    train_tf, eval_tf = build_transforms(cfg)

    # 同一份训练数据使用两套 transform:
    # 训练子集带增强；验证子集不带增强，避免验证指标被随机扰动。
    train_full = torchvision.datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=train_tf,
    )
    val_full = torchvision.datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=eval_tf,
    )
    test_set = torchvision.datasets.MNIST(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=eval_tf,
    )

    n_total = len(train_full)
    n_val = max(1, int(n_total * cfg.val_ratio))

    split_gen = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(n_total, generator=split_gen).tolist()

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_set = Subset(train_full, train_indices)
    val_set = Subset(val_full, val_indices)

    loader_gen = torch.Generator().manual_seed(cfg.seed)

    common: dict[str, Any] = {
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker if cfg.num_workers > 0 else None,
        "generator": loader_gen,
    }

    if cfg.num_workers > 0:
        common["persistent_workers"] = True
        common["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )

    print(f"训练 / 验证 / 测试 样本数: {len(train_set)} / {len(val_set)} / {len(test_set)}")
    return train_loader, val_loader, test_loader


# =============================================================================
# 模型
# =============================================================================
def conv_block(in_ch: int, out_ch: int, dropout2d: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    ]

    if dropout2d > 0:
        layers.append(nn.Dropout2d(dropout2d))

    return nn.Sequential(*layers)


class CNN(nn.Module):
    """MNIST CNN 分类模型。输入 [B, 1, 28, 28]，输出 [B, num_classes]。"""

    def __init__(
        self,
        num_classes: int = 10,
        channels: Tuple[int, int, int] = (32, 64, 128),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        c1, c2, c3 = channels
        dropout2d = dropout * 0.5

        self.features = nn.Sequential(
            conv_block(1, c1, dropout2d),
            conv_block(c1, c2, dropout2d),
            conv_block(c2, c3, dropout2d),
        )

        # 相比固定 Flatten(128*3*3)，AdaptiveAvgPool2d 更轻量，也不依赖中间特征图尺寸。
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="linear",
                )
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# =============================================================================
# 训练 / 评估
# =============================================================================
def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> int:
    return (logits.argmax(dim=1) == labels).sum().item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy_from_logits(logits, labels)
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    cfg: Config,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device):
            logits = model(images)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()

        if cfg.grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy_from_logits(logits.detach(), labels)
        total_samples += batch_size

        if step % cfg.log_interval == 0 or step == len(loader):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:02d} [{step:04d}/{len(loader)}] "
                f"loss={total_loss / total_samples:.4f} "
                f"acc={total_correct / total_samples * 100:.2f}% "
                f"lr={lr:.2e}"
            )

    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    val_loss: float,
    val_acc: float,
    cfg: Config,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    path = Path(path)

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def maybe_compile_model(
    model: nn.Module,
    cfg: Config,
    device: torch.device,
) -> nn.Module:
    if not cfg.compile_model:
        return model

    if device.type != "cuda":
        print("compile_model=True 但当前不是 CUDA 设备，已跳过 torch.compile。")
        return model

    if not hasattr(torch, "compile"):
        print("当前 PyTorch 版本不支持 torch.compile，已跳过。")
        return model

    print("启用 torch.compile。首次 epoch 可能较慢。")
    return torch.compile(model)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> float:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    total_steps = max(1, cfg.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=cfg.lr * 0.01,
    )

    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    eval_loss_fn = nn.CrossEntropyLoss()

    scaler = make_grad_scaler(device)
    ckpt_path = make_checkpoint_path(cfg)

    print("=" * 72)
    print(f"设备: {device} | AMP: {device.type == 'cuda'} | Epochs: {cfg.epochs}")
    print(f"Batch: {cfg.batch_size} | LR: {cfg.lr} | WD: {cfg.weight_decay}")
    print(f"Dropout: {cfg.dropout} | Label smoothing: {cfg.label_smoothing}")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 72)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=train_loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cfg=cfg,
            device=device,
            epoch=epoch,
        )
        val_loss, val_acc = evaluate(model, val_loader, eval_loss_fn, device)

        elapsed = time.perf_counter() - start

        is_better = (
            val_acc > best_val_acc
            or (abs(val_acc - best_val_acc) < 1e-12 and val_loss < best_val_loss)
        )

        msg = (
            f"[Epoch {epoch:02d}] {elapsed:.1f}s | "
            f"train: loss={train_loss:.4f} acc={train_acc * 100:.2f}% | "
            f"val: loss={val_loss:.4f} acc={val_acc * 100:.2f}%"
        )

        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(
                path=ckpt_path,
                model=model,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                cfg=cfg,
            )
            msg += "  [BEST -> saved]"
        else:
            no_improve += 1
            msg += f"  [no-improve: {no_improve}]"

        print(msg)

        if cfg.early_stop_patience is not None and no_improve >= cfg.early_stop_patience:
            print(f"早停: 验证集连续 {cfg.early_stop_patience} 轮未提升。")
            break

    print("=" * 72)
    print(f"训练完成，最佳验证准确率: {best_val_acc * 100:.2f}%")
    print("=" * 72)

    return best_val_acc


# =============================================================================
# 推理
# =============================================================================
@torch.no_grad()
def predict_single_tensor(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> Tuple[int, float]:
    """
    对单张 MNIST Tensor 预测。

    参数:
        image: shape 为 [1, 28, 28] 或 [1, 1, 28, 28]，
               且已完成 ToTensor + Normalize。
    """
    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    if image.dim() != 4 or image.size(1) != 1:
        raise ValueError(
            "输入维度应为 [1, 28, 28] 或 [1, 1, 28, 28]，"
            f"当前为: {tuple(image.shape)}"
        )

    image = image.to(device, non_blocking=True)
    logits = model(image)
    probs = torch.softmax(logits, dim=1)
    confidence, prediction = probs.max(dim=1)

    return prediction.item(), confidence.item()


@torch.no_grad()
def predict_image_file(
    model: nn.Module,
    image_path: str | Path,
    cfg: Config,
    device: torch.device,
) -> Tuple[int, float]:
    """
    对图片文件预测。图片会被转成灰度图并 resize 到 28x28。
    适合推理外部手写数字图片；MNIST 原始样本可直接用 predict_single_tensor。
    """
    _, eval_tf = build_transforms(cfg)
    image = Image.open(image_path).convert("L").resize((28, 28))
    tensor = eval_tf(image)
    return predict_single_tensor(model, tensor, device)


# =============================================================================
# CLI / 主流程
# =============================================================================
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="优化版 PyTorch CNN MNIST 训练脚本")

    parser.add_argument("--data-dir", type=str, default=Config.data_dir)
    parser.add_argument("--save-dir", type=str, default=Config.save_dir)
    parser.add_argument("--ckpt-name", type=str, default=Config.ckpt_name)

    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--test-batch-size", type=int, default=Config.test_batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--label-smoothing", type=float, default=Config.label_smoothing)
    parser.add_argument("--grad-clip", type=float, default=Config.grad_clip)

    parser.add_argument("--val-ratio", type=float, default=Config.val_ratio)
    parser.add_argument("--early-stop-patience", type=int, default=Config.early_stop_patience)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--log-interval", type=int, default=Config.log_interval)

    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--no-test", action="store_true")

    args = parser.parse_args()

    return Config(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        ckpt_name=args.ckpt_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        val_ratio=args.val_ratio,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        seed=args.seed,
        log_interval=args.log_interval,
        deterministic=args.deterministic,
        compile_model=args.compile_model,
        test_best_after_train=not args.no_test,
    )


def main() -> None:
    cfg = parse_args()
    device = get_device()

    set_seed(cfg.seed, deterministic=cfg.deterministic)

    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print("当前配置:")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    train_loader, val_loader, test_loader = build_dataloaders(cfg, device)

    model = CNN(
        num_classes=cfg.num_classes,
        channels=cfg.channels,
        dropout=cfg.dropout,
    ).to(device)

    model = maybe_compile_model(model, cfg, device)

    print(f"可训练参数量: {count_parameters(model) / 1e6:.3f}M")

    train(model, train_loader, val_loader, cfg, device)

    if cfg.test_best_after_train:
        ckpt_path = make_checkpoint_path(cfg)

        if ckpt_path.exists():
            checkpoint = load_checkpoint(ckpt_path, model, device)
            test_loss, test_acc = evaluate(
                model=model,
                loader=test_loader,
                loss_fn=nn.CrossEntropyLoss(),
                device=device,
            )

            print("=" * 72)
            print(f"测试集结果，基于 Epoch {checkpoint['epoch']} 的最佳模型:")
            print(f"  loss = {test_loss:.4f}")
            print(f"  acc  = {test_acc * 100:.2f}%")
            print("=" * 72)
        else:
            print("未找到最佳模型文件，跳过最终测试。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练已被用户手动中断。")