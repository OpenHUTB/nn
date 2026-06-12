"""
碰撞预测模型训练
使用 CNN 对深度图像进行二分类：安全(0) vs 危险(1)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from sklearn.model_selection import train_test_split


# 配置参数
DATA_DIR = os.path.join(os.path.dirname(__file__), "collision_dataset", "depth")
LABELS_FILE = os.path.join(os.path.dirname(__file__), "collision_dataset", "labels.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "collision_model.pth")

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
IMG_SIZE = 64  # 统一图像大小


class CollisionDataset(Dataset):
    """碰撞数据集"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 加载深度图像
        image = Image.open(img_path).convert('L')  # 灰度图

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


class CollisionCNN(nn.Module):
    """碰撞预测 CNN 模型"""

    def __init__(self):
        super(CollisionCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv1: 1x64x64 -> 16x32x32
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv2: 16x32x32 -> 32x16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv3: 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv4: 64x8x8 -> 128x4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # 输出 logits，BCEWithLogitsLoss 会自动 sigmoid
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_data():
    """加载数据集"""
    print("📂 加载数据...")
    print(f"   标签文件: {LABELS_FILE}")

    # 读取标签文件（没有表头，手动指定）
    df = pd.read_csv(LABELS_FILE, header=None,
                     names=['filename', 'label', 'risk', 'min_depth', 'mean_depth', 'pos_x', 'pos_y'])
    print(f"   总样本数: {len(df)}")

    # 统计标签分布
    safe_count = (df['label'] == 0).sum()
    danger_count = (df['label'] == 1).sum()
    print(f"   安全样本: {safe_count}, 危险样本: {danger_count}")

    # 构建图像路径（文件名需要加 _depth 后缀）
    image_dir = os.path.join(os.path.dirname(__file__), "collision_dataset", "depth")
    print(f"   图像目录: {image_dir}")

    df['full_path'] = df['filename'].apply(lambda x: os.path.join(image_dir, f"{x}_depth.png"))

    # 过滤存在的文件
    valid_df = df[df['full_path'].apply(os.path.exists)]
    print(f"   有效样本: {len(valid_df)}")

    if len(valid_df) == 0:
        print("❌ 未找到数据文件！请先运行 auto_collision_collector.py 采集数据")
        return None, None, None, None

    image_paths = valid_df['full_path'].values
    labels = valid_df['label'].values

    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 过采样：复制危险样本使两类数量相等（更好的平衡）
    safe_idx = np.where(y_train == 0)[0]
    danger_idx = np.where(y_train == 1)[0]

    if len(danger_idx) > 0 and len(danger_idx) < len(safe_idx):
        # 计算需要复制多少次才能让危险样本数量等于安全样本
        oversample_times = len(safe_idx) // len(danger_idx)
        remainder = len(safe_idx) % len(danger_idx)
        
        # 复制整数倍
        X_danger_oversampled = np.repeat(X_train[danger_idx], oversample_times, axis=0)
        y_danger_oversampled = np.repeat(y_train[danger_idx], oversample_times, axis=0)
        
        # 如果有余数，随机抽取剩余数量
        if remainder > 0:
            random_indices = np.random.choice(danger_idx, remainder, replace=False)
            X_danger_remainder = X_train[random_indices]
            y_danger_remainder = y_train[random_indices]
            X_danger_oversampled = np.concatenate([X_danger_oversampled, X_danger_remainder])
            y_danger_oversampled = np.concatenate([y_danger_oversampled, y_danger_remainder])
        
        # 合并：保留全部安全样本 + 过采样后的危险样本
        X_train = np.concatenate([X_train[safe_idx], X_danger_oversampled])
        y_train = np.concatenate([y_train[safe_idx], y_danger_oversampled])
        
        print(f"   过采样后训练集: {len(X_train)} (安全: {len(safe_idx)}, 危险: {len(X_danger_oversampled)})")


def train_model():
    """训练模型"""
    # 加载数据
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return

    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),           # ±15度旋转
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 平移10%
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 测试数据不变（仅resize和归一化）
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


    # 创建数据集
    train_dataset = CollisionDataset(X_train, y_train, train_transform)  # 用增强
    test_dataset = CollisionDataset(X_test, y_test, test_transform)      # 用原始

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  使用设备: {device}")

    model = CollisionCNN().to(device)
    print(model)

    # 损失函数和优化器 - 使用类别权重处理数据不平衡
    # 危险样本少，给它更高权重
    n_safe = (y_train == 0).sum()
    n_danger = (y_train == 1).sum()
    pos_weight = torch.tensor([n_safe / n_danger]).to(device)  # 危险样本权重更高
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    print(f"\n🚀 开始训练 (Epochs: {EPOCHS})...")
    best_acc = 0

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total

        # 测试阶段
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                predicted = (outputs > 0.5).float()
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)

        test_acc = 100 * test_correct / test_total

        print(f"Epoch [{epoch + 1}/{EPOCHS}] "
              f"Train Loss: {train_loss / len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss / len(test_loader):.4f} "
              f"Test Acc: {test_acc:.2f}%")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"   💾 保存最佳模型 (Acc: {test_acc:.2f}%)")

    print(f"\n✅ 训练完成！最佳测试准确率: {best_acc:.2f}%")
    print(f"📁 模型保存至: {MODEL_PATH}")

    return model


def evaluate_model():
    """评估模型"""
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CollisionCNN().to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"✅ 加载已有模型: {MODEL_PATH}")
    else:
        print("❌ 未找到训练好的模型")
        return

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_dataset = CollisionDataset(X_test, y_test, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 评估
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            # 使用 sigmoid 将 logits 转为概率，再用阈值判断
            predicted = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_preds.extend(predicted)
            all_labels.extend(labels.numpy())

    # 计算各项指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100

    # 混淆矩阵
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 40)
    print("📊 模型评估结果")
    print("=" * 40)
    print(f"准确率 (Accuracy): {accuracy:.2f}%")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print("-" * 40)
    print("混淆矩阵:")
    print(f"  真阴性(TN): {tn:3d}  假阳性(FP): {fp:3d}")
    print(f"  假阴性(FN): {fn:3d}  真阳性(TP): {tp:3d}")
    print("=" * 40)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="仅评估模型")
    args = parser.parse_args()

    if args.eval:
        evaluate_model()
    else:
        train_model()
