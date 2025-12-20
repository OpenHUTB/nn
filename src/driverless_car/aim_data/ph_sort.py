import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchgeo.datasets import RESISC45  # 无人机/遥感图像数据集
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 1. 配置参数 --------------------------
# 设备配置（优先使用GPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 图像大小（RESISC45原始大小为256x256，这里统一为224x224）
IMG_SIZE = 224
# 批次大小
BATCH_SIZE = 32
# 训练轮数
EPOCHS = 10
# 数据集根目录（自动下载到该路径）
DATA_ROOT = "./datasets/resisc45"
# 训练集占比
TRAIN_RATIO = 0.8

# -------------------------- 2. 数据预处理与加载（适配最新版torchgeo） --------------------------
# 定义数据变换（直接使用torchvision的变换）
# 训练集：数据增强+归一化
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
    transforms.RandomRotation(10),           # 随机旋转±10度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                         std=[0.229, 0.224, 0.225])   # ImageNet标准差
])

# 验证集：仅归一化
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载RESISC45数据集（自动下载，最新版torchgeo的参数调整）
# 1. 先加载完整数据集（无split参数，用transforms=None，后续手动拆分并应用变换）
full_dataset = RESISC45(
    root=DATA_ROOT,
    transforms=None,  # 最新版参数是transforms（复数），先设为None，后续拆分后再应用
    download=True     # 自动下载数据集
)

# 2. 手动拆分训练集和验证集（解决split参数缺失问题）
dataset_size = len(full_dataset)
train_size = int(TRAIN_RATIO * dataset_size)
val_size = dataset_size - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# 3. 封装带变换的数据集类（解决不同子集用不同变换的问题）
class DatasetWithTransform:
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        # 应用变换（img是PIL图像，直接传入transform）
        img = self.transform(img)
        return img, label

# 4. 为训练集和验证集应用各自的变换
train_dataset = DatasetWithTransform(train_subset, train_transform)
val_dataset = DatasetWithTransform(val_subset, val_transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 获取类别名称和数量（45类，如"farmland"（农田）、"road"（道路）、"building"（建筑）等）
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"数据集包含 {num_classes} 个类别，类别列表：{class_names[:10]}...")  # 打印前10个类别

# -------------------------- 3. 可视化1：数据集样本（无人机图像） --------------------------
def visualize_dataset_samples(dataset, class_names, num_samples=3):
    """可视化每个类别的若干样本图像（随机选几个类别，避免子图过多）"""
    # 随机选择8个类别（减少子图数量，更易查看）
    selected_classes = np.random.choice(range(num_classes), 8, replace=False)
    fig, axes = plt.subplots(nrows=len(selected_classes), ncols=num_samples, figsize=(10, 15))
    fig.suptitle("Drone/Remote Sensing Image Samples (RESISC45)", fontsize=16)

    for row_idx, cls_idx in enumerate(selected_classes):
        # 找到该类别的所有样本索引
        cls_samples = [i for i, (img, label) in enumerate(dataset) if label == cls_idx]
        # 随机选num_samples个样本（如果样本不足，用replace=True）
        selected_samples = np.random.choice(cls_samples, num_samples, replace=len(cls_samples)<num_samples)

        for col_idx, sample_idx in enumerate(selected_samples):
            img, _ = dataset[sample_idx]
            img = img.numpy().transpose((1, 2, 0))  # (C, H, W) -> (H, W, C)
            # 反归一化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)

            # 显示图像
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(class_names[cls_idx], fontsize=10)
    plt.tight_layout()
    plt.savefig("drone_dataset_samples.png", dpi=300, bbox_inches="tight")
    plt.show()

# 调用样本可视化函数（用train_dataset，也可以用val_dataset）
visualize_dataset_samples(train_dataset, class_names, num_samples=3)

# -------------------------- 4. 构建迁移学习模型（ResNet18） --------------------------
def build_model(num_classes):
    """构建预训练的ResNet18模型，微调最后一层"""
    model = models.resnet18(pretrained=True)
    # 冻结特征提取层（数据集足够大时可解冻，这里先冻结加快训练）
    for param in model.parameters():
        param.requires_grad = False
    # 替换最后一层全连接层，适配数据集类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

model = build_model(num_classes)

# -------------------------- 5. 模型训练 --------------------------
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 记录训练指标
train_losses = []
train_accs = []
val_losses = []
val_accs = []

def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in loader:  # 元组(inputs, labels)形式
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # 反向传播+优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)

# 开始训练
print(f"\n开始训练（设备：{DEVICE}）...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)

    # 记录指标
    train_losses.append(train_loss)
    train_accs.append(train_acc.item())
    val_losses.append(val_loss)
    val_accs.append(val_acc.item())

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# -------------------------- 6. 可视化2：训练过程（损失/准确率曲线） --------------------------
def visualize_training(train_losses, val_losses, train_accs, val_accs):
    """可视化训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Training Process (Drone Image Classification)", fontsize=16)

    # 损失曲线
    ax1.plot(range(1, EPOCHS+1), train_losses, label="Train Loss", marker="o", color="blue")
    ax1.plot(range(1, EPOCHS+1), val_losses, label="Val Loss", marker="s", color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 准确率曲线
    ax2.plot(range(1, EPOCHS+1), train_accs, label="Train Acc", marker="o", color="blue")
    ax2.plot(range(1, EPOCHS+1), val_accs, label="Val Acc", marker="s", color="red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("drone_training_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

visualize_training(train_losses, val_losses, train_accs, val_accs)

# -------------------------- 7. 可视化3：混淆矩阵（选前10类，避免过于密集） --------------------------
def visualize_confusion_matrix(model, loader, class_names, device, top_n=10):
    """可视化混淆矩阵（仅显示前top_n类）"""
    # 获取预测和标签
    _, _, all_preds, all_labels = validate(model, loader, criterion, device)
    # 选择前top_n类的索引
    top_classes = range(min(top_n, num_classes))
    # 筛选出前top_n类的样本
    mask = np.isin(all_labels, top_classes)
    all_preds_filtered = all_preds[mask]
    all_labels_filtered = all_labels[mask]
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels_filtered, all_preds_filtered)
    # 获取前top_n类的名称
    top_class_names = [class_names[i] for i in top_classes]

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=top_class_names, yticklabels=top_class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix (Top {top_n} Classes)")
    plt.tight_layout()
    plt.savefig("drone_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

visualize_confusion_matrix(model, val_loader, class_names, DEVICE, top_n=10)

# -------------------------- 8. 可视化4：单张图像预测 --------------------------
def predict_single_image(model, img_path, class_names, transform, device):
    """预测单张无人机图像并可视化"""
    from PIL import Image
    # 加载图像
    img = Image.open(img_path).convert("RGB")
    img_original = img.copy()
    # 预处理
    img_tensor = transform(img).unsqueeze(0).to(device)  # 添加batch维度

    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        pred_label = class_names[pred.item()]

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.imshow(img_original)
    plt.axis("off")
    plt.title(f"Predicted: {pred_label}", fontsize=14)
    plt.tight_layout()
    plt.savefig("drone_single_prediction.png", dpi=300, bbox_inches="tight")
    plt.show()

# 示例：使用数据集内的一张图像进行预测（获取数据集图像路径）
# 注意：full_dataset的files属性存储了图像路径（最新版torchgeo仍保留该属性）
sample_img_path = full_dataset.files[0]  # 取数据集第一张图像
predict_single_image(model, sample_img_path, class_names, val_transform, DEVICE)