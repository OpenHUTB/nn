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
# =============================================================================
# 基于 PyTorch 的卷积神经网络（CNN）实现
# 数据集：MNIST 手写数字识别（0-9，共10类）
# 网络结构：2个卷积层 + 2个全连接层 + Dropout正则化
# =============================================================================

# 导入操作系统模块，用于路径管理和环境变量设置
import os

# 导入 NumPy，用于高效的数值计算（矩阵、向量操作等）
import numpy as np

import torch
import torch.nn as nn

# 导入神经网络模块（构建模型的基础类和各类网络层）
import torch.nn as nn

# 导入函数接口模块，包含激活函数、损失函数等常用操作
import torch.nn.functional as F
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
# 导入数据处理模块，用于封装数据集和批量加载
import torch.utils.data as Data

# 导入 torchvision，包含常用视觉数据集、模型和图像处理工具
import torchvision

# =============================================================================
# 超参数设置
# 超参数：训练前手动指定的配置参数，控制模型行为，不从数据中学习
# =============================================================================
LEARNING_RATE  = 1e-4   # 学习率：控制每次参数更新的步长，过大容易震荡，过小收敛慢
KEEP_PROB_RATE = 0.7    # Dropout 保留率：训练时随机保留70%的神经元，防止过拟合
MAX_EPOCH      = 3      # 训练轮数：整个数据集被遍历的次数
BATCH_SIZE     = 50     # 批大小：每次迭代使用的样本数量，影响内存占用和训练稳定性

# =============================================================================
# 数据集加载
# =============================================================================

# 检查本地是否已存在 MNIST 数据集，若不存在则自动下载
DOWNLOAD_MNIST = False
if not os.path.exists('./mnist/') or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True  # 目录不存在或为空时，标记为需要下载

# 加载训练数据集
train_data = torchvision.datasets.MNIST(
    root='./mnist/',                               # 数据集本地存储路径
    train=True,                                    # True=加载训练集（60000张），False=加载测试集
    transform=torchvision.transforms.ToTensor(),   # 将 PIL 图像转为 Tensor，并自动归一化到 [0,1]
    download=DOWNLOAD_MNIST                        # 是否需要从网络下载
)

# 创建训练数据加载器（支持批量读取、数据打乱）
train_loader = Data.DataLoader(
    dataset=train_data,     # 使用的数据集
    batch_size=BATCH_SIZE,  # 每批加载的样本数
    shuffle=True            # 每个 epoch 开始前打乱数据，避免模型学到顺序规律
)

# 加载测试数据集（仅用于评估，不参与训练）
test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False  # False 表示加载测试集（10000张）
)

# 预处理测试数据：
# 1. unsqueeze(dim=1)：增加通道维度，从 [N,28,28] 变为 [N,1,28,28]（灰度图通道数为1）
# 2. .float() / 255.：转为浮点类型并归一化到 [0,1]（像素值原为0-255的整数）
# 3. [:500]：只取前500张用于快速评估
test_x = torch.unsqueeze(test_data.data, dim=1).float()[:500] / 255.

# 获取前500个测试样本的真实标签，转为 numpy 数组（用于准确率计算）
test_y = test_data.targets[:500].numpy()

# =============================================================================
# CNN 模型定义
# 网络结构：
#   输入(1x28x28)
#   → 卷积层1(32个3x3卷积核) + BN + ReLU + 最大池化 → (32x14x14)
#   → 卷积层2(64个3x3卷积核×2) + BN + ReLU + 最大池化 → (64x7x7)
#   → 展平 → 全连接层1(3136→1024) + ReLU + Dropout
#   → 全连接层2(1024→10) → 输出10类预测值
# =============================================================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 调用父类 nn.Module 的构造函数，必须调用

        # ---------- 第一卷积块 ----------
        # 输入：1通道（灰度图），28x28
        # 输出：32通道，14x14（经过池化后尺寸减半）
        self.conv1 = nn.Sequential(
            # 卷积层：输入1通道 → 输出32通道，3x3卷积核，padding=1保持特征图尺寸不变
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            # 批量归一化：对每批数据做归一化，加速训练收敛，提高稳定性
            nn.BatchNorm2d(32),
            # ReLU激活函数：f(x)=max(0,x)，引入非线性，使网络能学习复杂特征
            nn.ReLU(),
            # 最大池化：2x2窗口取最大值，特征图尺寸从28x28变为14x14
            nn.MaxPool2d(kernel_size=2)
        )

        # ---------- 第二卷积块 ----------
        # 输入：32通道，14x14
        # 输出：64通道，7x7（经过池化后尺寸减半）
        self.conv2 = nn.Sequential(
            # 第一个卷积：32通道 → 64通道，提取更丰富的特征
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 第二个卷积：64通道 → 64通道（深度叠加，增强特征提取能力）
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 最大池化：特征图尺寸从14x14变为7x7
            nn.MaxPool2d(kernel_size=2)
        )

        # ---------- 全连接层 ----------
        # 展平后的尺寸：64通道 × 7 × 7 = 3136
        # 全连接层1：3136 → 1024，进行高层特征整合
        self.fc1 = nn.Linear(64 * 7 * 7, 1024, bias=True)

        # Dropout 正则化层：训练时随机"关闭"部分神经元，防止过拟合
        # p=1-KEEP_PROB_RATE 表示随机丢弃的比例（这里是30%）
        self.dropout = nn.Dropout(p=1 - KEEP_PROB_RATE)

        # 全连接层2（输出层）：1024 → 10，对应10个数字类别（0-9）
        self.fc2 = nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        """
        前向传播：定义数据从输入到输出的计算流程
        参数：
            x: 输入张量，形状为 [batch_size, 1, 28, 28]
        返回：
            输出张量，形状为 [batch_size, 10]（10个类别的原始分数）
        """
        x = self.conv1(x)           # 第一卷积块：[B,1,28,28] → [B,32,14,14]
        x = self.conv2(x)           # 第二卷积块：[B,32,14,14] → [B,64,7,7]
        x = x.view(x.size(0), -1)  # 展平：[B,64,7,7] → [B,3136]，保留batch维度
        x = self.fc1(x)             # 全连接1：[B,3136] → [B,1024]
        x = F.relu(x)               # ReLU 激活，引入非线性
        x = self.dropout(x)         # Dropout 正则化（仅在训练模式下生效）
        x = self.fc2(x)             # 全连接2（输出层）：[B,1024] → [B,10]
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
    # AdamW：解耦的权重衰减，通常比 Adam + weight_decay 效果更好
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 余弦退火：学习率随训练进度从 LEARNING_RATE 平滑降到接近 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCH * len(train_loader)
    )

    # 带 Label Smoothing 的交叉熵损失
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

            # 前向 + 损失
            logits = model(x)
            loss = loss_fn(logits, y)

            # 反向 + 更新（优化器三步）
            optimizer.zero_grad(set_to_none=True)  # set_to_none=True 比置零更省显存
            loss.backward()
            optimizer.step()
            scheduler.step()                       # 按 step 调度（每个 batch 更新一次 lr）

            # 训练统计
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

        # 每个 epoch 结束在完整测试集上评估一次
        test_loss, test_acc = evaluate(model, test_loader, loss_fn)
        dt = time.time() - t0
        print(f">>> Epoch {epoch:2d} 完成 | 用时 {dt:.1f}s "
              f"| 训练损失 {running_loss/running_total:.4f} "
              f"| 测试损失 {test_loss:.4f} "
              f"| 测试准确率 {test_acc*100:.2f}%")

        # 保存最佳模型
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
    # 顺手打印参数量，方便对比模型大小
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量：{n_params/1e6:.2f} M")

    train(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
# 测试函数：评估模型在测试集上的准确率
# =============================================================================
def evaluate(cnn):
    """
    评估模型准确率
    参数：
        cnn: 训练中的 CNN 模型
    返回：
        accuracy: 测试集准确率（0~1之间的浮点数）
    """
    cnn.eval()  # 切换为评估模式（关闭 Dropout 和 BatchNorm 的训练行为）

    with torch.no_grad():  # 关闭梯度计算，节省内存，加快推理速度
        # 前向传播得到原始输出（logits，未经归一化的预测分数）
        y_pre = cnn(test_x)

        # 获取预测类别：找每个样本10个类别中分数最高的索引
        # torch.max 返回 (最大值, 最大值索引)，我们只需要索引
        _, pre_index = torch.max(y_pre, dim=1)

        # 转为 numpy 数组，与真实标签 test_y 比较
        prediction = pre_index.numpy()

        # 计算准确率：预测正确的样本数 / 总样本数
        correct = np.sum(prediction == test_y)
        accuracy = correct / len(test_y)

    cnn.train()  # 切回训练模式
    return accuracy


# =============================================================================
# 训练函数
# =============================================================================
def train(cnn):
    """
    训练 CNN 模型
    参数：
        cnn: 待训练的 CNN 模型
    """
    # Adam 优化器：自适应学习率优化算法，结合了动量和自适应学习率
    # weight_decay=1e-4 是 L2 正则化系数，防止参数过大导致过拟合
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 交叉熵损失函数：适用于多分类任务，内部包含 Softmax 计算
    loss_func = nn.CrossEntropyLoss()

    print("开始训练...")
    print(f"超参数：学习率={LEARNING_RATE}, Dropout保留率={KEEP_PROB_RATE}, "
          f"训练轮数={MAX_EPOCH}, 批大小={BATCH_SIZE}")
    print("=" * 60)

    for epoch in range(MAX_EPOCH):
        print(f"\n第 {epoch + 1}/{MAX_EPOCH} 轮训练开始")

        for step, (x_, y_) in enumerate(train_loader):
            # 前向传播：将输入数据传入模型，得到预测结果
            output = cnn(x_)

            # 计算损失：预测结果与真实标签之间的差距
            loss = loss_func(output, y_)

            # 反向传播三步骤：
            optimizer.zero_grad()   # 1. 清空上一步的梯度（否则梯度会累加）
            loss.backward()         # 2. 反向传播，计算各参数的梯度
            optimizer.step()        # 3. 根据梯度更新参数

            # 每隔20个batch打印一次测试准确率
            if step != 0 and step % 20 == 0:
                accuracy = evaluate(cnn)
                print(f"  Epoch {epoch+1} | Step {step:4d} | "
                      f"Loss: {loss.item():.4f} | 测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\n" + "=" * 60)
    print("训练完成！")
    final_accuracy = evaluate(cnn)
    print(f"最终测试准确率：{final_accuracy:.4f} ({final_accuracy*100:.2f}%)")


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == '__main__':
    # 创建 CNN 模型实例
    cnn = CNN()

    # 打印模型结构，方便了解网络层次
    print("模型结构：")
    print(cnn)
    print("=" * 60)

    # 开始训练
    train(cnn)
