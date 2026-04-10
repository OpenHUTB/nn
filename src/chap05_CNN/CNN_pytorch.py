#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# 基于 PyTorch 的卷积神经网络（CNN）实现
# 数据集：MNIST 手写数字识别（0-9，共10类）
# 网络结构：2个卷积层 + 2个全连接层 + Dropout正则化
# =============================================================================

# 导入操作系统模块，用于路径管理和环境变量设置
import os

# 导入 NumPy，用于高效的数值计算（矩阵、向量操作等）
import numpy as np

# 导入 PyTorch 主库
import torch

# 导入神经网络模块（构建模型的基础类和各类网络层）
import torch.nn as nn

# 导入函数接口模块，包含激活函数、损失函数等常用操作
import torch.nn.functional as F

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