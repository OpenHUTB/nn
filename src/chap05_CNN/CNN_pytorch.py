#!/usr/bin/env python
# coding: utf-8

# 导入必要的库
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision               # 包含常用的数据集和模型
import torch.nn.functional as F  # 包含常用的函数式API，如ReLU, softmax等
import numpy as np
from torch.autograd import Variable

# 设置超参数
learning_rate = 1e-4  #  学习率
keep_prob_rate = 0.7  #  Dropout保留神经元的比例
max_epoch = 3         # 训练的总轮数
BATCH_SIZE = 50       # 每批训练数据的大小为50

# 检查是否需要下载 MNIST 数据集
DOWNLOAD_MNIST = False
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # 如果不存在 mnist 目录或者目录为空，则需要下载
    DOWNLOAD_MNIST = True

# 加载训练数据集
train_data = torchvision.datasets.MNIST(
    root='./mnist/',                              # 数据集保存路径
    train=True,                                   # 加载训练集
    transform=torchvision.transforms.ToTensor(),  # 将图像转换为 Tensor 并归一化到[0,1]
    download=DOWNLOAD_MNIST                       # 如果需要则下载
)

# 创建数据加载器，用于批量加载数据
train_loader = Data.DataLoader(
    dataset=train_data,     # 使用的数据集
    batch_size=BATCH_SIZE,  # 每批数据量
    shuffle=True            # 是否打乱数据
)

# 加载测试数据集
# torchvision.datasets.MNIST用于加载 MNIST 数据集
# root='./mnist/'指定数据集的存储路径
# train=False表示加载测试集（而不是训练集）
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# 预处理测试数据：转换为 Variable ，调整维度，归一化，只取前500个样本
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:500]/255.
# 获取测试集的标签（前500个），并转换为 numpy 数组
test_y = test_data.test_labels[:500].numpy()

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()                                # 调用父类构造函数
        
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 3x3卷积核
            nn.BatchNorm2d(32),                                    # 添加批量归一化
            nn.ReLU(),                                             # ReLU激活函数，引入非线性
            nn.MaxPool2d(2)                                        # 最大池化，减小特征图尺寸
        )
        
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 3x3卷积核
            nn.BatchNorm2d(64),                                     # 添加批量归一化
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 增加一层3x3卷积
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)                                         # 最大池化，减小特征图尺寸
        )
        
        # 第一个全连接层：输入是7*7*64=3136（两次池化后图像尺寸变为7x7），输出1024维
        self.out1 = nn.Linear(7*7*64, 1024, bias=True)
        
        # Dropout层：训练时随机丢弃神经元，防止过拟合
        self.dropout = nn.Dropout(keep_prob_rate)
        
        # 第二个全连接层：1024维输入，10维输出（对应10个数字类别）
        self.out2 = nn.Linear(1024, 10, bias=True)

    #定义了一个神经网络的前向传播过程，进行特征提取和分类预测
    def forward(self, x):
        x = self.conv1(x)          # 第一卷积层特征提取，输入 -> 卷积 -> 激活 (ReLU由self.conv1定义)
        x = self.conv2(x)          # 第二卷积层特征提取，特征图 -> 卷积 -> 激活
        x = x.view(x.size(0), -1)  # 展平张量：保留批量维度，合并其他所有维度
        out1 = self.out1(x)        # 第一个全连接层 + 激活函数，线性变换: [B, in_features] -> [B, hidden_features]
        out1 = F.relu(out1)        # 应用ReLU激活函数引入非线性
        out1 = self.dropout(out1)  # 应用dropout正则化，随机丢弃部分神经元输出
        out2 = self.out2(out1)     # 将上一层输出out1传递给当前层self.out2进行处理
        return out2

# 测试函数 - 评估模型在测试集上的准确率
def test(cnn):
    global prediction  # 使用全局变量prediction保存预测结果
    
    # 模型预测：输入测试数据，得到原始输出logits（未归一化的预测值）
    y_pre = cnn(test_x)  
    
    # 计算softmax概率分布（将logits转换为概率值，dim=1表示对类别维度做归一化）
    y_prob = F.softmax(y_pre, dim=1)
    
    # 获取预测类别：找到每个样本概率最大的类别索引
    # torch.max返回(最大值, 最大值的索引)
    _, pre_index = torch.max(y_pre, 1)  
    
    # 调整张量形状为1维向量（例如从[N,1]变为[N]）
    pre_index = pre_index.view(-1)
    
    # 将预测结果从PyTorch张量转换为numpy数组
    prediction = pre_index.data.numpy()
    
    # 计算正确预测的数量（预测值与真实标签test_y比较）
    correct = np.sum(prediction == test_y)
    
    # 返回准确率（假设测试集共500个样本）
    return correct / 500.0  


# 训练函数
def train(cnn):
    # 使用Adam优化器，学习率为learning_rate
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    # 使用交叉熵损失函数
    loss_func = nn.CrossEntropyLoss()

    # 训练max_epoch轮
    for epoch in range(max_epoch):
        # 遍历训练数据加载器
        for step, (x_, y_) in enumerate(train_loader):
            # 将数据转换为Variable（自动求导需要）
            x, y = Variable(x_), Variable(y_)
            output = cnn(x)                         # 前向传播得到预测结果
            loss = loss_func(output, y)             # 计算损失
            optimizer.zero_grad(set_to_none=True)   # 清空模型参数的梯度缓存，set_to_none=True可减少内存占用
            loss.backward()                         # 反向传播计算梯度
            optimizer.step()                        # 更新参数

            # 每20个batch打印一次测试准确率
            if step != 0 and step % 20 == 0:
                print("=" * 10, step, "=" * 5, "=" * 5, "测试准确率: ", test(cnn), "=" * 10)
# 主程序入口
if __name__ == '__main__':
    cnn = CNN()  # 创建CNN实例
    train(cnn)   # 开始训练
