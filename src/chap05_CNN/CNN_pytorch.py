#!/usr/bin/env python
# coding: utf-8

# 导入必要的库
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision  # 包含常用的数据集和模型
import torch.nn.functional as F  # 包含常用的函数式API，如ReLU, softmax等
import numpy as np

# 设置超参数
learning_rate = 1e-4  # 学习率
keep_prob_rate = 0.7  # Dropout保留神经元的比例
max_epoch = 3  # 训练的总轮数
BATCH_SIZE = 50  # 每批训练数据的大小

# 检查是否需要下载MNIST数据集
DOWNLOAD_MNIST = False
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # 如果不存在mnist目录或者目录为空，则需要下载
    DOWNLOAD_MNIST = True

# 加载训练数据集
train_data = torchvision.datasets.MNIST(
    root='./mnist/',  # 数据集保存路径
    train=True,  # 加载训练集
    transform=torchvision.transforms.ToTensor(),  # 将图像转换为Tensor并归一化到[0,1]
    download=DOWNLOAD_MNIST  # 如果需要则下载
)

# 创建数据加载器，用于批量加载数据
train_loader = Data.DataLoader(
    dataset=train_data,  # 使用的数据集
    batch_size=BATCH_SIZE,  # 每批数据量
    shuffle=True  # 是否打乱数据
)

# 加载测试数据集
# torchvision.datasets.MNIST用于加载MNIST数据集
# root='./mnist/'指定数据集的存储路径
# train=False表示加载测试集（而不是训练集）
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# 预处理测试数据：转换为Variable，调整维度，归一化，只取前500个样本
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:500]/255.
# 获取测试集的标签（前500个），并转换为numpy数组
test_y = test_data.test_labels[:500].numpy()

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 调用父类构造函数
        
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            # 卷积层：1个输入通道，32个输出通道，7x7的卷积核
            # stride=1表示步长为1，padding=3表示边缘填充3层（保持尺寸不变）
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(2)  # 2x2的最大池化，尺寸减半
        )
        
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            # 卷积层：32个输入通道，64个输出通道，5x5的卷积核
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(2)  # 2x2的最大池化，尺寸再减半
        )
        
        # 第一个全连接层：输入是7*7*64=3136（两次池化后图像尺寸变为7x7），输出1024维
        self.out1 = nn.Linear(7*7*64, 1024, bias=True)
        
        # Dropout层：训练时随机丢弃神经元，防止过拟合
        self.dropout = nn.Dropout(keep_prob_rate)
        
        # 第二个全连接层：1024维输入，10维输出（对应10个数字类别）
        self.out2 = nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        # 前向传播过程
        x = self.conv1(x)  # 通过第一个卷积层
        x = self.conv2(x)  # 通过第二个卷积层
        x = x.view(x.size(0), -1)  # 将特征图展平为一维向量，保留batch维度
        out1 = self.out1(x)  # 第一个全连接层
        out1 = F.relu(out1)  # ReLU激活
        out1 = self.dropout(out1)  # Dropout
        out2 = self.out2(out1)  # 第二个全连接层
        output = F.softmax(out2)  # Softmax得到概率分布
        return output

# 测试函数
def test(cnn):
    global prediction  # 声明全局变量
    y_pre = cnn(test_x)  # 用模型预测测试数据
    _, pre_index = torch.max(y_pre, 1)  # 获取预测类别（最大概率的索引）
    pre_index = pre_index.view(-1)  # 调整张量形状
    prediction = pre_index.data.numpy()  # 转换为numpy数组
    correct = np.sum(prediction == test_y)  # 计算正确预测的数量
    return correct / 500.0  # 返回准确率

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
            output = cnn(x)  # 前向传播得到预测结果
            loss = loss_func(output, y)  # 计算损失
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零方式
            loss.backward()
            optimizer.step()
            
            # 每20个batch打印一次测试准确率
            if step != 0 and step % 20 == 0:
                print("=" * 10, step, "="*5, "="*5, "测试准确率: ", test(cnn), "=" * 10)

# 主程序入口
if __name__ == '__main__':
    cnn = CNN()  # 创建CNN实例
    train(cnn)  # 开始训练
