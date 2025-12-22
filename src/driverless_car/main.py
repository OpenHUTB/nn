# 导入必要的库
import torch  # PyTorch核心库，用于张量计算和模型构建
import torch.nn as nn  # 神经网络模块，包含各种层和损失函数
import torch.optim as optim  # 优化器模块，用于模型参数更新
from torch.utils.data import DataLoader  # 数据加载器，用于批量加载数据集
from torchvision import datasets, transforms  # 内置数据集和图像变换工具
import matplotlib.pyplot as plt  # 可视化库，用于绘制训练曲线和图像
import numpy as np  # 数值计算库（本代码中暂未直接使用，保留备用）
import os  # 操作系统库，用于文件和文件夹操作
from PIL import Image  # 图像处理库，用于读取和处理图像
import random  # 随机数库（本代码中暂未直接使用，保留备用）

# --------------------------
# 1. 基础配置（解决可视化和字体问题）
# --------------------------
import matplotlib
# 更换Matplotlib后端为TkAgg，兼容PyCharm等IDE的可视化显示（避免图像无法弹出）
matplotlib.use('TkAgg')

# 类别名称（对应CIFAR-10数据集的10个类别，可替换为无人机图像的自定义类别）
# 注：若用于真实无人机图像分类，需将此处改为无人机场景类别（如农田、道路、建筑等）
classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

# --------------------------
# 2. 数据预处理（简化，减少计算量）
# --------------------------
# 定义图像预处理流水线：将图像统一尺寸、转换为张量、归一化
# 简化变换：移除数据增强（如随机翻转、旋转），加快训练速度，牺牲少量泛化能力
basic_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 将图像缩放到32x32（CIFAR-10原始尺寸就是32x32，此处为统一处理）
    transforms.ToTensor(),  # 将PIL图像转换为PyTorch张量，形状为(C, H, W)，数值范围0-1
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化：将数值范围调整为-1到1，参数为RGB三通道的均值和标准差
])

# 加载CIFAR-10数据集（若用于无人机图像分类，可替换为自定义数据集）
# root：数据集保存路径；train=True表示加载训练集；download=True表示若本地无数据集则自动下载
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=basic_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=basic_transform)

# 创建数据加载器，批量加载数据
# batch_size=128：每次加载128个样本，增大批次可加快训练（需根据内存调整）
# shuffle=True：训练集打乱顺序，提升模型泛化能力；测试集无需打乱
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# --------------------------
# 3. 搭建轻量化CNN模型（参数少，速度快，适配无人机端的低算力场景）
# --------------------------
class DroneCNN(nn.Module):
    """
    轻量化卷积神经网络模型，适用于无人机图像分类（低算力设备）
    结构：3层卷积 + 3层池化 + 2层全连接
    """
    def __init__(self):
        super(DroneCNN, self).__init__()  # 继承父类nn.Module的初始化
        # 卷积层1：输入通道3（RGB），输出通道32，卷积核3x3，填充1（保持尺寸不变）
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # 卷积层2：输入通道32，输出通道64，卷积核3x3，填充1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # 卷积层3：输入通道64，输出通道128，卷积核3x3，填充1
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # 最大池化层：核2x2，步长2（将图像尺寸缩小一半）
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层1：输入维度128*4*4（卷积池化后的数据维度），输出维度512
        # 计算过程：32x32 → 池化后16x16 → 池化后8x8 → 池化后4x4，故128*4*4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        # 全连接层2：输入维度512，输出维度10（对应CIFAR-10的10个类别）
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()  # 激活函数，引入非线性
        self.dropout = nn.Dropout(0.3)  # Dropout层：随机丢弃30%的神经元，防止过拟合（降低比例加快计算）

    def forward(self, x):
        """
        前向传播函数：定义数据在模型中的流动路径
        :param x: 输入张量，形状为(batch_size, 3, 32, 32)
        :return: 输出张量，形状为(batch_size, 10)（每个样本的类别得分）
        """
        # 卷积1 → ReLU → 池化：32x32 → 16x16
        x = self.pool(self.relu(self.conv1(x)))
        # 卷积2 → ReLU → 池化：16x16 → 8x8
        x = self.pool(self.relu(self.conv2(x)))
        # 卷积3 → ReLU → 池化：8x8 → 4x4
        x = self.pool(self.relu(self.conv3(x)))
        # 展平：将4维张量(batch_size, 128, 4, 4)转换为2维张量(batch_size, 128*4*4)
        x = x.view(-1, 128 * 4 * 4)  # -1表示自动计算batch_size
        # 全连接1 → ReLU
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # 应用Dropout
        x = self.fc2(x)  # 全连接2，输出类别得分
        return x

# 设备配置：优先使用GPU（CUDA），无GPU则使用CPU（适配无人机的CPU场景）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化模型并移至指定设备
model = DroneCNN().to(device)
# 损失函数：交叉熵损失（适用于多分类任务，包含Softmax操作）
criterion = nn.CrossEntropyLoss()
# 优化器：Adam优化器（自适应学习率，收敛速度快），学习率0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 4. 训练函数（大幅优化速度，适配实时训练需求）
# --------------------------
# 全局开启Matplotlib交互模式，用于实时绘制训练曲线（无需等待绘图完成）
plt.ion()
# 创建1行2列的子图，用于显示损失曲线和准确率曲线，设置窗口大小
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 训练曲线窗口

def plot_training_curve(train_losses, train_accs, test_accs):
    """
    更新训练曲线窗口，减少绘图开销（只清空重绘，不重建窗口）
    :param train_losses: 训练损失列表
    :param train_accs: 训练准确率列表
    :param test_accs: 测试准确率列表
    """
    ax1.clear()  # 清空第一个子图
    ax2.clear()  # 清空第二个子图
    # 绘制训练损失曲线
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_xlabel('Iteration (Batch)')  # x轴：批次迭代次数
    ax1.set_ylabel('Loss')  # y轴：损失值
    ax1.set_title('Training Loss Change')  # 标题
    ax1.legend()  # 显示图例
    ax1.grid(True)  # 显示网格
    # 绘制训练/测试准确率曲线
    ax2.plot(train_accs, label='Training Accuracy', color='green')
    ax2.plot(test_accs, label='Test Accuracy', color='red')
    ax2.set_xlabel('Iteration (Batch)')  # x轴：批次迭代次数
    ax2.set_ylabel('Accuracy (%)')  # y轴：准确率（百分比）
    ax2.set_title('Training/Test Accuracy Change')  # 标题
    ax2.legend()  # 显示图例
    ax2.grid(True)  # 显示网格
    plt.draw()  # 重绘图像
    plt.pause(0.01)  # 暂停极短时间，让图像更新（减少暂停时间加快绘图）

def calculate_test_acc_fast(test_loader, sample_batches=5):
    """
    快速计算测试集准确率：只抽取少量批次，不遍历整个测试集（大幅减少耗时）
    :param test_loader: 测试集数据加载器
    :param sample_batches: 抽取的批次数量，默认5
    :return: 测试集准确率（百分比）
    """
    test_correct = 0  # 正确预测的样本数
    test_total = 0  # 总样本数
    model.eval()  # 将模型设置为评估模式（关闭Dropout、BatchNorm等层的训练行为）
    with torch.no_grad():  # 关闭梯度计算（节省内存，加快计算）
        # 只遍历前sample_batches个批次
        for i, (test_inputs, test_labels) in enumerate(test_loader):
            if i >= sample_batches:
                break  # 达到指定批次后停止
            # 将数据移至指定设备
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_outputs = model(test_inputs)  # 模型预测
            # 获取预测结果：取类别得分最大的索引
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)  # 累计样本数
            test_correct += (test_predicted == test_labels).sum().item()  # 累计正确数
    model.train()  # 将模型恢复为训练模式
    if test_total == 0:
        return 0.0
    return 100 * test_correct / test_total  # 计算准确率（百分比）

def train_model(epochs=1):  # 训练轮数默认1轮（减少轮数加快训练，可根据需求调整）
    """
    模型训练函数，优化训练速度，实时绘制训练曲线
    :param epochs: 训练轮数，默认1
    """
    train_losses = []  # 存储训练损失
    train_accs = []  # 存储训练准确率
    test_accs = []  # 存储测试准确率
    model.train()  # 将模型设置为训练模式

    for epoch in range(epochs):  # 遍历训练轮数
        running_loss = 0.0  # 累计批次损失
        correct = 0  # 累计正确预测数
        total = 0  # 累计样本数
        # 每200个批次更新一次曲线（原100，减少更新频率，降低绘图开销）
        update_interval = 200
        # 遍历训练集数据加载器
        for i, (inputs, labels) in enumerate(train_loader):
            # 将数据移至指定设备
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播+反向传播+优化
            optimizer.zero_grad()  # 清空梯度（避免梯度累积）
            outputs = model(inputs)  # 模型前向传播，得到类别得分
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 优化器更新模型参数

            # 统计指标
            running_loss += loss.item()  # 累计损失值（item()将张量转换为标量）
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            total += labels.size(0)  # 累计样本数
            correct += (predicted == labels).sum().item()  # 累计正确数

            # 达到更新间隔时，计算并记录指标，更新曲线
            if i % update_interval == update_interval - 1:
                train_loss = running_loss / update_interval  # 计算平均损失
                train_acc = 100 * correct / total  # 计算训练准确率
                train_losses.append(train_loss)
                train_accs.append(train_acc)

                # 快速计算测试集准确率（只取5个批次）
                test_acc = calculate_test_acc_fast(test_loader, sample_batches=5)
                test_accs.append(test_acc)

                # 打印训练信息
                print(f'Epoch {epoch+1}, Batch {i+1} | Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')
                # 重置统计变量
                running_loss = 0.0
                correct = 0
                total = 0

                # 更新训练曲线
                plot_training_curve(train_losses, train_accs, test_accs)

    # 训练完成后保存模型参数（后续可加载模型进行推理）
    torch.save(model.state_dict(), 'drone_model.pth')
    print('模型已保存为drone_model.pth')
    model.eval()  # 模型设置为评估模式
    plt.ioff()  # 关闭交互模式
    plt.show(block=False)  # 显示曲线窗口，不阻塞程序

# --------------------------
# 5. 模拟无人机实时图像输入（优化推理速度，适配无人机实时流处理）
# --------------------------
def load_drone_images(folder_path):
    """
    读取本地文件夹中的图像，模拟无人机采集的图像流
    :param folder_path: 图像文件夹路径
    :return: 图像路径列表
    """
    # 支持的图像扩展名
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    img_paths = []
    # 遍历文件夹中的文件
    for file in os.listdir(folder_path):
        # 判断文件扩展名是否在支持的列表中
        if os.path.splitext(file)[1].lower() in img_extensions:
            img_paths.append(os.path.join(folder_path, file))  # 拼接完整路径
    if not img_paths:
        raise ValueError(f'文件夹{folder_path}中未找到任何图像文件！')
    return img_paths

def preprocess_image(img_path):
    """
    预处理单张图像，适配模型输入要求
    :param img_path: 图像路径
    :return: 原始图像（用于显示），预处理后的张量（用于模型推理）
    """
    # 读取图像（RGB模式，避免灰度图问题）
    img = Image.open(img_path).convert('RGB')
    original_img = img.copy()  # 保存原始图像，用于可视化显示
    # 应用预处理流水线
    img = basic_transform(img)
    # 添加batch维度（模型输入要求batch_size在前，形状为(1, 3, 32, 32)）
    img = torch.unsqueeze(img, 0)
    return original_img, img.to(device)

def drone_real_time_inference(folder_path, delay=0.1):
    """
    模拟无人机实时图像输入，优化推理速度，实时显示预测结果
    :param folder_path: 无人机图像文件夹路径
    :param delay: 每张图像的显示延迟（秒），默认0.1秒（低延迟，快速播放）
    """
    print(f'\n开始模拟无人机实时图像流（读取文件夹：{folder_path}），每{delay}秒处理一张图像...')
    img_paths = load_drone_images(folder_path)
    # 创建单个显示窗口，减少窗口创建开销（优化可视化速度）
    fig, ax = plt.subplots(figsize=(6, 4))  # 缩小窗口，加快绘图
    plt.ion()  # 开启交互模式

    # 遍历图像路径，模拟实时图像流
    for img_path in img_paths:
        try:
            # 预处理图像
            original_img, input_tensor = preprocess_image(img_path)

            # 模型预测（推理阶段，关闭梯度计算）
            with torch.no_grad():
                outputs = model(input_tensor)  # 模型前向传播，得到类别得分
                probabilities = torch.softmax(outputs, dim=1)  # 将得分转换为概率（0-1）
                pred_idx = torch.argmax(probabilities, dim=1).item()  # 获取概率最大的类别索引
                pred_class = classes[pred_idx]  # 获取类别名称
                pred_conf = probabilities[0][pred_idx].item() * 100  # 计算预测置信度（百分比）

            # 可视化优化：只更新图像和文本，不重建窗口
            ax.clear()  # 清空窗口
            ax.imshow(original_img)  # 显示原始图像
            ax.axis('off')  # 关闭坐标轴，美观显示
            # 显示预测结果和置信度，添加白色背景框增强可读性
            text = f'{pred_class} ({pred_conf:.1f}%)'
            ax.text(5, 5, text, fontsize=10, color='red',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax.set_title('Drone Real-Time View', fontsize=12)  # 窗口标题
            plt.draw()  # 重绘图像
            plt.pause(delay)  # 暂停指定时间，模拟实时流

            # 简化控制台输出，只显示关键信息
            print(f'图像：{os.path.basename(img_path)} → {pred_class} ({pred_conf:.1f}%)')

        except Exception as e:
            # 捕获异常，避免单张图像处理失败导致程序终止
            print(f'处理图像{img_path}时出错：{e}')
            continue

    # 推理完成后，保持窗口显示
    plt.ioff()  # 关闭交互模式
    # 在窗口中显示完成提示
    ax.text(0.5, 0.5, 'Done!', fontsize=14, ha='center', va='center',
            transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.8))
    plt.draw()
    plt.show(block=True)  # 阻塞程序，保持窗口

# --------------------------
# 主程序运行入口
# --------------------------
if __name__ == '__main__':
    # 第一步：训练模型（优化后速度大幅提升，默认1轮）
    # 可根据需求改为2轮，仍比原始版本快很多
    train_model(epochs=1)

    # 第二步：加载模型（可选，若已训练过模型，可直接加载无需重新训练）
    # model.load_state_dict(torch.load('drone_model.pth', map_location=device))
    # model.eval()
    # print('模型已加载')

    # 第三步：模拟无人机实时图像输入
    # 无人机图像文件夹路径（相对路径，可替换为绝对路径）
    drone_image_folder = r".\driverless_car\data\potoh"
    # 若文件夹不存在，则创建并提示用户放入图片
    if not os.path.exists(drone_image_folder):
        os.makedirs(drone_image_folder)
        print(f'已创建文件夹：{drone_image_folder}，请放入测试图片后重新运行！')
    else:
        # 执行实时推理
        drone_real_time_inference(drone_image_folder, delay=0.1)