import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random

# --------------------------
# 1. 基础配置（解决可视化和字体问题）
# --------------------------
import matplotlib
matplotlib.use('TkAgg')  # 更换后端，兼容PyCharm的可视化

# 类别名称（对应CIFAR-10的10类）
classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

# --------------------------
# 2. 数据预处理（简化，减少计算量）
# --------------------------
# 简化变换：移除数据增强（加快训练，牺牲一点泛化能力）
basic_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集（使用简化变换）
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=basic_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=basic_transform)
# 增大批次大小，加快训练（根据内存调整，默认128）
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# --------------------------
# 3. 搭建轻量化CNN模型（原模型，参数少，速度快）
# --------------------------
class DroneCNN(nn.Module):
    def __init__(self):
        super(DroneCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 降低dropout比例，加快计算

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 初始化模型、损失函数、优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DroneCNN().to(device)
criterion = nn.CrossEntropyLoss()
# 使用SGD优化器（比Adam稍快，或保留Adam，影响不大）
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 4. 训练函数（大幅优化速度）
# --------------------------
# 全局开启交互模式，用于实时绘图
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 训练曲线窗口

def plot_training_curve(train_losses, train_accs, test_accs):
    """更新训练曲线窗口，减少绘图开销"""
    ax1.clear()
    ax2.clear()
    # 损失曲线
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_xlabel('Iteration (Batch)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Change')
    ax1.legend()
    ax1.grid(True)
    # 准确率曲线
    ax2.plot(train_accs, label='Training Accuracy', color='green')
    ax2.plot(test_accs, label='Test Accuracy', color='red')
    ax2.set_xlabel('Iteration (Batch)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training/Test Accuracy Change')
    ax2.legend()
    ax2.grid(True)
    plt.draw()
    plt.pause(0.01)  # 减少暂停时间，加快绘图

def calculate_test_acc_fast(test_loader, sample_batches=5):
    """快速计算测试集准确率：只抽取少量批次，不遍历整个测试集"""
    test_correct = 0
    test_total = 0
    model.eval()
    with torch.no_grad():
        # 只取前sample_batches个批次，大幅减少耗时
        for i, (test_inputs, test_labels) in enumerate(test_loader):
            if i >= sample_batches:
                break
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_outputs = model(test_inputs)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()
    model.train()
    if test_total == 0:
        return 0.0
    return 100 * test_correct / test_total

def train_model(epochs=1):  # 减少训练轮数，默认1轮
    train_losses = []
    train_accs = []
    test_accs = []
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        # 每200个批次更新一次曲线（原100，减少更新频率）
        update_interval = 200
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播+反向传播+优化
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计指标
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 降低更新频率，减少计算和绘图开销
            if i % update_interval == update_interval - 1:
                train_loss = running_loss / update_interval
                train_acc = 100 * correct / total
                train_losses.append(train_loss)
                train_accs.append(train_acc)

                # 快速计算测试集准确率（只取5个批次）
                test_acc = calculate_test_acc_fast(test_loader, sample_batches=5)
                test_accs.append(test_acc)

                print(f'Epoch {epoch+1}, Batch {i+1} | Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0

                plot_training_curve(train_losses, train_accs, test_accs)

    # 保存模型
    torch.save(model.state_dict(), 'drone_model.pth')
    print('模型已保存为drone_model.pth')
    model.eval()
    plt.ioff()  # 关闭交互模式
    plt.show(block=False)

# --------------------------
# 5. 模拟无人机实时图像输入（优化推理速度）
# --------------------------
def load_drone_images(folder_path):
    """读取本地文件夹中的图像，模拟无人机采集的图像流"""
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    img_paths = []
    for file in os.listdir(folder_path):
        if os.path.splitext(file)[1].lower() in img_extensions:
            img_paths.append(os.path.join(folder_path, file))
    if not img_paths:
        raise ValueError(f'文件夹{folder_path}中未找到任何图像文件！')
    return img_paths

def preprocess_image(img_path):
    """预处理单张图像，简化操作"""
    # 读取图像（RGB模式）
    img = Image.open(img_path).convert('RGB')
    original_img = img.copy()
    # 预处理
    img = basic_transform(img)
    # 添加batch维度
    img = torch.unsqueeze(img, 0)
    return original_img, img.to(device)

def drone_real_time_inference(folder_path, delay=0.1):  # 降低延迟，默认0.1秒
    """模拟无人机实时图像输入，优化推理速度"""
    print(f'\n开始模拟无人机实时图像流（读取文件夹：{folder_path}），每{delay}秒处理一张图像...')
    img_paths = load_drone_images(folder_path)
    # 创建单个显示窗口，减少窗口创建开销
    fig, ax = plt.subplots(figsize=(6, 4))  # 缩小窗口，加快绘图
    plt.ion()

    for img_path in img_paths:
        try:
            # 预处理图像
            original_img, input_tensor = preprocess_image(img_path)

            # 模型预测（简化，减少计算）
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probabilities, dim=1).item()
                pred_class = classes[pred_idx]
                pred_conf = probabilities[0][pred_idx].item() * 100

            # 可视化优化：只更新图像和文本，不重建窗口
            ax.clear()
            ax.imshow(original_img)
            ax.axis('off')
            # 简化文本显示，减少渲染开销
            text = f'{pred_class} ({pred_conf:.1f}%)'
            ax.text(5, 5, text, fontsize=10, color='red',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax.set_title('Drone Real-Time View', fontsize=12)
            plt.draw()
            plt.pause(delay)

            # 简化控制台输出
            print(f'图像：{os.path.basename(img_path)} → {pred_class} ({pred_conf:.1f}%)')

        except Exception as e:
            print(f'处理图像{img_path}时出错：{e}')
            continue

    # 结束后保持窗口
    plt.ioff()
    ax.text(0.5, 0.5, 'Done!', fontsize=14, ha='center', va='center',
            transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.8))
    plt.draw()
    plt.show(block=True)

# --------------------------
# 主程序运行
# --------------------------
if __name__ == '__main__':
    # 第一步：训练模型（优化后速度大幅提升）
    train_model(epochs=1)  # 可改为2轮，仍比原来快很多

    # 第二步：加载模型（可选）
    # model.load_state_dict(torch.load('drone_model.pth', map_location=device))
    # model.eval()
    # print('模型已加载')

    # 第三步：模拟无人机实时图像输入
    drone_image_folder = r".\driverless_car\data\potoh"
    if not os.path.exists(drone_image_folder):
        os.makedirs(drone_image_folder)
        print(f'已创建文件夹：{drone_image_folder}，请放入测试图片后重新运行！')
    else:
        drone_real_time_inference(drone_image_folder, delay=0.1)  # 低延迟，快速播放