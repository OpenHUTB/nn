import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time


# 修复1：解决中文字体问题（改用英文显示，避免字体依赖）
# 若需中文，可注释以下行并使用系统自带中文字体，如：
# plt.rcParams["font.family"] = ["Microsoft YaHei", "SimSun", "Arial"]
# plt.rcParams["axes.unicode_minus"] = False
# --------------------------
# 直接使用英文显示类别，避免中文字体问题
classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

# --------------------------
# 修复2：设置Matplotlib后端为TkAgg，解决tostring_rgb报错
# --------------------------
import matplotlib
matplotlib.use('TkAgg')  # 更换后端，兼容PyCharm的可视化

# --------------------------
# 1. 数据预处理与加载（模拟无人机采集的图像数据）
# --------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --------------------------
# 2. 搭建轻量化CNN模型（无人机端深度学习模型）
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
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DroneCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 3. 可视化函数（修复非阻塞显示问题）
# --------------------------
def show_dataset_samples():
    """显示数据集的样本图像（模拟无人机采集的图像）"""
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    images = images / 2 + 0.5  # 反归一化

    plt.figure(figsize=(10, 6))
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.suptitle('Drone Collected Image Samples (CIFAR-10 Simulation)', fontsize=14)
    plt.tight_layout()
    plt.show(block=False)  # 保留非阻塞
    plt.pause(0.1)  # 修复：添加pause，解决后端渲染问题
    # 移除time.sleep，改用plt.pause更稳定

def plot_training_curve(train_losses, train_accs, test_accs):
    """绘制训练损失和准确率曲线"""
    plt.figure(figsize=(12, 4))
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Iteration (Batch)')
    plt.ylabel('Loss')
    plt.title('Training Loss Change')
    plt.legend()
    plt.grid(True)
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Iteration (Batch)')
    plt.ylabel('Accuracy (%)')
    plt.title('Training/Test Accuracy Change')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # 修复：添加pause

def show_predictions():
    """显示模型的预测结果"""
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    images = images / 2 + 0.5  # 反归一化

    plt.figure(figsize=(12, 8))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
        true_label = classes[labels[i]]
        pred_label = classes[predicted[i]]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}', color=color)
        plt.axis('off')
    plt.suptitle('Drone Image Classification Model Predictions', fontsize=14)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # 修复：添加pause

# --------------------------
# 4. 训练模型并实时可视化
# --------------------------
def train_model(epochs=2):
    train_losses = []
    train_accs = []
    test_accs = []
    model.train()

    # 显示数据集样本
    show_dataset_samples()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 每100个批次记录并可视化
            if i % 100 == 99:
                train_loss = running_loss / 100
                train_acc = 100 * correct / total
                train_losses.append(train_loss)
                train_accs.append(train_acc)

                # 计算测试集准确率
                test_correct = 0
                test_total = 0
                with torch.no_grad():
                    for test_inputs, test_labels in test_loader:
                        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                        test_outputs = model(test_inputs)
                        _, test_predicted = torch.max(test_outputs.data, 1)
                        test_total += test_labels.size(0)
                        test_correct += (test_predicted == test_labels).sum().item()
                test_acc = 100 * test_correct / test_total
                test_accs.append(test_acc)

                print(f'Epoch {epoch+1}, Batch {i+1} | Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0

                # 实时绘制曲线
                plot_training_curve(train_losses, train_accs, test_accs)

    # 显示预测结果
    show_predictions()
    # 修复：最后添加plt.show(block=True)，防止窗口闪退
    plt.show(block=True)
    print('Training Finished!')

# --------------------------
# 运行主程序
# --------------------------
if __name__ == '__main__':
    train_model(epochs=2)
