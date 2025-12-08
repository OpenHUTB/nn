import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 路径设置
base_dir = "./data"  # 修改为你的数据目录
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# 检查路径是否存在
print("=" * 50)
print("路径检查:")
print(f"基础目录: {base_dir}, 存在: {os.path.exists(base_dir)}")
print(f"训练目录: {train_dir}, 存在: {os.path.exists(train_dir)}")
print(f"测试目录: {test_dir}, 存在: {os.path.exists(test_dir)}")
print("=" * 50)

# 参数配置
img_size = (128, 128)
batch_size = 32
epochs = 70
num_classes = 0  # 将在后面根据数据确定

# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        print(f"\n正在初始化数据集: {data_dir}")
        
        # 检查数据目录是否存在
        if not os.path.exists(data_dir):
            print(f"错误: 数据目录 {data_dir} 不存在!")
            return
        
        # 遍历目录获取图像和标签
        classes = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
        
        if not classes:
            print(f"警告: 在 {data_dir} 中没有找到任何类别文件夹!")
            return
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        print(f"找到类别: {self.class_to_idx}")
        
        total_images = 0
        for class_name in classes:
            class_dir = os.path.join(data_dir, class_name)
            class_images = []
            
            # 获取所有支持的图像文件
            supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(supported_formats):
                    img_path = os.path.join(class_dir, img_name)
                    class_images.append(img_path)
            
            print(f"类别 '{class_name}': 找到 {len(class_images)} 张图片")
            
            self.images.extend(class_images)
            self.labels.extend([self.class_to_idx[class_name]] * len(class_images))
            total_images += len(class_images)
        
        print(f"数据集总计: {total_images} 张图片")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"加载图像失败: {img_path}, 错误: {e}")
            # 返回一个黑色图像作为占位符
            image = Image.new('RGB', (128, 128), color='black')
            if self.transform:
                image = self.transform(image)
            return image, label

# 数据增强和预处理
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集
print("\n" + "=" * 50)
print("创建数据集...")
train_dataset = ImageDataset(train_dir, transform=train_transform)
test_dataset = ImageDataset(test_dir, transform=test_transform)

# 检查数据集是否为空
if len(train_dataset) == 0:
    print("\n错误: 训练集为空!")
    print("请检查:")
    print("1. data/train/ 目录是否存在")
    print("2. data/train/ 下面是否有类别文件夹（如 cat/, dog/ 等）")
    print("3. 类别文件夹中是否有图片文件")
    print("4. 图片格式是否支持（jpg, png, jpeg 等）")
    exit(1)

if len(test_dataset) == 0:
    print("\n警告: 测试集为空!")

# 确定类别数量
num_classes = len(train_dataset.class_to_idx)
print(f"\n检测到 {num_classes} 个类别: {train_dataset.class_to_idx}")
print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集批次数量: {len(train_loader)}")
print(f"测试集批次数量: {len(test_loader)}")

# 构建模型（使用预训练的ResNet18）
# 构建模型（使用预训练的ResNet18）
class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        
        # 使用新的weights参数（兼容新版本torchvision）
        try:
            # 新版本用法（torchvision >= 0.13）
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except TypeError:
            # 旧版本兼容（torchvision < 0.13）
            self.backbone = models.resnet18(pretrained=True)
        
        # 冻结预训练层的参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# 初始化模型
model = ImageClassifier(num_classes=num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练函数
def train_model(model, train_loader, test_loader, epochs, patience=5):
    best_accuracy = 0.0
    patience_counter = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        # 计算平均训练损失
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 验证阶段
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算准确率
        accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        # 早停机制
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(base_dir, "best_model.pth"))
            print(f"保存最佳模型，准确率: {accuracy:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"早停：验证准确率 {patience} 轮未提升")
            break
        
        # 更新学习率
        scheduler.step()
    
    return train_losses, val_accuracies

# 训练模型
print("\n" + "=" * 50)
print("开始训练模型...")
train_losses, val_accuracies = train_model(model, train_loader, test_loader, epochs)

# 保存最终模型
torch.save(model.state_dict(), os.path.join(base_dir, "final_model.pth"))

# 绘制训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig(os.path.join(base_dir, "training_plot.png"))
plt.show()

print("模型训练完成！")
