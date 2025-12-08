import os
import sys

# 获取当前脚本所在的目录
if hasattr(sys, '_MEIPASS'):
    # 如果是打包后的exe，使用临时解压目录
    current_dir = sys._MEIPASS
else:
    # 否则使用脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

# 设置工作目录为脚本所在目录
os.chdir(current_dir)
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 导入其他模块的功能
from Data_classfication import split_dataset
from image_classification import ImageDataset, ImageClassifier
from visual_navigation import main as run_visual_navigation
from forecast import predict_image, batch_predict

# 路径设置
base_dir = "data"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
dataset_dir = os.path.join(base_dir, "dataset")

def setup_directories():
    """设置数据目录"""
    print("=" * 50)
    print("设置数据目录...")
    
    # 检查并创建目录
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 检查是否需要分割数据集
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        print("训练集不存在或为空，开始自动分割数据集...")
        if os.path.exists(dataset_dir):
            success = split_dataset(dataset_dir, train_dir, test_dir, split_ratio=0.8)
            if not success:
                print("❌ 数据集分割失败，请检查原始数据集路径")
                return False
        else:
            print(f"❌ 原始数据集路径不存在: {dataset_dir}")
            print("请将数据集放入 ./data/dataset/ 目录")
            print("数据集结构应为:")
            print("data/dataset/")
            print("├── 类别1/")
            print("│   ├── image1.jpg")
            print("│   └── image2.jpg")
            print("├── 类别2/")
            print("│   ├── image1.jpg")
            print("│   └── image2.jpg")
            print("└── ...")
            return False
    else:
        print("✅ 训练集已存在，跳过数据集分割步骤")
    
    return True

def train_pytorch_model():
    """使用PyTorch训练模型"""
    print("\n" + "=" * 50)
    print("开始PyTorch模型训练...")
    
    # 参数配置
    img_size = (128, 128)
    batch_size = 32
    epochs = 70
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据预处理
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
    train_dataset = ImageDataset(train_dir, transform=train_transform)
    test_dataset = ImageDataset(test_dir, transform=test_transform)
    
    if len(train_dataset) == 0:
        print("❌ 训练集为空，无法训练模型")
        return None, [], []
    
    num_classes = len(train_dataset.class_to_idx)
    print(f"检测到 {num_classes} 个类别: {train_dataset.class_to_idx}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = ImageClassifier(num_classes=num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 训练模型
    best_accuracy = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 验证
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
        
        accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(base_dir, "best_model.pth"))
            print(f"✅ 保存最佳模型，准确率: {accuracy:.4f}")
        
        scheduler.step()
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(base_dir, "final_model.pth"))
    print("✅ 最终模型已保存")
    
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
    
    return model, train_losses, val_accuracies

def main():
    """主函数"""
    print("🚀 开始图像分类系统...")
    # 1. 设置数据目录
    if not setup_directories():
        return
    
    # 2. 训练PyTorch模型
    model, train_losses, val_accuracies = train_pytorch_model()
    
    if model is None:
        print("❌ 模型训练失败")
        return
    
    # 3. 提供预测功能
    print("\n" + "=" * 50)
    choice = input("是否进行图像预测？(y/n): ")
    if choice.lower() == 'y':
        test_image_path = input("请输入测试图像路径: ")
        if os.path.exists(test_image_path):
            result = predict_image(
                os.path.join(base_dir, "best_model.pth"),
                test_image_path,
                train_dir
            )
        else:
            print("❌ 指定的路径不存在")
    
    # 4. 启动视觉导航（可选）
    print("\n" + "=" * 50)
    choice = input("是否启动视觉导航系统？(y/n): ")
    if choice.lower() == 'y':
        try:
            from visual_navigation import main as nav_main
            nav_main()
        except ImportError as e:
            print(f"❌ 无法启动视觉导航: {e}")
    
    print("\n🎉 所有任务完成！")

if __name__ == "__main__":
    # 添加必要的导入
    import torch.optim as optim
    main()