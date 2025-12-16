import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# 构建与训练时相同的模型结构
class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        
        # 使用与训练时相同的模型结构
        try:
            # 新版本用法（torchvision >= 0.13）
            self.backbone = models.resnet18(weights=None)  # 不加载预训练权重，因为我们会加载自己的
        except TypeError:
            # 旧版本兼容（torchvision < 0.13）
            self.backbone = models.resnet18(pretrained=False)
        
        # 冻结预训练层的参数（与训练时一致）
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 替换最后的全连接层（必须与训练时结构相同）
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def predict_image(model_path, img_path, train_dir, img_size=(128, 128)):
    """
    使用训练好的PyTorch模型进行图像预测
    
    参数:
        model_path: 模型文件路径
        img_path: 要预测的图像路径
        train_dir: 训练数据目录（用于获取类别标签）
        img_size: 图像尺寸，必须与训练时相同
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取类别标签（与训练时相同的方式）
    class_labels = sorted([d for d in os.listdir(train_dir) 
                          if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_labels)
    
    if num_classes == 0:
        print("错误: 在训练目录中未找到任何类别!")
        return None
    
    print(f"检测到 {num_classes} 个类别: {class_labels}")
    
    # 初始化模型
    model = ImageClassifier(num_classes=num_classes)
    
    # 加载训练好的权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # 设置为评估模式
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 图像预处理（必须与训练时的测试预处理相同）
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 检查img_path是文件还是目录
    if os.path.isdir(img_path):
        print(f"检测到目录路径: {img_path}")
        # 如果是目录，找到目录中的第一个图像文件
        image_files = [f for f in os.listdir(img_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not image_files:
            print("错误: 目录中没有找到图像文件!")
            return None
        # 使用第一个图像文件
        img_path = os.path.join(img_path, image_files[0])
        print(f"使用目录中的第一个图像: {image_files[0]}")
    
    # 加载和预处理图像
    try:
        image = Image.open(img_path).convert('RGB')
        print(f"成功加载图像: {img_path}")
        print(f"图像尺寸: {image.size}")
    except Exception as e:
        print(f"加载图像失败: {e}")
        return None
    
    # 应用预处理
    input_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    input_tensor = input_tensor.to(device)
    
    # 预测
    with torch.no_grad():  # 禁用梯度计算
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_idx].item()
    
    # 获取预测结果
    predicted_class = class_labels[predicted_class_idx]
    
    # 显示详细信息
    print("\n" + "=" * 50)
    print("📊 预测结果:")
    print(f"🔍 预测类别: {predicted_class}")
    print(f"📈 置信度: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"🏷️ 类别索引: {predicted_class_idx}")
    
    # 显示所有类别的概率
    print("\n所有类别概率:")
    for i, class_name in enumerate(class_labels):
        prob = probabilities[i].item()
        print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    
    print("=" * 50)
    
    return predicted_class, confidence

def main():
    """主函数 - 使用示例"""
    # 路径设置
    base_dir = "./data"  # 与训练代码相同的基准目录
    model_path = os.path.join(base_dir, "best_model.pth")  # 使用训练代码保存的最佳模型
    train_dir = os.path.join(base_dir, "train")
    
    # 要预测的图像路径 - 可以修改为你的测试图像路径
    # 可以选择使用目录或具体图像文件
    test_dir = os.path.join(base_dir, "test", "Fire")  # 目录路径
    # 或者直接指定具体图像文件：
    # test_image_path = os.path.join(base_dir, "test", "Fire", "具体的图像文件名.jpg")
    
    # 检查路径是否存在
    print("=" * 50)
    print("路径检查:")
    print(f"模型路径: {model_path}, 存在: {os.path.exists(model_path)}")
    print(f"训练目录: {train_dir}, 存在: {os.path.exists(train_dir)}")
    print(f"测试目录: {test_dir}, 存在: {os.path.exists(test_dir)}")
    
    # 如果指定的是目录，检查其中是否有图像文件
    if os.path.exists(test_dir) and os.path.isdir(test_dir):
        image_files = [f for f in os.listdir(test_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"测试目录中的图像文件: {len(image_files)} 个")
        if image_files:
            print(f"前几个文件: {image_files[:3]}")  # 显示前3个文件
    
    print("=" * 50)
    
    if not all([os.path.exists(model_path), os.path.exists(train_dir)]):
        print("错误: 模型或训练目录不存在!")
        return
    
    if not os.path.exists(test_dir):
        print("错误: 测试路径不存在!")
        return
    
    # 执行预测
    result = predict_image(model_path, test_dir, train_dir)
    
    if result:
        predicted_class, confidence = result
        print(f"\n🎯 最终预测: {predicted_class} (置信度: {confidence*100:.2f}%)")

# 批量预测单个目录中的所有图像（不要求子目录结构）
def predict_directory(model_path, directory_path, train_dir, img_size=(128, 128)):
    """
    预测指定目录中的所有图像文件
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取类别标签
    class_labels = sorted([d for d in os.listdir(train_dir) 
                          if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_labels)
    
    # 初始化模型
    model = ImageClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    
    # 获取目录中的所有图像文件
    image_files = [f for f in os.listdir(directory_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"在目录 {directory_path} 中没有找到图像文件!")
        return results
    
    print(f"\n开始批量预测 {len(image_files)} 个图像...")
    
    for img_name in image_files:
        img_path = os.path.join(directory_path, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_class_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class_idx].item()
            
            predicted_class = class_labels[predicted_class_idx]
            
            results.append({
                'image_name': img_name,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
            
            print(f"📸 {img_name}: {predicted_class} (置信度: {confidence*100:.2f}%)")
            
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
    
    # 统计预测结果
    if results:
        print(f"\n📊 批量预测完成!")
        class_counts = {}
        for result in results:
            cls = result['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print("预测结果统计:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} 个图像")
    
    return results

if __name__ == "__main__":
    main()