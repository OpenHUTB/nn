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
    
    # 加载和预处理图像
    try:
        image = Image.open(img_path).convert('RGB')
        print(f"成功加载图像: {img_path}")
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
    img_path = os.path.join(base_dir, "test", "Fire", "fi10.jpg")  # 示例路径
    
    # 检查路径是否存在
    print("=" * 50)
    print("路径检查:")
    print(f"模型路径: {model_path}, 存在: {os.path.exists(model_path)}")
    print(f"训练目录: {train_dir}, 存在: {os.path.exists(train_dir)}")
    print(f"图像路径: {img_path}, 存在: {os.path.exists(img_path)}")
    print("=" * 50)
    
    if not all([os.path.exists(model_path), os.path.exists(train_dir), os.path.exists(img_path)]):
        print("错误: 必要的文件或目录不存在!")
        return
    
    # 执行预测
    result = predict_image(model_path, img_path, train_dir)
    
    if result:
        predicted_class, confidence = result
        print(f"\n🎯 最终预测: {predicted_class} (置信度: {confidence*100:.2f}%)")

# 批量预测函数
def batch_predict(model_path, test_dir, train_dir, img_size=(128, 128)):
    """
    批量预测测试目录中的所有图像
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
    
    # 遍历测试目录
    for class_name in class_labels:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(class_dir, img_name)
                
                try:
                    image = Image.open(img_path).convert('RGB')
                    input_tensor = transform(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        predicted_class_idx = torch.argmax(outputs[0]).item()
                        confidence = torch.nn.functional.softmax(outputs[0], dim=0)[predicted_class_idx].item()
                    
                    predicted_class = class_labels[predicted_class_idx]
                    is_correct = (predicted_class == class_name)
                    
                    results.append({
                        'image_path': img_path,
                        'true_class': class_name,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'correct': is_correct
                    })
                    
                    status = "✅" if is_correct else "❌"
                    print(f"{status} {img_name}: 真实={class_name}, 预测={predicted_class}, 置信度={confidence:.4f}")
                    
                except Exception as e:
                    print(f"处理图像 {img_path} 时出错: {e}")
    
    # 计算准确率
    if results:
        correct_predictions = sum(1 for r in results if r['correct'])
        accuracy = correct_predictions / len(results)
        print(f"\n📊 批量预测准确率: {accuracy:.4f} ({correct_predictions}/{len(results)})")
    
    return results

if __name__ == "__main__":
    main()
    
   
