#!/usr/bin/env python3
"""
PyTorch模型工具类
用于加载和运行PyTorch模型
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os


class PyTorchDroneModel:
    """PyTorch无人机视觉模型类"""

    def __init__(self, model_path=None, device=None):
        self.model = None
        self.device = None
        self.class_names = ['Forest', 'Fire', 'City', 'Animal', 'Vehicle', 'Water']
        self.img_size = (224, 224)

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"✅ 使用设备: {self.device}")

        # 图像预处理变换
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 如果提供了模型路径，自动加载
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def define_model_architecture(self):
        """定义PyTorch模型架构（需要与训练时一致）"""

        class DroneCNN(nn.Module):
            def __init__(self, num_classes=6):
                super(DroneCNN, self).__init__()
                self.features = nn.Sequential(
                    # 第一层卷积
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.25),

                    # 第二层卷积
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.25),

                    # 第三层卷积
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.25),

                    # 第四层卷积
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.25),
                )

                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 14 * 14, 512),  # 224/2/2/2/2 = 14
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x

        return DroneCNN(num_classes=len(self.class_names))

    def load_resnet18_model(self):
        """加载预训练的ResNet18模型"""
        from torchvision import models

        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.class_names))

        return model

    def load_mobilenetv2_model(self):
        """加载预训练的MobileNetV2模型"""
        from torchvision import models

        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.class_names))

        return model

    def load_model(self, model_path, model_type='custom'):
        """加载PyTorch模型"""
        print(f"🔄 正在加载PyTorch模型: {model_path}")

        try:
            # 根据类型创建模型架构
            if model_type == 'resnet18':
                self.model = self.load_resnet18_model()
            elif model_type == 'mobilenet':
                self.model = self.load_mobilenetv2_model()
            else:
                self.model = self.define_model_architecture()

            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict):
                # 如果保存的是检查点字典
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # 尝试直接加载
                    self.model.load_state_dict(checkpoint)
            else:
                # 如果保存的是模型本身
                self.model = checkpoint

            # 移动到设备
            self.model = self.model.to(self.device)

            # 设置为评估模式
            self.model.eval()

            print(f"✅ PyTorch模型加载成功")
            print(f"📊 模型结构: {self.model.__class__.__name__}")
            print(f"📊 参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.model = None
            return False

    def preprocess_image(self, image):
        """预处理图像以供PyTorch模型使用"""
        # 转换OpenCV BGR图像为PIL RGB图像
        if isinstance(image, np.ndarray):
            # OpenCV图像 (BGR) -> PIL图像 (RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image

        # 应用变换
        tensor = self.transform(pil_image)

        # 添加批次维度
        tensor = tensor.unsqueeze(0)

        # 移动到设备
        tensor = tensor.to(self.device)

        return tensor

    def predict(self, image):
        """对图像进行预测"""
        if self.model is None:
            print("⚠️  模型未加载")
            return None, 0

        try:
            # 预处理图像
            input_tensor = self.preprocess_image(image)

            # 禁用梯度计算
            with torch.no_grad():
                # 前向传播
                outputs = self.model(input_tensor)

                # 获取预测结果
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                # 转换为Python标量
                class_idx = predicted.item()
                confidence_value = confidence.item()

                # 获取类别名称
                if 0 <= class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                else:
                    class_name = f"Class_{class_idx}"

                return class_name, confidence_value

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None, 0

    def predict_batch(self, images):
        """批量预测图像"""
        if self.model is None:
            return [], []

        try:
            # 预处理所有图像
            tensors = []
            for img in images:
                tensor = self.preprocess_image(img)
                tensors.append(tensor)

            # 堆叠为批次
            batch = torch.cat(tensors, dim=0)
            batch = batch.to(self.device)

            # 预测
            with torch.no_grad():
                outputs = self.model(batch)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)

            # 转换结果
            results = []
            conf_values = []

            for i in range(len(images)):
                class_idx = predicted[i].item()
                if 0 <= class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                else:
                    class_name = f"Class_{class_idx}"

                results.append(class_name)
                conf_values.append(confidences[i].item())

            return results, conf_values

        except Exception as e:
            print(f"❌ 批量预测失败: {e}")
            return [], []


# 模型工厂函数
def load_pytorch_model(model_path, model_type='custom'):
    """加载PyTorch模型的便捷函数"""
    model = PyTorchDroneModel()
    success = model.load_model(model_path, model_type)
    return model if success else None


# 测试函数
def test_model():
    """测试模型加载和预测"""
    print("🧪 测试PyTorch模型...")

    # 创建一个测试图像
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # 加载模型
    model = PyTorchDroneModel()

    # 测试自定义模型
    print("\n1. 测试自定义模型架构...")
    custom_model = model.define_model_architecture()
    print(f"✅ 自定义模型创建成功，参数数量: {sum(p.numel() for p in custom_model.parameters()):,}")

    # 测试ResNet18
    print("\n2. 测试ResNet18架构...")
    resnet_model = model.load_resnet18_model()
    print(f"✅ ResNet18模型创建成功，参数数量: {sum(p.numel() for p in resnet_model.parameters()):,}")

    # 测试MobileNetV2
    print("\n3. 测试MobileNetV2架构...")
    mobilenet_model = model.load_mobilenetv2_model()
    print(f"✅ MobileNetV2模型创建成功，参数数量: {sum(p.numel() for p in mobilenet_model.parameters()):,}")

    print("\n🧪 测试完成！")


if __name__ == "__main__":
    test_model()