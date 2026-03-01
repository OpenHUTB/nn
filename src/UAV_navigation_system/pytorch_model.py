#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch模型工具类
用于加载和运行无人机场景分类的PyTorch模型
"""

# 临时修复：在导入 torch 前尝试处理 typing_extensions 问题
try:
    import typing_extensions
    # 检查 TypeIs 是否可用，如果不可用则模拟一个
    if not hasattr(typing_extensions, 'TypeIs'):
        typing_extensions.TypeIs = type(lambda: None)
except ImportError:
    pass

# 导入核心依赖库
import torch  # PyTorch核心库，用于模型构建和推理
import torch.nn as nn  # 神经网络层模块
import torchvision.transforms as transforms  # 图像预处理工具
from PIL import Image  # PIL库，用于图像读取和格式转换
import numpy as np  # 数值计算库，处理图像数组
import cv2  # OpenCV库，处理视频/图像数据
import os  # 文件路径操作库

# ... 其余代码保持不变 ...
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch模型工具类
用于加载和运行无人机场景分类的PyTorch模型
核心功能：
1. 支持自定义CNN/ResNet18/MobileNetV2三种模型架构加载
2. 实现图像预处理、单张/批量图像预测
3. 适配CPU/GPU设备，自动检测硬件环境
"""

# 导入核心依赖库
import torch  # PyTorch核心库，用于模型构建和推理
import torch.nn as nn  # 神经网络层模块
import torchvision.transforms as transforms  # 图像预处理工具
from PIL import Image  # PIL库，用于图像读取和格式转换
import numpy as np  # 数值计算库，处理图像数组
import cv2  # OpenCV库，处理视频/图像数据
import os  # 文件路径操作库


class PyTorchDroneModel:
    """
    PyTorch无人机视觉场景分类模型类
    封装模型加载、图像预处理、场景分类预测等核心功能
    支持的场景类别：Forest(森林)、Fire(火灾)、City(城市)、Animal(动物)、Vehicle(车辆)、Water(水域)
    """

    def __init__(self, model_path=None, device=None):
        """
        初始化模型类

        Args:
            model_path (str, optional): 预训练模型权重文件路径，默认None
            device (torch.device, optional): 模型运行设备（cpu/cuda），默认自动检测
        """
        # 模型实例初始化
        self.model = None
        # 运行设备初始化
        self.device = None
        # 场景分类类别名称（与训练时标签对应）
        self.class_names = ['Forest', 'Fire', 'City', 'Animal', 'Vehicle', 'Water']
        # 模型输入图像尺寸（需与训练时一致）
        self.img_size = (224, 224)

        # 自动检测/指定运行设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"✅ 使用设备: {self.device}")

        # 定义图像预处理流水线（与训练时预处理逻辑一致）
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),  # 调整图像尺寸
            transforms.ToTensor(),  # 转换为Tensor（0-1归一化）
            # 标准化（使用ImageNet均值/标准差，适配预训练模型）
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 如果传入有效模型路径，自动加载模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def define_model_architecture(self):
        """
        定义自定义CNN模型架构（需与训练时的模型结构完全一致）
        适用于无人机场景分类的轻量级卷积神经网络

        Returns:
            nn.Module: 自定义CNN模型实例
        """

        class DroneCNN(nn.Module):
            """内部自定义CNN模型类"""
            def __init__(self, num_classes=6):
                super(DroneCNN, self).__init__()
                # 特征提取层（4层卷积+批归一化+激活+池化+dropout）
                self.features = nn.Sequential(
                    # 第一层卷积：3通道输入→32通道输出，3×3卷积核，填充1
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),  # 批归一化，加速训练，防止过拟合
                    nn.ReLU(inplace=True),  # ReLU激活函数，inplace=True节省内存
                    nn.MaxPool2d(kernel_size=2, stride=2),  # 2×2最大池化，步长2
                    nn.Dropout(0.25),  # Dropout层，随机丢弃25%神经元，防止过拟合

                    # 第二层卷积：32通道→64通道
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.25),

                    # 第三层卷积：64通道→128通道
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.25),

                    # 第四层卷积：128通道→256通道
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.25),
                )

                # 分类器层（全连接层）
                self.classifier = nn.Sequential(
                    nn.Flatten(),  # 展平特征图：256×14×14 → 256×14×14
                    # 全连接层1：特征展平后→512维隐藏层（224/2^4=14，4次池化后尺寸）
                    nn.Linear(256 * 14 * 14, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),  # 丢弃50%神经元，防止过拟合
                    nn.Linear(512, num_classes)  # 输出层：512→分类类别数
                )

            def forward(self, x):
                """前向传播逻辑"""
                x = self.features(x)  # 特征提取
                x = self.classifier(x)  # 分类预测
                return x

        # 返回自定义模型实例（分类类别数与场景类别数一致）
        return DroneCNN(num_classes=len(self.class_names))

    def load_resnet18_model(self):
        """
        加载预训练的ResNet18模型并适配自定义分类任务
        修改最后一层全连接层，适配无人机6类场景分类

        Returns:
            nn.Module: 适配后的ResNet18模型实例
        """
        from torchvision import models

        # 加载ResNet18骨架（不加载ImageNet预训练权重，避免与自定义任务冲突）
        model = models.resnet18(pretrained=False)
        # 获取最后一层全连接层的输入特征数
        num_features = model.fc.in_features
        # 替换最后一层全连接层：原1000类→自定义6类
        model.fc = nn.Linear(num_features, len(self.class_names))

        return model

    def load_mobilenetv2_model(self):
        """
        加载预训练的MobileNetV2模型并适配自定义分类任务
        轻量级模型，适配无人机嵌入式设备

        Returns:
            nn.Module: 适配后的MobileNetV2模型实例
        """
        from torchvision import models

        # 加载MobileNetV2骨架（不加载预训练权重）
        model = models.mobilenet_v2(pretrained=False)
        # 替换分类器最后一层：原1000类→自定义6类
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.class_names))

        return model

    def load_model(self, model_path, model_type='custom'):
        """
        加载预训练PyTorch模型权重

        Args:
            model_path (str): 模型权重文件路径（.pth/.pt）
            model_type (str): 模型架构类型，可选['custom', 'resnet18', 'mobilenet']

        Returns:
            bool: 加载成功返回True，失败返回False
        """
        print(f"🔄 正在加载PyTorch模型: {model_path}")

        try:
            # 根据模型类型创建对应架构的模型实例
            if model_type == 'resnet18':
                self.model = self.load_resnet18_model()
            elif model_type == 'mobilenet':
                self.model = self.load_mobilenetv2_model()
            else:  # 默认加载自定义CNN
                self.model = self.define_model_architecture()

            # 加载模型权重文件（兼容多种保存格式）
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict):
                # 情况1：保存的是检查点字典（包含state_dict/优化器参数等）
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # 情况2：字典仅包含模型权重
                    self.model.load_state_dict(checkpoint)
            else:
                # 情况3：直接保存的模型实例
                self.model = checkpoint

            # 将模型移动到指定设备（CPU/GPU）
            self.model = self.model.to(self.device)

            # 设置模型为评估模式（禁用Dropout/BatchNorm的训练行为）
            self.model.eval()

            # 打印加载成功信息
            print(f"✅ PyTorch模型加载成功")
            print(f"📊 模型结构: {self.model.__class__.__name__}")
            # 计算并打印模型总参数数量
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"📊 参数数量: {total_params:,}")

            return True

        except Exception as e:
            # 捕获加载过程中的所有异常
            print(f"❌ 模型加载失败: {e}")
            self.model = None
            return False

    def preprocess_image(self, image):
        """
        图像预处理：将输入图像转换为模型可接受的Tensor格式

        Args:
            image (np.ndarray/PIL.Image): 输入图像（OpenCV格式(BGR)或PIL格式(RGB)）

        Returns:
            torch.Tensor: 预处理后的4维Tensor (batch_size, channels, height, width)
        """
        # 处理OpenCV格式图像（BGR→RGB，转换为PIL图像）
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR→RGB转换
            pil_image = Image.fromarray(image_rgb)  # 数组→PIL图像
        else:
            # 直接使用PIL图像
            pil_image = image

        # 应用预处理流水线
        tensor = self.transform(pil_image)

        # 添加批次维度（模型要求批量输入，单张图像batch_size=1）
        tensor = tensor.unsqueeze(0)

        # 将Tensor移动到指定设备
        tensor = tensor.to(self.device)

        return tensor

    def predict(self, image):
        """
        单张图像场景分类预测

        Args:
            image (np.ndarray/PIL.Image): 输入图像（OpenCV/PIL格式）

        Returns:
            tuple: (预测类别名称, 置信度)，预测失败返回(None, 0)
        """
        # 检查模型是否加载
        if self.model is None:
            print("⚠️  模型未加载，无法预测")
            return None, 0

        try:
            # 图像预处理
            input_tensor = self.preprocess_image(image)

            # 禁用梯度计算（推理阶段无需计算梯度，提升速度，节省内存）
            with torch.no_grad():
                # 模型前向传播，获取预测logits
                outputs = self.model(input_tensor)

                # 计算softmax概率（将logits转换为0-1的概率分布）
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                # 获取最大概率的类别索引和置信度
                confidence, predicted = torch.max(probabilities, 1)

                # 转换为Python标量（从Tensor→数值）
                class_idx = predicted.item()
                confidence_value = confidence.item()

                # 获取类别名称
                if 0 <= class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                else:
                    class_name = f"Class_{class_idx}"  # 未知类别兜底

                return class_name, confidence_value

        except Exception as e:
            # 捕获预测过程中的异常
            print(f"❌ 预测失败: {e}")
            return None, 0

    def predict_batch(self, images):
        """
        批量图像场景分类预测（提升批量处理效率）

        Args:
            images (list): 图像列表，每个元素为np.ndarray/PIL.Image格式

        Returns:
            tuple: (预测类别名称列表, 置信度列表)，失败返回([], [])
        """
        if self.model is None:
            print("⚠️  模型未加载，无法批量预测")
            return [], []

        try:
            # 预处理所有图像，生成Tensor列表
            tensors = []
            for img in images:
                tensor = self.preprocess_image(img)
                tensors.append(tensor)

            # 堆叠为批量Tensor（batch_size=N）
            batch = torch.cat(tensors, dim=0)
            batch = batch.to(self.device)

            # 推理阶段禁用梯度计算
            with torch.no_grad():
                outputs = self.model(batch)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)

            # 解析批量预测结果
            results = []
            conf_values = []
            for i in range(len(images)):
                class_idx = predicted[i].item()
                # 映射类别索引到名称
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


def load_pytorch_model(model_path, model_type='custom'):
    """
    加载PyTorch模型的便捷工厂函数

    Args:
        model_path (str): 模型权重文件路径
        model_type (str): 模型架构类型，可选['custom', 'resnet18', 'mobilenet']

    Returns:
        PyTorchDroneModel: 模型实例（加载成功）/None（加载失败）
    """
    model = PyTorchDroneModel()
    success = model.load_model(model_path, model_type)
    return model if success else None


def test_model():
    """
    测试函数：验证模型架构创建、加载等核心功能
    无需实际权重文件，仅测试模型结构完整性
    """
    print("🧪 开始测试PyTorch模型工具类...")

    # 创建随机测试图像（224×224×3，模拟RGB图像）
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # 初始化模型实例
    model = PyTorchDroneModel()

    # 测试1：自定义CNN模型架构创建
    print("\n1. 测试自定义CNN模型架构...")
    custom_model = model.define_model_architecture()
    total_params = sum(p.numel() for p in custom_model.parameters())
    print(f"✅ 自定义模型创建成功，参数数量: {total_params:,}")

    # 测试2：ResNet18模型架构加载
    print("\n2. 测试ResNet18模型架构...")
    resnet_model = model.load_resnet18_model()
    total_params = sum(p.numel() for p in resnet_model.parameters())
    print(f"✅ ResNet18模型创建成功，参数数量: {total_params:,}")

    # 测试3：MobileNetV2模型架构加载
    print("\n3. 测试MobileNetV2模型架构...")
    mobilenet_model = model.load_mobilenetv2_model()
    total_params = sum(p.numel() for p in mobilenet_model.parameters())
    print(f"✅ MobileNetV2模型创建成功，参数数量: {total_params:,}")

    print("\n🧪 所有测试完成！")


# 主函数：仅在直接运行脚本时执行测试
if __name__ == "__main__":
    test_model()