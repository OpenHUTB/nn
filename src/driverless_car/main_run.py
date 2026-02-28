# 导入必要的库
import numpy as np  # 数值计算库，用于处理数组、矩阵和数值运算
import torch  # PyTorch核心库，用于构建和训练神经网络
import torch.nn as nn  # PyTorch的神经网络模块，包含各类层和损失函数
import torch.optim as optim  # PyTorch的优化器模块，用于模型参数优化
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器，用于批量处理数据
import matplotlib.pyplot as plt  # 绘图库，用于可视化结果
import random  # 随机数生成库，用于生成异常区域的随机坐标
import os  # 操作系统库，用于创建目录、保存文件等操作

# -------------------------- 修复Matplotlib后端问题（核心新增代码） --------------------------
# 方案一：强制设置Matplotlib后端，避开PyCharm等IDE的兼容问题
# Matplotlib的后端负责图像的渲染和显示，不同环境下默认后端可能存在兼容问题
plt.switch_backend('TkAgg')  # 优先用TkAgg后端（需安装tkinter，大部分环境已预装，支持交互式显示）
# 如果TkAgg报错，可替换为以下后端：
# plt.switch_backend('Agg')  # 无界面后端，仅支持保存图片到文件，不支持交互式显示
# plt.switch_backend('Qt5Agg')  # 需安装PyQt5，支持交互式显示

# -------------------------- 1. 配置全局参数 --------------------------
# 设置计算设备：优先使用GPU（CUDA），若无则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = (64, 64)  # 无人机巡检图像的尺寸：高度64，宽度64
batch_size = 32  # 批次大小：每次训练加载32个样本
epochs = 30  # 训练轮数：整个数据集训练30次
lr = 1e-3  # 学习率：优化器的步长，控制参数更新的幅度
threshold = 0.02  # 异常检测的阈值：重构误差超过该值则判定为异常

# -------------------------- 2. 生成模拟无人机巡检数据 --------------------------
def generate_normal_sample():
    """
    生成单张正常的无人机巡检图像（3通道RGB，64x64分辨率）
    返回：
        np.ndarray: 形状为(64, 64, 3)的浮点数组，值范围[0,1]
    """
    # 初始化图像为灰度值0.5的背景（3通道）
    img = np.ones((img_size[0], img_size[1], 3), dtype=np.float32) * 0.5
    # 绘制网格纹理（模拟巡检图像的背景纹理，每4个像素画一条深灰色线）
    for i in range(0, img_size[0], 4):
        img[i, :, :] = 0.1  # 水平网格线
    for j in range(0, img_size[1], 4):
        img[:, j, :] = 0.1  # 垂直网格线
    # 添加高斯噪声（模拟图像的自然噪声）
    noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)  # 均值0，标准差0.02的高斯噪声
    img = np.clip(img + noise, 0, 1)  # 限制值范围在[0,1]，避免超出图像像素范围
    return img

def generate_abnormal_sample():
    """
    生成单张异常的无人机巡检图像（在正常图像基础上添加异常区域）
    返回：
        np.ndarray: 形状为(64, 64, 3)的浮点数组，值范围[0,1]
    """
    # 先生成正常图像
    img = generate_normal_sample()
    # 随机生成异常区域的左上角和右下角坐标（范围：10~50，避免超出图像边界）
    x1 = random.randint(10, 30)
    y1 = random.randint(10, 30)
    x2 = random.randint(30, 50)
    y2 = random.randint(30, 50)
    # 在异常区域填充0.7~1.0的随机亮度（模拟缺陷、异物等异常）
    img[x1:x2, y1:y2, :] = np.random.uniform(0.7, 1.0, (x2-x1, y2-y1, 3))
    return img

# 自定义数据集类：继承PyTorch的Dataset，用于加载模拟的无人机巡检数据
class DroneInspectionDataset(Dataset):
    def __init__(self, is_normal, sample_num=1000):
        """
        初始化数据集
        参数：
            is_normal (bool): True表示加载正常样本，False表示加载异常样本
            sample_num (int): 数据集的样本数量，默认1000
        """
        self.sample_num = sample_num  # 样本总数
        self.is_normal = is_normal  # 样本类型标记

    def __len__(self):
        """返回数据集的样本数量（必须实现的方法）"""
        return self.sample_num

    def __getitem__(self, idx):
        """
        根据索引获取单个样本（必须实现的方法）
        参数：
            idx (int): 样本索引
        返回：
            torch.Tensor: 形状为(3, 64, 64)的张量，值范围[-1,1]
        """
        # 根据标记生成对应类型的图像
        if self.is_normal:
            img = generate_normal_sample()
        else:
            img = generate_abnormal_sample()
        # 转换为张量并调整维度：(H, W, C) → (C, H, W)（PyTorch默认的图像张量格式）
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        # 归一化到[-1,1]：将原[0,1]的范围转换为[-1,1]，符合神经网络的输入习惯
        img_tensor = (img_tensor - 0.5) / 0.5
        return img_tensor

# -------------------------- 3. 定义轻量化自编码器模型 --------------------------
class LightweightAutoencoder(nn.Module):
    """
    轻量化自编码器模型，用于无人机巡检图像的重构和异常检测
    结构：编码器（卷积层）→ 解码器（反卷积层）
    输入：3通道64x64图像张量
    输出：3通道64x64重构图像张量
    """
    def __init__(self):
        super(LightweightAutoencoder, self).__init__()  # 继承父类的初始化方法
        # 编码器：通过卷积层降维，提取图像特征（3通道→64通道，64x64→8x8）
        self.encoder = nn.Sequential(
            # 卷积层1：3输入通道→16输出通道，卷积核3x3，步长2，填充1 → 输出尺寸32x32
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(True),  # 激活函数：ReLU，inplace=True表示原地操作，节省内存
            # 卷积层2：16输入通道→32输出通道，卷积核3x3，步长2，填充1 → 输出尺寸16x16
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            # 卷积层3：32输入通道→64输出通道，卷积核3x3，步长2，填充1 → 输出尺寸8x8
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True)
        )
        # 解码器：通过反卷积层升维，重构图像（64通道→3通道，8x8→64x64）
        self.decoder = nn.Sequential(
            # 反卷积层1：64输入通道→32输出通道，卷积核3x3，步长2，填充1，输出填充1 → 输出尺寸16x16
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            # 反卷积层2：32输入通道→16输出通道，卷积核3x3，步长2，填充1，输出填充1 → 输出尺寸32x32
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            # 反卷积层3：16输入通道→3输出通道，卷积核3x3，步长2，填充1，输出填充1 → 输出尺寸64x64
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # 激活函数：Tanh，输出值范围[-1,1]，与输入的归一化范围匹配
        )

    def forward(self, x):
        """
        前向传播：输入图像经过编码器和解码器，输出重构图像
        参数：
            x (torch.Tensor): 输入张量，形状为(B, 3, 64, 64)或(3, 64, 64)（B为批次大小）
        返回：
            torch.Tensor: 重构张量，形状与输入一致
        """
        # 处理单张图像的情况：如果输入是3维（C,H,W），添加batch维度变为4维（1,C,H,W）
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # 增加batch维度
        encoded = self.encoder(x)  # 编码：提取特征
        decoded = self.decoder(encoded)  # 解码：重构图像
        return decoded

# -------------------------- 4. 训练自编码器模型 --------------------------
def train_model():
    """
    训练自编码器模型（仅使用正常样本训练，让模型学习正常图像的特征）
    返回：
        LightweightAutoencoder: 训练好的模型
    """
    # 创建训练数据集：仅使用正常样本，数量1000
    train_dataset = DroneInspectionDataset(is_normal=True, sample_num=1000)
    # 创建数据加载器：批量加载数据，打乱顺序
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型并移至指定设备（GPU/CPU）
    model = LightweightAutoencoder().to(device)
    # 定义损失函数：均方误差（MSE），衡量重构图像与原图像的差异
    criterion = nn.MSELoss()
    # 定义优化器：Adam优化器，用于更新模型参数
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 开启训练模式（启用批归一化、dropout等训练相关的层）
    model.train()
    # 遍历训练轮数
    for epoch in range(epochs):
        total_loss = 0.0  # 累计总损失
        # 遍历数据加载器中的每个批次
        for data in train_loader:
            img = data.to(device)  # 将图像移至指定设备
            output = model(img)  # 前向传播，得到重构图像
            loss = criterion(output, img)  # 计算损失值

            # 反向传播与参数更新
            optimizer.zero_grad()  # 清空梯度（避免梯度累积）
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数

            total_loss += loss.item() * data.size(0)  # 累加损失（乘以批次大小，得到该批次的总损失）
        # 计算该轮的平均损失
        avg_loss = total_loss / len(train_loader.dataset)
        # 打印训练信息
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}")
    print("训练完成！")
    return model

# -------------------------- 5. 异常检测测试 --------------------------
def test_anomaly_detection(model):
    """
    使用训练好的模型进行异常检测测试
    参数：
        model (LightweightAutoencoder): 训练好的自编码器模型
    """
    # 开启评估模式（关闭批归一化、dropout等训练相关的层）
    model.eval()
    # 生成测试样本：5个正常样本 + 5个异常样本
    test_normal = DroneInspectionDataset(is_normal=True, sample_num=5)
    test_abnormal = DroneInspectionDataset(is_normal=False, sample_num=5)
    # 拼接测试样本：形状为(10, 3, 64, 64)
    test_samples = torch.stack([test_normal[i] for i in range(5)] + [test_abnormal[i] for i in range(5)])
    # 定义测试标签：0表示正常，1表示异常
    test_labels = [0]*5 + [1]*5

    # 计算每个样本的重构误差
    errors = []
    # 禁用梯度计算（评估阶段不需要计算梯度，节省内存和计算资源）
    with torch.no_grad():
        for img in test_samples:
            img = img.to(device)  # 将图像移至指定设备
            output = model(img)  # 前向传播，得到重构图像
            # 将图像从[-1,1]还原到[0,1]，方便计算视觉上的误差
            img_01 = (img + 1) / 2  # 原图像还原
            output_01 = (output.squeeze(0) + 1) / 2  # 重构图像还原（去掉batch维度）
            # 计算重构误差（MSE）
            error = nn.MSELoss()(output_01, img_01).item()
            errors.append(error)  # 保存误差值

    # 可视化结果
    plt.figure(figsize=(15, 8))  # 设置画布大小：宽15英寸，高8英寸
    for i in range(len(test_samples)):
        # 处理图像用于显示：(C, H, W) → (H, W, C)，并还原到[0,1]
        img = (test_samples[i].permute(1, 2, 0).cpu().numpy() + 1) / 2
        error = errors[i]
        is_anomaly = error > threshold  # 判断是否为异常
        # 绘制子图：2行5列，第i+1个位置
        plt.subplot(2, 5, i+1)
        plt.imshow(img)  # 显示图像
        # 设置标题：包含真实标签、重构误差、检测结果
        plt.title(f"Label: {'Normal' if test_labels[i]==0 else 'Abnormal'}\nError: {error:.4f}\nDetect: {'Anomaly' if is_anomaly else 'Normal'}")
        plt.axis('off')  # 关闭坐标轴显示
    plt.tight_layout()  # 调整子图间距，避免重叠

    # 尝试显示图像，若失败则保存到本地
    try:
        plt.show()  # 显示图像窗口
    except Exception as e:
        # 捕获异常，保存图像到本地
        print(f"显示图像失败：{e}，将保存图像到本地")
        # 创建保存目录（如果不存在）
        if not os.path.exists("drone_anomaly_results"):
            os.makedirs("drone_anomaly_results")
        # 保存图像：分辨率300，保留所有内容
        plt.savefig("drone_anomaly_results/anomaly_detection_result.png", dpi=300, bbox_inches='tight')
        print("图像已保存到：drone_anomaly_results/anomaly_detection_result.png")
    finally:
        plt.close()  # 关闭画布，释放资源

# -------------------------- 主函数：程序入口 --------------------------
if __name__ == "__main__":
    # 训练模型
    model = train_model()
    # 测试异常检测
    test_anomaly_detection(model)