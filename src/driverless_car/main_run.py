import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import os

# -------------------------- 修复Matplotlib后端问题（核心新增代码） --------------------------
# 方案一：强制设置Matplotlib后端，避开PyCharm的兼容问题
plt.switch_backend('TkAgg')  # 优先用TkAgg后端（需安装tkinter，大部分环境已预装）
# 如果TkAgg报错，可替换为以下后端：
# plt.switch_backend('Agg')  # 无界面后端，需配合保存图片使用
# plt.switch_backend('Qt5Agg')  # 需安装PyQt5

# -------------------------- 1. 配置参数 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = (64, 64)
batch_size = 32
epochs = 30
lr = 1e-3
threshold = 0.02

# -------------------------- 2. 代码生成模拟无人机巡检数据 --------------------------
def generate_normal_sample():
    """生成单张正常的无人机巡检图像（3通道，64x64）"""
    img = np.ones((img_size[0], img_size[1], 3), dtype=np.float32) * 0.5
    # 绘制网格纹理
    for i in range(0, img_size[0], 4):
        img[i, :, :] = 0.1
    for j in range(0, img_size[1], 4):
        img[:, j, :] = 0.1
    # 添加噪声
    noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 1)
    return img

def generate_abnormal_sample():
    """生成单张异常的无人机巡检图像（3通道，64x64）"""
    img = generate_normal_sample()
    # 制造异常区域
    x1 = random.randint(10, 30)
    y1 = random.randint(10, 30)
    x2 = random.randint(30, 50)
    y2 = random.randint(30, 50)
    img[x1:x2, y1:y2, :] = np.random.uniform(0.7, 1.0, (x2-x1, y2-y1, 3))
    return img

# 自定义数据集
class DroneInspectionDataset(Dataset):
    def __init__(self, is_normal, sample_num=1000):
        self.sample_num = sample_num
        self.is_normal = is_normal

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        if self.is_normal:
            img = generate_normal_sample()
        else:
            img = generate_abnormal_sample()
        # 转换为张量：(H, W, C) → (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        img_tensor = (img_tensor - 0.5) / 0.5  # 归一化到[-1,1]
        return img_tensor

# -------------------------- 3. 定义轻量化自编码器模型 --------------------------
class LightweightAutoencoder(nn.Module):
    def __init__(self):
        super(LightweightAutoencoder, self).__init__()
        # 编码器：3通道输入
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 3→16，64→32
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16→32，32→16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32→64，16→8
            nn.ReLU(True)
        )
        # 解码器：3通道输出
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 8→16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 16→32
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # 32→64，输出3通道
            nn.Tanh()
        )

    def forward(self, x):
        # 自动处理维度：3维→4维（添加batch维度）
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # (C,H,W) → (1,C,H,W)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# -------------------------- 4. 训练自编码器 --------------------------
def train_model():
    train_dataset = DroneInspectionDataset(is_normal=True, sample_num=1000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LightweightAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data in train_loader:
            img = data.to(device)
            output = model(img)
            loss = criterion(output, img)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}")
    print("训练完成！")
    return model

# -------------------------- 5. 异常检测测试 --------------------------
def test_anomaly_detection(model):
    model.eval()
    # 生成测试样本：5正常+5异常
    test_normal = DroneInspectionDataset(is_normal=True, sample_num=5)
    test_abnormal = DroneInspectionDataset(is_normal=False, sample_num=5)
    # 拼接测试样本：(10,3,64,64)
    test_samples = torch.stack([test_normal[i] for i in range(5)] + [test_abnormal[i] for i in range(5)])
    test_labels = [0]*5 + [1]*5  # 0=正常，1=异常

    # 计算重构误差
    errors = []
    with torch.no_grad():
        for img in test_samples:
            img = img.to(device)
            output = model(img)
            # 还原到[0,1]计算误差
            img_01 = (img + 1) / 2  # (3,64,64)
            output_01 = (output.squeeze(0) + 1) / 2  # 去掉batch维度
            error = nn.MSELoss()(output_01, img_01).item()
            errors.append(error)

    # 可视化结果（修复后）
    plt.figure(figsize=(15, 8))
    for i in range(len(test_samples)):
        # 处理图像用于显示：(C,H,W) → (H,W,C)
        img = (test_samples[i].permute(1, 2, 0).cpu().numpy() + 1) / 2
        error = errors[i]
        is_anomaly = error > threshold
        # 绘制
        plt.subplot(2, 5, i+1)
        plt.imshow(img)
        plt.title(f"Label: {'Normal' if test_labels[i]==0 else 'Abnormal'}\nError: {error:.4f}\nDetect: {'Anomaly' if is_anomaly else 'Normal'}")
        plt.axis('off')
    plt.tight_layout()

    # -------------------------- 修复可视化的核心修改 --------------------------
    try:
        # 尝试显示图像
        plt.show()
    except Exception as e:
        # 如果显示失败，保存图像到本地（方案二兜底）
        print(f"显示图像失败：{e}，将保存图像到本地")
        # 创建保存目录（如果不存在）
        if not os.path.exists("drone_anomaly_results"):
            os.makedirs("drone_anomaly_results")
        # 保存图像
        plt.savefig("drone_anomaly_results/anomaly_detection_result.png", dpi=300, bbox_inches='tight')
        print("图像已保存到：drone_anomaly_results/anomaly_detection_result.png")
    finally:
        # 关闭画布，释放资源
        plt.close()

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 训练模型
    model = train_model()
    # 测试异常检测
    test_anomaly_detection(model)