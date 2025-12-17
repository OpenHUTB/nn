import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import random
from tqdm import tqdm

# --------------------------
# 基础配置：设备选择+可视化设置
# --------------------------
# 自动选择GPU/CPU，提速训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备：{device}')

# 解决Matplotlib中文显示问题
plt.rcParams["font.family"] = ["SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.switch_backend('TkAgg')  # 解决PyCharm可视化问题

# --------------------------
# 步骤1：生成模拟无人机航拍数据集（道路/非道路二分类）
# --------------------------
class DroneRoadDataset(Dataset):
    """模拟无人机航拍道路分割数据集（自动生成图像和掩码）"""
    def __init__(self, num_samples=100, image_size=(128, 128), transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        # 预生成所有图像和掩码
        self.images, self.masks = self._generate_data()

    def _generate_data(self):
        """生成模拟无人机航拍图像和对应的分割掩码"""
        images = []
        masks = []
        w, h = self.image_size

        for _ in range(self.num_samples):
            # 1. 创建图像（模拟航拍：浅灰色背景，道路为深灰色）
            img = Image.new('RGB', (w, h), color=(230, 230, 230))  # 非道路背景
            mask = Image.new('L', (w, h), color=0)  # 掩码：0=非道路，255=道路
            draw_img = ImageDraw.Draw(img)
            draw_mask = ImageDraw.Draw(mask)

            # 绘制主道路（随机角度的矩形，模拟航拍道路）
            road_angle = random.randint(-30, 30)  # 道路倾斜角度
            road_width = random.randint(20, 30)
            # 道路中心坐标
            cx, cy = w // 2, h // 2
            # 绘制道路（填充深灰色）
            # 简化处理：用水平道路替代旋转，降低计算量（新手友好）
            road_x1 = cx - road_width // 2
            road_x2 = cx + road_width // 2
            draw_img.rectangle([0, cy - road_width//2, w, cy + road_width//2], fill=(100, 100, 100))
            draw_mask.rectangle([0, cy - road_width//2, w, cy + road_width//2], fill=255)

            # 绘制次要元素（建筑/行人，非道路）
            # 建筑：随机矩形（浅棕色）
            for _ in range(random.randint(2, 5)):
                bx1 = random.randint(10, w-10)
                by1 = random.randint(10, h-10)
                bx2 = bx1 + random.randint(10, 20)
                by2 = by1 + random.randint(10, 20)
                draw_img.rectangle([bx1, by1, bx2, by2], fill=(160, 120, 80))
            # 行人：随机小圆点（红色）
            for _ in range(random.randint(3, 8)):
                px = random.randint(10, w-10)
                py = random.randint(10, h-10)
                draw_img.ellipse([px-2, py-2, px+2, py+2], fill=(255, 0, 0))

            images.append(img)
            masks.append(mask)

        return images, masks

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]

        # 数据变换：转为张量+归一化
        if self.transform:
            img = self.transform(img)
            # 掩码归一化到0/1（二分类）
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0).float()  # 255→1，0→0

        return img, mask

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 创建数据集和数据加载器
train_dataset = DroneRoadDataset(num_samples=100, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# --------------------------
# 步骤2：定义轻量化UNet（UNet-small）
# --------------------------
class DoubleConv(nn.Module):
    """UNet的基本模块：两次卷积+BN+ReLU（轻量化：减少通道数）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样：最大池化+DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样：反卷积+拼接+DoubleConv（轻量化：减少通道数）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 反卷积上采样
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 拼接（对应UNet的skip connection）
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出层：1x1卷积，将通道数转为1（二分类）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetSmall(nn.Module):
    """轻量化UNet：减少通道数和下采样层数，适合小图像分割"""
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 编码器（下采样）：通道数从3→16→32
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        # 移除原UNet的深层下采样，减少计算量

        # 解码器（上采样）：通道数从32→16
        self.up1 = Up(32, 16)
        self.outc = OutConv(16, n_classes)

        # Sigmoid激活，输出0-1之间的概率
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)

        # 解码器
        x = self.up1(x2, x1)
        logits = self.outc(x)
        return self.sigmoid(logits)

# 初始化模型并移到设备上
model = UNetSmall(n_channels=3, n_classes=1).to(device)

# --------------------------
# 步骤3：定义损失函数和优化器
# --------------------------
# 二分类损失：BCELoss（适用于概率输出）
criterion = nn.BCELoss()
# 优化器：Adam，学习率适中
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --------------------------
# 步骤4：训练模型（少量epochs，快速收敛）
# --------------------------
def train_model(model, loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        # 进度条显示训练进度
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 反向传播+优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            # 更新进度条显示损失
            pbar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(loader.dataset)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1} 平均损失：{epoch_loss:.4f}')

    print('\n训练完成！')
    return model, train_losses

# 开始训练（10个epoch，快速收敛）
model, train_losses = train_model(model, train_loader, criterion, optimizer, epochs=10)

# --------------------------
# 步骤5：可视化分割结果（原图+掩码+预测结果）
# --------------------------
def visualize_segmentation(model, dataset, num_samples=5):
    """可视化分割结果：三图并列展示"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    fig.suptitle('无人机航拍道路分割结果（原图→真实掩码→预测结果）', fontsize=16, fontweight='bold')

    with torch.no_grad():
        for i in range(num_samples):
            # 取数据
            img, mask = dataset[i]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            # 反归一化：(x*0.5)+0.5 → 0-1
            img_np = img_np * 0.5 + 0.5
            mask_np = mask.squeeze().cpu().numpy()

            # 模型预测
            img_input = img.unsqueeze(0).to(device)
            pred = model(img_input)
            pred_np = pred.squeeze().cpu().numpy()
            # 二值化：大于0.5为道路（1），否则为非道路（0）
            pred_np = (pred_np > 0.5).astype(np.float32)

            # 绘制原图
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title('无人机航拍原图', fontsize=12)
            axes[i, 0].axis('off')

            # 绘制真实掩码（伪色彩：道路为红色）
            axes[i, 1].imshow(mask_np, cmap='Reds')
            axes[i, 1].set_title('真实道路掩码', fontsize=12)
            axes[i, 1].axis('off')

            # 绘制预测结果（伪色彩：道路为红色）
            axes[i, 2].imshow(pred_np, cmap='Reds')
            axes[i, 2].set_title('模型预测结果', fontsize=12)
            axes[i, 2].axis('off')

    # 保存结果图
    plt.tight_layout()
    plt.savefig('drone_segmentation_results.png', bbox_inches='tight', dpi=100)
    plt.show()

# 可视化分割结果
visualize_segmentation(model, train_dataset, num_samples=5)

# --------------------------
# 额外：可视化训练损失曲线
# --------------------------
def plot_loss_curve(losses):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o', color='b', label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('无人机分割模型训练损失曲线', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('train_loss_curve.png', bbox_inches='tight')
    plt.show()

plot_loss_curve(train_losses)