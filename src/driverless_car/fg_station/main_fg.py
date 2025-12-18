import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
# scikit-image相关库（生成纹理数据）
from skimage import data, util
from skimage.transform import resize

# ====================== 1. 自动生成带纹理的模拟航拍数据集（修复维度问题） ======================
def generate_synthetic_drone_data(save_root, num_train=10, num_test=1, img_size=256):
    """
    用scikit-image生成带纹理的模拟无人机航拍道路数据（修复维度不匹配问题）
    :param save_root: 数据集保存根路径
    :param num_train: 训练集图片数量
    :param num_test: 测试集图片数量
    :param img_size: 图片尺寸（默认256x256）
    """
    # 创建文件夹结构
    train_img_dir = os.path.join(save_root, "train", "images")
    train_mask_dir = os.path.join(save_root, "train", "masks")
    test_img_dir = os.path.join(save_root, "test", "images")
    test_mask_dir = os.path.join(save_root, "test", "masks")
    for dir_path in [train_img_dir, train_mask_dir, test_img_dir, test_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 生成训练数据
    for i in range(num_train):
        # 1. 生成自然纹理背景：用skimage的草地图片（二维灰度图）转换为三维RGB图
        grass = data.grass()  # 获取skimage内置的草地纹理图（二维灰度图：img_size×img_size）
        grass = resize(grass, (img_size, img_size), anti_aliasing=True)  # 缩放至256x256
        grass = util.img_as_ubyte(grass)  # 转为uint8格式（0-255）
        # 关键修复：将二维灰度图转为三维RGB图（复制三个通道）
        grass = np.stack([grass, grass, grass], axis=-1)  # 变为：img_size×img_size×3

        # 2. 生成道路纹理：灰色混凝土纹理（三维RGB格式，和背景一致）
        road = np.random.rand(img_size, img_size, 3) * 50 + 100  # 灰色噪声纹理（100-150亮度）
        road = road.astype(np.uint8)

        # 3. 随机生成道路区域（矩形），合并背景和道路（处理坐标顺序问题，避免空切片）
        img = grass.copy()
        # 随机确定道路的左上角和右下角坐标，确保y1 < y2，x1 < x2（避免切片为空）
        road_x1 = np.random.randint(0, img_size // 2)
        road_x2 = np.random.randint(img_size // 2, img_size)
        road_y1 = np.random.randint(0, img_size)
        road_y2 = np.random.randint(road_y1, img_size)  # 确保y2 >= y1
        # 将道路区域替换到背景图中（现在维度匹配，不会报错）
        img[road_y1:road_y2, road_x1:road_x2, :] = road[road_y1:road_y2, road_x1:road_x2, :]

        # 4. 生成对应的掩码图：道路区域为白色（255），背景为黑色（0）（二维）
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        mask[road_y1:road_y2, road_x1:road_x2] = 255  # 白色道路掩码

        # 5. 保存图片（JPG格式存图像，PNG格式存掩码）
        img_pil = Image.fromarray(img)
        mask_pil = Image.fromarray(mask)
        img_pil.save(os.path.join(train_img_dir, f"{i+1}.jpg"))
        mask_pil.save(os.path.join(train_mask_dir, f"{i+1}.png"))

    # 生成测试数据（逻辑和训练数据一致）
    for i in range(num_test):
        grass = data.grass()
        grass = resize(grass, (img_size, img_size), anti_aliasing=True)
        grass = util.img_as_ubyte(grass)
        grass = np.stack([grass, grass, grass], axis=-1)  # 转为三维RGB图

        road = np.random.rand(img_size, img_size, 3) * 50 + 100
        road = road.astype(np.uint8)

        img = grass.copy()
        road_x1 = np.random.randint(0, img_size // 2)
        road_x2 = np.random.randint(img_size // 2, img_size)
        road_y1 = np.random.randint(0, img_size)
        road_y2 = np.random.randint(road_y1, img_size)  # 确保y2 >= y1
        img[road_y1:road_y2, road_x1:road_x2, :] = road[road_y1:road_y2, road_x1:road_x2, :]

        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        mask[road_y1:road_y2, road_x1:road_x2] = 255

        img_pil = Image.fromarray(img)
        mask_pil = Image.fromarray(mask)
        img_pil.save(os.path.join(test_img_dir, f"{i+1}.jpg"))
        mask_pil.save(os.path.join(test_mask_dir, f"{i+1}.png"))

    print(f"✅ 带纹理的模拟数据集生成完成！保存路径：{save_root}")
    return train_img_dir, train_mask_dir, os.path.join(test_img_dir, "1.jpg"), os.path.join(test_mask_dir, "1.png")

# ====================== 2. 定义简化版U-Net模型 ======================
class DoubleConv(nn.Module):
    """双卷积层（U-Net的基础模块，提取特征）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """下采样（最大池化+双卷积）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    """上采样（转置卷积+拼接+双卷积）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 上采样后的特征图
        # 拼接：将上采样的特征图和对应下采样的特征图拼接（U-Net的核心跳连）
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出层（1x1卷积，调整通道数到类别数）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    """简化版U-Net（减少了一层下采样，适合新手快速训练）"""
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.n_channels = n_channels  # 输入通道：3（RGB图像）
        self.n_classes = n_classes    # 输出通道：1（二分类：道路/背景）

        self.inc = DoubleConv(n_channels, 64)    # 输入层：3→64
        self.down1 = Down(64, 128)               # 下采样1：64→128
        self.down2 = Down(128, 256)              # 下采样2：128→256
        self.up1 = Up(256, 128)                  # 上采样1：256→128
        self.up2 = Up(128, 64)                   # 上采样2：128→64
        self.outc = OutConv(64, n_classes)       # 输出层：64→1

    def forward(self, x):
        # 下采样（编码）
        x1 = self.inc(x)   # [batch, 64, H, W]
        x2 = self.down1(x1)# [batch, 128, H/2, W/2]
        x3 = self.down2(x2)# [batch, 256, H/4, W/4]
        # 上采样（解码）
        x = self.up1(x3, x2)# [batch, 128, H/2, W/2]
        x = self.up2(x, x1) # [batch, 64, H, W]
        logits = self.outc(x)# [batch, 1, H, W]
        return logits

# ====================== 3. 定义数据集类（适配生成的数据集） ======================
class RoadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # 只取jpg格式的图片（避免其他文件干扰）
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        # 容错：如果没有jpg文件，给出提示
        if len(self.img_names) == 0:
            raise ValueError(f"路径 {img_dir} 下没有找到.jpg格式的图片！")

    def __len__(self):
        # 数据集长度
        return len(self.img_names)

    def __getitem__(self, idx):
        # 读取单张图像和对应的掩码
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        # 掩码文件是png格式，替换后缀
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        # 容错：检查掩码文件是否存在
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"掩码文件 {mask_path} 不存在，请检查数据集是否完整！")

        # 打开图像（RGB）和掩码（灰度图）
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # 数据预处理（resize、转tensor）
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # 掩码二值化：大于0.5的为1（道路），否则为0（背景）
        mask = (mask > 0.5).float()
        return image, mask

# ====================== 4. 主训练和测试流程 ======================
def main():
    # -------------------------- 第一步：自动生成带纹理的模拟数据集 --------------------------
    # 数据集保存路径（可自定义，比如D:\texture_drone_data）
    save_root = r"D:\texture_drone_data"
    # 生成10张训练图，1张测试图（数量可调整）
    train_img_dir, train_mask_dir, test_img_path, test_mask_path = generate_synthetic_drone_data(
        save_root, num_train=10, num_test=1
    )

    # -------------------------- 配置参数 --------------------------
    batch_size = 2  # 批次大小（CPU建议设为1/2，GPU可设更大）
    num_epochs = 5   # 训练轮数（纹理数据简单，5轮足够收敛）
    lr = 1e-4        # 学习率

    # -------------------------- 数据预处理 --------------------------
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 统一尺寸为256x256（和生成的图片一致）
        transforms.ToTensor(),          # 转为tensor，且像素值归一化到[0,1]
    ])

    # -------------------------- 加载数据集 --------------------------
    try:
        train_dataset = RoadDataset(train_img_dir, train_mask_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"✅ 数据集加载成功！共 {len(train_dataset)} 张训练图片")
    except ValueError as e:
        print(f"❌ 数据集加载失败：{e}")
        return
    except FileNotFoundError as e:
        print(f"❌ 数据集加载失败：{e}")
        return

    # -------------------------- 初始化模型 --------------------------
    # 自动选择CPU/GPU（有GPU会自动使用，训练更快）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNet(n_channels=3, n_classes=1).to(device)
    # 损失函数：二分类用BCEWithLogitsLoss（自带sigmoid，数值更稳定）
    criterion = nn.BCEWithLogitsLoss()
    # 优化器：Adam（常用的优化器，收敛速度快）
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------------------------- 训练模型 --------------------------
    model.train()  # 切换到训练模式
    print(f"\n开始训练（使用 {device}）...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            # 数据移到设备上（CPU/GPU）
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播：输入图片，得到模型输出
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, masks)

            # 反向传播+优化：更新模型参数
            optimizer.zero_grad()  # 清空梯度（避免梯度累积）
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 更新模型参数

            running_loss += loss.item() * images.size(0)

            # 每5个批次打印一次损失（便于观察训练进度）
            if (batch_idx + 1) % 5 == 0:
                print(f"Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 打印每轮的平均损失
        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}\n')

    # -------------------------- 测试单张图片 --------------------------
    model.eval()  # 切换到评估模式（禁用Dropout等层）
    # 读取并预处理测试图片
    test_image = Image.open(test_img_path).convert('RGB')
    test_mask = Image.open(test_mask_path).convert('L')
    # 增加batch维度（模型输入需要batch维度，用unsqueeze(0)添加）
    test_image_tensor = transform(test_image).unsqueeze(0).to(device)

    # 预测（禁用梯度计算，加快速度并节省内存）
    with torch.no_grad():
        pred_output = model(test_image_tensor)
        # 用sigmoid转为概率，再二值化得到分割掩码（大于0.5为道路）
        pred_mask = torch.sigmoid(pred_output) > 0.5

    # -------------------------- 可视化结果 --------------------------
    plt.figure(figsize=(15, 5))
    # 原图
    plt.subplot(1, 3, 1)
    plt.imshow(test_image)
    plt.title('Original Drone Image (Texture)')
    plt.axis('off')
    # 真实掩码
    plt.subplot(1, 3, 2)
    plt.imshow(test_mask, cmap='gray')
    plt.title('Ground Truth (Road)')
    plt.axis('off')
    # 预测掩码
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask.cpu().squeeze().numpy(), cmap='gray')
    plt.title('Predicted Road')
    plt.axis('off')

    # 保存结果图片到当前目录
    plt.savefig("drone_segmentation_texture_result.png")
    plt.show()
    print("✅ 纹理数据分割结果已保存为 drone_segmentation_texture_result.png")

if __name__ == '__main__':
    main()