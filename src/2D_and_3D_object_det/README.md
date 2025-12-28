# Monocular 3D Object Detector

基于单目图像的轻量级 3D 目标检测器，结合 YOLOv8 进行 2D 检测与 MiDaS 进行深度估计，实现近似 3D 中心点定位。

---

## 📦 功能特点

- 使用 **YOLOv8n** 快速检测图像中的目标
- 利用 **MiDaS_small** 实时估计单目深度图
- 基于相机内参将 2D 检测框中心反投影为近似 3D 点
- 自动下载预训练模型，无需手动配置权重
- 输出带标注的检测结果图 + 深度图（`depth_map.png`）

> ⚠️ 注意：MiDaS 输出为**相对深度**，3D 坐标无真实物理尺度，仅适用于定性分析或相对位置估计。

---

## 🛠️ 依赖要求

```txt
YOLOv8核心库（必须）
ultralytics>=8.0.0,<9.0.0

图像处理（必须）
opencv-python>=4.5.0,<5.0.0

可视化（必须）
matplotlib>=3.0.0,<4.0.0

PyTorch核心（必须）
torch>=2.0.0,<2.4.0
torchvision>=0.15.0,<0.17.0

基础数值计算（建议指定）
numpy>=1.21.0,<2.0.0

可选（增强兼容性）
pillow>=8.0.0,<11.0.0
psutil>=5.8.0
pyyaml>=6.0,<7.0.0
```

### 安装命令
```bash
pip install ultralytics opencv-python matplotlib torch torchvision numpy pillow psutil pyyaml
```

> 💡 建议从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取匹配 CUDA 版本的安装命令以启用 GPU 加速。

---

## ▶️ 快速开始

1. **准备测试图像**
   ```bash
   mkdir -p data
   cp your_image.jpg data/sample_image.jpg
   ```

2. **运行检测脚本**
   ```bash
   python monocular_3d_detector.py
   ```

3. **查看输出**
   - `output_3d.jpg`：带 2D 检测框和类别标签的结果图
   - `depth_map.png`：归一化后的深度图（使用 plasma 色彩映射）
   - 控制台打印每个检测目标的近似 3D 中心坐标 `(X, Y, Z)`

---

## 📂 项目结构

```
.
├── monocular_3d_detector.py   # 主程序
├── data/
│   └── sample_image.jpg       # 示例输入图像
├── output_3d.jpg              # 输出检测结果（自动生成）
└── depth_map.png              # 输出深度图（自动生成）
```

---

## ⚙️ 相机内参说明

当前代码使用 KITTI 数据集典型内参作为默认值：
```python
K = [[721.5,     0,   W/2],
     [    0, 721.5,   H/2],
     [    0,     0,     1]]
```
若用于其他相机，请替换为实际标定参数以提高 3D 估计精度。

---

## 📌 注意事项

- 首次运行会自动下载 `yolov8n.pt` 和 `MiDaS_small` 模型（需联网）。
- 深度图由 MiDaS 生成，为**逆深度（inverse depth）**，非真实米制距离。
- 3D 坐标基于图像中心像素点估算，未考虑物体尺寸或姿态。
- 推荐使用 GPU（CUDA）以获得更快推理速度。

---

## 📚 参考项目

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [MiDaS (Intel ISL)](https://github.com/isl-org/MiDaS)

---

> ✨ 本项目适用于教学演示、快速原型开发或单目 3D 感知初步探索。
