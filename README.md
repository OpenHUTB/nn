# Lane Line Prediction with OpenPilot Supercombo Model
基于 OpenPilot 的 supercombo 预训练模型实现视频车道线实时预测，通过 OpenCV 可视化车道线（左车道-蓝、右车道-红、预测路径-绿），适配 Ubuntu 虚拟机环境。

## 📌 项目功能
- 读取 HEVC 格式视频文件，提取帧并预处理为模型输入格式
- 使用 supercombo 模型推理车道线坐标
- 基于 OpenCV 实时可视化车道线（放大圆点，解决偏左问题）
- 适配 Ubuntu 虚拟机环境，无 Matplotlib 渲染依赖
- 完善的异常处理（文件不存在、帧数不足、模型加载失败等）

## 🛠️ 环境要求
| 依赖项 | 版本要求 |
|--------|----------|
| Python | 3.8+ (测试环境：3.10) |
| TensorFlow | 2.x |
| OpenCV-Python | 4.5.5.62+ |
| NumPy | 1.21+ |
| Ubuntu | 20.04/22.04 (虚拟机/物理机) |

## 🚀 快速开始

### 1. 克隆仓库（可选）
```bash
git clone <你的GitHub仓库地址>
cd <仓库名称>
```

### 2. 创建并激活虚拟环境
```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

### 3. 安装依赖
```bash
# 安装核心依赖
pip install tensorflow opencv-python==4.5.5.62 numpy

# 安装虚拟机显示依赖（可选，解决OpenCV窗口问题）
sudo apt update && sudo apt install -y libgtk-3-dev libgl1-mesa-glx
```

### 4. 准备模型和视频文件
- 将 `supercombo.h5` 模型文件放入路径：`/home/dacun/桌面/openpilot-modeld-main/models/`
- 将 HEVC 格式视频文件（如 `sample.hevc`）放入路径：`/home/dacun/nn/`

### 5. 运行程序
```bash
# 进入脚本目录
cd /home/dacun/nn/src/openpilot_model/

# 直接运行（无需传参，路径已写死适配环境）
python3 main.py
```

## 📝 使用说明
1. 程序启动后会依次显示：
   - `Loading model...`：模型加载中
   - `Preprocessing frames...`：视频帧预处理中
   - 视频画面 + 车道线可视化（蓝=左车道、红=右车道、绿=预测路径）
2. 按 `Q` 键可退出程序（需先点击视频窗口激活）
3. 终端会输出推理进度和异常信息（如某帧推理失败）

## 🔍 核心逻辑
### 1. 视频预处理
- 将视频帧转换为 YUV420 格式，缩放至 512x384（模型输入尺寸）
- 对帧进行归一化和张量转换，适配 supercombo 模型输入要求

### 2. 模型推理
- 使用连续 2 帧作为模型输入，推理得到车道线 x 坐标（共 192 个点）
- 模型输出坐标映射到视频显示尺寸（800x600），并右移 100 像素解决偏左问题

### 3. 可视化
- 放大车道线圆点（8px），确保肉眼可见
- 仅依赖 OpenCV 绘制，避免 Matplotlib 虚拟机渲染问题
- 异常帧保底显示原始视频画面，不中断程序

## 📸 效果展示
> 请替换为你的实际运行截图（虚拟机截屏方法见下文）
![Lane Line Prediction Demo](demo.png)
- 左侧为原始视频帧，右侧为叠加车道线后的效果
- 蓝色圆点：左车道线；红色圆点：右车道线；绿色圆点：预测行驶路径

## ❌ 常见问题解决
### 1. 窗口显示问号/乱码
- 原因：Ubuntu 虚拟机缺少中文字体
- 解决：程序已替换为全英文提示，无需额外配置

### 2. 车道线偏左
- 原因：模型输出坐标范围与视频显示尺寸不匹配
- 解决：程序已默认将 x 坐标右移 100 像素，可修改 `+100` 调整偏移量

### 3. OpenCV 窗口黑屏/无内容
- 步骤1：启用虚拟机 3D 加速（设置 → 显示 → 勾选 3D 加速）
- 步骤2：重装 OpenCV 依赖：
  ```bash
  pip uninstall -y opencv-python
  pip install opencv-python==4.5.5.62
  ```
- 步骤3：验证视频文件完整性：`ffplay /home/dacun/nn/sample.hevc`

### 4. 模型加载失败
- 检查模型路径：`/home/dacun/桌面/openpilot-modeld-main/models/supercombo.h5`
- 确保模型文件完整（未损坏、未截断）

## 🖥️ 虚拟机截屏方法
```bash
# Ubuntu 虚拟机内部截屏
# 全屏截图
Print Screen

# 区域截图
Shift + Print Screen

# 保存到剪贴板（当前窗口）
Alt + Print Screen
```

## 📄 许可证
本项目仅用于学习交流，supercombo 模型版权归 OpenPilot 官方所有。

## 📧 联系作者
如有问题，可提交 Issue 或联系：<你的邮箱/联系方式>