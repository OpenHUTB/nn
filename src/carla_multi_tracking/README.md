# CARLA 模拟器中的多车辆跟踪系统
# Multi Vehicle tracking in carla simulator
#
本项目提供了一个在 CARLA 自动驾驶模拟器中实现多目标跟踪（Multi-Object Tracking, MOT）的完整解决方案，结合了 YOLO 目标检测算法和 DeepSORT 跟踪算法。

**技术说明：**
- **YOLO (You Only Look Once)**：一种实时目标检测算法，能够在图像中快速识别和定位多个目标
- **DeepSORT**：一种基于深度学习的多目标跟踪算法，通过结合运动信息和外观特征实现对目标的持续跟踪
- **CARLA**：一个开源的自动驾驶模拟器，提供逼真的城市环境和交通场景

本项目使用的 YOLOv8 模型在 CARLA 数据集上进行了训练，该数据集可在 Kaggle 上获取：https://www.kaggle.com/datasets/alechantson/carladataset

![演示效果](https://github.com/Bsornapudi/Carla-YOLO-DeepSort-Multi-Object-Tracking/assets/48683074/c365a981-e314-4cae-b4aa-d234b3de5cfa)

---

## 环境要求与安装指南

在安装所需的 Python 包之前，请确保已完成以下软件的安装和配置：

### 1. CARLA 模拟器
**用途**：提供自动驾驶仿真环境，生成车辆和行人等交通场景

下载 CARLA 模拟器并按照官方指南进行安装：
- 下载地址：https://github.com/carla-simulator/carla/releases
- 安装指南：https://carla.readthedocs.io/en/latest/start_quickstart/

### 2. CUDA
**用途**：NVIDIA 的并行计算平台，用于加速深度学习模型的 GPU 运算

下载并安装 CUDA Toolkit：
- 下载地址：https://developer.nvidia.com/cuda-downloads
- **注意**：请根据您的显卡型号选择兼容的 CUDA 版本

### 3. cuDNN
**用途**：NVIDIA 的深度神经网络加速库，为深度学习框架提供优化

安装 CUDA Deep Neural Network 库：
- 安装指南：https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
- **注意**：cuDNN 版本需与 CUDA 版本匹配

### 4. Anaconda
**用途**：Python 环境管理工具，用于创建独立的虚拟环境

下载并安装 Anaconda：
- 下载地址：https://www.anaconda.com/download

### 5. PyMOT（可选，用于评估）
**用途**：多目标跟踪评估工具，用于计算 MOTA、MOTP 等跟踪性能指标

下载或克隆此仓库用于后续评估：
- 仓库地址：https://github.com/Videmo/pymot

---

## 安装步骤

### 步骤 6：创建并激活 Conda 虚拟环境

```bash
conda create --name <env-name> python=3.8
conda activate <env-name>
```

**参数说明：**
- `<env-name>`：您的虚拟环境名称，可自定义（如 `carla-tracking`）
- `python=3.8`：指定 Python 版本为 3.8，确保与项目依赖兼容

### 步骤 7：安装项目依赖包

```bash
pip install -r requirements.txt
```

**注意**：原命令 `pip install requirements.txt` 有误，正确应为 `pip install -r requirements.txt`

---

## 运行指南

### 步骤 8：启动 CARLA 模拟器
完成环境配置后，运行 `Carla.exe` 文件启动模拟器

**说明**：模拟器启动后会监听默认端口（2000-2002），等待客户端连接

### 步骤 9：打开命令行或 Jupyter
从 Anaconda Prompt 启动命令行或 Jupyter Notebook

### 步骤 10：运行目标跟踪
在 Jupyter 中运行 `track.ipynb` 文件，开始多目标跟踪

**功能说明**：此脚本将连接 CARLA 模拟器，实时捕获图像并使用 YOLO+DeepSORT 进行目标检测与跟踪

### 步骤 11：评估跟踪性能
依次运行以下文件生成评估指标：
1. `gt_deepsort.ipynb`：生成 Ground Truth 和 DeepSORT 跟踪结果
2. `evaluate.ipynb`：计算 MOTA 和 MOTP 值

**评估指标说明：**
- **MOTA (Multiple Object Tracking Accuracy)**：多目标跟踪准确度，衡量跟踪的整体准确性
- **MOTP (Multiple Object Tracking Precision)**：多目标跟踪精确度，衡量目标位置估计的精确程度

---

## PyTorch 安装注意事项

在安装 PyTorch 时，请根据您的 CUDA 和 cuDNN 配置选择兼容的版本。

**安装步骤：**
1. 访问 PyTorch 官网：https://pytorch.org/
2. 向下滚动找到版本选择器
3. 选择您的系统配置（操作系统、包管理器、语言、CUDA 版本）
4. 网站将自动生成对应的安装命令

**示例命令（CUDA 11.7 版本）：**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

**版本兼容性提示：**
- PyTorch 版本需与 CUDA 版本匹配
- 建议使用 `nvidia-smi` 命令查看您的显卡支持的 CUDA 版本
- 不同版本的 CUDA 对应不同的 PyTorch 安装源（如 cu117、cu118、cu121 等）

---

## 常见问题

1. **模拟器连接失败**：确保 CARLA 模拟器已启动并正在运行
2. **CUDA 内存不足**：尝试减小批处理大小或使用更小的模型
3. **跟踪效果不佳**：检查 YOLO 模型是否正确加载，确认数据集训练是否充分
