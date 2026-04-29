# CARLA 多目标跟踪（YOLOv8 + DeepSORT）
> 基于 CARLA 仿真环境的数据集与视觉检测（YOLOv8），结合 DeepSORT 数据关联，实现多目标检测与轨迹跟踪。

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![CARLA](https://img.shields.io/badge/CARLA-Simulator-green)

## 项目简介
该方案面向自动驾驶/智能交通的典型研究流程：先在 CARLA 场景中进行目标检测（YOLOv8），再将检测结果输入 DeepSORT，通过“运动状态 + 外观特征”的方式完成跨帧关联，从而得到多目标的轨迹。最终可借助评估脚本输出经典跟踪指标（例如 `MOTA` / `MOTP`）。

YOLOv8 模型训练数据来自 Kaggle 的 Carla 数据集：  
https://www.kaggle.com/datasets/alechantson/carladataset

![ezgif com-optimize](https://github.com/Bsornapudi/Carla-YOLO-DeepSort-Multi-Object-Tracking/assets/48683074/c365a981-e314-4cae-b4aa-d234b3de5cfa)

> 注意：本仓库目录当前仅包含文档（`README.MD`）。如果你在本地还集成了原项目的代码/Notebook（例如 `track.ipynb`、`evaluate.ipynb`），请把下面的步骤对应到你实际的文件名与路径。

## 核心功能
* **目标检测**：使用 YOLOv8 在仿真画面中检测多目标，并输出边界框（bbox）等信息。
* **多目标数据关联**：DeepSORT 根据检测结果完成跨帧身份保持（ID persistence），生成轨迹。
* **跟踪效果评估**：可生成 `MOTA` / `MOTP` 等指标，用于量化跟踪质量。
* **仿真验证流程**：在 CARLA 运行后对检测与跟踪效果进行对比与调参。

## 环境准备
在安装 Python 依赖前，需要先准备以下软件环境：

1. **CARLA Simulator**
   - 下载并按官方快速开始安装/启动：  
     https://carla.readthedocs.io/en/latest/start_quickstart/
2. **CUDA**
   - 安装 CUDA：  
     https://developer.nvidia.com/cuda-downloads
3. **cuDNN**
   - 安装对应版本的 cuDNN：  
     https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
4. **Anaconda/Miniconda**
   - 用于创建隔离 Python 环境：  
     https://www.anaconda.com/download
5. **PyMOT（用于评估）**
   - 建议下载/克隆用于跟踪评估：  
     https://github.com/Videmo/pymot

### 创建并激活 Conda 环境
```bash
conda create --name <env-name> python=3.8
conda activate <env-name>
```

### 安装依赖包
```bash
pip install -r requirements.txt
```

> `requirements.txt` 路径以你实际集成位置为准。

## 快速开始
下面给出一个典型的运行顺序（Notebook/脚本文件名以原项目为准）：

1. **启动 CARLA**
   - 运行 `Carla.exe`（或按你安装方式启动 CARLA 服务/客户端）。
2. **在 Conda 环境下打开命令行 / Jupyter**
   - 从 Conda 提示符进入后再启动 Jupyter，以确保依赖版本一致。
3. **运行跟踪流程（Track）**
   - 在 Jupyter 中运行跟踪 Notebook（原项目常见为 `track.ipynb`）。
4. **生成 DeepSORT 相关的 GT/关联（如适用）**
   - 依次运行生成 GT 或 DeepSORT 关联所需的 Notebook（原项目示例常见为 `gt_deepsort.ipynv`，请根据你实际文件名修正）。
5. **评估生成 `MOTA` / `MOTP`**
   - 最后运行评估 Notebook（原项目常见为 `evaluate.ipynb`），输出指标结果。

## PyTorch + CUDA/cuDNN 兼容性（重要）
由于 PyTorch 的二进制包与 CUDA/cuDNN 强相关，需要先根据你机器的 CUDA 版本选择匹配的安装命令。

参考官方选择器：https://pytorch.org/ （滚动选择系统配置，生成对应的 `pip install` 命令）

例如（仅作为示例，具体版本请按你的 CUDA 版本替换）：
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

## 配置说明
本模块以“检测 -> 关联 -> 评估”为主线，通常会涉及以下配置项（以你实际使用的原项目为准）：
* **模型权重路径**：YOLOv8 weights / DeepSORT 相关权重
* **输入数据路径**：CARLA 导出数据、视频帧或数据集目录
* **检测阈值**：置信度阈值、NMS 阈值等（影响后续关联效果）
* **评估输出目录**：MOTA/MOTP 结果与中间产物保存位置

建议优先查看你实际 Notebook/代码里的 `config` 或参数区块，并用中文标注理解每个参数的作用。

## 项目结构
当前 `carla_yolo_deepsort_multi_object_tracking` 目录下仅包含文档文件：
```
README.MD
```

如果你希望做到“文档与工程一一对应”，建议将原项目的代码、Notebook 与本 README 对齐（例如补齐 `track.ipynb`、`evaluate.ipynb` 等文件名）。

## 开发日志
* `2026-04`：根据 `carla_yolo_planner/README.md` 的结构化模板，将本 README 的内容进行了中文化与可读性优化，并补充运行顺序与 CUDA/cuDNN 兼容性说明。
