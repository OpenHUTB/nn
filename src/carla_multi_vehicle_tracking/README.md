在 Carla 模拟器中使用 YOLOv8 + DeepSORT 实现多车辆目标跟踪

项目说明

本项目基于 YOLOv8 与 DeepSORT 算法，在 Carla 模拟器环境下实现多车辆实时跟踪。模型在 Carla 数据集上训练，可直接用于仿真环境目标跟踪、指标评估与二次开发。

环境依赖（必须先安装）

Carla 模拟器https://carla.readthedocs.io/en/latest/start\_quickstart/

CUDAhttps://developer.nvidia.com/cuda-downloads

cuDNNhttps://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

Anacondahttps://www.anaconda.com/download

PyMOT（用于指标评估）https://github.com/Videmo/pymot

运行步骤

创建并激活 Conda 虚拟环境

plaintext

conda create --name carla\_tracking python=3.8

conda activate carla\_tracking

安装项目依赖

plaintext

pip install -r requirements.txt

安装匹配 CUDA 版本的 PyTorch前往 https://pytorch.org/ 获取对应命令示例（CUDA 11.7）：

plaintext

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

启动 Carla 模拟器运行 Carla.exe 或对应启动脚本

启动 Jupyter

plaintext

jupyter notebook

运行跟踪主程序打开并执行 track.ipynb

生成真值与跟踪结果打开并执行 gt\_deepsort.ipynb

计算跟踪指标（MOTA / MOTP）打开并执行 evaluate.ipynb

数据集

模型训练使用 Carla 数据集，下载地址：https://www.kaggle.com/datasets/alechantson/carladataset

后续更新说明

本项目将逐步优化：

跟踪效果调参

代码结构整理

新增命令行运行方式

支持更多类别跟踪

可视化与日志优化

