# Multi Vehicle Tracking

This repo hosts a comprehensive solution for multi-object tracking in combination with YOLO and DeepSORT in CARLA simulator.
YOLOv8 model is trained on CARLA dataset, which is available in Kaggle: https://www.kaggle.com/datasets/alechantson/carladataset

![ezgif com-optimize](https://github.com/Bsornapudi/Carla-YOLO-DeepSort-Multi-Object-Tracking/assets/48683074/c365a981-e314-4cae-b4aa-d234b3de5cfa)

## Requirements

Following software are required before installing required packages:

1. CARLA simulator: Download CARLA simulator and follow the instruction guide - https://carla.readthedocs.io/en/latest/start_quickstart/
2. CUDA: Download and install CUDA - https://developer.nvidia.com/cuda-downloads
3. CuDNN: Install CUDA DNN - https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
4. Anaconda: https://www.anaconda.com/download
5. PyMOT: Download / clone this repo for evaluations - https://github.com/Videmo/pymot

## Setup

6. Create and activate a new virtual conda env:

```bash
conda create --name <env-name> python=3.8
conda activate <env-name>
```

NOTE: `<env-name>` should be the name of your virtual env.

7. Install required packages:

```bash
pip install -r requirements.txt
```
## 📦 模型权重文件下载
由于模型文件体积较大，未直接上传至仓库，请通过以下方式获取：

### 下载链接
链接: https://pan.baidu.com/s/1qubYmwj-uwLHeTeCifNjJw?pwd=ujqw 提取码: ujqw

### 配置方法
1.  下载模型权重文件（`.pt` 格式）
2.  将文件放入项目根目录下的 `weights` 文件夹中
3.  确保路径为：`weights/你的模型文件名.pt`，即可正常运行代码


8. Once the setup is done, run Carla.exe file to launch simulator.
9. Open command prompt or launch Jupyter from conda prompt.
10. Run `track.ipynb` file in Jupyter.
11. Run `gt_deepsort.ipynb` file followed by `evaluate.ipynb` to generate MOTA and MOTP values.

## PyTorch Compatibility

Check PyTorch compatibility and install an appropriate version based on your CUDA and cuDNN configurations from https://pytorch.org/. Scroll down and you will see an option to select your system configs and this will generate pip install command for CUDA+torch which is compatible.

Example:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

