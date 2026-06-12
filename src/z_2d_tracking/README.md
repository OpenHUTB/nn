# CARLA 仿真 2D 跟踪：多算法对比与课程实践

## 1. 项目选题
本项目为机器学习课程大作业，旨在自动驾驶仿真平台 CARLA 中实现 2D 多目标跟踪算法，并对比 SORT、DeepSORT、OC-SORT 等经典跟踪器的性能差异。

## 2. 项目目标
- 在 CARLA 环境中实现车辆、行人的 2D 目标跟踪
- 对比不同多目标跟踪算法的效果与鲁棒性
- 完成环境搭建、代码调试、实验分析
- 有余力时扩展至 3D 目标跟踪，更贴近真实自动驾驶感知

## 4. 运行环境
| 环境 | 版本要求 |
| Python | 3.7.16 |
| CARLA | 0.9.14 |
| 操作系统 |Windows 11 |
| GPU | 推荐 NVIDIA GPU（用于深度学习加速） |

## 5. 主要依赖
项目所有依赖均基于 **Python 3.7.16** 可正常运行的环境导出，统一收录在 `requirements.txt` 中。

### 核心基础依赖
- `numpy`：数值计算
- `opencv-python`：图像处理、画面可视化
- `carla`：CARLA 仿真平台 Python API（需手动安装对应 whl 包）

### 深度学习框架
- `torch`、`torchvision`：目标检测器运行依赖
- `tensorflow`、`Keras`：图像特征编码相关功能

### 跟踪算法依赖
- `filterpy`：卡尔曼滤波实现
- `lap`：匈牙利匹配算法
- `scipy`、`scikit-image`：矩阵运算、距离计算与图像处理

### 辅助工具依赖
包含日志输出、进度展示、数据解析、网络请求等配套工具库，为项目运行必需依赖。

## 6. 安装与运行

### 6.1 安装通用依赖
先激活项目对应的 conda 环境，再执行以下命令安装全部基础依赖：

pip install -r requirements.txt


### 6.2 安装 CARLA Python API
- 下载并解压 CARLA 0.9.14 WindowsNoEditor
- 进入路径：CARLA_0.9.14/WindowsNoEditor/PythonAPI/carla/dist/
- 找到文件：carla-0.9.14-cp37-cp37m-win_amd64.whl
- 终端切换至该文件所在目录，执行安装：
	pip install carla-0.9.14-cp37-cp37m-win_amd64.whl
- 验证安装结果：
	python -c "import carla; print(carla.__version__)"
	输出 0.9.14 即代表安装成功。

### 6.3 启动并运行项目
- 双击运行 CarlaUE4.exe 启动 CARLA 仿真器，等待场景加载完成
- 切换至项目根目录，执行程序：
	python 2d-detection-frcnn.py

## 7. 实现计划
- 完成 CARLA 环境配置与调试
- 实现基于真值的 2D 目标检测与跟踪
- 集成 YOLO 检测器 + SORT/DeepSORT 跟踪器
- 对比不同算法在不同场景下的性能
- 结果可视化与分析总结
- 可选扩展：3D 目标跟踪实现与对比

## 8. 参考与致谢
本项目基于优秀开源项目学习与复现：
基础框架参考：wuhanstudio/2d-carla-tracking
在此向原作者表示感谢，本项目在此基础上完成课程实践、算法对比与扩展尝试。