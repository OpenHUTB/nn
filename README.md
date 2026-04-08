<<<<<<< HEAD
# PilotNet  
基于 TensorFlow 实现的论文复现：《解释端到端学习训练的深度神经网络如何操控汽车》([Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car](https://arxiv.org/pdf/1704.07911.pdf))，作者来自 NVIDIA、Google Research 与纽约大学。

## 安装  
按照以下步骤在本地安装并运行本项目。

### 环境准备  
1. [Anaconda/Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)  
2. [CARLA 仿真模拟器](http://carla.org)

### 操作步骤  
1. 克隆仓库  
   ```
   https://github.com/vishalkrishnads/pilotnet.git
   ```
2. 进入工作目录  
   ```
   cd pilotnet
   ```
3. 创建一个指定 Python 3.8 版本的 conda 环境并激活  
   ```
   conda create -n "pilotnet" python=3.8.0
   conda activate pilotnet
   ```
4. 按照[官方文档说明](https://www.tensorflow.org/install/)在环境中安装 TensorFlow。如果你的设备没有[支持 CUDA 的 GPU](https://developer.nvidia.com/cuda-gpus) 或根本没有 GPU，请跳过此步继续。  
5. 安装其他所需模块  
   ```
   pip install -r requirements.txt
   ```
6. 运行应用  
   ```
   python app.py
   ```

## 使用方法  

* 运行 `main.py` 文件，这是程序的入口。你将看到如下菜单：  
   ```
   $ python main.py
   # 一段横幅文字
   1.  使用已有数据训练模型
   2.  生成新的驾驶数据
   3.  对单张视频帧进行预测
   4.  对实时视频流进行预测
   5.  结束，我要退出。
   请输入你的选择 >> 
   ```
* 输入对应数字并遵循后续提示操作。菜单会无限循环显示，选择 5 即可退出。

## 常见问题  
1. **训练阶段**  
   * 若你的计算机算力有限，该模型无法强制处理高分辨率图像（如 1920×1080）。如果强行尝试，程序会因资源耗尽而抛出异常并退出。建议从默认的较低画质逐步上调，测试系统能承受的极限。  
   * 有时你可能会过于急切，试图用仅 2 分钟的录制数据训练 100 个 epoch。这显然会因数据生成器无法为所有 epoch 提供足够数据而报错。解决方法是增加录制数据量，或者减少 epoch 数量。

2. **数据生成器**  
   * 在 WSL 环境下，数据生成器针对 WSL 连接设有回退机制。但若仍连接失败，可尝试用 `ping $(hostname).local` 命令获取宿主机 IP 地址。随后打开 `app.py`，在 `Collector.run2()` 中将 IP 从 `127.27.144.1` 修改为你的实际 IP 地址，并重启程序。  
     ```python
     # ...
     warn('你的 CARLA 服务器似乎存在问题。正在尝试使用 WSL 地址重试...')

     # 在此修改
     # client = carla.Client('172.27.144.1', 2000)
     client = carla.Client('<你的IP>', 2000)
     
     world = client.get_world()
     # ...
     ```
   * 网上有许多关于无法连接 CARLA 服务器的报告，多数与端口阻塞或网络配置有关。你可以用以下代码测试连接是否正常。若此段代码运行失败，请优先解决连接问题，之后数据生成器应能正常工作。  
     ```python
     import carla

     client = carla.Client('localhost', 2000)
     world = client.get_world()
     ```
   * 若磁盘剩余空间不足，生成器自然无法写入数据。录制内容默认存储在 `recordings/` 目录下，请尝试清理磁盘空间。

   * 性能优化：数据采集时已移除 pygame 实时预览窗口，显著降低 CPU/GPU 占用，录制过程更加流畅。

3. **对单帧图像进行预测**  
   * 此步骤唯一可能的问题是指定了不存在的文件路径，例如测试图片路径或已保存的模型路径。输入错误路径会导致脚本直接崩溃，因此请仔细核对路径是否正确。

## 目录结构  
```
pilotnet
    |
    |-pilotnet
    |   |
    |   |- data.py （用于训练的自定义数据类型）
    |   |- model.py （模型定义及相关辅助函数）
    |
    |-utils
    |   |-piloterror.py
    |   |-collect.py （数据采集器）
    |   |-screen.py （屏幕工具）
    |
    |-app.py （程序入口）
    |-requirements.txt （Python 依赖清单）
    |-README.md （本文档）
```
=======
# 神经网络实现代理

利用神经网络/ROS 实现 Carla（车辆、行人的感知、规划、控制）、AirSim、Mujoco 中人和载具的代理。

## 环境配置

* 平台：Windows 10/11，Ubuntu 20.04/22.04
* 软件：Python 3.7-3.12（需支持3.7）、Pytorch（尽量不使用Tensorflow）
* 相关软件下载 [链接](https://pan.baidu.com/s/1IFhCd8X9lI24oeYQm5-Edw?pwd=hutb)

## 功能模块表
| 模块类别                | 模块名 | 链接                                                                            | 其他                                                                  |
|-------------------|------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------|
| 感知             | 车辆检测以及跟踪 | [Yolov4_Vehicle_inspection](https://github.com/OpenHUTB/nn/tree/main/src/Yolov4_Vehicle_inspection)                                                                          | -                                          |
| 规划             | 车辆全局路径规划 | [carla_slam_gmapping](https://github.com/OpenHUTB/nn/tree/main/src/carla_slam_gmapping)                                                                          | -                                          |
| 控制             | 手势控制无人机 | [autonomus_drone_hand_gesture_project](https://github.com/OpenHUTB/nn/tree/main/src/autonomus_drone_hand_gesture_project)                                                                          | -                                          |
| 控制             | 倒车入库 | [autonomus_drone_hand_gesture_project](https://github.com/OpenHUTB/nn/tree/main/src/autonomus_drone_hand_gesture_project)                                                                          | [效果](https://github.com/OpenHUTB/nn/pull/4399)                                          |


## 贡献指南

准备提交代码之前，请阅读 [贡献指南](https://github.com/OpenHUTB/.github/blob/master/CONTRIBUTING.md) 。
代码的优化包括：注释、[PEP 8 风格调整](https://peps.pythonlang.cn/pep-0008/) 、将神经网络应用到Carla模拟器中、撰写对应 [文档](https://openhutb.github.io/nn/) 、添加 [源代码对应的自动化测试](https://docs.github.com/zh/actions/use-cases-and-examples/building-and-testing/building-and-testing-python) 等（从Carla场景中获取神经网络所需数据或将神经网络的结果输出到场景中）。

### 约定

* 每个模块位于`src/{模块名}`目录下，`模块名`需要用2-3个单词表示，首字母不需要大写，下划线`_`分隔，不能宽泛，越具体越好
* 每个模块的入口须为`main.`开头，比如：main.py、main.cpp、main.bat、main.sh等，提供的ROS功能以`main.launch`文件作为启动配置文件
* 每次pull request都需要保证能够通过main脚本直接运行整个模块，在提交信息中提供运行动图或截图；Pull Request的标题不能随意，需要概括具体的修改内容；README.md文档中提供运行环境和运行步骤的说明
* 仓库尽量保存文本文件，二进制文件需要慎重，如运行需要示例数据，可以保存少量数据，大量数据可以通过提供网盘链接并说明下载链接和运行说明


### 文档生成

测试生成的文档：
1. 使用以下命令安装`mkdocs`和相关依赖：
```shell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
（可选）安装完成后使用`mkdocs --version`查看是否安装成功。

2. 在命令行中进入`nn`目录下，运行：
```shell
mkdocs build
mkdocs serve
```
然后使用浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)，查看文档页面能否正常显示。

## 参考

* [代理模拟器文档](https://openhutb.github.io)
* 已有相关 [无人车](https://openhutb.github.io/doc/used_by/) 、[无人机](https://openhutb.github.io/air_doc/third/used_by/) 、[具身人](https://openhutb.github.io/doc/pedestrian/humanoid/) 的实现
* [神经网络原理](https://github.com/OpenHUTB/neuro)
>>>>>>> 377e0f2ca50fbcc5b8db8bc7644a76799e99dd1b
