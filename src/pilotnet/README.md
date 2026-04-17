# PilotNet  
基于 TensorFlow 实现的论文复现：《解释端到端学习训练的深度神经网络如何操控汽车》([Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car](https://arxiv.org/pdf/1704.07911.pdf))，作者来自 NVIDIA、Google Research 与纽约大学。

# 安装  
按照以下步骤在本地安装并运行本项目。

# 环境准备  
1. [Anaconda/Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)  
2. [CARLA 仿真模拟器](http://carla.org)

# 操作步骤  
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
   python main.py
   ```

# 使用方法  

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

# 常见问题  
1. 训练阶段  
   * 若你的计算机算力有限，该模型无法强制处理高分辨率图像（如 1920×1080）。如果强行尝试，程序会因资源耗尽而抛出异常并退出。建议从默认的较低画质逐步上调，测试系统能承受的极限。  
   * 有时你可能会过于急切，试图用仅 2 分钟的录制数据训练 100 个 epoch。这显然会因数据生成器无法为所有 epoch 提供足够数据而报错。解决方法是增加录制数据量，或者减少 epoch 数量。

2. 数据生成器 
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

# 目录结构  
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
    |-main.py （程序入口）
    |-requirements.txt （Python 依赖清单）
    |-README.md （本文档）
```