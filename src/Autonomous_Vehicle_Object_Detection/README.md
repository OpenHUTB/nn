# 自动驾驶车辆目标检测与轨迹规划

## 项目概述
本毕业设计聚焦于基于YOLOv3算法和CARLA仿真器的自动驾驶车辆目标检测与轨迹规划。该项目展示了如何将最先进的目标检测技术与真实的自动驾驶仿真器相结合，以提升车辆的感知能力和决策能力。

### 核心功能：
⦁	采用YOLOv3算法实现目标检测

⦁	与CARLA仿真器集成，构建真实的自动驾驶场景

⦁	基于检测到的目标和环境约束进行轨迹规划

⦁	使用TensorBoard实现性能监控与可视化

## 安装步骤

### 前置条件：
⦁	Python 3.7

⦁	CARLA仿真器 0.9.11 版本

### 操作步骤：

1. 下载适用于Windows系统的指定版本CARLA仿真器压缩包：
````plaintext
  https://github.com/carla-simulator/carla/releases
````

2. 参考官方文档安装CARLA\_0.9.11

3. 克隆本项目仓库：
```plaintext
  https://github.com/William4861/Autonomous-Vehicle-Object-Detection-and-Trajectory-Planning-using-YOLOv3-and-CARLA-Simulator
```
4. 安装所需的Python库： （执行该命令前确保cd到requirements.txt文件目录）
```plaintext
  pip install -r requirements.txt
```  
```commandline
  pip install setuptools
```
(执行该命令前确保cd到CARLA_0.9.11\WindowsNoEditor\PythonAPI\carla\dist目录)
```commandline
  easy_install carla-0.9.11-py3.7-win-amd64.egg
```

  

## 使用方法

1. 将 object\_detection.py 文件和 generate_traffic.py 文件复制到CARLA的PythonAPI示例目录中： 
```plaintext
  cp object\_detection.py \\CARLA\_0.9.11\\WindowsNoEditor\\PythonAPI\\examples
```
```plaintext
  cp generate_traffic.py \\CARLA\_0.9.11\\WindowsNoEditor\\PythonAPI\\examples
```

2. 下载yolov3.weights和yolo3.cfg文件至examples目录
```
  https://huggingface.co/spaces/Epitech/Scarecrow/resolve/main/yolov3.weights
```
```
  https://www.kaggle.com/datasets/ravi02516/trained-weights-and-cfg?select=yolov3.cfg
```
3. 运行CARLA仿真器： 
```plaintext
  cd CARLA\_0.9.11
```
```
  ./CarlaUE4.exe
```

4. 打开新的终端，导航至PythonAPI示例目录并运行脚本：
```plaintext
  cd CARLA\_0.9.11/PythonAPI/examples
```
```commandline
  python generate_traffic.py
```
5. 打开新的终端，导航至PythonAPI示例目录并运行脚本：
```plaintext
  cd CARLA\_0.9.11/PythonAPI/examples
```
```
  python object\_detection.py
```

  

## 模型性能监控
通过TensorBoard监控和跟踪YOLOv3模型性能的步骤如下：

1. 启动TensorBoard：
```plaintext
  tensorboard --logdir=path/to/logs
```

2. 打开网页浏览器，访问 http://localhost:6006 以查看TensorBoard仪表板。

  

  ## 项目结构
  ⦁	 object\_detection.py ：目标检测与轨迹规划的主Python脚本

  ⦁	 requirements.txt ：所需Python库的清单

  ⦁	 models/ ：存放训练好的YOLOv3模型权重的目录

  ⦁	 config/ ：存放训练好的YOLOv3模型配置文件的目录

  ⦁	 logs/ ：用于性能监控的TensorBoard日志文件

  

  ## 贡献方式
  欢迎通过贡献代码改进本项目。您可自由分叉（fork）仓库、修改代码，并提交拉取请求（pull request）。

  

  ## 许可协议
  本项目基于MIT许可协议开源，详情请参见LICENSE文件。

  

  ## 致谢
  ⦁	感谢CARLA仿真器团队提供了稳健的自动驾驶仿真平台

  ⦁	感谢YOLOv3开发者研发了高效的目标检测算法 



  ## 参考文档
  *[自动驾驶汽车物体检测和轨迹规划使用YOLOv3和CARLA模拟器](https://github.com/ROBERT-ADDO-ASANTE-DARKO/Autonomous-Vehicle-Object-Detection-and-Trajectory-Planning-using-YOLOv3-and-CARLA-Simulator)

