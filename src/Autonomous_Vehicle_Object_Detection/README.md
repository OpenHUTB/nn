# 自动驾驶车辆目标检测系统（基于YOLO和CARLA）

## 项目概述
本项目聚焦于基于 YOLO 算法和 CARLA 仿真器的自动驾驶车辆目标检测。通过 CARLA 仿真环境采集真实场景数据，利用 YOLOv5 进行模型微调，实现对车辆、行人等目标的实时检测。系统支持数据采集、模型训练和实时检测一体化流程。

### 核心功能：
⦁	基于 YOLOv3-tiny 进行初始目标检测（通过 OpenCV DNN 模块实现），支持加载自定义训练模型

⦁	与 CARLA 仿真器集成，实时获取场景图像（640x480 分辨率）和目标真实位置

⦁	自动采集带标注的数据集（YOLO 格式标签），支持的目标类别包括：车辆（car）、卡车（truck）、公交车（bus）、行人（person）

⦁	基于 YOLOv5s 模型进行微调，提升特定场景检测性能

⦁	实时可视化真实目标（红框）与检测结果（绿框），计算检测精度（精确率、召回率，IOU 阈值 0.3）
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
```bash
  git clone https://github.com/William4861/Autonomous-Vehicle-Object-Detection-and-Trajectory-Planning-using-YOLOv3-and-CARLA-Simulator
```
4. 安装所需的Python库： （执行该命令前确保cd到requirements.txt文件目录）
```bash
  pip install -r requirements.txt
  pip install setuptools==40.8.0
```
5. 安装 CARLA Python API：（执行该命令前确保 cd 到 CARLA_0.9.11\WindowsNoEditor\PythonAPI\carla\dist 目录）
(执行该命令前确保cd到CARLA_0.9.11\WindowsNoEditor\PythonAPI\carla\dist目录)
```bash
  easy_install carla-0.9.11-py3.7-win-amd64.egg
```

  

## 使用方法

1. 将 object\_detection.py 文件和 generate_traffic.py 文件复制到CARLA的PythonAPI示例目录中： 
```plaintext
  # Linux/macOS 系统
  cp object_detection.py /CARLA_0.9.11/WindowsNoEditor/PythonAPI/examples
  cp generate_traffic.py /CARLA_0.9.11/WindowsNoEditor/PythonAPI/examples

  # Windows 系统（PowerShell 或 cmd）
  copy object_detection.py \CARLA_0.9.11\WindowsNoEditor\PythonAPI\examples
  copy generate_traffic.py \CARLA_0.9.11\WindowsNoEditor\PythonAPI\examples
```

2. 下载yolov3-tiny.weights和yolov3-tiny.cfg文件至examples目录
```plaintext
  https://pjreddie.com/media/files/yolov3-tiny.weights
```
```plaintext
  https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
```
3. 运行CARLA仿真器： 
```bash
  cd CARLA_0.9.11
  ./CarlaUE4.exe
```

4. 打开新的终端，cd至PythonAPI示例目录并运行脚本：
```bash
  cd CARLA_0.9.11/PythonAPI/examples
  python generate_traffic.py
```
5. 打开新的终端，cd至PythonAPI示例目录并运行脚本：
```bash
  cd CARLA_0.9.11/PythonAPI/examples
  python object_detection.py
```
6. 操作指令：  
ESC：退出程序  
C：开启 / 关闭数据采集（默认关闭，开启后自动保存带标注的图像）  
T：开始模型训练（需先采集至少 10 张图像，训练基于 YOLOv5s 进行）

  

## 模型训练与评估
1. 数据采集：  
⦁	按C键开启数据采集，系统会自动保存图像（至carla_dataset/images）和对应 YOLO 格式标签（至carla_dataset/labels）  
⦁	标签格式遵循 YOLO 标准：[类别ID] [中心x归一化] [中心y归一化] [宽度归一化] [高度归一化]  
⦁	类别 ID 映射：car→0，truck→1，bus→2，person→3  
⦁	建议采集多样化场景数据（不同光照、交通密度）以提升模型泛化能力  

2. 模型训练：  
按T键启动训练，系统会自动：  
⦁	克隆 YOLOv5 仓库并安装依赖（包括 ultralytics 库）  
⦁	划分训练集（80%）和验证集（20%）  
⦁	生成 YOLOv5 所需的数据集配置文件（carla_dataset.yaml）  
⦁	基于 YOLOv5s 模型进行微调（默认 50 轮训练，批次大小 16）  
⦁	训练结果保存在carla_train_results/exp/weights，其中best.pt为最佳权重文件  

3. 性能评估：  
⦁	实时显示检测结果（绿框）与真实目标（红框），并标注类别名称  
⦁	日志文件dataset_training.log记录数据采集和训练过程  
⦁	内部通过 IOU（交并比）计算检测精度，IOU 阈值为 0.3  
⦁	训练过程中可通过 YOLOv5 自带的可视化工具查看损失曲线  
  

  ## 项目结构
⦁	object\_detection.py：主程序脚本

⦁   generate_traffic.py ：交通流生成脚本

⦁	requirements.txt：所需Python库的清单

⦁	carla_dataset/：自动生成，存储采集的图像（images）和标签（labels），训练时会自动划分为 train/val 子集

⦁	carla_train_results/：自动生成，存储训练后的模型权重  

⦁   dataset_training.log：系统运行日志（数据采集、训练记录）

  ## 贡献方式
  欢迎通过贡献代码改进本项目。您可自由分叉（fork）仓库、修改代码，并提交拉取请求（pull request）。

  

  ## 许可协议
  本项目基于MIT许可协议开源，详情请参见LICENSE文件。

  

  ## 致谢
  ⦁	感谢CARLA仿真器团队提供了稳健的自动驾驶仿真平台

  ⦁	感谢YOLOv3开发者研发了高效的目标检测算法 



  ## 参考文档
  * [自动驾驶汽车物体检测和轨迹规划使用YOLOv3和CARLA模拟器](https://github.com/ROBERT-ADDO-ASANTE-DARKO/Autonomous-Vehicle-Object-Detection-and-Trajectory-Planning-using-YOLOv3-and-CARLA-Simulator)

#  常见问题
  ###  CARLA 连接失败：
  确保 CARLA 仿真器已启动，且脚本与仿真器版本严格一致（均为 0.9.11）；若仍失败，检查终端是否有权限访问 CARLA 进程。
  ###  模型加载错误：
  确认 yolov3-tiny.weights、yolov3-tiny.cfg 和 coco.names 已放在 examples 目录下，文件未损坏（可重新下载官方链接文件）。
  ###  训练失败：  
⦁	提示 "数据集样本不足"：需按C键采集至少 10 张带目标的图像  
⦁	依赖安装错误：确保网络通畅，训练时会自动安装 YOLOv5 依赖（yolov5/requirements.txt），网络问题可手动安装  
⦁	权限问题：部分系统可能需要管理员权限执行训练命令  
###  数据采集无反应：  
⦁	检查场景中是否有目标（车辆、行人），系统会跳过无真实目标的帧  
⦁	确认按C键后终端显示 "数据采集已开启"  
⦁	检查存储路径是否有写入权限（carla_dataset目录需可写）  
###  可视化窗口问题：  
⦁    窗口分辨率固定为 640x480（与相机参数一致）  
⦁    若窗口无响应，检查 CARLA 仿真器是否正常运行（需保持仿真器窗口打开）  