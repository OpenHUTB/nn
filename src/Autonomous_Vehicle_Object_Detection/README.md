# 自动驾驶车辆目标检测系统（基于 YOLOv8 和 CARLA）

## 项目概述
本项目聚焦于基于 YOLOv8 算法和 CARLA 仿真器的自动驾驶车辆目标检测。通过 CARLA 仿真环境实时采集场景图像，利用 YOLOv8 模型实现对车辆、行人、摩托车、公交车、卡车等目标的实时检测，并支持数据采集、模型训练（含断点续训）、精度评测一体化流程，通过 Pygame 进行可视化展示，支持中文显示。

### 核心功能：
⦁	基于 YOLOv8 模型（默认使用 yolov8m.pt）进行实时目标检测，支持指定置信度阈值（默认 0.5）和检测类别

⦁	与 CARLA 0.9.11 仿真器深度集成，生成自动驾驶车辆并挂载摄像头，实时获取动态视角图像（1024x768 分辨率）

⦁	自动采集带标注的数据集（YOLO 格式标签），支持行人、汽车、摩托车、公交车、卡车五类目标

⦁	实时评测检测精度（mAP@0.5、Precision、Recall），自动保存最佳模型权重

⦁	实时可视化检测结果（不同类别目标用不同颜色边框标注），显示类别名称及置信度

## 安装步骤

### 前置条件：
⦁	Python 3.7

⦁	CARLA仿真器 0.9.11 版本  

⦁	NVIDIA 显卡（支持 CUDA 11.7，显存≥4GB，可选但推荐，用于加速训练和检测）

### 操作步骤：

1. 下载适用于Windows系统的指定版本CARLA仿真器压缩包：
````plaintext
  https://github.com/carla-simulator/carla/releases
````

2. 参考官方文档安装 CARLA_0.9.11，解压至任意路径（如 D:\CARLA_0.9.11）

3. 克隆本项目仓库或下载代码文件：
```bash
  git clone https://github.com/William4861/Autonomous-Vehicle-Object-Detection-and-Trajectory-Planning-using-YOLOv3-and-CARLA-Simulator
```
4. 安装所需的 Python 库（在 PyCharm 终端中执行，确保已进入项目目录）：
```bash
  # 安装 PyTorch（支持 CUDA 11.7，加速训练）
  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
  # 安装 YOLOv8 依赖（兼容 Python 3.7.4）
  pip install ultralytics==8.0.151
  # 安装其他依赖
  pip install carla==0.9.11 numpy==1.21.6 opencv-python==4.5.5.64 pygame==2.1.0 scikit-learn==1.0.2
  pip install setuptools==40.8.0
```
5. 安装 CARLA Python API（在 PyCharm 终端中执行，先进入对应路径）：
```bash
  # 进入 CARLA 安装目录下的 PythonAPI 路径
  cd D:\CARLA_0.9.11\WindowsNoEditor\PythonAPI\carla\dist
  easy_install carla-0.9.11-py3.7-win-amd64.egg
```

  

## 使用方法

1. 将 object\_detection.py 文件和 generate_traffic.py 文件复制到 CARLA 的 PythonAPI 示例目录中： 
```plaintext
  # Linux/macOS 系统
  cp object_detection.py /CARLA_0.9.11/WindowsNoEditor/PythonAPI/examples
  cp generate_traffic.py /CARLA_0.9.11/WindowsNoEditor/PythonAPI/examples

  # Windows 系统（PowerShell 或 cmd）
  copy object_detection.py \CARLA_0.9.11\WindowsNoEditor\PythonAPI\examples
  copy generate_traffic.py \CARLA_0.9.11\WindowsNoEditor\PythonAPI\examples
```

2. 准备 YOLOv8 模型文件：
手动下载 yolov8m.pt 至上述 examples 目录（与 carla_test.py 同目录），下载地址：  
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

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
6. 操作指令（Pygame 窗口焦点状态下）： 
⦁ ESC：退出程序  
⦁ C：采集当前帧数据（自动保存图像至 carla_yolo_dataset/images，标签至 labels）  
⦁ T：开始模型训练（需先采集≥50 个样本，训练过程在终端显示）  
⦁ P：训练中暂停 / 继续训练（仅训练时有效）  
⦁ V：手动评测当前模型精度（输出 mAP、Precision 等指标）



## 模型训练与评估
1. 数据采集：
⦁	按 C 键采集数据，系统自动生成 YOLO 格式标签（格式：[类别 ID] [中心 x 归一化] [中心 y 归一化] [宽度归一化] [高度归一化]）
⦁	类别 ID 映射：行人→0，汽车→1，摩托车→2，公交车→3，卡车→4 
⦁	建议采集≥1000 个样本，覆盖不同场景（光照、遮挡、距离）以提升模型泛化能力

2. 模型训练：  
⦁ 按 T 键启动训练，系统自动：
⦁ 生成数据集配置文件 carla_yolo_dataset/data.yaml
⦁ 每 5 轮保存一次权重（至 carla_yolo_results/train/weights）
⦁ 每 10 轮自动评测精度，保存最佳模型（best.pt）
⦁ 训练参数可在代码 Config 类中调整：
⦁ TRAIN_EPOCHS：训练轮次（默认 50）
⦁ BATCH_SIZE：批次大小（8G 显存推荐 8，可按需调整）
⦁ LEARNING_RATE：学习率（默认 0.001）  

3. 断点续训：  
⦁ 若训练中断，修改代码中 Config 类参数：  
```python
Config.RESUME_TRAIN = True  # 开启续训
Config.LAST_WEIGHTS = "carla_yolo_results/train/weights/last.pt"  # 上次训练的权重路径
```  
⦁ 重新运行程序并按 T 键，自动从上次进度继续训练

4. 性能评估：  
⦁ 按 V 键手动评测，终端输出关键指标：  
⦁ mAP@0.5：目标检测核心指标（越高越好）  
⦁ Precision（精确率）：检测结果中真实目标的比例  
⦁ Recall（召回率）：所有真实目标中被检测到的比例  
⦁ 最佳模型自动保存为 carla_yolo_results/train/weights/best.pt

  ## 项目结构
⦁	object\_detection.py：主程序脚本，包含 CARLA 交互、YOLOv8 检测、数据采集、模型训练逻辑

⦁   generate_traffic.py ：交通流生成脚本

⦁   carla_yolo_dataset/：自动生成，存储采集的图像（images）和标签（labels）

⦁   carla_yolo_results/：自动生成，存储训练结果（模型权重 weights、评测指标 metrics.csv 等）


  ## 贡献方式
  欢迎通过贡献代码改进本项目。您可自由分叉（fork）仓库、修改代码，并提交拉取请求（pull request）。



  ## 许可协议
  本项目基于MIT许可协议开源，详情请参见LICENSE文件。



  ## 致谢
  ⦁	感谢CARLA仿真器团队提供了稳健的自动驾驶仿真平台

  ⦁ 感谢 Ultralytics 团队开发的 YOLOv8 算法及开源库



  ## 参考文档
  * [CARLA 0.9.11 官方文档](https://carla.readthedocs.io/en/0.9.11/)
  * [自动驾驶汽车物体检测和轨迹规划使用YOLOv3和CARLA模拟器](https://github.com/ROBERT-ADDO-ASANTE-DARKO/Autonomous-Vehicle-Object-Detection-and-Trajectory-Planning-using-YOLOv3-and-CARLA-Simulator)
  * [YOLOv8 官方文档](https://docs.ultralytics.com/)

#  常见问题
  ###  CARLA 连接失败：
  确保 CARLA 仿真器已启动，且脚本与仿真器版本严格一致（均为 0.9.11）；若仍失败，检查终端是否有权限访问 CARLA 进程。
  ###  模型加载错误：
⦁ 确认 yolov8m.pt 已放在 examples 目录，文件名与代码中 Config.MODEL_PATH 一致（区分大小写）  
⦁ 若提示文件不存在，检查路径是否正确（可使用绝对路径，如 D:\CARLA_0.9.11\...\yolov8m.pt）
  ###  训练失败：
⦁ 提示 “数据集为空”：需按 C 键采集至少 50 个样本  
⦁ “CUDA out of memory”：修改 Config.BATCH_SIZE 为 4（减小批次大小，适配小显存显卡）  
⦁ “AMP 不支持”：设置 Config.MIXED_PRECISION = False（关闭混合精度训练）
  ###  数据采集无反应：
⦁ 检查 CARLA 场景中是否有目标（车辆、行人），无目标时会跳过采集  
⦁ 确认 carla_yolo_dataset 目录有写入权限（可手动创建目录测试）
  ###  可视化窗口问题：  
⦁	窗口无响应时，检查 CARLA 仿真器是否正常运行（需保持仿真器窗口打开）  
⦁	中文显示异常时，确保系统中已安装 SimHei 或 WenQuanYi Micro Hei 字体