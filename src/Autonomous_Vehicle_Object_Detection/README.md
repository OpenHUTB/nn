# 自动驾驶车辆目标检测系统（基于 YOLOv8 和 CARLA）

## 项目概述
本项目聚焦于基于 YOLOv8 算法和 CARLA 仿真器的自动驾驶车辆目标检测。通过 CARLA 仿真环境实时采集场景图像，利用 YOLOv8 模型实现对车辆、行人、摩托车、公交车、卡车等目标的实时检测，并通过 Pygame 进行可视化展示，支持中文显示。  

### 核心功能：
⦁	基于 YOLOv8 模型（默认使用 yolov8n.pt）进行实时目标检测，支持指定置信度阈值和检测类别

⦁	与 CARLA 0.9.11 仿真器深度集成，实时获取车辆搭载摄像头的场景图像（1024x768 分辨率）

⦁	支持对行人、汽车、摩托车、公交车、卡车五类目标的检测与分类

⦁	实时可视化检测结果（不同类别目标使用不同颜色边框标注），显示类别名称及置信度

⦁	自动清理 CARLA 仿真资源（车辆、摄像头），确保程序退出时资源释放  

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
  pip install ultralytics==8.0.151
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

2. YOLOv8 模型文件（yolov8n.pt）会在首次运行时由 ultralytics 库自动下载，无需手动下载  

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
关闭 Pygame 窗口：退出程序

  

## 检测类别与可视化说明
1. 支持检测的目标类别及对应边框颜色： 
⦁	行人（person）：红色边框
⦁	汽车（car）：绿色边框  
⦁	摩托车（motorcycle）：蓝色边框
⦁	公交车（bus）：黄色边框  
⦁	卡车（truck）：洋红色边框

2. 可视化窗口显示摄像头实时画面，每个检测目标会标注边框、类别名称及置信度（保留两位小数）

3. 窗口分辨率固定为 1024x768（与摄像头参数一致）
  

  ## 项目结构
⦁	object\_detection.py：主程序脚本

⦁   generate_traffic.py ：交通流生成脚本

⦁	requirements.txt：所需Python库的清单

  ## 贡献方式
  欢迎通过贡献代码改进本项目。您可自由分叉（fork）仓库、修改代码，并提交拉取请求（pull request）。



  ## 许可协议
  本项目基于MIT许可协议开源，详情请参见LICENSE文件。



  ## 致谢
  ⦁	感谢CARLA仿真器团队提供了稳健的自动驾驶仿真平台

  ⦁	感谢YOLOv8开发者研发了高效的目标检测算法 



  ## 参考文档
  * [自动驾驶汽车物体检测和轨迹规划使用YOLOv3和CARLA模拟器](https://github.com/ROBERT-ADDO-ASANTE-DARKO/Autonomous-Vehicle-Object-Detection-and-Trajectory-Planning-using-YOLOv3-and-CARLA-Simulator)
  * [YOLOv8 官方文档](https://docs.ultralytics.com/)


#  常见问题
  ###  CARLA 连接失败：
  确保 CARLA 仿真器已启动，且脚本与仿真器版本严格一致（均为 0.9.11）；若仍失败，检查终端是否有权限访问 CARLA 进程。
  ###  模型加载错误：
⦁	确认网络通畅，YOLOv8 模型（yolov8n.pt）会自动下载至用户目录下的 .cache/ultralytics 文件夹  
⦁	若自动下载失败，可手动下载模型文件并放置于脚本运行目录，下载地址：https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
  ###  可视化窗口问题：  
⦁	窗口无响应时，检查 CARLA 仿真器是否正常运行（需保持仿真器窗口打开）  
⦁	中文显示异常时，确保系统中已安装 SimHei 或 WenQuanYi Micro Hei 字体