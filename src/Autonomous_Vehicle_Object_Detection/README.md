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
```bash
  git clone https://github.com/William4861/Autonomous-Vehicle-Object-Detection-and-Trajectory-Planning-using-YOLOv3-and-CARLA-Simulator
```
4. 安装所需的Python库： （执行该命令前确保cd到requirements.txt文件目录）
```plaintext
  pip install -r requirements.txt
```  
```commandline
  pip install setuptools==40.8.0
```
(执行该命令前确保cd到CARLA_0.9.11\WindowsNoEditor\PythonAPI\carla\dist目录)
```commandline
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
```
  https://pjreddie.com/media/files/yolov3-tiny.weights
```
```
  https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
```
3. 运行CARLA仿真器： 
```plaintext
  cd CARLA_0.9.11
```
```
  ./CarlaUE4.exe
```

4. 打开新的终端，cd至PythonAPI示例目录并运行脚本：
```plaintext
  cd CARLA_0.9.11/PythonAPI/examples
```
```commandline
  python generate_traffic.py
```
5. 打开新的终端，cd至PythonAPI示例目录并运行脚本：
```plaintext
  cd CARLA_0.9.11/PythonAPI/examples
```
```plaintext
  python object_detection.py
```

  

## 模型性能监控
通过TensorBoard监控和跟踪YOLOv3模型性能的步骤如下：

1. 启动TensorBoard：
```plaintext
  # 日志文件默认生成在CARLA_0.9.11/PythonAPI/examples/logs目录下（与object_detection.py同级）
  tensorboard --logdir=./logs
```

2. 打开网页浏览器，访问 http://localhost:6006 以查看TensorBoard仪表板。

  

  ## 项目结构
  ⦁	 object\_detection.py ：目标检测与轨迹规划的主Python脚本

  ⦁ generate_traffic.py ：交通流生成脚本

  ⦁	 requirements.txt ：所需Python库的清单

  ⦁	 logs/ ：用于性能监控的TensorBoard日志文件

  

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
  确认 yolov3.weights 和 yolov3.cfg 已放在 examples 目录下，文件未损坏（可重新下载官方链接文件）；检查 object_detection.py 中模型路径是否指向 examples 目录（默认无需修改，若修改过需还原为相对路径）。
  ###  依赖安装失败：
  建议使用 Python 3.7 虚拟环境（如 conda create -n carla_env python=3.7），避免与其他 Python 版本的库冲突；执行 pip install 时若网络超时，可添加国内镜像源（如 - i https://pypi.tuna.tsinghua.edu.cn/simple）。
  ###  TensorBoard 无法打开：
  检查 logs 目录是否存在且有日志文件，若未生成日志，需先运行 object_detection.py 至少 1 次；确保终端当前目录为 examples 目录（与 logs 同级），再执行 tensorboard 命令。