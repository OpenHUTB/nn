# 自动驾驶车辆目标检测系统（基于 YOLOv8 和 CARLA）

## 项目概述
本项目聚焦于基于 YOLOv8 算法和 CARLA 仿真器的自动驾驶车辆目标检测。通过 CARLA 仿真环境实时采集场景图像，利用 YOLOv8 模型实现对车辆、行人、摩托车、公交车、卡车等目标的实时检测，并支持数据采集、模型训练（含断点续训）、精度评测一体化流程，通过 Pygame 进行可视化展示，支持中文显示。

**本项目目前支持两种运行模式：**
1.  **Windows 单机模式**：仅使用 Python 脚本，适合快速验证算法、训练模型和采集数据。
2.  **ROS 分布式模式**：Windows (Server) + Ubuntu (Client)，适合集成到 ROS 机器人操作系统中进行算法开发。

### 核心功能：
⦁	基于 YOLOv8 模型（默认使用 yolov8m.pt）进行实时目标检测，支持指定置信度阈值（默认 0.5）和检测类别  
⦁	与 CARLA 0.9.11 仿真器深度集成，生成自动驾驶车辆并挂载摄像头，实时获取动态视角图像（1024x768 分辨率）  
⦁	支持自动/手动采集带 YOLO 格式标签的数据集，自动过滤无目标帧（单帧至少 1 个目标才保存）  
⦁	实时评测检测精度（mAP@0.5、Precision、Recall），自动更新并保存最佳模型权重（best.pt）  
⦁	实时可视化检测结果（不同类别目标用不同颜色边框标注），显示类别名称及置信度  

## 📂 项目结构

```text
.
├── original_windows_scripts/   # [备份] Windows端原始单机脚本 (对应模式 1)
│   ├── object_detection.py
│   └── generate_traffic.py
├── catkin_ws_src/              # [新增] Ubuntu端 ROS 源码包 (对应模式 2)
│   └── carla_yolo_ros/
│       ├── launch/             # 启动文件
│       ├── src/                # ROS 节点源码
│       └── models/             # 模型权重
├── requirements.txt            # Python 依赖列表
└── README.md                   # 说明文档
```

---

## 模式一：Windows 单机模式 (原始功能)

### 安装步骤

#### 前置条件：
⦁	Python 3.7（必选，与 CARLA 0.9.11 兼容）  
⦁	CARLA仿真器 0.9.11 版本（必选，版本需严格匹配）    
⦁	NVIDIA 显卡（支持 CUDA 11.7，显存≥4GB，可选但推荐，用于加速训练和检测）  

#### 操作步骤：

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
  pip install -r requirements.txt
```

5. 安装 CARLA Python API（在 PyCharm 终端中执行，先进入对应路径）：
```bash
  # 进入 CARLA 安装目录下的 PythonAPI 路径
  cd D:\CARLA_0.9.11\WindowsNoEditor\PythonAPI\carla\dist
  easy_install carla-0.9.11-py3.7-win-amd64.egg
```

### 使用方法

1. **代码文件准备**  
   本项目的单机脚本已移动至 `original_windows_scripts/` 目录。请将该目录下的 `object_detection.py` 和 `generate_traffic.py` 复制到 CARLA 的 PythonAPI 示例目录中：
   ```bash
   # Windows 系统（PowerShell 或 cmd）
   copy original_windows_scripts\object_detection.py \CARLA_0.9.11\WindowsNoEditor\PythonAPI\examples
   copy original_windows_scripts\generate_traffic.py \CARLA_0.9.11\WindowsNoEditor\PythonAPI\examples
   ```

2. **准备 YOLOv8 预训练模型**：
   手动下载 yolov8m.pt 至上述 examples 目录（与 carla_test.py 同目录），下载地址：  
   https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

3. **运行流程（按顺序执行）**  
   
   **步骤 1：启动 CARLA 仿真器**  
   ```bash
   cd CARLA_0.9.11
   ./CarlaUE4.exe
   ```
   等待地图加载完成（显示 3D 城市场景，约 1-2 分钟），保持窗口打开（可最小化）。  

   **步骤 2：生成交通实体（丰富检测场景）**  
   打开新的 PyCharm 终端，执行以下命令生成车辆和行人：
   ```bash
   # 进入 examples 目录
   cd CARLA_0.9.11/PythonAPI/examples
   # 生成 70 辆车 + 50 个行人（可按需调整数量）
   python generate_traffic.py
   ```  
   终端显示 “spawned 70 vehicles and 50 walkers” 即成功。

   **步骤 3：运行主程序（检测 + 采集 + 训练）**  
   打开新的 PyCharm 终端，执行：
   ```bash
   # 进入 examples 目录
   cd CARLA_0.9.11/PythonAPI/examples  
   # 启动主程序
   python object_detection.py
   ```    
   成功启动后会弹出 Pygame 窗口，显示自动驾驶车辆视角的实时画面。

4. **操作指令（Pygame 窗口焦点状态下）：**   
⦁ ESC：退出程序    
⦁ C：采集当前帧数据（自动保存图像至 carla_yolo_dataset/images，标签至 labels）    
⦁ A：切换自动采集开关（默认关闭，开启后每 0.5 秒保存 1 帧有效数据）  
⦁ T：开始模型训练（需先采集 ≥50 个样本，训练过程在终端显示，每 5 轮保存权重）  
⦁ P：训练暂停 / 继续（仅训练时有效，终端会提示 “训练暂停 / 继续”）  
⦁ V：手动评测模型精度（需指定权重路径，终端输出 mAP@0.5、Precision、Recall）  

---

## 模式二：ROS 分布式模式 (新增功能)

此模式适用于需要与 ROS 生态结合的场景。

### 架构说明
*   **Host (Server)**: Windows 10/11 运行 CARLA 仿真器 (IP 例如: `192.168.1.100`)
*   **Client**: Ubuntu 16.04 + ROS Kinetic 运行算法节点

### 安装步骤 (Ubuntu 端)
1.  **复制源码包**：
    将 `catkin_ws_src/carla_yolo_ros` 文件夹复制到你的 Ubuntu 工作空间的 `src` 目录下 (例如 `~/catkin_ws/src/`)。

2.  **编译**：
    ```bash
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    ```

3.  **安装依赖**：
    确保 Ubuntu 端的 Python 环境已安装 `ultralytics` 和 `carla` 库。

### 运行方法
1.  在 Windows 端启动 CARLA (`CarlaUE4.exe`).
2.  在 Ubuntu 端运行 launch 文件（需指定 Windows 主机的 IP）：
    ```bash
    roslaunch carla_yolo_ros start.launch host:=192.168.1.100
    ```
3.  节点说明：
    *   `traffic_generator_node`: 负责生成交通流。
    *   `object_detection_node`: 负责 YOLO 检测并发布 `/carla/camera/detection` 图像话题。

---

## 模型训练与评估 (通用)
1. 数据采集细节：  
⦁	自动采集：按 A 键开启，系统自动过滤 “无目标帧”，每 0.5 秒保存 1 帧，数据存至 carla_yolo_dataset 目录：  
⦁	images：RGB 图像（.jpg 格式，文件名含时间戳）  
⦁	labels：YOLO 格式标签（.txt 格式，每行：[类别 ID] [中心 x 归一化] [中心 y 归一化] [宽度归一化] [高度归一化]）  
⦁	类别 ID 映射（训练后模型专用）：  
⦁	0：person（行人）、1：car（汽车）、2：motorcycle（摩托车）、3：bus（公交车）、4：truck（卡车）  
⦁	采集建议：至少采集 1000 个样本，覆盖不同场景（如白天 / 夜晚、拥堵 / 畅通、近距 / 远距），提升模型泛化能力。  

2. 模型训练配置（可在代码 Config 类中调整）：  
⦁	TRAIN_EPOCHS：训练轮次 (默认 50)  
⦁	BATCH_SIZE：批次大小 (建议：8G 显存设为 8，4G 显存设为 2)  
⦁	LEARNING_RATE：初始学习率 (默认 0.001)  

3. 断点续训：  
⦁	找到上次训练的权重路径（默认：carla_yolo_results/train2/weights/last.pt）；
⦁	修改 Config 类参数：`RESUME_TRAIN = True` 并指定 `LAST_WEIGHTS` 路径。

4. 精度评估：  
⦁ 自动评测：训练每 10 轮自动评测，终端输出指标，最佳模型保存为 best.pt；  
⦁ 手动评测：按 V 键触发。

## 贡献方式
欢迎通过贡献代码改进本项目。您可自由分叉（fork）仓库、修改代码，并提交拉取请求（pull request）。

## 许可协议
本项目基于MIT许可协议开源，详情请参见LICENSE文件。

## 致谢
⦁	感谢CARLA仿真器团队提供了稳健的自动驾驶仿真平台
⦁   感谢 Ultralytics 团队开发的 YOLOv8 算法及开源库
⦁   感谢 Pygame 团队提供可视化支持。

## 参考文档
* [CARLA 0.9.11 官方文档](https://carla.readthedocs.io/en/0.9.11/)
* [自动驾驶汽车物体检测和轨迹规划使用YOLOv3和CARLA模拟器](https://github.com/ROBERT-ADDO-ASANTE-DARKO/Autonomous-Vehicle-Object-Detection-and-Trajectory-Planning-using-YOLOv3-and-CARLA-Simulator)
* [YOLOv8 官方文档](https://docs.ultralytics.com/)

#  常见问题
###  CARLA 连接失败：
确保 CARLA 仿真器已启动，且脚本与仿真器版本严格一致（均为 0.9.11）；若仍失败，检查终端是否有权限访问 CARLA 进程。对于 ROS 模式，请检查虚拟机与主机能否 Ping 通。

###  模型加载错误：
⦁ 确认 yolov8m.pt 已放在正确目录，文件名与代码中 Config.MODEL_PATH 一致（区分大小写）  

###  训练失败：
⦁ 提示 “数据集为空”：需按 C 键采集至少 50 个样本  
⦁ “CUDA out of memory”：修改 Config.BATCH_SIZE 为 4（减小批次大小，适配小显存显卡）  

###  可视化窗口问题：  
⦁	窗口无响应时，检查 CARLA 仿真器是否正常运行（需保持仿真器窗口打开）  
⦁	中文显示异常时，确保系统中已安装 SimHei 或 WenQuanYi Micro Hei 字体
```