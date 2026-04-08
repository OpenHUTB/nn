# 项目简介 
CVIPS (Connected Vision for Increased Pedestrian Safety) 是一个致力于通过协同视觉技术（Connected Vision）和 V2X（Vehicle-to-Everything）通信来提升弱势道路使用者（VRU，特别是行人）安全的研究项目。
在复杂的城市交通场景中，单车智能往往受到视线遮挡（Occlusion）、感知范围受限和恶劣天气的影响。本项目利用 CARLA 仿真器 构建高保真的测试环境，旨在研究：
协同感知 (Collaborative Perception): 融合多视点（车辆、路侧单元 RSU）信息。
1. 遮挡处理: 解决“鬼探头”等高危场景下的行人检测问题。
2. 全天候鲁棒性: 在雨天、夜间等极端光照和天气下的感知性能。
3. 本仓库包含 CVIPS 的核心仿真场景生成工具、数据集采集脚本以及协同感知算法实现。
# 安装与依赖 (Prerequisites & Installation)
# 系统要求
- Ubuntu 20.04 / Windows 10+
- NVIDIA GPU (推荐 8GB+ 显存)
- CARLA Simulator 0.9.14
## 数据集生成 (Dataset Generation)
我们的数据集是使用 CARLA 模拟器生成的，为协同感知提供了多样化的场景。
### carla模拟器构建
下载以下版本的carla预编译包（可直接运行，无需使用Epic Games Launcher）
- [carla0.9.14](https://github.com/carla-simulator/carla/releases/tag/0.9.14)
### 运行数据生成脚本 (Running the Data Generation Script)
1. 克隆此仓库
   ```bash
   git clone https://github.com/cvips/CVIPS.git
   cd CVIPS
   ```
2. 安装所需的依赖包:
   ```bash
   pip install -r requirements.txt
   ```
3. 确保 CARLA 已正确安装并运行。在一个单独的终端中启动 CARLA 服务器:
   ```bash
   /path/to/carla/CarlaUE4.exe
   ```
4. 运行数据生成脚本:
   ```bash
   python cvips_generation.py
   ```
注：该脚本将连接到 CARLA 服务器，并根据指定的参数生成数据集,需要单独创建运行脚本。 请根据你特定的设置需求，调整 cvips_generation.py 中的 CARLA 服务器路径以及任何配置参数。
### 配置虚拟环境
- CARLA 对 Python 版本（推荐 3.7-3.9）和依赖库版本有严格要求，虚拟环境可避免与其他项目的依赖冲突
### 下载Anaconda
- 下载 [Anaconda](https://repo.anaconda.com/archive/Anaconda3-2021.05-Windows-x86_64.exe)。
- 安装时勾选 “Add Anaconda to PATH”（Windows 需手动勾选，Linux/Mac 默认添加）。
- 验证安装：打开终端，输入conda --version，显示版本号即成功。
### 创建激活虚拟环境（仅需执行一次）
1. 打开终端（或 Anaconda Prompt），运行以下命令：
   ```bash
   # 创建名为cvips_env的虚拟环境（Python3.7适配CARLA 0.9.14）
   conda create -n cvips_env python=3.7
   ```
2. 激活虚拟环境（每次运行脚本前都需执行）
   ```bash
   conda activate cvips_env  # Windows
   ```
### 在虚拟环境中安装 CARLA 依赖（仅需执行一次）
- 激活环境后，直接安装匹配版本的 CARLA Python API：
   ```bash
   pip install carla==0.9.14
   ```
### 使用说明
- 激活虚拟环境（每次运行脚本前都需执行）
   ```bash
   conda activate cvips_env  # Windows
   ```
- 打开 CARLA 0.9.14 模拟器（CarlaUE4.exe/./CarlaUE4.sh）。
- 进入 cvips 文件夹目录（cd 你的cvips路径）
- 输入想要运行的脚本
## 使用示例及格式说明
- 命令格式
   ```shell
   python cvips_generation.py --town <城镇名称> [--num_vehicles < 数量 >] [--num_pedestrians < 数量 >] [--weather < 天气类型 >] [--time_of_day < 时段 >] [--seed < 种子值 >]
## 参数说明
|参数名|类型|默认值|可选值|作用说明|
|:---:|:---:|:---:|:---:|:---:|
|--town|字符串|town01|town01,town04等|指定要加载的CARLA地图|
|--num_vehicles|整数|20|>=0|要生成的自动驾驶车辆数量|
|--num_pedestrians|整数|100|>=0|要生成的自主行走的行人数量|
|--seed|整数|None|任意整数|随机种子，设置后可保证每次运行完全相同的场景，便于复现。|
|--weather|字符串|clear|clear，rainy，cloudy|设置天气，rainy会有明显的降雨和地面反光效果|
|--time_of_day|字符串|noon|noon，sunset，night|设置时间，night会切换到夜晚并开启月光|
### 一、基础场景命令 (核心参数覆盖)
1. Town01 + 晴天 + 中午 (默认配置)
   ```shell
   python cvips_generation.py --town Town01
   ```
2. Town01 + 雨天 + 夜晚
   ```shell
   python cvips_generation.py --town Town01 --weather rainy --time_of_day night
   ```
### 二、不同密度场景命令
1. Town01 + 低密度 (10 辆车，50 个行人)
   ```shell
   python cvips_generation.py --town Town01 --num_vehicles 10 --num_pedestrians 50
2. Town01 + 中密度 (25 辆车，150 个行人)
   ```shell
   python cvips_generation.py --town Town01 --num_vehicles 25 --num_pedestrians 150
### 三、随机种子与场景复现命令

1. Town01 + 种子 123 (可复现)
   ```shell
   python cvips_generation.py --town Town01 --seed 123
2. Town04 + 种子 456 (可复现)
   ```shell
   python cvips_generation.py --town Town04 --seed 456
### 四、多参数组合场景命令
1. Town01 + 15 辆车 + 80 个行人 + 雨天 + 日落 + 种子 111
   ```shell
   python cvips_generation.py --town Town01 --num_vehicles 15 --num_pedestrians 80 --weather rainy --time_of_day sunset --seed 111
## 数据集生成
数据集由cvip_collector.py生成，保存在_out_dataset_final(主文件夹)
- 📂 ego_rgb/ : 存放主车（Tesla）视角的图片（命名格式为 帧号.png）。
- 📂 rsu_rgb/ : 存放路侧单元（RSU，安装在高处）视角的图片。
- 📂 label/ : 标注文件，用于作为标签提供目标的 3D 空间属性和类别信息。
data_check.py：用于验证收集数据的可视化工具。
cvips_utils.py：用于坐标变换和投影的实用函数,为cvip_collector.py提供计算服务。
### 运行命令
### 基础模式 (默认参数)
### 模板指令
- 可根据需求调整城镇、天气、时间、车辆数、行人数和最大帧数
   ```shell
   python cvips_collector.py --town Town03 --weather Rain --time Sunset --num_vehicles 50 --num_walkers 30 --max_frames 1000
   ```
-  默认参数运行
（对应配置：Town01、Clear天气、Day时间、40辆车、20个行人、最多500帧）
   ```shell
   python cvips_collector.py
   ```
### 更换地图 (例如 Town02, Town10HD)
使用 --town 参数。
- 在 Town02 采集
   ```shell
   python cvips_collector.py --town Town02
   ```
- 在 Town10HD (高清地图) 采集
   ```shell
   python cvips_collector.py --town Town10HD
   ```
### 调整交通密度 (拥堵/空旷)
使用 --num_vehicles 和 --num_walkers。
- 拥堵场景 (50辆车, 80人):
   ```shell
   python cvips_collector.py --num_vehicles 50 --num_walkers 80
   ```
- 空旷场景 (5辆车, 0人):
   ```shell
   python cvips_collector.py --num_vehicles 5 --num_walkers 0
   ```
## 使用官方脚本生成目标
由于在Carla中不使用官方的脚本产生行走的行人和车辆，使用自己的脚本产生需要耗费大量的GPU导致非常的卡顿，所以提供另外一种使用Carla官方自带的脚本文件生成数据集的思路。
- 进入CARLA安装目录（替换为你实际的路径）
   ```shell
   cd F:\carla\CARLA\WindowsNoEditor
   ```
- 启动CARLA服务器（窗口模式，默认端口2000）
   ```shell
   CarlaUE4.exe
   ```
### 生成动态行人和车辆（用 generate_traffic.py）
- 打开新的终端（CMD/PowerShell），PythonAPI/examples目录（不同版本目录不同，根据实际版本来定），运行官方交通生成脚本：
- 进入examples目录（替换为你实际的路径）
   ```shell
   cd F:\carla\CARLA\WindowsNoEditor\PythonAPI\examples
   ```
- 生成动态交通：30辆车 + 15个行人（可自定义数量）
   ```shell
   python generate_traffic.py --vehicles 30 --walkers 15
   ```
### 运行 cvips_collector.py 采集数据集
打开第三个终端，进入你的cvips_collector.py脚本所在目录，执行采集指令（关键：设num_vehicles和num_walkers为 0，避免重复生成）：
- 进入你的cvips_collector.py所在目录（替换为实际路径）
   ```shell
   cd F:\你的脚本存放路径
   ```
- 运行采集脚本（不额外生成车辆/行人，仅采集已有交通流）
   ```shell
   python cvips_collector.py 
   - --town Town01 
   - --weather Clear 
   - --time Day 
   - --num_vehicles 0 
   - --num_walkers 0 
   - --max_frames 500
   ```
注意：Windows 终端中换行用^，如果是单行写就不需要加；max_frames 500表示采集 500 帧数据，可按需调整。
关键补充说明
脚本功能对应：
CARLA 旧版本：spawn_npc.py → 生成 NPC（车辆 + 行人）
CARLA 新版本：generate_traffic.py → 功能完全替代spawn_npc.py，参数更清晰
避免重复生成：
必须把cvips_collector.py的--num_vehicles和--num_walkers设为 0，否则脚本会额外生成车辆 / 行人，导致场景中交通流过多、卡顿。
常见问题解决：
若运行generate_traffic.py提示 “找不到 carla 模块”：
先在该终端执行pip install -r requirements.txt（requirements.txt 在你截图的 examples 目录里），安装依赖。
若采集时帧率低：减少generate_traffic.py的车辆 / 行人数目（比如设--vehicles 20 --walkers 10）。
### 运行数据检查脚本（data_check.py）
- 命令格式：
   ```shell
   python data_check.py
   ```
## 六视角的环境感知可视化系统（可用于自动驾驶场景中的目标检测、多摄像头标定等相关算法的测试与验证）
cvips_collector_6cam.py 是一个基于 CARLA 自动驾驶仿真环境的多摄像头数据采集与目标可视化程序，主要功能是通过安装在主车上的 6 个摄像头采集周围环境图像，并在图像上实时绘制动态物体（车辆、行人）和静态车辆的 3D 边界框。以下是其主要组成部分和功能说明：
1. 依赖与配置
依赖库：使用 carla 库连接仿真环境，numpy 处理矩阵运算，cv2 进行图像处理与显示，以及自定义工具库 cvips_utils。
核心配置：设置图像分辨率（640x360）、摄像头视场角（FOV=90°）、目标帧率（30 FPS）。
- 注意：依赖库需在自己所创建的虚拟环境中克隆。
2. 运行指令
   ```shell
   python cvips_collector_6cam.py
   ```
- 注意：该脚本的行人和车辆都是运用Carla官方文件生成，详细步骤见 [使用官方脚本生成目标](#使用官方脚本生成目标)
## 样本可视化 (Sample Visualizations)
我们提供了可视化结果来展示我们数据集中的不同视角:
- 车辆自动行驶检测（视角为车后）
- <img src="https://github.com/user-attachments/assets/4b520d73-5f00-4326-b072-d07de0772f04" alt="Infrastructure View" width="640" height="360">
- 六摄像头自动检测
- <img src="https://github.com/user-attachments/assets/f2182831-3719-44cb-acfa-1369121ca227" alt="Vehicle View" width="640" height="360">


## 致谢 (Acknowledgement)
本项目基于以下开源项目: BEVerse, Fiery, open-mmlab, 以及 DeepAccident。

