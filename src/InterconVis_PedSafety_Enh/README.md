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
### 样本可视化 (Sample Visualizations)

我们提供了可视化结果来展示我们数据集中的不同视角:

## 致谢 (Acknowledgement)
本项目基于以下开源项目: BEVerse, Fiery, open-mmlab, 以及 DeepAccident。

