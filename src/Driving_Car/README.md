Carla 无人车仿真开发项目

基于 Carla 仿真平台的全栈自动驾驶算法开发与验证框架

项目概述

1.1 项目定位

本项目是一套基于 Carla 仿真平台的全栈自动驾驶开发工具链，覆盖感知、定位、规划、控制四大核心模块，支持从算法原型开发、模块联调到场景化验证的全流程需求。适用于自动驾驶算法工程师、高校科研人员及相关专业学生进行技术研究与工程实践。

1.2 核心价值

低门槛入门：提供完整环境配置脚本与最小化演示用例，新手可快速上手

高可扩展性：模块间解耦设计，支持替换自定义算法（如将YOLOv8替换为Faster R-CNN）

贴近工程实践：还原真实自动驾驶系统的数据流向与容错机制

完善的评估体系：内置多维度性能指标统计与可视化工具

1.3 支持的 Carla 特性

特性类别

支持内容

使用场景

传感器类型

RGB摄像头、深度摄像头、语义分割摄像头、激光雷达（LiDAR）、毫米波雷达、GPS/IMU、超声波雷达

多源数据融合、感知算法开发

仿真场景

城市道路（Town01-Town12）、高速公路、雨天/雾天/夜间环境、动态交通流

算法鲁棒性测试、场景化验证

车辆控制

油门/刹车/转向控制、车辆动力学模型、多车协同

控制算法开发、编队行驶仿真

环境配置

2.1 系统配置要求

硬件/软件

基础配置（可运行）

推荐配置（流畅开发）

备注

操作系统

Ubuntu 18.04 LTS

Ubuntu 20.04 LTS

不推荐Windows（Carla兼容性较差）

CPU

Intel i5-8400 / AMD Ryzen 5 3600

Intel i7-12700H / AMD Ryzen 7 5800X

多线程性能影响交通流仿真效率

GPU

NVIDIA GTX 1660 Ti（6GB）

NVIDIA RTX 3070（8GB）及以上

必须支持CUDA，显存影响LiDAR点云处理

内存

16GB DDR4

32GB DDR4

多传感器数据缓存需大内存支持

磁盘

100GB SSD（空闲）

200GB NVMe SSD

Carla安装包+数据集需大量存储空间

2.2.2 Carla 仿真平台安装

提供两种安装方式，推荐预编译版本（适合开发），源码编译适合二次开发：

方式1：预编译版本（推荐）

# 选择版本（0.9.14稳定版），创建安装目录
mkdir -p ~/carla && cd ~/carla

# 下载预编译包（约20GB，建议用迅雷等工具加速后传输）
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.14.tar.gz

# 解压（耗时约5分钟）
tar -xzf CARLA_0.9.14.tar.gz

# 安装Carla额外资产（交通标志、植被等，约10GB）
cd CARLA_0.9.14
./ImportAssets.sh


方式2：源码编译版本（进阶）

# 安装Unreal Engine 4.26（Carla依赖版本）
git clone --depth 1 --branch 4.26 https://github.com/EpicGames/UnrealEngine.git ~/UnrealEngine
cd ~/UnrealEngine
./Setup.sh && ./GenerateProjectFiles.sh && make

# 编译Carla源码
git clone https://github.com/carla-simulator/carla.git ~/carla-source
cd ~/carla-source
make launch  # 编译并启动编辑器


2.2.3 项目环境配置

# 1. 克隆项目仓库
git clone https://github.com/your-username/carla-autonomous-driving.git ~/carla-project
cd ~/carla-project

# 2. 创建Python虚拟环境（隔离依赖）
python3.8 -m venv venv
# 激活环境（每次开发前需执行）
source venv/bin/activate

# 3. 安装Python依赖（分基础依赖与可选依赖）
# 基础依赖（核心功能）
pip install -r requirements/base.txt
# 可选依赖（可视化、模型训练等）
pip install -r requirements/optional.txt

# 4. 配置Carla Python API环境变量（永久生效）
echo "export PYTHONPATH=\$PYTHONPATH:~/carla/CARLA_0.9.14/PythonAPI/carla/dist/carla-0.9.14-py3.8-linux-x86_64.egg" >> ~/.bashrc
source ~/.bashrc


2.2.4 环境验证

# 1. 启动Carla服务（新开终端1）
cd ~/carla/CARLA_0.9.14
./CarlaUE4.sh -windowed -ResX=1024 -ResY=768  # 窗口模式启动

# 2. 运行连接测试脚本（终端2，需激活虚拟环境）
cd ~/carla-project
source venv/bin/activate
python scripts/test_carla_connection.py


若终端输出 [SUCCESS] Connected to Carla server at 127.0.0.1:2000，说明环境配置成功。

快速开始

3.1 核心演示流程

按以下步骤快速运行自动驾驶演示，体验完整功能：

1. 启动Carla服务# 基础启动（带可视化界面）
./CarlaUE4.sh -windowed -ResX=1024 -ResY=768 -carla-port=2000

# 高性能启动（无头模式，无界面，适合服务器运行）
./CarlaUE4.sh -RenderOffScreen -carla-port=2000 -carla-world-port=2001


2. 运行全栈自动驾驶演示cd ~/carla-project
source venv/bin/activate
# 选择Town07场景，特斯拉Model 3车辆，开启可视化
python scripts/full_stack_demo.py --town Town07 --vehicle tesla3 --visualize True


3. 观察演示效果Carla窗口：车辆自动沿车道行驶，躲避行人与障碍物，遵守红绿灯

4. 终端输出：实时打印车速、航向角、障碍物距离等信息

5. 可视化窗口：显示LiDAR点云、摄像头图像、规划路径等数据

3.2 常用命令速查

功能

命令

说明

启动指定场景

./CarlaUE4.sh -windowed -carla-town=Town05

直接加载Town05场景，无需手动切换

多车仿真

python scripts/multi_vehicle_demo.py --num-vehicles 5

启动5辆自动驾驶车辆协同行驶

数据采集

python scripts/data_collector.py --output ./data --duration 120

采集2分钟多传感器数据，保存至./data

性能评估

python scripts/evaluate_performance.py --log ./logs/driving.log

分析驾驶日志，生成性能报告

核心模块详解

项目核心代码位于 src/ 目录，模块间通过ROS2消息机制通信（可选），也支持直接函数调用，模块结构如下：

src/
├── perception/  # 感知模块（目标检测、车道线识别等）
├── localization/ # 定位模块（GPS+IMU融合、SLAM）
├── planning/    # 规划模块（全局路径+局部避障）
├── control/     # 控制模块（PID/MPC控制器）
├── scenario/    # 场景管理（交通流、天气控制）
└── common/      # 公共工具（数据结构、日志、配置）


4.1 感知模块（src/perception/）

4.1.1 模块功能

输入多传感器数据，输出障碍物位置、车道线参数、交通灯状态等环境信息，核心流程：传感器数据同步 → 数据预处理 → 目标检测 → 融合后处理。

4.1.2 关键算法实现

子模块

算法选型

输入数据

输出结果

配置文件

2D目标检测

YOLOv8（预训练模型）

RGB摄像头图像（1280×720）

目标类别、2D边界框、置信度

config/perception/yolov8.yaml

3D目标检测

PointPillars（TensorRT加速）

LiDAR点云（10万点/秒）

目标3D边界框、速度、航向角

config/perception/pointpillars.yaml

车道线检测

语义分割（SegFormer）+ 多项式拟合

语义分割图像

车道线多项式参数、车道宽度

config/perception/lane_detection.yaml

传感器融合

卡尔曼滤波 + 匈牙利算法

2D检测结果、3D检测结果

融合后的目标信息（置信度提升）

config/perception/fusion.yaml

4.1.3 快速测试

# 启动感知模块单独测试
python src/perception/perception_demo.py --visualize True


将弹出可视化窗口，显示原始图像、检测结果、点云融合效果。

4.2 规划模块（src/planning/）

4.2.1 分层规划架构

1. 全局路径规划：基于A*算法，输入起点/终点与地图拓扑，输出全局导航路径（道路级）

2. 行为决策：基于有限状态机（FSM），处理跟车、变道、红绿灯等场景决策

3. 局部路径规划：基于MPC（模型预测控制），输入全局路径与障碍物信息，输出可执行的局部路径

4.2.2 关键参数调整

核心参数位于 config/planning/mpc_params.yaml，常用调整项：

mpc:
  horizon: 10          # 预测步长（越大越稳定，耗时越长）
  speed_weight: 1.0    # 速度跟踪权重
  tracking_weight: 5.0 # 路径跟踪权重
  control_weight: 0.1  # 控制量平滑权重
  max_steer: 0.5       # 最大转向角（弧度）
  max_accel: 2.0       # 最大加速度（m/s²）


4.3 控制模块（src/control/）

支持两种控制器，可通过配置文件切换：

控制器类型

优点

缺点

适用场景

切换命令

PID控制器

实现简单、响应快、参数易调

高速场景鲁棒性差

低速行驶、停车场景

--controller pid

MPC控制器

考虑车辆动力学约束，稳定鲁棒

计算耗时较长，需GPU加速

高速行驶、复杂避障

--controller mpc

数据采集与评估

5.1 多传感器数据采集

5.1.1 采集配置

修改 config/data_collection/sensor_config.yaml 配置需要采集的传感器：

sensors:
  front_rgb:          # 前向RGB摄像头
    type: "sensor.camera.rgb"
    position: [1.5, 0, 2.4]  # 相对于车辆的安装位置（x前，y左，z上）
    resolution: [1280, 720]
    fps: 15
  top_lidar:          # 车顶LiDAR
    type: "sensor.lidar.ray_cast"
    position: [0, 0, 2.8]
    range: 100.0
    points_per_second: 1000000
  gps_imu:            # GPS+IMU组合
    type: "sensor.other.gnss"
    frequency: 10


5.1.2 启动采集

# 启动采集脚本，指定输出目录、采集时长、场景
python scripts/data_collector.py \
  --output ./data/town07_rain \
  --duration 300 \  # 采集5分钟
  --town Town07 \   # 场景
  --weather rain    # 雨天环境


5.1.3 数据格式

采集的数据按时间戳对齐，存储结构如下：

data/town07_rain/
├── 20251126_100000/  # 采集时间戳
│   ├── front_rgb/    # RGB图像（PNG格式）
│   │   ├── 1732584000.0.png
│   │   └── ...
│   ├── top_lidar/    # LiDAR点云（PCD格式）
│   ├── gps_imu/      # GPS/IMU数据（CSV格式）
│   └── annotations/  # 自动标注的目标信息（JSON格式）
└── collect_info.yaml # 采集配置信息


5.2 性能评估体系

5.2.1 评估指标

评估维度

核心指标

计算方式

优秀阈值

安全性

碰撞率、最小安全距离

碰撞次数/行驶里程，最小距离统计

碰撞率=0，最小距离>1.5m

舒适性

加加速度（Jerk）、转向波动

加速度变化率、转向角标准差

Jerk<2m/s³

效率

平均车速、行程时间

总里程/总时间

≥设计车速的80%

稳定性

路径跟踪误差、车速跟踪误差

横向误差标准差、车速误差绝对值

横向误差<0.3m

5.2.2 生成评估报告

# 1. 先运行自动驾驶并保存日志
python scripts/full_stack_demo.py --save-log ./logs/town07_demo.log

# 2. 生成评估报告（支持HTML/CSV格式）
python scripts/evaluate_performance.py \
  --log ./logs/town07_demo.log \
  --output ./reports/town07_report.html \
  --format html


打开HTML报告可查看指标图表与详细分析。

调试与扩展

6.1 常见问题排查

6.2 模块扩展指南

6.2.1 新增自定义算法（以替换目标检测算法为例）

1. 添加算法代码：在 src/perception/detectors/ 目录下新建 faster_rcnn_detector.py

2. 实现统一接口：必须包含 init()（初始化模型）和 detect(image)（执行检测）方法

3. 修改配置文件：在 config/perception/yolov8.yaml 中修改 detector_type: faster_rcnn

4. 添加依赖：在 requirements/optional.txt 中添加Faster R-CNN相关依赖（如torchvision）

5. 测试验证：运行感知模块测试脚本，确认新算法正常输出结果

6.2.2 自定义仿真场景

通过 src/scenario/scenario_builder.py 构建自定义场景，示例：

def build_intersection_scenario(world):
    # 1. 设置天气
    weather = carla.WeatherParameters(rain_intensity=0.8)
    world.set_weather(weather)
    
    # 2. 生成动态障碍物（3辆社会车辆+2个行人）
    spawn_points = world.get_map().get_spawn_points()
    # 生成社会车辆
    for i in range(3):
        vehicle_bp = world.get_blueprint_library().find("vehicle.audi.a2")
        world.spawn_actor(vehicle_bp, spawn_points[i+5])
    # 生成行人
    for i in range(2):
        pedestrian_bp = world.get_blueprint_library().find("walker.pedestrian.0001")
        world.spawn_actor(pedestrian_bp, spawn_points[i+10])
    
    # 3. 设置目标点
    goal_location = carla.Location(x=100, y=200, z=0)
    return goal_location


贡献指南

7.1 代码提交规范

提交代码时遵循以下规范，便于代码审查与版本管理：

commit格式：[模块名] 功能描述（动词开头）
示例1：[perception] 新增Faster R-CNN目标检测算法
示例2：[control] 修复MPC控制器高速抖动问题
示例3：[docs] 补充环境配置的常见问题


7.2 贡献流程

1. Fork本仓库到个人GitHub账号

2. 创建功能分支：git checkout -b feature/your-feature-name

3. 提交代码并按规范写commit信息

4. 推送分支到个人仓库：git push origin feature/your-feature-name

5. 在GitHub上提交Pull Request，描述功能细节与测试结果

许可证与参考资料

8.1 许可证

本项目采用 MIT许可证 开源，允许非商业与商业使用，但需保留原作者信息。

8.2 参考资料

- Carla官方文档：https://carla.readthedocs.io/

- YOLOv8官方仓库：https://github.com/ultralytics/ultralytics

- 自动驾驶算法入门：Awesome-Autonomous-Driving

- Model Predictive Control for Autonomous Vehicles: Theory and Practice

