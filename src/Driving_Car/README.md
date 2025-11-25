Carla 无人车仿真项目
项目概述
本项目基于 Carla 开源自动驾驶仿真器，构建了一套功能完善的无人车仿真平台，支持自主导航、障碍物避障、交通灯识别、车道保持、动态场景交互等核心自动驾驶任务。通过模块化设计，实现了 Carla 客户端连接、车辆控制、环境感知、场景配置与数据可视化的完整流程，适用于自动驾驶算法（路径规划、感知融合、决策控制）的研究、开发与验证。
环境准备
依赖安装
1. Carla 仿真器安装
下载 Carla 对应版本（推荐 0.9.15 稳定版）：Carla 官方下载地址
解压后设置环境变量（Windows/Linux/macOS 通用）：
Windows：将 Carla 解压目录下的 PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg 添加到系统 PYTHONPATH
Linux/macOS：将 PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg 添加到 PYTHONPATH
验证安装：终端执行 python -c "import carla; print('Carla installed successfully')"
2. Python 依赖库安装
bash
运行
pip install numpy opencv-python matplotlib absl-py pyyaml scipy
carla：Carla 仿真器核心 API，负责车辆控制、环境交互
numpy：数值计算与传感器数据处理
opencv-python：图像传感器数据解析与可视化
matplotlib：行驶轨迹、速度曲线等数据绘图
absl-py：命令行参数解析
pyyaml：配置文件读取
scipy：路径规划算法（如 A*、RRT*）支持
项目结构
文件名	功能描述
main.py	核心控制程序，实现 Carla 客户端连接、车辆加载、基础控制与传感器配置
navigate_autonomous.py	自主导航任务程序，结合路径规划（A*/RRT*）与车道保持控制
obstacle_avoidance.py	障碍物避障专项任务，支持动态 / 静态障碍物检测与规避
traffic_light_detection.py	交通灯识别与响应任务，实现红灯停、绿灯行的交通规则遵守
scene_generator.py	动态场景生成工具，支持随机车辆、行人、交通事件（如突发障碍物）配置
carla_utils.py	工具类封装，包含传感器数据解析、坐标转换、路径平滑等通用功能
config.yaml	配置文件，存储车辆参数、传感器类型、任务参数（车速限制、安全距离等）
README.md	项目说明文档
核心功能
1. 基础控制与环境交互（main.py）
自动客户端连接：启动时自动检测 Carla 服务器，支持 IP / 端口配置，提供连接状态反馈
灵活车辆配置：支持加载不同车型（如 Tesla Model 3、林肯 MKZ），自定义传感器组合（摄像头、激光雷达、毫米波雷达、GPS/IMU）
实时数据采集：同步获取传感器数据（图像、点云、定位信息），支持数据本地存储
基础控制模式：支持手动键盘控制与自动速度 / 转向控制切换
python
运行
核心客户端连接与车辆加载示例（main.py）
client = carla.Client('localhost', 2000)  # 连接Carla服务器（默认IP:localhost，端口:2000）
client.set_timeout(10.0)  # 超时时间10秒
world = client.load_world('Town05')  # 加载地图（Town01-Town10可选）
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')  # 选择车型
spawn_point = world.get_map().get_spawn_points()[0]  # 选择初始 spawn 点
vehicle = world.spawn_actor(vehicle_bp, spawn_point)  # 生成车辆
2. 自主导航系统（navigate_autonomous.py）
多算法路径规划：集成 A*、RRT * 路径规划算法，支持从起点到目标点的最优路径生成
地图解析：自动解析 Carla 地图的车道线、路口、交通标志位置信息
车道保持控制：基于 PID 控制实现精准车道居中，支持弯道速度自适应调整
目标点配置：支持手动设置目标点或随机生成目标点，适配不同测试场景
python
运行
路径规划配置示例（navigate_autonomous.py）
from carla_utils import AStarPlanner
planner = AStarPlanner(world_map)  # 初始化A*规划器
start_pos = vehicle.get_transform().location  # 获取车辆当前位置
target_pos = carla.Location(x=100, y=50, z=0)  # 设置目标位置（世界坐标系）
waypoints = planner.plan(start_pos, target_pos)  # 生成路径航点
3. 避障与交通规则遵守（obstacle_avoidance.py & traffic_light_detection.py）
多传感器融合避障：结合激光雷达点云和摄像头图像，检测车辆、行人、静态障碍物（如路沿、锥桶）
动态避障策略：根据障碍物速度、距离调整避障路径，支持减速、绕行两种模式
交通灯识别：基于图像颜色识别与 Carla 交通灯 API，实现红绿灯状态实时检测与响应
安全距离控制：可配置最小安全距离（默认 2 米），自动调整车速避免碰撞
4. 动态场景生成（scene_generator.py）
随机化场景元素：支持配置随机车辆数量（5-50 辆）、行人数量（10-30 人）、障碍物类型（锥桶、石墩）
交通事件模拟：可生成突发场景（如行人横穿马路、前车急刹），用于算法鲁棒性测试
场景参数可配置：通过config.yaml调整车辆密度、行人速度、事件触发概率
5. 可视化与数据记录
实时画面展示：支持车辆第一视角（摄像头）、全局俯瞰视角、传感器数据（点云 / 图像）同步显示
数据记录功能：自动保存行驶轨迹、车速曲线、传感器原始数据，用于后续算法分析
可视化工具：通过 matplotlib 绘制实时速度、转向角、距离障碍物距离等关键指标
使用方法
1. 启动 Carla 服务器
Windows：双击 Carla 解压目录下的 CarlaUE4.exe（默认端口 2000）
Linux：终端进入 Carla 目录，执行 ./CarlaUE4.sh
可选参数（如加载特定地图）：./CarlaUE4.sh /Game/Carla/Maps/Town03
2. 运行仿真任务
基础控制（支持键盘手动控制）
bash
运行
python main.py --map Town05 --vehicle tesla.model3
自主导航（自动规划路径到目标点）
bash
运行
python navigate_autonomous.py --target_x 150 --target_y 80 --speed_limit 30
障碍物避障测试
bash
运行
python obstacle_avoidance.py --obstacle_count 10 --safe_distance 2.5
交通灯识别与响应
bash
运行
python traffic_light_detection.py --map Town04  # Town04包含完整交通灯系统
动态场景生成 + 自主导航
bash
运行
python scene_generator.py --vehicle_count 30 --pedestrian_count 20 --run_navigation
交互操作
操作	功能
鼠标左键拖拽	调整仿真全局视角
鼠标滚轮	视角缩放
WASD 键	手动控制车辆（仅 main.py 手动模式）
空格键	紧急刹车
Ctrl+C	终止仿真程序
F1 键	切换第一视角 / 全局视角
F2 键	显示 / 隐藏激光雷达点云
参数调整指南
参数	调整范围	效果说明
speed_limit	10~60（km/h）	无人车最大行驶速度，提高值增加任务难度
safe_distance	1.0~5.0（米）	与障碍物的最小安全距离，增大值提高安全性但降低通行效率
pid_kp（转向 PID 比例增益）	0.5~2.0	增大会加快转向响应，过大会导致车道震荡
obstacle_count	5~50（个）	动态障碍物数量，增多会提升避障算法测试强度
lidar_points	8192~65536	激光雷达点数，增多提高检测精度但增加计算量
planning_algorithm	A*/RRT*	路径规划算法选择，A效率高，RRT更适用于复杂障碍物场景
参考资料
Carla 官方文档
Carla GitHub 仓库
自动驾驶路径规划算法详解（A*/RRT*）
OpenCV 图像识别官方教程
激光雷达点云处理入门
