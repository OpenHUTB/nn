# 人车模拟
# CARLA 全局路径规划和SLAM节点

实现基于 CARLA 模拟器的车辆全局路径规划功能，支持从车辆自身位置（通过里程计输入）到地图中随机目标点的路径生成，并通过 ROS 2 话题可视化规划结果，模拟用户预订行程后车辆的全局路径规划过程。 新增支持 3D 建图功能，集成 GMapping 算法实现三维环境下的地图构建。

## 功能概述
该节点是 CARLA 自动驾驶仿真系统中的核心模块之一，主要功能包括：
- **CARLA 交互**：建立与 CARLA 服务器的连接，获取地图数据和可导航的生成点（目标点候选）
- **路径规划服务**：提供 ROS 2 服务 `plan_to_random_goal`，接收起始点里程计信息，生成到随机目标点的全局路径
- **路径可视化**：通过 `visualization_marker` 话题发布路径标记，在 RViz 中以绿色线条直观展示规划结果
- **鲁棒性保障**：包含路径有效性校验（确保路径长度满足需求）、异常捕获和错误日志输出，避免无限循环或崩溃
- **3D 建图**：集成 GMapping 算法，利用激光雷达数据进行实时三维地图构建

## 环境配置
### 基础环境
| 类别       | 要求                          | 说明                                  |
|------------|-------------------------------|---------------------------------------|
| 操作系统   | Ubuntu 20.04 LTS              | 兼容 ROS 2 Foxy 版本，是 CARLA 推荐系统 |
| ROS 版本   | ROS 2 Foxy Fitzroy            | 需确保环境变量无 ROS 1 冲突          |
| Python 版本| Python 3.8                    | 匹配 ROS 2 Foxy 依赖及 CARLA 0.9.15 要求 |
| CARLA 版本 | CARLA 0.9.15                  | 需与代码中客户端连接逻辑兼容          |

### 依赖安装
1. **ROS 2 Foxy 安装**  
   参考 [ROS 2 官方文档](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html) 完成基础安装，确保 `ros-foxy-nav-msgs`、`ros-foxy-visualization-msgs` 等依赖已安装：
   ```bash
   sudo apt update && sudo apt install -y ros-foxy-nav-msgs ros-foxy-visualization-msgs ros-foxy-geometry-msgs
   ```

2. **CARLA Python 客户端**  
   从 [CARLA 官网](https://carla.org/2023/07/11/release-0.9.15/) 下载对应版本的 CARLA 压缩包，解压后将 `PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg` 添加到 Python 路径：
   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/your/carla/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg
   ```
3. **其他 Python 依赖**  
   安装路径计算和坐标转换所需依赖：
   ```bash
   pip install tf-transformations  # 用于欧拉角转四元数
   ```

4. **编译工具**  
   安装 ROS 2 编译工具 `colcon`：
   ```bash
   sudo apt install -y python3-colcon-common-extensions
   ```
5. **GMapping 依赖**  安装 slam_gmapping 包以支持 3D 建图功能：
'''
bash
sudo apt install ros-foxy-slam-gmapping
'''
## 快速开始
### 1. 工作空间准备
假设你的 ROS 2 工作空间为 `carla_ws`，将该节点所在的功能包 `carla_global_planner` 放入 `src` 目录：
```bash
mkdir -p ~/carla_ws/src
cd ~/carla_ws/src
# 克隆或复制 carla_global_planner 功能包到此处
```

### 2. 编译节点
在工作空间根目录执行编译命令，确保无编译错误：
```bash
cd ~/carla_ws
# 激活 ROS 2 环境
source /opt/ros/foxy/setup.bash
# 编译功能包（仅编译指定包可加快速度）
colcon build --packages-select carla_global_planner
```

### 3. 启动流程
#### 步骤 1：启动 CARLA 服务器
打开新终端，进入 CARLA 安装目录，启动无渲染模式（节省资源）：
```bash
cd ~/carla/CARLA_0.9.15  # 替换为你的 CARLA 实际路径
# 低画质+无渲染启动（适合后台运行）
./CarlaUE4.sh -quality-level=Low -RenderOffScreen
```
- 若需可视化 CARLA 场景，可去掉 `-RenderOffScreen` 参数

#### 步骤 2：启动全局规划节点
打开新终端，激活工作空间环境并启动节点：
```bash
cd ~/carla_ws
# 激活 ROS 2 环境和工作空间环境
source /opt/ros/foxy/setup.bash
source install/setup.bash
# 启动节点和建图功能
ros2 launch carla_global_planner carla_slam_launch.py
```
- 节点启动成功后，会输出日志 `CARLA全局路径规划服务已启动`
- GMapping 建图节点同时启动，开始接收激光雷达数据并构建地图
- 若脚本卡住无报错，说明节点正常运行（等待服务请求）；若报错 "CARLA客户端初始化失败"，需检查 CARLA 服务器是否已启动，或连接地址/端口是否正确

#### 步骤 3：调用路径规划服务
打开新终端，通过 `ros2 service call` 命令测试服务（需替换 `start` 参数为实际里程计数据）：
```bash
# 激活环境
source /opt/ros/foxy/setup.bash
source ~/carla_ws/install/setup.bash
# 调用服务（示例：起始点为(0,0,0)，姿态为默认）
ros2 service call /plan_to_random_goal carla_global_planner/srv/PlanGlobalPath "{start: {header: {frame_id: 'map'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}}"
```
- 服务调用成功后，节点会输出路径信息（如 `成功规划路径，包含XX个路点`）

#### 步骤 4：可视化路径
打开新终端，启动 RViz 查看路径和建图结果：
```bash
source /opt/ros/foxy/setup.bash
rviz2
```
在 RViz 中配置：
1. 设置 `Fixed Frame` 为 `map`
2. 添加 `Marker` 显示项，设置 `Topic` 为 `/visualization_marker`
3. 路径会以绿色线条显示在场景中
4.  添加 Map 显示项，设置 Topic 为 /map 查看实时构建的地图

SLAM 算法详解
SLAM 概述
SLAM (Simultaneous Localization and Mapping) 是移动机器人领域的一项关键技术，它允许机器人在未知环境中同时构建地图并确定自身位置。这对于自主导航至关重要，因为机器人必须知道自己的位置才能有效地规划路径并避开障碍物。

SLAM 问题的核心挑战在于：

机器人不知道自己在哪里（定位问题）
机器人不知道环境是什么样子（建图问题）
这两个问题相互依赖，必须同时解决
GMapping 算法原理
本项目采用经典的 GMapping 算法实现 2D 建图功能。GMapping 是一种基于粒子滤波的 SLAM 算法，它通过以下步骤实现地图构建：

粒子滤波: 使用多个粒子表示机器人的可能位姿，每个粒子都携带一个地图副本
运动模型更新: 根据里程计数据预测每个粒子的新位姿
观测模型评估: 利用激光雷达扫描数据评估每个粒子的权重
重采样: 根据权重重新分布粒子集，保留高似然的粒子
地图更新: 基于最优粒子的位姿估计更新全局地图
GMapping 工作流程
初始化阶段
创建 N 个粒子，每个粒子代表机器人的一种可能位姿
每个粒子维护一张独立的地图
所有粒子初始权重相等
循环迭代过程
当收到新的传感器数据时，算法执行以下步骤：

预测步骤

使用里程计读数更新每个粒子的位置
考虑运动噪声，使粒子分布更加分散
更新步骤

对于每个粒子，比较激光雷达扫描数据与该粒子地图的预期观测
计算粒子权重，权重越高表示该粒子越接近真实情况
重采样步骤

根据粒子权重进行重采样
权重高的粒子被多次复制，权重低的粒子被淘汰
保证粒子集始终代表最可能的机器人位姿分布
地图更新

基于最高权重粒子的信息更新全局地图
将新的观测数据融合到现有地图中
数据流
输入传感器数据
激光雷达扫描数据 (/scan): 提供周围环境的距离测量
里程计数据 (/odom): 提供相对运动信息
输出结果
2D 占据栅格地图 (/map): 表示环境的占用情况
机器人位姿估计 (/map 坐标系下的变换): 提供机器人在地图中的位置
参数配置
GMapping 算法的关键参数已在 launch 文件中预设，可根据具体场景调整：

参数名	默认值	描述
maxUrange	10.0	激光雷达最大使用距离 (单位: 米)
sigma	0.05	扫描匹配的标准差
particles	30	粒子数量
delta	0.05	地图分辨率 (单位: 米/cell)
temporalUpdate	-1.0	地图更新频率 (-1 表示自适应更新)
linearUpdate	1.0	线性移动多少米后更新地图
angularUpdate	0.5	角度旋转多少弧度后更新地图
resampleInterval	1	重采样间隔(以scan为单位)
xmin, ymin, xmax, ymax	-100.0 ~ 100.0	地图边界
算法优势与局限性
优势
实现相对简单，易于理解和部署
在中小规模环境中表现良好
对传感器噪声具有较好的鲁棒性
能够处理闭环检测问题
局限性
计算复杂度随粒子数量增加而线性增长
在大规模环境中效率较低
对快速运动或剧烈震动敏感
难以处理动态环境中的移动物体

## 核心接口说明
### 1. ROS 2 服务
| 服务名称               | 服务类型                          | 功能描述                                  |
|------------------------|-----------------------------------|-------------------------------------------|
| `/plan_to_random_goal` | `carla_global_planner/srv/PlanGlobalPath` | 输入起始点里程计信息，返回全局路径（`nav_msgs/Path`） |

#### 服务请求（Request）
```msg
# 起始点里程计信息
nav_msgs/Odometry start
```

#### 服务响应（Response）
```msg
# 规划后的全局路径
nav_msgs/Path path
```

### 2. ROS 2 话题
话题名称	消息类型	功能描述
/visualization_marker	visualization_msgs/Marker	发布路径可视化标记（绿色 LINE_STRIP 类型）
/scan	sensor_msgs/LaserScan	激光雷达扫描数据，用于建图
/odom	nav_msgs/Odometry	里程计数据，用于定位和建图
/map	nav_msgs/OccupancyGrid	构建的 2D 占据栅格地图



## 常见问题排查
1. **节点启动报错 "CARLA客户端初始化失败"**  
   - 检查 CARLA 服务器是否已启动：`ps aux | grep CarlaUE4`
   - 确认 CARLA 服务器地址/端口与代码一致（默认 `localhost:2000`）
   - 检查 Python 路径中是否正确添加了 CARLA 的 `egg` 文件

2. **服务调用后无路径生成**  
   - 检查 CARLA 地图是否加载完成（服务器启动后需等待地图加载，约10-20秒）
   - 若提示 "未找到可用的生成点"，需更换 CARLA 地图（如 `Town01`）：
     ```bash
     # 在 CARLA 服务器终端中输入（需安装 CARLA 附加地图）
     ./CarlaUE4.sh -quality-level=Low -RenderOffScreen -map Town01
     ```

3. **RViz 中看不到路径**  
   - 确认 `Fixed Frame` 已设置为 `map`
   - 检查 `Marker` 话题是否正确订阅，且消息是否有更新（`ros2 topic echo /visualization_marker`）
   - 检查路径点的坐标是否在 RViz 可视范围内（可通过 "Reset View" 重置视角）

4. **编译报错 "找不到 carla_global_planner/srv/PlanGlobalPath"**  
   - 确认 `srv` 目录下已定义 `PlanGlobalPath.srv` 文件，且 `CMakeLists.txt`/`package.xml` 已正确配置服务依赖
5. **RViz 中看不到地图**  

确认 Map 显示项已添加且 Topic 设置为 /map
检查是否有地图数据发布：ros2 topic echo /map
确保激光雷达数据正常：ros2 topic echo /scan


## 代码结构说明
```
carla_global_planner/
├── src/
│   └── carla_global_planner_node.py  # 核心节点代码
├── srv/
│   └── PlanGlobalPath.srv            # 路径规划服务定义
├── utilities/
│   └── planner.py                    # 路径计算工具（compute_route_waypoints 函数）
├── launch/
│   └── carla_slam_launch.py          # 启动文件，包含建图功能
├── CMakeLists.txt                    # 编译配置
└── package.xml                       # 依赖和包信息配置
```
核心节点代码关键函数：
- `_initialize_carla_client()`：初始化 CARLA 客户端和地图
- `plan_path_cb()`：服务回调函数，处理路径规划请求
- `_get_valid_route()`：生成满足长度要求的有效路径
- `_build_path_message()`：将 CARLA 路径转换为 ROS 2 Path 消息
- `_visualize_path()`：发布路径可视化标记


## 贡献指南
1. **代码规范**  
   - 遵循 [PEP 8 Python 代码风格](https://peps.pythonlang.cn/pep-0008/)
   - 新增功能需添加对应的注释和文档字符串
   - 关键逻辑变更需同步更新 README.md

2. **功能扩展建议**  
   支持自定义目标点（而非仅随机目标点）
   增加路径规划算法选择（如 A*、RRT*）
   添加路径平滑处理，优化车辆行驶轨迹
   增加单元测试（基于 pytest 和 ros2 test）
   集成其他 SLAM 算法（如 Hector SLAM、Karto SLAM）以提供更多建图选项

3. **提交流程**  
   准备提交代码前，请确保：
   1. 代码无编译错误和运行时异常
   2. 新增功能包含对应的测试用例
   3. 文档（如 README.md）已同步更新


## 参考资料
1. [CARLA 官方文档](https://carla.readthedocs.io/en/0.9.15/) - 了解 CARLA Python API 和地图数据结构
2. [ROS 2 Foxy 官方文档](https://docs.ros.org/en/foxy/) - 学习 ROS 2 服务、话题和节点开发
3. [nav_msgs/Path 消息定义](https://docs.ros.org/en/foxy/api/nav_msgs/html/msg/Path.html) - 路径消息格式说明
4. [visualization_msgs/Marker 消息定义](https://docs.ros.org/en/foxy/api/visualization_msgs/html/msg/Marker.html) - 可视化标记格式说明
5. slam_gmapping 官方文档 - GMapping 算法详细说明