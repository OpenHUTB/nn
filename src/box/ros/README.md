### ROS 2 机械臂感知+数据获取+虚拟运动可视化 完整封装步骤（最终版）
「感知模块发布关节数据 + 数据获取模块保存数据 + RViz2 虚拟运动可视化」的核心步骤，每一步可落地、无冗余，适配 ROS 2 Humble + WSL2 环境：

#### 一、基础环境与工作空间准备
1. **确认依赖安装**（确保ROS 2核心依赖齐全）：
   ```
   sudo apt update && sudo apt install -y ros-humble-rclpy ros-humble-sensor-msgs ros-humble-robot-state-publisher ros-humble-rviz2
   ```
2. **创建并初始化ROS 2工作空间**：
   ```
   # 创建目录
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws
   # 初始化编译（生成基础配置）
   colcon build --symlink-install
   # 加载环境变量（每次新开终端需执行，或添加到~/.bashrc）
   source install/setup.bash
   ```

#### 二、创建核心功能包（感知+数据获取）
1. **创建感知模块功能包**（发布机械臂关节数据）：
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python perception_module --dependencies rclpy std_msgs sensor_msgs
   ```
2. **创建数据获取模块功能包**（订阅并保存关节数据）：
   ```bash
   ros2 pkg create --build-type ament_python data_acquisition_module --dependencies rclpy std_msgs sensor_msgs
   ```

#### 三、编写核心代码（感知+数据获取）
##### 1. 感知模块代码（发布关节数据）
- 编辑文件：`~/ros2_ws/src/perception_module/perception_module/perception_node.py`
  写入可运行的关节数据发布代码（核心逻辑：定时发布6轴机械臂关节角度，模拟joint2抬升运动）；
- 配置编译入口：修改`~/ros2_ws/src/perception_module/setup.py`，在`entry_points`中添加：
  ```python
  'console_scripts': ['arm_perception_node = perception_module.perception_node:main']
  ```

##### 2. 数据获取模块代码（保存数据到CSV）
- 编辑文件：`~/ros2_ws/src/data_acquisition_module/data_acquisition_module/acquisition_node.py`
  写入订阅关节话题、保存数据到CSV的代码（核心逻辑：订阅`/arm/joint_states`，按时间戳保存关节名+角度）；
- 配置编译入口：修改`~/ros2_ws/src/data_acquisition_module/setup.py`，在`entry_points`中添加：
  ```python
  'console_scripts': ['arm_acquisition_node = data_acquisition_module.acquisition_node:main']
  ```

#### 四、编写URDF模型+可视化配置（虚拟运动）
1. **创建URDF文件**（定义6轴机械臂结构）：
   ```bash
   mkdir -p ~/ros2_ws/src/perception_module/urdf
   nano ~/ros2_ws/src/perception_module/urdf/6dof_arm.urdf
   ```
   写入6轴机械臂URDF代码（定义底座、连杆、关节、颜色，适配`joint1~joint6`）；
2. **配置URDF打包**：修改`~/ros2_ws/src/perception_module/setup.py`，在`data_files`中添加URDF文件路径：
   ```python
   ('share/' + package_name + '/urdf', ['urdf/6dof_arm.urdf'])
   ```

#### 五、编译工作空间（使代码/配置生效）
```bash
cd ~/ros2_ws
colcon build --symlink-install  # --symlink-install：修改代码无需重新编译
source install/setup.bash       # 重新加载环境变量
```

#### 六、启动节点+验证功能（分终端操作，最稳定）
| 终端序号 | 执行命令（核心操作）| 功能说明 |
|----------|---------------------------------------------|--------------------------|
| 终端1    | `ros2 run perception_module arm_perception_node` | 启动感知模块，发布关节数据 |
| 终端2    | `ros2 run data_acquisition_module arm_acquisition_node` | 启动数据获取模块，保存CSV |
| 终端3    | `ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(cat ~/ros2_ws/src/perception_module/urdf/6dof_arm.urdf)"` | 解析URDF，发布坐标系（TF） |
| 终端4    | `rviz2` | 启动RViz2，手动配置可视化 |

##### RViz2 手动配置（必做，确保模型显示）
1. 点击「Add」→ 选择「RobotModel」→ 「OK」；
2. RobotModel 面板配置：
   - `Description Source` → `Topic`；
   - `Description Topic` → `/robot_description`；
   - `Joint State Topic` → `/arm/joint_states`；
3. 顶部「Global Options」→ `Fixed Frame` → `base_link`。

