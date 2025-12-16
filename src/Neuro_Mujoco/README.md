# 基于 MuJoCo 的神经网络代理实现
实现基于 MuJoCo 物理引擎的机器人、机械结构等智能体的感知、规划与控制，结合神经网络实现动态环境中的自主决策与行为生成。

# 环境配置
平台：Windows 10/11，Ubuntu 20.04/22.04（ROS推荐），macOS（Intel/Apple Silicon）  
软件：Python 3.7-3.12（需支持 3.7 及以上版本）、PyTorch（不依赖 TensorFlow）  
核心依赖：MuJoCo 物理引擎、mujoco-python 绑定  

# 基础依赖安装
安装 Python 3.11（推荐版本）  
安装 MuJoCo 及相关依赖：  
```shell
# 安装MuJoCo Python绑定
pip install mujoco -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

# 安装PyTorch（根据系统配置选择合适版本）
pip3 install torch torchvision torchaudio -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

# 安装文档生成工具
pip install mkdocs -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
pip install -r requirements.txt
```

（可选）验证安装：  
```shell
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
mkdocs --version
```

# ROS 集成配置（Ubuntu 专属）
### 1. ROS 版本要求
- Ubuntu 20.04 → ROS Noetic  
- Ubuntu 22.04 → ROS Humble（需适配 Python 3.10+）  

### 2. 安装 ROS 依赖
```shell
# ROS Noetic 依赖（Ubuntu 20.04）
sudo apt install ros-noetic-rospy ros-noetic-catkin ros-noetic-geometry-msgs ros-noetic-sensor-msgs
sudo apt install python3-catkin-tools python3-rosdep
```

### 3. Catkin 工作空间配置
```shell
# 创建/进入 Catkin 工作空间（示例路径：~/桌面/nn/catkin_ws）
mkdir -p ~/桌面/nn/catkin_ws/src && cd ~/桌面/nn/catkin_ws/src

# 将 mujoco_ros 功能包软链接到 Catkin 工作空间
ln -s ~/桌面/nn/src/Neuro_Mujoco/mujoco_ros ./mujoco_ros

# 初始化 rosdep（首次使用）
sudo rosdep init && rosdep update
rosdep install --from-paths ./ --ignore-src -r -y
```

### 4. 编译 Catkin 工作空间
```shell
cd ~/桌面/nn/catkin_ws
# 清理旧编译缓存
rm -rf build devel
# 编译（支持虚拟环境 Python）
catkin_make -DPYTHON_EXECUTABLE=$(which python3)
```

### 5. ROS 启动步骤
#### 5.1 激活环境
```shell
# 激活 Python 虚拟环境
source ~/桌面/nn/.venv/bin/activate

# 加载 Catkin 环境
source ~/桌面/nn/catkin_ws/devel/setup.bash
```

#### 5.2 启动仿真与 ROS 节点
```shell
# 启动 Anymal B 模型 + ROS 控制/订阅节点
roslaunch mujoco_ros main.launch
# 或使用绝对路径（避免软链接歧义）
roslaunch ~/桌面/nn/src/Neuro_Mujoco/mujoco_ros/launch/main.launch
```

#### 5.3 核心节点说明
- `mujoco_core`：MuJoCo 仿真核心，发布关节状态（`/mujoco/joint_states`）、基座姿态（`/mujoco/pose`）  
- `mujoco_ctrl_publisher`：控制指令发布者，发布 `/mujoco/ctrl_cmd` 话题（适配 Anymal B 控制维度 nu=12）  
- `mujoco_state_subscriber`：状态订阅者，实时打印关节状态与基座姿态  

# 文档查看
在命令行中进入项目根目录，运行：  
```shell
mkdocs build
mkdocs serve
```
使用浏览器打开 http://127.0.0.1:8000，查看项目文档是否正常显示。

# 常见问题解决
1. **`ModuleNotFoundError: No module named 'rclpy'`**  
   → 错误导入 ROS 2 库，替换为 ROS 1 的 `rospy` 即可。  

2. **`unrecognized arguments: __name:=mujoco_core`**  
   → 修改 `main.py` 中参数解析：  
   ```python
   # 原代码：args = parser.parse_args()
   args, unknown = parser.parse_known_args()  # 忽略 ROS 自动参数
   ```

3. **`模型文件不存在`**  
   → 修改 `launch/main.launch` 中的模型路径，指向真实 MuJoCo 模型（如 Anymal B：`/home/lan/桌面/nn/mujoco_menagerie/anybotics_anymal_b/anymal_b.xml`）。  

4. **`NameError: name 'count' is not defined`**  
   → 初始化 `count` 变量或注释冗余计数逻辑：  
   ```python
   count = 0  # 循环前初始化
   while ...:
       count += 1
       if count % 5 == 0:  # 按需保留
           ...
   ```

# 贡献指南
提交代码前，请阅读 贡献指南。代码优化方向包括：  
- 遵循 PEP 8 代码风格 并完善注释  
- 实现神经网络在 MuJoCo 模拟环境中的应用（如强化学习控制、运动规划等）  
- 撰写对应功能的 文档  
- 添加自动化测试（包括模型加载验证、物理模拟稳定性测试、神经网络推理性能测试等）  
- 优化物理模拟与神经网络的交互效率（如数据采集、动作执行链路）  
- 扩展 ROS 功能（如多机通信、RViz 可视化、Gazebo 联合仿真）  

# 参考资源
- MuJoCo 官方文档  
- MuJoCo GitHub 仓库  
- MuJoCo Python 绑定教程  
- MuJoCo 模型库（Menagerie）  
- 神经网络基础原理  
- MuJoCo 强化学习教程（MJX）  
- ROS Noetic 官方文档  
- Catkin 工作空间教程