# 基于 MuJoCo 的神经网络代理实现
实现基于 MuJoCo 物理引擎的机器人、机械结构等智能体的感知、规划与控制，结合神经网络实现动态环境中的自主决策与行为生成。新增**强化学习策略推理**与**ROS 1 实时通信**功能，支持端到端的智能控制流程。

# 环境配置
平台：Windows 10/11，Ubuntu 20.04/22.04（ROS推荐），macOS（Intel/Apple Silicon）  
软件：Python 3.7-3.12（需支持 3.7 及以上版本）、PyTorch（策略网络依赖）  
核心依赖：MuJoCo 物理引擎、mujoco-python 绑定  

# 基础依赖安装
安装 Python 3.11（推荐版本）  
安装核心依赖：  
```shell
# 安装MuJoCo Python绑定
pip install mujoco -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

# 安装PyTorch（策略网络必需，根据系统配置选择）
pip3 install torch torchvision torchaudio -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

# 安装文档生成工具
pip install mkdocs -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
pip install -r requirements.txt
```

（可选）验证安装：  
```shell
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__)"  # 验证策略依赖
mkdocs --version
```

# 核心功能使用
## 1. 模型可视化与控制
支持直接加载模型文件或目录，集成策略控制与ROS通信：  
```shell
# 基础可视化（无控制）
python mujoco_utils.py visualize /path/to/model.xml

# 启用ROS 1模式（发布关节状态/基座姿态，接收控制指令）
python mujoco_utils.py visualize /path/to/model.xml --ros

# 加载预训练策略模型（自动生成控制指令）
python mujoco_utils.py visualize /path/to/model.xml --policy /path/to/policy.pth

# 联合模式（ROS + 策略控制）
python mujoco_utils.py visualize /path/to/model.xml --ros --policy /path/to/policy.pth
```
- 交互说明：可视化窗口支持鼠标拖拽旋转、滚轮缩放，按ESC键退出
- 控制优先级：ROS指令 > 策略推理 > 无控制

## 2. 模型格式转换
支持XML与MJB（二进制）格式互转（MJB加载速度更快）：  
```shell
# XML转MJB
python mujoco_utils.py convert input.xml output.mjb

# MJB转XML
python mujoco_utils.py convert input.mjb output.xml
```

## 3. 模拟速度测试
多线程测试模型仿真性能，评估实时性：  
```shell
# 默认参数（1线程，10000步）
python mujoco_utils.py testspeed /path/to/model.xml

# 自定义配置（4线程，50000步，控制噪声0.02）
python mujoco_utils.py testspeed /path/to/model.xml --nthread 4 --nstep 50000 --ctrlnoise 0.02
```
输出指标包括：每秒步数、实时因子（仿真时间/真实时间）、线程耗时统计

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


# 策略网络说明
## 网络结构
轻量级多层感知器（MLP），适用于机器人关节控制：  
```python
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),  # 观测维度=关节位置数+速度数
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),  # 输出维度=控制维度(nu)
            nn.Tanh()  # 输出范围[-1,1]，自动映射到实际控制范围
        )
```

## 观测与控制映射
- 输入观测：`[关节位置(qpos) + 关节速度(qvel)]`
- 输出控制：归一化指令`[-1,1]`，通过模型`actuator_ctrlrange`自动映射到实际范围

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
   → 修改参数解析逻辑，忽略ROS自动添加的参数：  
   ```python
   args, unknown = parser.parse_known_args()  # 替代 parser.parse_args()
   ```

3. **`模型文件不存在`**  
   → 检查模型路径是否正确，可视化命令支持自动搜索目录下的`.xml`/`.mjb`文件。  

4. **`策略模型加载失败`**  
   → 确认：① 模型文件存在 ② PyTorch已安装 ③ 模型输入/输出维度与环境匹配（obs_dim=qpos+qvel维度，action_dim=nu）

5. **`ROS话题无数据`**  
   → 检查：① `roscore`是否启动 ② Catkin环境是否加载（`source devel/setup.bash`）③ 关节名称映射是否正确

# 贡献指南
提交代码前，请阅读 贡献指南。代码优化方向包括：  
- 遵循 PEP 8 代码风格 并完善注释  
- 扩展策略网络类型（如CNN、RNN，适配视觉观测）  
- 优化策略推理效率（如ONNX导出、TensorRT加速）  
- 完善ROS功能（如服务调用、参数服务器配置）  
- 增加自动化测试（模型加载/转换验证、策略性能基准测试）  
- 支持多智能体仿真与分布式控制  

# 参考资源
- MuJoCo 官方文档  
- MuJoCo Python 绑定教程  
- MuJoCo 模型库（Menagerie）  
- PyTorch 神经网络教程  
- ROS 1 Noetic 官方文档  
- 强化学习在MuJoCo中的应用（MJX框架）