# 基于 MuJoCo 的智能体控制工具
整合 MuJoCo 物理引擎的模型加载、格式转换、速度测试、可视化功能，新增强化学习策略推理与ROS 1 实时通信能力，支持机器人端到端智能控制流程。
## 环境配置
1. 支持平台
Windows 10/11、Ubuntu 20.04/22.04（ROS 推荐）、macOS（Intel/Apple Silicon）
2. 软件要求
Python 3.7-3.12（推荐 3.11）、MuJoCo 物理引擎、PyTorch（策略网络依赖，可选）、ROS 1 Noetic（ROS 功能依赖，可选）
3. 核心依赖安装
shell
## 安装 MuJoCo Python 绑定
pip install mujoco -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

## PyTorch（策略网络推理必需）
pip install torch torchvision torchaudio -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

## ROS 1 依赖（仅 Ubuntu，需先安装 ROS Noetic）
sudo apt install ros-noetic-rospy ros-noetic-geometry-msgs ros-noetic-sensor-msgs
## 验证安装
shell
### 验证 MuJoCo
python -c "import mujoco; print(f'MuJoCo 版本: {mujoco.__version__}')"

# 验证 PyTorch（策略功能）
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')"

### 验证 ROS（ROS 功能）
# python -c "import rospy; print('ROS 1 导入成功')"
## 快速开始
### 模型可视化（核心功能）
支持加载单个模型文件或目录（自动识别目录内 .xml/.mjb 文件），控制优先级：ROS 指令 > 策略推理 > 无控制。
1.1 运行命令
shell
# 基础可视化（无控制）
python mujoco_utils.py visualize /path/to/your/model.xml

# 启用 ROS 1 模式（发布关节状态/接收控制指令）
python mujoco_utils.py visualize /path/to/your/model.xml --ros

# 加载预训练策略模型（自动生成控制指令）
python mujoco_utils.py visualize /path/to/your/model.xml --policy /path/to/your/policy.pth

# 混合模式（ROS + 策略控制）
python mujoco_utils.py visualize /path/to/your/model.xml --ros --policy /path/to/your/policy.pth

# 目录自动加载（选目录内第一个模型文件）
python mujoco_utils.py visualize /home/lan/桌面/nn/mujoco_menagerie/anybotics_anymal_b

# 启动交互式模型选择菜单（支持PD控制、预设位置切换）
python mujoco_utils.py menu

1.2 交互说明
窗口操作：鼠标拖拽旋转、滚轮缩放，按 ESC 键退出；
关节状态：仅发布非自由关节（排除 mjJNT_FREE 类型）的位置 / 速度；
策略映射：输出 [-1,1] 自动线性缩放至执行器 ctrlrange 范围，并强制裁剪至物理极限。
菜单模式专属指令：
- 预设位置切换：直接输入预设名称（如home/up/left，根据模型不同有所差异）
- 自定义关节角度：set <关节号> <角度>（示例：set 0 0.5，单位rad）
- 仿真控制：pause（暂停）/ resume（继续）/ exit（退出）
- ROS键盘控制（需ROS环境）：在新终端运行 `python keyboard_control.py <关节数>`


## 支持的预设模型
工具内置4种常用机器人模型，可通过交互式菜单直接选择：
1. Franka Panda（机械臂）：7关节，支持home/up/left/right预设位置
2. UR5 机械臂：6关节，支持home/up/forward预设位置
3. Franka Panda（带手爪）：8关节（含手爪），支持home/open_gripper/up_open预设位置
4. Walker2d 机器人：6关节，支持stand/walk_left/walk_right预设位置


### 模型格式转换
2.1 运行命令
shell
# XML 转 MJB（提升加载速度）
python mujoco_utils.py convert input.xml output.mjb

# MJB 转 XML（便于编辑）
python mujoco_utils.py convert input.mjb output.xml
### 仿真速度测试
多线程测试模型仿真性能，优化点：线程内动态生成控制噪声，降低内存占用。
3.1 运行命令
shell
# 默认参数（1线程，10000步，噪声0.01）
python mujoco_utils.py testspeed /path/to/your/model.xml

# 自定义配置（4线程，50000步，噪声0.02）
python mujoco_utils.py testspeed /path/to/your/model.xml --nthread 4 --nstep 50000 --ctrlnoise 0.02
3.2 输出指标
总步数 / 总耗时 / 每秒步数；
实时因子（仿真时间 / 真实时间）；
线程耗时统计（平均值 ± 标准差）。
## ROS 1 集成（Ubuntu 专属）
1. ROS 话题说明
类型	   话题名	                    消息类型	                    说明
发布	/mujoco/joint_states	sensor_msgs/JointState	      非自由关节位置 / 速度
发布	/mujoco/pose	        geometry_msgs/PoseStamped	  基座位置 / 姿态（world 帧）
订阅	/mujoco/ctrl_cmd	    std_msgs/Float32MultiArray	  控制指令（长度 = 模型 nu）
2. 运行步骤
shell
# 1. 启动 ROS 核心（新开终端）
roscore

# 2. 启动 MuJoCo ROS 节点
python mujoco_utils.py visualize /path/to/your/model.xml --ros

# 3. （可选）发布测试控制指令（100Hz）
rostopic pub /mujoco/ctrl_cmd std_msgs/Float32MultiArray "data: [0.1, 0.2, 0.3]" -r 100
## 策略网络说明
1. 网络结构
轻量级 MLP，适配机器人关节控制场景：
python
运行
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()  # 输出 [-1,1]
        )
2. 输入输出
输入：观测（关节位置 qpos + 关节速度 qvel）；
输出：归一化控制指令（[-1,1]），自动映射至模型 actuator_ctrlrange 范围：
python
运行

## PD控制器说明
适用于模型精确位置控制，核心参数与逻辑：
1. 控制参数：每个模型预配置KP（比例增益）和KD（微分增益），如Franka Panda默认KP=800.0、KD=60.0
2. 控制逻辑：
   ```python
   # 核心PD计算
   pos_error = 目标关节位置 - 当前关节位置
   vel_error = -当前关节速度
   关节力矩 = KP * pos_error + KD * vel_error

   
### 核心映射逻辑
action = ctrl_min + (ctrl_max - ctrl_min) * (action + 1) / 2
action = np.clip(action, ctrl_min, ctrl_max)  # 强制裁剪至物理极限
## 性能优化点
速度测试：线程内动态生成控制噪声，避免主线程预生成大数组；
策略推理：复用张量对象，避免重复内存分配；
控制指令：强制裁剪至执行器物理极限，防止超限报错；
路径处理：支持目录自动识别模型文件，提升易用性；
关节映射：精准过滤自由关节，仅发布有效关节状态。
## 常见问题
问题现象	               解决方案
模型加载失败	检查路径是否正确，确保文件为 .xml/.mjb 格式
策略加载失败	确认 PyTorch 已安装、策略文件路径正确、输入输出维度匹配
ROS 无数据	    确保 roscore 已启动、话题名称匹配、控制指令长度 = 模型 nu
仿真实时性差	使用 MJB 格式模型、减少线程数、降低控制噪声强度
控制指令无效果	检查模型 nu（控制维度）是否 > 0，策略输出是否映射至正确范围
## 参考资源
1、MuJoCo 官方文档
2、PyTorch 神经网络教程
3、ROS 1 Noetic 文档
4、MuJoCo 官方模型库