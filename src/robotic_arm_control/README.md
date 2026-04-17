# 机械臂控制与仿真项目
A Robotic Arm Control and Simulation Project

## 项目简介
本项目旨在实现对六轴工业机械臂的运动控制、路径规划与仿真验证，支持离线编程、实时姿态解算和简单的抓取任务调度。项目采用模块化设计，可快速适配不同型号的机械臂硬件，同时提供可视化仿真界面，降低开发与调试成本。

## 核心功能
- 正逆运动学解算：基于D-H参数法实现机械臂关节角与末端位姿的相互转换
- 路径规划：支持直线插补、圆弧插补、关节空间轨迹规划
- 控制模式：提供手动示教、自动运行、仿真验证三种控制模式
- 状态监控：实时显示关节角度、末端坐标、执行速度等关键参数
- 硬件适配：支持串口/以太网与机械臂控制器通信（兼容主流工业协议）

## 项目结构
```
Robotic_Arm_Control/
├── main.py                  # 主程序入口，GUI界面与控制逻辑
├── mujoco_sim.py            # MuJoCo仿真环境配置
├── requirements.txt         # Python依赖
├── config/
│   └── arm_config.yaml      # 机械臂D-H参数与关节限位配置
├── core/
│   ├── base_arm.py          # 机械臂基类（BaseRoboticArm, BaseMuJoCoSim）
│   ├── kinematics.py        # 运动学解算（RoboticArmKinematics）
│   ├── arm_functions.py     # 机械臂功能函数（ArmFunctions）
│   └── arm_extensions.py    # 扩展功能（PID控制、轨迹管理）
├── model/
│   └── six_axis_arm.xml     # MuJoCo机械臂模型文件
├── params/
│   ├── default.json         # 默认参数
│   └── last.json            # 上次运行参数
├── trajectories/            # 轨迹文件存储目录
├── data/                    # 数据存储目录
└── logs/                    # 日志目录
```

## 技术栈
| 模块         | 技术/工具                          |
|--------------|------------------------------------|
| 核心算法     | Python (NumPy, SciPy)              |
| 运动学解算   | PyKDL / 自定义D-H解算库            |
| 仿真可视化   | MuJoCo / PyQt5 / matplotlib        |
| 硬件通信     | pyserial / socket                  |
| 依赖管理     | pip / requirements.txt             |

## 快速开始

### 环境要求
- Python 3.8+
- Windows/Linux/macOS
- MuJoCo 物理引擎
- 机械臂控制器（如支持Modbus/TCP的控制器，或仿真模式无需硬件）

### 安装依赖
```bash
# 克隆项目
git clone https://github.com/your-username/robotic-arm-project.git
cd robotic-arm-project

# 安装依赖包
pip install -r requirements.txt

# 安装 MuJoCo（如果需要）
pip install mujoco
```

### 运行程序
```bash
# 启动仿真界面
python main.py
```

### 3. MuJoCo 仿真控制
```python
import mujoco
from core.base_arm import BaseMuJoCoSim

# 创建仿真器（需要先加载模型和数据）
sim = BaseMuJoCoSim(model_path="model/six_axis_arm.xml")

# 获取当前关节角度
current_joints = sim.get_joint_angles()
print(f"当前关节角: {current_joints}")

# 设置目标关节角度
target_joints = [45, 30, -45, 60, 0, 0]
sim.set_joint_angles(target_joints)

# 碰撞检测
has_collision, pairs = sim.check_collision()
print(f"碰撞状态: {has_collision}")
```

### 4. 轨迹生成
```python
from core.arm_functions import ArmFunctions
from core.kinematics import RoboticArmKinematics
from core.base_arm import BaseMuJoCoSim

# 初始化
kin = RoboticArmKinematics()
sim = BaseMuJoCoSim(model_path="model/six_axis_arm.xml")
arm_func = ArmFunctions(kin)

# 生成直线轨迹
start_joints = [0, 0, 0, 0, 0, 0]
target_joints = [90, 45, -30, 60, 0, 0]
trajectory = arm_func.generate_linear_trajectory(start_joints, target_joints, num_points=50)

# 逐点执行
for joint_angles in trajectory:
    sim.set_joint_angles(joint_angles)
    mujoco.mj_step(sim.model, sim.data)
```

### 5. 配置自定义机械臂
修改 `config/arm_config.yaml` 文件：
```yaml
DH_PARAMS:
  joint1: {a: 0, alpha: 90, d: 100, theta: 0}
  joint2: {a: 200, alpha: 0, d: 0, theta: 0}
  # ... 添加更多关节

JOINT_LIMITS:
  joint1: [-180, 180]
  joint2: [-90, 90]
  # ... 设置关节限位
```

## API 说明

### core.kinematics.RoboticArmKinematics

| 方法 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `forward_kinematics(joint_angles)` | 正运动学求解 | 关节角度列表(度) | 包含位置和姿态的字典 |
| `inverse_kinematics(target_pose, ...)` | 逆运动学求解 | 目标位姿字典 | 关节角度列表 |

### core.base_arm.BaseRoboticArm

| 方法 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `forward_kinematics(joint_angles)` | 正运动学求解 | 关节角度列表(度) | 位姿字典 |
| `inverse_kinematics(target_pose, ...)` | 逆运动学求解 | 目标位姿字典 | 关节角度列表 |

### core.base_arm.BaseMuJoCoSim

| 方法 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `get_joint_angles()` | 获取当前关节角 | - | 关节角度列表(度) |
| `set_joint_angles(joint_angles)` | 设置关节角 | 关节角度列表(度) | - |
| `check_collision()` | 碰撞检测 | - | (是否碰撞, 碰撞对列表) |

### core.arm_functions.ArmFunctions

| 方法 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `generate_linear_trajectory(start, end, num_points)` | 生成直线轨迹 | 起点终点角度, 轨迹点数 | 轨迹点列表 |
| `manual_joint_control(current, key, step)` | 手动关节控制 | 当前角度, 按键, 步长 | 更新后的角度 |
| `follow_moving_target(current, target_pos)` | 追踪移动目标 | 当前角度, 目标位置 | 关节角度列表 |

### core.arm_extensions

| 类 | 说明 |
|------|------|
| `PIDController` | PID 控制器，用于关节位置控制 |
| `TrajectoryManager` | 轨迹管理器，保存/加载轨迹数据 |
| `TargetVisualizer` | 目标可视化器，在仿真中显示目标点 |

## D-H 参数说明

本项目采用标准 D-H 参数法建模：

| 参数 | 说明 | 单位 |
|------|------|------|
| a | 连杆长度 | mm |
| alpha | 连杆扭角 | 度 |
| d | 连杆偏移 | mm |
| theta | 关节转角 | 度 |
