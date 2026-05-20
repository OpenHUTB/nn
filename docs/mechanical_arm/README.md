# 基于 DH 参数的四自由度机械臂可视化与控制系统

## 项目概述

本项目实现了一个基于 DH（Denavit-Hartenberg）参数的四自由度机械臂可视化与交互控制系统。系统采用正向运动学计算各关节空间位置，配备三指夹爪执行器，支持 3D 实时渲染、多视图投影、交互式滑块控制和轨迹动画演示。

在工程化改进过程中，针对原始代码中存在的**关节角度无验证导致越界异常**和**动画不可中断导致窗口关闭崩溃**两个核心问题，分别引入了角度范围裁剪机制和中断安全动画框架，使系统从"能演示"提升为"可交互、不崩溃、可中断"的稳定状态。

## 项目目标

- **正向运动学建模**：基于 DH 参数实现四自由度机械臂的位姿计算，输出各关节和末端执行器的三维坐标。
- **三指夹爪可视化**：在末端执行器上附加三指夹爪模型，支持开口大小和旋转角度的独立控制。
- **多视图实时渲染**：同时展示 3D 立体视图、XY 俯视图和 XZ 正视图，提供全方位空间感知。
- **交互式控制**：通过 matplotlib 滑块实时调整关节角度和夹爪参数，通过按钮触发抓取演示和轨迹动画。
- **运行时健壮性**：添加关节角度范围验证和动画中断安全机制，确保任意操作下系统不崩溃。

## 一、核心算法原理（DH 参数法）

### 1.1 DH 参数定义

Denavit-Hartenberg（DH）参数法是机器人学中描述串联连杆机构运动学的标准方法。每个关节用四个参数描述相邻连杆之间的空间关系：

| 参数 | 符号 | 含义 |
|------|------|------|
| 连杆长度 | $a_i$ | 沿 $x_i$ 轴，从 $z_{i-1}$ 到 $z_i$ 的距离 |
| 连杆扭角 | $\alpha_i$ | 绕 $x_i$ 轴，从 $z_{i-1}$ 到 $z_i$ 的旋转角 |
| 连杆偏移 | $d_i$ | 沿 $z_{i-1}$ 轴，从 $x_{i-1}$ 到 $x_i$ 的距离 |
| 关节角度 | $\theta_i$ | 绕 $z_{i-1}$ 轴，从 $x_{i-1}$ 到 $x_i$ 的旋转角 |

### 1.2 DH 变换矩阵

每个关节的齐次变换矩阵为：

$$
T_i = \begin{bmatrix} \cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\ \sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\ 0 & \sin\alpha_i & \cos\alpha_i & d_i \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

### 1.3 本项目 DH 参数配置

本项目四自由度机械臂的 DH 参数：

| 关节 | $a$（连杆长度） | $\alpha$（扭角） | $d$（偏移） | $\theta$（关节角） | 物理含义 |
|------|---------|---------|---------|------------|---------|
| 1 | 2.0 | 0 | 0 | $\theta_1$ | 底座旋转 |
| 2 | 1.5 | $\pi/2$ | 0 | $\theta_2$ | 肩关节俯仰 |
| 3 | 1.0 | 0 | 0 | $\theta_3$ | 肘关节弯曲 |
| 4 | 0.5 | 0 | 0 | $\theta_4$ | 腕关节旋转 |

末端执行器位置通过链式矩阵乘法得到：$T_{total} = T_1 \cdot T_2 \cdot T_3 \cdot T_4$，取 $T_{total}$ 的平移分量即为末端三维坐标。

## 二、主要改进与技术突破

### 2.1 改进前的核心问题

原始代码（`test_six.py`）中存在两个严重的工程缺陷：

**问题一：动画不可中断导致崩溃**

```python
# test_six.py 中的原始动画代码（改进前）
def grasp_demo(self, event):
    for i in range(20):
        self.gripper_opening = original_opening * (1 - i / 20)
        self.update_plot()
        plt.pause(0.05)  # 用户关闭窗口后直接抛出异常

def animate_movement(self, event):
    for target_angles in path:
        for i in range(20):
            # ... 插值计算 ...
            self.update_plot()
            plt.pause(0.03)  # 无法中断，窗口关闭后报错
```

用户在动画播放期间关闭 matplotlib 窗口，`plt.pause()` 抛出异常，程序崩溃。

**问题二：关节角度无验证**

```python
# test_six.py 中直接使用滑块值，无范围校验
def update_from_slider(self, val):
    self.joint_angles = [s.val for s in self.sliders]  # 可能越界
    self.update_dh_params()
    self.update_plot()

# 动画路径中的角度也未验证
path = [
    [0.5, 0.3, -0.2, 0.1],    # 硬编码，无验证
    [-0.3, 0.6, -0.4, 0.2],
    [0.8, -0.2, 0.5, -0.3],   # Joint 2: 0.8 rad > π/2 上限
]
```

虽然滑块本身有范围限制，但动画路径中的角度值未经验证，且缺少显式的防御性校验，存在潜在越界风险。

### 2.2 改进一：关节角度输入验证

**改进文件**：`src/mechanical_arm/main.py`

在 `RoboticArmWithGripper` 类中新增 `JOINT_LIMITS` 常量、`_clamp()` 工具方法和 `validate_angles()` 验证函数：

```python
class RoboticArmWithGripper:
    # 关节角度范围限制（弧度）
    JOINT_LIMITS = [
        (-np.pi, np.pi),          # Joint 1: 底座旋转
        (-np.pi / 2, np.pi / 2),  # Joint 2: 肩关节
        (-np.pi / 2, np.pi / 2),  # Joint 3: 肘关节
        (-np.pi, np.pi),          # Joint 4: 腕关节
    ]

    @staticmethod
    def _clamp(value, lower, upper):
        """将值限制在指定范围内"""
        return max(lower, min(upper, value))

    def validate_angles(self, angles):
        """验证并修正关节角度，确保在合法范围内"""
        if len(angles) != len(self.JOINT_LIMITS):
            raise ValueError(
                f"关节角度数量错误：期望 {len(self.JOINT_LIMITS)} 个，实际 {len(angles)} 个"
            )
        return [
            self._clamp(a, lo, hi)
            for a, (lo, hi) in zip(angles, self.JOINT_LIMITS)
        ]
```

所有角度更新路径均通过 `validate_angles()` 校验：

```python
# 滑块回调：验证后更新
def update_from_slider(self, val):
    self.joint_angles = self.validate_angles([s.val for s in self.sliders])
    self.update_dh_params()
    self.update_plot()

# 动画路径：每条路径都验证
def animate_movement(self, event):
    raw_path = [[0.5, 0.3, -0.2, 0.1], [-0.3, 0.6, -0.4, 0.2], ...]
    path = [self.validate_angles(p) for p in raw_path]
```

夹爪参数同样添加了范围裁剪：

```python
def update_gripper(self, val):
    self.gripper_opening = self._clamp(
        self.slider_gripper_open.val,
        self.GRIPPER_OPENING_RANGE[0], self.GRIPPER_OPENING_RANGE[1])
    self.gripper_angle = self._clamp(
        self.slider_gripper_rotate.val,
        self.GRIPPER_ANGLE_RANGE[0], self.GRIPPER_ANGLE_RANGE[1])
```

### 2.3 改进二：动画中断安全机制

引入 `_animating` 标志位和 `_safe_pause()` 中断安全暂停方法，替代原始代码中不可中断的 `plt.pause()` 调用：

```python
# 新增：中断安全暂停
def _safe_pause(self, seconds):
    """带中断检查的暂停，窗口关闭时提前返回"""
    if not self._animating:
        return False
    try:
        plt.pause(seconds)
        return True
    except Exception:
        self._animating = False
        return False
```

所有动画函数（`grasp_demo`、`animate_movement`）在每个关键步骤检查中断标志：

```python
def animate_movement(self, event):
    self._animating = True
    try:
        for target_angles in path:
            if not self._animating:     # 中断检查
                break
            for i in range(20):
                if not self._animating:  # 中断检查
                    break
                # ... 插值计算 ...
                self.update_plot()
                if not self._safe_pause(0.03):  # 安全暂停
                    break
            if not self._safe_pause(0.5):
                break
    finally:
        self._animating = False  # 确保状态复位
```

`grasp_demo()` 函数同样添加了中断检查：

```python
def grasp_demo(self, event):
    self._animating = True
    # 闭合夹爪
    for i in range(20):
        if not self._animating:
            break
        self.gripper_opening = original_opening * (1 - i / 20)
        self.slider_gripper_open.set_val(self.gripper_opening)
        self.update_plot()
        if not self._safe_pause(0.05):
            break
```

## 三、改进前后代码对比

| 对比项 | 改进前（test_six.py） | 改进后（main.py） |
|--------|----------------------|-------------------|
| 关节角度限制 | 无常量定义 | `JOINT_LIMITS` 四关节范围常量 |
| 滑块角度校验 | 直接使用 `s.val` | `validate_angles()` 裁剪 |
| 动画路径校验 | 硬编码无验证 | `self.validate_angles(p)` 逐路径验证 |
| 夹爪参数校验 | 直接使用滑块值 | `_clamp()` 范围裁剪 |
| 动画暂停方式 | `plt.pause()` 不可中断 | `_safe_pause()` 支持中断 |
| 动画状态管理 | 无状态标志 | `_animating` 标志位 |
| 窗口关闭行为 | 抛出异常崩溃 | 捕获异常，安全退出 |
| 动画函数结构 | 无 try/finally | try/finally 确保状态复位 |

## 四、系统技术架构

### 4.1 整体架构

本项目采用"运动学计算层 - 可视化渲染层 - 交互控制层"三层架构：

- **运动学计算层**：`dh_matrix()` 计算单关节变换矩阵，`forward_kinematics()` 链式相乘得到各关节空间位置，`calculate_gripper_positions()` 计算夹爪手指端点。
- **可视化渲染层**：`update_plot()` 同时更新 3D 主视图、XY 俯视图、XZ 正视图，`draw_gripper_3d()` 和 `draw_gripper_2d()` 分别在 3D 和 2D 视图中渲染夹爪。
- **交互控制层**：4 个关节角度滑块 + 2 个夹爪参数滑块 + 3 个功能按钮（Reset / Grasp Demo / Animate），通过回调函数驱动运动学计算和视图更新。

### 4.2 核心模块职责

| 方法 | 输入 | 输出 | 功能 |
|------|------|------|------|
| `dh_matrix()` | $a, \alpha, d, \theta$ | 4x4 齐次变换矩阵 | 计算单关节 DH 变换 |
| `forward_kinematics()` | 关节角度 | 关节位置列表 + 末端变换矩阵 | 链式计算各关节空间位置 |
| `calculate_gripper_positions()` | 末端变换矩阵 + 夹爪参数 | 三个手指的基座/尖端坐标 | 计算夹爪几何 |
| `validate_angles()` | 角度列表 | 裁剪后的角度列表 | 角度范围校验 |
| `_safe_pause()` | 延时秒数 | 是否继续 | 中断安全暂停 |
| `update_plot()` | 当前关节角度 | 渲染画面 | 更新三个视图 |

## 五、技术栈与快速开始

### 5.1 核心技术栈

| 技术类别 | 具体选型 | 选型理由 |
|---------|---------|---------|
| 编程语言 | Python 3.8+ | 科学计算生态丰富 |
| 数值计算 | NumPy | 矩阵运算高效，适合 DH 变换计算 |
| 可视化 | Matplotlib 3.x | 支持 3D 渲染和交互控件（Slider/Button） |
| 运动学建模 | DH 参数法 | 串联连杆运动学标准方法 |

### 5.2 快速开始

```bash
# 安装依赖
pip install numpy matplotlib

# 运行主程序（改进后版本）
cd src/mechanical_arm
python main.py

# 运行对比版本（改进前版本，观察差异）
python test_six.py
```

### 5.3 操作指南

运行 `main.py` 后将显示交互式窗口：

- **滑块控制**：拖动 Joint 1-4 滑块调整关节角度，拖动 Gripper Opening/Rotation 调整夹爪
- **Reset All 按钮**：将所有参数恢复初始值
- **Grasp Demo 按钮**：自动演示夹爪闭合→打开动画
- **Animate 按钮**：沿预设路径执行机械臂轨迹动画，**播放期间可随时关闭窗口安全退出**

## 六、项目结构

```plaintext
mechanical_arm/
├── main.py                    # 主程序（改进后，含角度验证+中断安全动画）
├── test_five.py               # 基础版（无夹爪，FuncAnimation 轨迹演示）
├── test_six.py                # 改进前版本（有夹爪但无验证、无中断处理）
├── full_gripper_demo.py       # 完整三指夹爪演示
├── arm_test.py                # 基本测试
├── model_test.py              # 模型测试
├── motion_test.py             # 运动测试
├── new_arm_test.py            # 新臂测试
├── three_arm_test.py          # 三指臂测试
├── three_arm_test_two.py      # 三指臂测试（二）
├── three_arm_test_three.py    # 三指臂测试（三）
├── 6dof_arm.xml               # 六自由度 MuJoCo 模型
├── new_arm.xml                # 新臂 MuJoCo 模型
├── three_fingered_arm.xml     # 三指臂 MuJoCo 模型
├── three_fingered_arm_two.xml # 三指臂 MuJoCo 模型（二）
└── README.md
```

## 七、现存不足与后续规划

### 7.1 现存不足

- **无逆运动学求解**：当前仅支持正向运动学（给定角度算位置），不支持给定目标位置反算关节角度，轨迹动画中的"逆运动学"是简化的硬编码值。
- **无碰撞检测**：机械臂运动时不检测关节与连杆之间的自碰撞，极端角度下可能出现穿越。
- **无物理仿真**：纯几何运动学计算，不考虑重力、惯性和关节力矩。
- **2D 投影无深度信息**：俯视图和正视图的夹爪投影可能出现重叠遮挡。

### 7.2 后续优化方向

1. **逆运动学求解**：实现数值迭代法（如牛顿-拉夫逊法）或解析法，支持点击目标位置自动规划关节角度。
2. **碰撞检测**：引入轴对齐包围盒（AABB）或球体近似，检测连杆间的自碰撞。
3. **轨迹规划**：实现关节空间的梯形速度规划或笛卡尔空间的直线插补，替代当前的简单线性插值。
4. **MuJoCo 物理仿真集成**：利用目录中已有的 XML 模型文件，接入 MuJoCo 进行动力学仿真。

## 八、总结

本项目实现了一个基于 DH 参数的四自由度机械臂可视化系统，核心改进包括：(1) 新增 `validate_angles()` + `_clamp()` 机制对所有角度输入路径进行范围校验，防止越界导致的运动学计算异常；(2) 引入 `_animating` 标志 + `_safe_pause()` 中断安全暂停方法替代裸 `plt.pause()`，配合 try/finally 结构确保动画可安全中断、窗口可正常关闭。改进后的系统从"演示型"提升为"可交互型"，用户可随时中断动画、自由调整参数，系统在任意操作下均保持稳定。
