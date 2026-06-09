# 快速开始

## 环境要求

| 依赖 | 最低版本 | 说明 |
|:---|:---|:---|
| Python | 3.8+ | 建议使用 3.9 或更高版本 |
| MuJoCo | 3.0+ | 物理引擎核心 |
| NumPy | 1.20+ | 数值计算 |

## 安装依赖

### 方式一：使用 requirements.txt

```bash
pip install -r requirements.txt
```

### 方式二：手动安装

```bash
pip install mujoco numpy
```

## 基本运行

进入项目目录后，运行以下命令启动仿真：

```bash
cd src/mujoco01
python main.py
```

或使用模块方式运行：

```bash
python -m mujoco01.main
```

启动后，将看到 MuJoCo 渲染窗口，机器人处于平躺姿态，右肘关节周期性执行抬手动作。

## 交互操作

| 操作 | 效果 |
|:---|:---|
| **鼠标左键拖动** | 旋转视角 |
| **鼠标右键拖动** | 平移视角 |
| **滚轮** | 缩放 |
| **Shift + 鼠标左键拖动** | 精细调整视角 |

### 切换预设相机

在代码中可通过修改相机名称切换视角：

- `back` — 后视相机（默认）
- `side` — 侧视相机
- `egocentric` — 第一人称视角

## 命令行参数

当前版本暂不支持命令行参数。如需调整仿真行为，请直接修改 `main.py` 中的配置常量：

- `kp` / `kd` — PD 控制增益
- `elbow_lift_max` — 肘关节最大力矩
- `cycle_period` — 抬手周期
- `smooth_alpha` — 平滑系数
- `max_torque` — 力矩限幅

详见[配置说明](configuration.md)。

## 常见问题

### 模型加载失败

如果出现 `模型加载失败` 错误，请检查：

1. `humanoid.xml` 文件是否存在于 `src/mujoco01/` 目录下
2. 当前工作目录是否正确（需要在 `src/mujoco01/` 下运行）
3. MuJoCo 版本是否 >= 3.0

### 窗口一闪而过

这是正常现象——仿真在打开窗口后立即进入主循环，机器人保持平躺姿态。如果窗口自动关闭，说明程序发生了异常，请查看终端输出的错误信息。

### 如何停止仿真

直接关闭 MuJoCo 渲染窗口即可。程序会自动保存数据到 `simulation_data.csv`。
