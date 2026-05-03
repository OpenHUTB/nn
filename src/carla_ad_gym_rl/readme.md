本项目旨在为 CARLA 模拟器提供了一个轻便易用的兼容 Gym 的界面，专为强化学习（RL）应用量身定制。它集成了激光雷达扫描、车辆状态、附近车辆信息和航点等关键观察组件。环境支持安全意识学习，包括奖励和成本信号、路标可视化以及可定制参数，包括交通设置、车辆数量和传感器范围。

安装所需依赖：

```bash
pip install -r requirements.txt
```

将本项目作为本地 Python 包安装：

```bash
pip install -e .
```

## 快速开始

运行一个简单的演示，与环境进行交互：

```bash
python easycarla_demo.py
```

该脚本演示了如何：
- 创建并重置 CARLA Gym 环境
- 使用 `autopilot`、`random`、`safe_random` 和 `manual` 等不同控制模式与环境交互
- 在环境中逐步执行动作，并获取观测、奖励、代价以及结束信号
- 保存每个 episode 的交互数据和运行统计结果
- 将观测数据转换为一维状态向量，便于后续强化学习训练或数据分析
- 在 CARLA 仿真画面中显示车辆速度、奖励、代价、碰撞和越界等运行状态

其中，`manual` 模式基于 pygame 实现。使用该模式时，需要先在代码中设置：

```python
CONTROL_MODE = "manual"S

在运行该演示之前，请确保你的 CARLA 服务端已经启动。