无人车自主行驶与自动避让系统

本项目仅使用MuJoCo（Multi-Joint dynamics with Contact）物理仿真引擎与Python编程语言，构建高保真的无人车自主行驶与自动避让仿真系统。系统核心目标是在虚拟仿真场景中，实现无人车沿预设路径（或自由探索）行驶，并能实时检测周围障碍物，通过Python实现的决策算法完成安全避让动作。项目无需依赖真实硬件，可快速迭代验证避障算法、路径规划策略的有效性，适用于自动驾驶算法研发、强化学习训练、机器人运动控制等相关领域的学习与研究。

本系统具备场景可配置、障碍物类型多样、算法可替换等特性，为无人车自主导航相关算法的快速验证提供了高效的仿真平台。

环境要求
- 操作系统：Windows 10/11 64位，Ubuntu 20.04/22.04 64位

- MuJoCo版本：2.3.7 及以上（需获取官方许可证，学生可申请免费许可证）

- 编程语言：Python 3.8 - 3.11

核心依赖库：mujoco（MuJoCo仿真引擎Python绑定）、numpy（数值计算与数据处理）、matplotlib（数据可视化与结果绘图）、pyyaml（配置文件解析）、opencv-python（图像数据处理，用于传感器数据可视化）

环境搭建步骤
1. 安装Python 3.8 - 3.11，配置Python环境变量。

2. 获取MuJoCo许可证，下载对应版本的MuJoCo引擎，解压至指定目录（如Windows：C:\Users\用户名\.mujoco，Ubuntu：~/.mujoco）。

3. 设置MuJoCo环境变量：Windows系统需添加系统变量指定MuJoCo解压目录路径和许可证文件路径；Ubuntu系统需在终端执行相关命令配置环境变量，可写入.bashrc文件实现永久生效。

4. 安装依赖库：通过Python包管理工具安装所需核心依赖库，若需使用强化学习功能，额外安装对应的强化学习相关依赖库。

5. 验证环境：运行简单的Python命令导入mujoco库，无报错则说明环境搭建完成。

## 仿真指标统计与 CSV 报告

本项目新增了无界面运行和仿真指标导出功能，便于在 PyCharm 终端或 CI 环境中复现实验结果。

运行示例：

```bash
python qwer.py --headless --duration 8 --output-dir output --log-interval 1
```

参数说明：

- `--headless`：不打开 MuJoCo viewer，直接运行仿真并输出数据。
- `--duration`：仿真时长，单位为秒。
- `--output-dir`：CSV 报告输出目录，默认值为 `output`。
- `--log-interval`：终端状态打印间隔，单位为秒。

运行结束后会生成 `output/simulation_metrics.csv`，包含每一步的车辆位置、速度、油门、转向角、让行状态、急刹状态、避障状态、最近障碍距离和两车距离等指标。终端会同步打印最小车距、平均速度、急刹次数和距离目标点的剩余距离。

## 避障决策优化

车辆避障逻辑在原有道路边界斥力和让行规则基础上，新增动态车辆斥力：

- 当两车距离进入 `avoidance_distance` 范围时，根据距离和相对速度生成连续斥力。
- 使用 `avoidance_active` 标记当前车辆是否处于主动避障状态。
- 使用 `min_obstacle_distance` 记录最近动态障碍距离，方便在 CSV 报告中分析避障过程。
- 急刹阈值拆分为独立的 `emergency_distance`，避免和普通避障距离混用。
