# 生态驾驶绿波速度建议模块

本模块模拟智能车辆在多信号灯城市走廊中行驶，通过绿波速度建议减少红灯停车和不必要的加速，从而降低能耗。项目对比固定巡航策略与生态速度建议策略，并输出速度曲线、时空图、指标对比图和 CSV 指标数据。

## 项目特点

- 构建包含 4 个信号灯的城市道路走廊。
- 每个信号灯具有周期、绿灯时长和相位偏移。
- 规划车辆通过每个信号灯的目标到达时间。
- 在合法速度范围内生成平滑速度建议。
- 与固定巡航基线进行对比。
- 统计旅行时间、停车次数、绿灯通过次数和能耗指标。
- 输出速度-距离图、时空信号灯图和综合指标对比图。

## 文件结构

```text
src/eco_driving_speed_advisor/
├── main.py                  # 命令行入口
├── scenario.py              # 道路、车辆、信号灯场景定义
├── advisor.py               # 绿波速度建议与固定巡航基线
├── visualization.py         # 可视化图表生成
├── requirements.txt         # 依赖说明
├── README.md                # 项目说明
├── assets/                  # 运行结果
└── tests/test_advisor.py    # 基础测试
```

## 运行方法

```bash
cd src/eco_driving_speed_advisor
python main.py
```

运行后会在 `assets/` 目录生成：

```text
speed_profile.png        # 速度-距离曲线
time_space_diagram.png   # 时空图与绿灯窗口
eco_metrics.png          # 生态驾驶与固定巡航指标对比
metrics.json             # 汇总指标
speed_profile.csv        # 速度轨迹数据
```

## 核心思路

固定巡航车辆以固定速度行驶，遇到红灯只能停车等待。生态驾驶策略提前利用信号灯相位信息，调整车辆速度，使车辆尽量在绿灯窗口内通过路口，减少停车和急加速。

本项目先根据每个信号灯位置、周期、绿灯时长和相位偏移，计算车辆可行的绿灯到达时间；再基于这些目标时间生成速度曲线。能耗指标使用轻量级牵引能耗代理模型，综合考虑速度、正加速度和舒适性相关的 jerk 惩罚。

## 运行结果示例

```text
Eco-driving speed advisor demo finished
Signals: 4
Energy saving: ...
Stops eco/baseline: 0 / ...
Green passes eco/baseline: 4 / ...
```

## 可扩展方向

- 接入真实 SPaT 信号灯相位数据。
- 加入前车跟驰约束。
- 加入坡度、限速和车辆动力学模型。
- 与 Carla 或 SUMO 联合仿真。
- 使用动态规划或模型预测控制生成更平滑的速度曲线。
