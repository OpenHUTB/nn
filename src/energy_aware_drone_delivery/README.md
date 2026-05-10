# 能量感知无人机配送路径规划

本模块模拟城市低空配送场景中的无人机任务规划，重点考虑电池容量、安全余量、载重能耗、风场影响和中途充电站选择。项目可直接运行并生成路线图、电量曲线和能耗分解图，适合用于无人机配送、低空物流和智能路径规划方向的课程作业。

## 项目特点

- 构建带禁飞区、配送点、充电站和风场的二维城市地图。
- 定义多个 pickup-delivery 配送任务，并设置载重和优先级。
- 使用能量代价版 A* 搜索规划路径。
- 能耗模型同时考虑基础飞行代价、载重代价和逆风惩罚。
- 根据安全电量余量自动选择充电站补能。
- 与不考虑充电和任务重排的朴素方案进行电量曲线对比。
- 输出路线图、电池曲线、能耗分解图、JSON 指标和 CSV 电量轨迹。

## 文件结构

```text
src/energy_aware_drone_delivery/
├── main.py                     # 命令行入口
├── scenario.py                 # 城市地图、风场、任务和充电站定义
├── planner.py                  # 能量感知 A*、任务排序和充电决策
├── visualization.py            # 可视化图表生成
├── requirements.txt            # 依赖说明
├── README.md                   # 项目说明
├── assets/                     # 运行结果
└── tests/test_delivery.py      # 基础测试
```

## 运行方法

```bash
cd src/energy_aware_drone_delivery
python main.py
```

运行后会在 `assets/` 目录生成：

```text
energy_aware_route.png     # 能量感知配送路线图
battery_profile.png        # 电池余量曲线图
energy_breakdown.png       # 各任务段能耗分解图
metrics.json               # 汇总指标
battery_trace.csv          # 电量轨迹数据
```

## 核心思路

普通路径规划通常只考虑距离最短，但无人机配送还需要考虑电池约束。载重越大，能耗越高；逆风飞行时，能耗也会升高。如果只按距离规划，无人机可能在任务中途低于安全电量余量。

本模块将路径规划代价从“距离”改为“能量”，并在任务执行过程中持续检查剩余电量。如果完成下一个任务会低于安全余量，系统会先规划到最近可达充电站，补满电后继续配送。

## 运行结果示例

```text
Energy-aware drone delivery demo finished
Tasks: 4  Charging stations: 4
Feasible: True
Charges: 1
```

## 可扩展方向

- 加入多架无人机协同配送。
- 加入动态天气和实时风场变化。
- 加入配送时间窗约束。
- 接入 AirSim、PX4 或 ROS2 无人机仿真。
- 将 A* 替换为 D* Lite 或强化学习策略。
