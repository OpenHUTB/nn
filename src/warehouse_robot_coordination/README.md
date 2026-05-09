# 多机器人仓储协同调度与避碰演示

本模块模拟仓储环境中的多机器人任务分配、路径规划与时空避碰调度。项目不依赖大型仿真平台，可以直接运行并生成图片、图表、GIF 和指标文件，适合作为机器人导航与智能物流方向的课程项目。

## 项目特点

- 构建带货架、通道、取货点和投放点的二维仓库地图。
- 支持 4 台仓储机器人和 6 个 pickup-delivery 任务。
- 使用 A* 为机器人到取货点、投放点规划路径。
- 使用优先级感知的贪心策略进行任务分配。
- 检测多机器人路径中的顶点冲突和边交换冲突。
- 通过插入等待动作进行时空避碰调度。
- 对比多机器人方案与单机器人基线方案。
- 输出仓库布局图、路线图、负载均衡图、冲突消除图、时间线图和 GIF 动图。

## 文件结构

```text
src/warehouse_robot_coordination/
├── main.py              # 命令行入口
├── warehouse.py         # 仓库地图、机器人、任务定义
├── planner.py           # A* 路径规划、任务分配、单机器人基线
├── scheduler.py         # 时空冲突检测与等待避碰
├── visualization.py     # 图片、图表和 GIF 生成
├── requirements.txt     # 依赖说明
├── README.md            # 项目说明
├── assets/              # 运行结果
└── tests/test_coordination.py
```

## 运行方法

```bash
cd src/warehouse_robot_coordination
python main.py
```

运行后会在 `assets/` 中生成：

```text
warehouse_task_map.png       # 仓库、机器人、任务分布图
assigned_routes.png          # 多机器人任务路线图
workload_balance.png         # 多机器人负载均衡与单机器人基线对比
conflict_reduction.png       # 调度前后冲突数量对比
robot_timeline.png           # 多机器人执行时间线
warehouse_coordination.gif   # 多机器人协同执行动图
metrics.json                 # 汇总指标
robot_loads.csv              # 各机器人任务步数
```

## 核心流程

1. 生成仓库地图、机器人和任务。
2. 按任务优先级排序。
3. 对每个任务，计算每台机器人从当前位置完成该任务的路径代价。
4. 使用优先级感知的贪心策略把任务分配给代价较低、负载较轻的机器人。
5. 合成每台机器人的完整路径。
6. 在时间维度上检测机器人之间的顶点冲突和边交换冲突。
7. 对冲突机器人插入等待动作，直到冲突消除或达到迭代上限。
8. 输出指标和可视化结果。

## 运行结果示例

```text
Warehouse robot coordination demo finished
Robots: 4  Tasks: 6
Conflicts before/after: 1 -> 0
```

## 创新点

相比单机器人路径规划项目，本模块强调多机器人协同：任务分配、负载均衡、路径冲突检测、时间维度避碰和执行时间线分析。它更贴近智能仓储、物流机器人和多智能体协同调度场景。

## 后续可扩展方向

- 加入更复杂的任务收益函数和截止时间约束。
- 将等待避碰升级为 CBS（Conflict-Based Search）等多智能体路径规划算法。
- 加入机器人电量消耗和充电站调度。
- 接入 ROS、Gazebo、Webots 或真实 AGV 调度系统。
