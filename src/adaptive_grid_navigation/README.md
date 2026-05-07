# 自适应栅格导航与动态重规划演示

本模块演示二维栅格地图中的机器人路径规划与动态障碍物重规划流程，适合用于无人车、无人机、移动机器人在仿真环境中的路径规划入门实验。

## 项目功能

- 构建二维占据栅格地图，支持固定障碍物、起点和终点。
- 实现三种路径规划算法：
  - Dijkstra
  - A*
  - Greedy best-first search
- 对比不同规划器的路径长度和搜索节点数量。
- 模拟动态障碍物沿固定轨迹移动。
- 当障碍物阻挡当前路径时，机器人自动触发 A* 重规划。
- 输出路线图片、算法对比图、动态重规划 GIF 和指标文件。

## 文件结构

```text
src/adaptive_grid_navigation/
├── main.py                  # 命令行入口
├── grid_map.py              # 栅格地图、地图生成和基础工具
├── planners.py              # Dijkstra、A*、贪心搜索规划器
├── simulator.py             # 动态障碍物和在线重规划仿真
├── visualization.py         # 图片、图表和 GIF 生成
├── requirements.txt         # 模块依赖
├── assets/                  # 运行后生成的图片、动图和指标
└── tests/                   # 基础测试
```

## 环境依赖

```bash
pip install -r requirements.txt
```

主要依赖：

- numpy
- matplotlib
- Pillow

## 运行方法

在项目根目录运行：

```bash
cd src/adaptive_grid_navigation
python main.py
```

也可以指定输出目录和地图类型：

```bash
python main.py --map demo --output assets --max-steps 120
python main.py --map random --seed 10 --output assets_random
```

## 输出结果

运行后会在 `assets/` 目录生成：

```text
astar_route.png              # A* 路线图
planner_comparison.png       # 不同规划算法对比图
dynamic_replanning.gif       # 动态障碍物重规划演示动图
metrics.json                 # 汇总指标
planner_metrics.csv          # 规划器指标表
```

## 算法流程

1. 生成二维占据栅格地图。
2. 使用 Dijkstra、A* 和 Greedy best-first search 分别规划路径。
3. 记录每个算法的路径长度和扩展节点数量。
4. 在同一地图上加入动态障碍物。
5. 机器人沿 A* 路径移动。
6. 如果动态障碍物阻挡下一步路径，则从当前位置重新规划。
7. 输出图片、动图和评价指标。

## 改进意义

本模块相比简单的静态路径规划示例，多加入了动态障碍物和在线重规划机制，更接近真实机器人导航任务中的情况。它既可以作为路径规划算法教学示例，也可以作为后续迁移到 ROS、Gazebo、Webots 或 Carla 仿真环境的基础模块。
