# 演讲提纲：自适应栅格导航项目

## 开场

老师好，我这次汇报的是我在 `nn` 项目中新增加的 `adaptive_grid_navigation` 模块。这个模块模拟二维栅格环境中的机器人路径规划，并加入动态障碍物和在线重规划功能。

## 为什么做这个项目

`nn` 仓库中有很多无人车、无人机和机器人导航相关内容。为了让我的修改更像一个完整工程，而不是只改几行代码，我选择做一个可以独立运行、可以生成图片和动图、可以用于课堂演示的导航算法模块。

## 项目主要内容

本项目包括五个核心部分：

1. 栅格地图建模。
2. Dijkstra、A* 和贪心搜索三种路径规划算法。
3. 动态障碍物模拟。
4. 机器人在线重规划。
5. 路线图片、算法对比图和 GIF 动图输出。

## 代码结构

我把项目放在：

```text
src/adaptive_grid_navigation/
```

其中：

- `grid_map.py` 负责地图。
- `planners.py` 负责路径规划算法。
- `simulator.py` 负责动态障碍物和重规划。
- `visualization.py` 负责生成图片和 GIF。
- `main.py` 是运行入口。
- `tests/test_navigation.py` 是测试文件。

## 运行结果

运行命令是：

```bash
cd src/adaptive_grid_navigation
python main.py
```

运行后生成三类主要成果：

- A* 路线图。
- 三种算法对比图。
- 动态重规划 GIF。

本次运行结果显示：

```text
Dijkstra: path_length=54, expanded_nodes=899
A*: path_length=54, expanded_nodes=648
Greedy best-first: path_length=56, expanded_nodes=58
Dynamic replanning: reached_goal=True replans=2 travelled=54 frames=55
```

## 结果分析

Dijkstra 和 A* 都找到了长度为 54 的路径，但 A* 扩展节点更少，说明启发式搜索提高了效率。

贪心搜索扩展节点最少，但路径长度变成 56，说明它速度快，但不保证最优。

动态重规划实验中，机器人最终成功到达目标点，并进行了 2 次重规划，说明模块可以模拟动态环境下的路径调整。

## 项目价值

这个项目的价值在于：

- 它是一个完整的工程模块。
- 它和机器人导航、无人车路径规划主题相关。
- 它可以直接生成图片和动图，方便课堂展示。
- 后续可以迁移到 ROS、Gazebo、Webots 或 Carla 中。

## 结尾

我的汇报到这里结束。后续我计划继续加入路径平滑、速度规划和更真实的动态障碍物，使它进一步接近真实机器人导航系统。
