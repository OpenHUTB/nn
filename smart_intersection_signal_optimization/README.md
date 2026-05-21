# 城市交叉口信号配时智能优化

本项目面向城市道路交叉口拥堵治理，构建四相位交通信号仿真环境，并使用精英保留遗传算法优化信号周期和绿信比分配。项目可以直接运行，自动输出指标文件、对比图、收敛曲线和 GIF 动图，适合作为智能交通与神经网络课程中的优化实践项目。

## 项目特点

- 构建四进口交叉口，包含直行与左转相位、交通需求、饱和流率和优先级。
- 使用 Webster 固定配时作为基线方案。
- 设计队列演化仿真器，统计平均延误、最大排队、停车次数、通行率和公平性。
- 使用精英保留遗传算法搜索周期长度和绿信比分配。
- 输出优化前后指标对比、绿信比对比、队列变化曲线、遗传算法收敛图和动态 GIF。
- 附带单元测试，验证配时归一化、仿真输出和优化效果。
- 提供 `index.html` 汇报网页，可以发布到 GitHub Pages。

## 文件结构

```text
src/smart_intersection_signal_optimization/
├── main.py                  # 命令行入口
├── scenario.py              # 交叉口场景、相位、Webster 基线
├── simulator.py             # 队列演化仿真与指标计算
├── optimizer.py             # 遗传算法信号配时优化
├── visualization.py         # 图片、图表和 GIF 生成
├── index.html               # GitHub Pages 汇报网页
├── style.css                # 汇报页样式
├── script.js                # 汇报页交互动画
├── requirements.txt         # 依赖说明
├── README.md                # 项目说明
├── assets/                  # 运行结果图片、动图和数据
└── tests/test_signal_optimization.py
```

## 运行方法

```bash
cd src/smart_intersection_signal_optimization
python main.py
```

运行后会在 `assets/` 中生成：

```text
optimized_queue_profiles.png          # 优化后各进口排队曲线
green_split_comparison.png            # 基线与优化绿信比分配对比
metric_comparison.png                 # 延误、排队、停车、综合评分对比
ga_convergence.png                    # 遗传算法收敛曲线
intersection_signal_optimization.gif  # 信号控制与排队动态演示
metrics.json                          # 汇总指标
optimized_greens.csv                  # 优化绿灯时间表
convergence.csv                       # 每代优化过程
```

## 核心流程

1. 构建高峰期四相位交叉口交通需求。
2. 根据 Webster 方法计算固定配时基线。
3. 用队列演化模型模拟每一秒车辆到达、放行和排队。
4. 将周期长度、各相位绿信比编码为遗传算法个体。
5. 按平均延误、最大排队、通行率和停车次数构建综合目标函数。
6. 通过选择、交叉、变异和精英保留迭代优化。
7. 输出优化方案、指标对比、图表和 GIF 动图。

## 运行结果示例

```text
Smart intersection signal optimization finished
Algorithm: elitist genetic algorithm
Delay reduction: 10%+
Score reduction: 10%+
```

## 创新点

相比普通固定信号灯演示，本项目把交通工程中的 Webster 配时、队列仿真和智能优化算法结合起来。它不仅给出一个更优信号方案，还展示了优化过程、指标改善和动态运行状态，工程量覆盖建模、算法、仿真、可视化、网页汇报和测试。

## 后续可扩展方向

- 将遗传算法替换为 DQN、PPO 或多智能体强化学习。
- 加入真实路口检测器数据或 SUMO/CARLA 仿真数据。
- 增加行人相位、公交优先、应急车辆优先和协调绿波控制。
- 将单路口扩展为多路口区域信号协同优化。
