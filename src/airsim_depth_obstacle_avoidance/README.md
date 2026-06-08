# AirSim 深度相机避障决策模块

本项目使用 AirSim `DepthPerspective` 前向深度相机条带数据，分析无人机前方、左侧和右侧障碍距离，并生成避障转向建议。

## 主要内容

- 使用 AirSim DepthPerspective 深度数据 `airsim_depth_strips.csv`。
- 读取左/中/右三个方向深度、飞行高度和前向速度。
- 计算最小深度和碰撞风险。
- 输出 keep_course / slow_down / turn_left / turn_right 避障动作。
- 生成 AirSim 深度运行视图和避障动作曲线图。

## 运行

```bash
python src/airsim_depth_obstacle_avoidance/depth_avoidance.py --output docs/pr_assets/airsim_depth_obstacle_avoidance
python src/airsim_depth_obstacle_avoidance/tests/test_depth_avoidance.py
```
