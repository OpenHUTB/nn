# AirSim 无人机 LiDAR 风险监测与避障动作建议模块

本项目使用 AirSim 风格的无人机 LiDAR 与飞行状态日志，分析无人机在近障、姿态变化和低空下降过程中的安全风险，并根据最危险方向生成避障动作建议。样例数据位于 `sample_data/airsim_lidar_log.csv`。

## 主要内容

- 读取 AirSim 飞行日志中的高度、速度、姿态角和三向障碍物距离。
- 计算最小净空距离和综合安全风险分数。
- 标记 safe / warning / critical 三类风险状态。
- 根据前方、左侧、右侧最小净空距离生成 `brake_and_climb`、`shift_left`、`shift_right` 等动作建议。
- 生成 LiDAR 净空-风险曲线、高度-姿态风险散点图和避障动作分布图。
- 将运行效果图输出到 `docs/pr_assets/airsim_lidar_safety_monitor`。

## 运行

```bash
python src/airsim_lidar_safety_monitor/airsim_safety.py --output docs/pr_assets/airsim_lidar_safety_monitor
python src/airsim_lidar_safety_monitor/tests/test_airsim_safety.py
```
