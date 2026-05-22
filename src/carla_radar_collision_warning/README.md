# CARLA Radar 前向碰撞预警模块

本项目使用 CARLA RadarDetection 传感器风格数据，分析前方目标距离、方位角和径向速度，计算 TTC 碰撞风险并给出制动建议。

## 主要内容

- 使用 CARLA radar detection 数据 `carla_radar_detections.csv`。
- 读取 depth、azimuth、altitude、radial_velocity 和目标类型。
- 计算 closing speed、TTC 和综合风险分数。
- 输出 monitor / soft_brake / emergency_brake 三类制动建议。
- 生成 TTC 曲线和雷达目标风险分布图。

## 运行

```bash
python src/carla_radar_collision_warning/radar_warning.py --output docs/pr_assets/carla_radar_collision_warning
python src/carla_radar_collision_warning/tests/test_radar_warning.py
```
