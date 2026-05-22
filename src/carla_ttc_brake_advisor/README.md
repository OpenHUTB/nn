# CARLA 目标跟踪 TTC 碰撞风险与制动建议模块

本项目使用 CARLA 风格的目标跟踪数据，基于相对位置、相对速度和目标类别计算 Time-To-Collision，并输出分级制动建议。

## 主要内容

- 读取 CARLA 目标跟踪日志，包括车辆、行人和自行车。
- 根据相对速度投影计算 TTC。
- 结合横向冲突门控和目标类别权重计算碰撞风险。
- 输出 keep_speed / prepare_brake / soft_brake / emergency_brake 四类制动建议。
- 生成 TTC 时序图和目标位置风险热力散点图。
- 将运行效果图输出到 `docs/pr_assets/carla_ttc_brake_advisor`。

## 运行

```bash
python src/carla_ttc_brake_advisor/ttc_advisor.py --output docs/pr_assets/carla_ttc_brake_advisor
python src/carla_ttc_brake_advisor/tests/test_ttc_advisor.py
```
