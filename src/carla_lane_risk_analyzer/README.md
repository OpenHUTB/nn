# CARLA 车道保持风险分析模块

本项目使用 CARLA 风格的车道跟踪日志，分析车辆在弯道和交通密度变化下的横向偏移、航向误差与车道偏离风险。样例数据位于 `sample_data/carla_lane_log.csv`，字段包含车辆速度、车道中心偏移、航向误差、车道线距离、道路曲率和交通密度。

## 主要内容

- 读取 CARLA 车道跟踪时序数据。
- 计算基于横向偏移、航向误差、速度、曲率和交通密度的风险分数。
- 输出 low / medium / high 三档风险等级。
- 生成车道偏移-风险曲线和速度-交通密度-风险散点图。
- 将运行效果图输出到 `docs/pr_assets/carla_lane_risk_analyzer`。

## 运行

```bash
python src/carla_lane_risk_analyzer/lane_risk.py --output docs/pr_assets/carla_lane_risk_analyzer
python src/carla_lane_risk_analyzer/tests/test_lane_risk.py
```

## 风险等级复核

建议在查看输出图时优先核对高风险时刻对应的横向偏移、航向误差和道路曲率。如果高风险集中出现在弯道或高交通密度片段，应结合车辆速度判断是车道保持控制不足，还是场景本身需要更保守的速度策略。这样可以让风险曲线不仅用于展示，也能辅助定位控制参数需要优化的位置。
