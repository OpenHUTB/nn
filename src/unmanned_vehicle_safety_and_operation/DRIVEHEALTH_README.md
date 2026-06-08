# DriveHealth：面向无人车运行安全的多源异常诊断与自健康黑匣子系统

## 项目定位

DriveHealth 是一个不依赖 CARLA、AirSim、MuJoCo 或 Gazebo 的无人车运行安全监测项目。它使用多源传感器流和运行日志完成车辆自健康诊断，适合课程展示、毕设原型和离线算法验证。

本项目把原目录下分散的温度报警、烟雾报警、速度异常、紧急刹车、人机接管和效率检测能力，整合成一个统一的“自健康黑匣子”入口：

```bash
python src/unmanned_vehicle_safety_and_operation/drivehealth_blackbox.py
```

## 新增功能

1. 多源传感器流生成：速度、电机温度、电池温度、控制器温度、烟雾浓度、障碍物距离、路面摩擦、人机接管状态、效率评分。
2. 多子系统风险评分：热管理、烟雾、电控/速度传感器、刹车安全裕度、驾驶员接管、运行效率。
3. 黑匣子事件提取：自动识别连续高风险区间，记录事件 ID、峰值风险、起止时间、严重等级和建议动作。
4. 健康分数：将各风险按权重融合为 0-100 的车辆健康分。
5. 自动报告生成：CSV 日志、JSON 事件、JSON 摘要、HTML 报告和多张 PNG 可视化图。

## 运行方式

基础运行：

```bash
python src/unmanned_vehicle_safety_and_operation/drivehealth_blackbox.py
```

自定义运行时长和输出目录：

```bash
python src/unmanned_vehicle_safety_and_operation/drivehealth_blackbox.py --duration 600 --event-threshold 60 --output-dir src/unmanned_vehicle_safety_and_operation/drivehealth_visualizations
```

依赖库：

```bash
pip install numpy matplotlib
```

## 输出目录

默认输出到：

```text
src/unmanned_vehicle_safety_and_operation/drivehealth_visualizations/
```

主要文件：

- `drivehealth_blackbox_log.csv`：逐秒传感器、风险分数、健康分数和等级日志。
- `drivehealth_events.json`：黑匣子事件清单。
- `drivehealth_summary.json`：项目摘要和总体指标。
- `drivehealth_report.html`：可直接打开的图文报告。
- `01_system_overview_dashboard.png`：系统总览仪表盘。
- `02_health_score_timeline.png`：健康分数时间线。
- `03_sensor_streams_overview.png`：多源传感器曲线。
- `04_subsystem_risk_heatmap.png`：子系统风险热力图。
- `05_blackbox_event_timeline.png`：黑匣子事件时间线。
- `06_risk_radar_summary.png`：风险雷达图。
- `07_event_distribution.png`：事件类型与严重等级分布。
- `08_feature_health_correlation.png`：特征与健康分相关性矩阵。
- `09_health_score_distribution.png`：健康分分布。
- `10_action_priority_board.png`：高风险动作优先级看板。

## 为什么不需要模拟器

DriveHealth 关注的是“无人车自身运行状态是否健康”，不是测试车辆在 3D 世界里的驾驶行为。因此它不需要启动城市道路模拟器。它可以使用：

- 合成传感器流，适合课堂演示；
- CSV 运行日志，适合离线复盘；
- 真实硬件传感器数据，适合后续扩展。

这和 CARLA 中做车道保持、目标检测、交通规则控制的项目不同。DriveHealth 的核心价值是运行安全监测、故障早期发现和事件取证。
