# rain_sensor_fusion

## 拟定课题

**基于深度学习置信度加权的自动驾驶雨天多传感器融合感知优化**

## 项目简介

原项目只有课题说明，没有实际代码实现。本次改造把它补全为一个可运行的轻量深度学习项目：使用 PyTorch 构建相机、LiDAR、毫米波雷达三路特征编码器，通过雨天质量先验和神经网络注意力门控动态分配传感器权重，输出目标置信度、粗 3D 框、风险评分和驾驶动作建议。

该版本不强依赖 CARLA 才能演示。`sensor/synthetic.py` 提供雨天退化传感器模拟，后续可以把同样的数据结构接到 CARLA 同步帧。

## 深度学习优化点

1. **多模态 MLP 编码器**：分别提取 camera、LiDAR、radar 的高维隐特征。
2. **雨天感知注意力融合**：网络学习三类传感器权重，并融合雨量、雾气、湿滑路面质量先验。
3. **风险检测头**：输出目标存在概率、网络风险值、雨天严重度，并结合物理距离/闭合速度得到可解释风险分数。
4. **合成训练闭环**：提供 `training.py`，可用自动生成的雨天样本快速训练网络，再替换成真实 CARLA 数据。
5. **可视化实验输出**：输出融合权重、风险等级、动作建议、粗 3D 检测框、场景对比图和雨强扫描曲线。

## 快速运行

```bash
pip install -r src/rain_sensor_fusion/requirements.txt
python -m src.rain_sensor_fusion.main --scenario heavy_rain
```

可选：先做一次轻量合成训练再推理。

```bash
python -m src.rain_sensor_fusion.main --scenario foggy_storm --train-steps 80
```

生成单场景可视化：

```bash
python -m src.rain_sensor_fusion.main --scenario heavy_rain --visualize
```

生成完整实验图集：

```bash
python -m src.rain_sensor_fusion.main --scenario heavy_rain --visualize --compare
```

默认输出到 `outputs/rain_sensor_fusion/`：

- `heavy_rain_dashboard.png`：单场景融合权重、风险分解、传感器特征热力图和 3D 目标俯视图；
- `clear_dashboard.png`、`light_rain_dashboard.png`、`foggy_storm_dashboard.png`：其他场景对照；
- `scenario_comparison.png`：不同天气下传感器权重、风险和置信度对比；
- `rain_intensity_sweep.png`：雨强从 0 到 1 时融合权重和风险曲线；
- `scenario_results.csv`：四个场景的数值结果，便于写实验表格。

## 项目结构

```bash
rain_sensor_fusion/
├── main.py                  # 命令行演示入口
├── config.py                # 融合模型和风险阈值配置
├── training.py              # 合成雨天样本训练循环
├── sensor/
│   └── synthetic.py         # 雨天传感器退化模拟与训练数据生成
├── fusion/
│   └── deep_fusion.py       # PyTorch 多模态深度融合网络
├── detection/
│   └── risk.py              # 风险评分、动作建议、粗 3D 框
└── visualization/
    ├── report.py            # CLI 结果格式化
    └── plots.py             # 仪表盘、场景对比和雨强扫描可视化
```

## 示例输出

```text
Scenario: heavy_rain
Weather: rain= 82.0%, fog= 42.0%, wet_road= 90.0%
Deep fusion weights: camera= 16.2%, lidar= 32.7%, radar= 51.1%
Object confidence:  70.1%
Risk score:  66.3% (network= 51.4%, physical= 69.6%)
Recommended action: slow_down
3D box: x=17.0m, y=-0.4m, lwh=(4.46, 2.03, 1.63)m, conf= 70.1%
```
