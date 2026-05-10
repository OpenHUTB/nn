# 自动驾驶多传感器故障诊断与鲁棒融合

本模块模拟自动驾驶车辆在行驶过程中同时接收 GPS、雷达和里程计传感器数据，并在部分传感器出现漂移、尖峰噪声或尺度误差时，利用残差门限进行故障诊断，再通过鲁棒卡尔曼融合降低故障传感器对定位结果的影响。

## 项目特点

- 生成二维车辆真值轨迹。
- 模拟 GPS、雷达、里程计三类传感器观测。
- 注入三类故障：
  - GPS bias drift
  - Radar spike noise
  - Odometry scale drift
- 使用常速度模型进行卡尔曼预测。
- 根据传感器残差进行故障诊断。
- 对疑似故障传感器进行降权融合。
- 对比朴素平均融合与鲁棒融合的轨迹 RMSE。
- 输出轨迹图、残差诊断图、RMSE 对比图、metrics.json 和 fault_flags.csv。

## 文件结构

```text
src/sensor_fusion_fault_diagnosis/
├── main.py                 # 命令行入口
├── scenario.py             # 轨迹、传感器观测和故障注入
├── fusion.py               # 鲁棒卡尔曼融合与故障诊断
├── visualization.py        # 可视化图表生成
├── README.md               # 项目说明
├── requirements.txt        # 依赖说明
├── assets/                 # 运行结果
└── tests/test_fusion.py    # 基础测试
```

## 运行方法

```bash
cd src/sensor_fusion_fault_diagnosis
python main.py
```

运行后会在 `assets/` 中生成：

```text
fusion_trajectory.png   # 真值、原始传感器、朴素融合和鲁棒融合轨迹
fault_residuals.png     # 三类传感器残差与诊断结果
rmse_comparison.png     # 朴素平均与鲁棒融合 RMSE 对比
metrics.json            # 汇总指标
fault_flags.csv         # 每个时间步的故障诊断标记
```

## 核心思路

普通多传感器融合常假设传感器噪声稳定，但真实自动驾驶系统中，传感器可能出现短时漂移、异常尖峰或尺度误差。如果直接把所有传感器平均或等权融合，故障传感器会明显拉偏定位结果。

本模块使用卡尔曼预测位置作为参考，计算各传感器观测与预测位置之间的残差。当归一化残差超过阈值时，将该传感器标记为疑似故障，并在融合更新时降低它的权重。这样可以在不完全丢弃传感器的情况下减少异常数据的影响。

## 验证指标

模块会输出：

- 朴素平均融合 RMSE。
- 鲁棒融合 RMSE。
- RMSE 改善百分比。
- 各传感器故障诊断 precision / recall。

## 可扩展方向

- 将二维位置扩展为 3D 定位。
- 加入 IMU、Camera、LiDAR 等更多传感器。
- 将固定阈值替换为自适应阈值或学习型异常检测器。
- 接入 Carla 或真实 rosbag 数据进行验证。
