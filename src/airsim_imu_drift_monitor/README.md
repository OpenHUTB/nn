# AirSim 多旋翼 IMU/GPS 漂移监测模块

本项目使用 AirSim 多旋翼飞行遥测数据，分析 IMU 振动、GPS 漂移和飞行轨迹偏差，并生成重定位或控制平滑建议。

## 主要内容

- 使用 AirSim 多旋翼 IMU/GPS 遥测数据 `airsim_multirotor_imu.csv`。
- 读取位置、速度、加速度、陀螺仪和 GPS 估计位置。
- 计算加速度范数、角速度范数、GPS 漂移和健康分数。
- 输出 normal / smooth_control / relocalize 三类建议。
- 生成 AirSim 飞行轨迹漂移图和 IMU 健康曲线图。

## 环境要求

- Python 3.7+
- NumPy
- pandas
- matplotlib

## 数据格式

样例数据包含 AirSim 多旋翼飞行遥测数据：位置、速度、加速度、陀螺仪和 GPS 估计位置。

## 运行

```bash
python src/airsim_imu_drift_monitor/imu_drift.py --output docs/pr_assets/airsim_imu_drift_monitor
python src/airsim_imu_drift_monitor/tests/test_imu_drift.py
```
