# MuJoCo 机械臂关节力矩异常检测模块

本项目使用 MuJoCo 风格机械臂关节力矩日志，结合关节速度和末端执行器误差计算异常分数，用于识别过载、振荡和控制不稳定片段。

## 主要内容
- 读取 MuJoCo 机械臂力矩、速度和末端误差数据。
- 计算力矩范数、速度范数和综合异常分数。
- 输出 normal / warning / critical 三类状态。
- 生成异常分数曲线和力矩-误差风险散点图。

## 运行
```bash
python src/mujoco_torque_anomaly_detector/torque_anomaly.py --output docs/pr_assets/mujoco_torque_anomaly_detector
python src/mujoco_torque_anomaly_detector/tests/test_torque_anomaly.py
```
