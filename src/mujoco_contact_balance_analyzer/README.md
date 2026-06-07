# MuJoCo 足底接触力平衡分析模块

本项目使用 MuJoCo MJCF 模型和 `sensordata` 中的 force sensor 接触力导出日志，分析左右支撑力差、质心偏移和躯干横滚角对机器人稳定性的影响，并生成姿态恢复建议。

## 主要内容
- 读取 MuJoCo MJCF 模型 `mujoco_quadruped_balance.xml`。
- 运行 `generate_mujoco_contact_data.py` 调用官方 `mujoco` 包 step 仿真并导出数据。
- 读取 MuJoCo `sensordata` 中的四个足底 force sensor 和质心状态数据。
- 计算左右支撑力不平衡、质心偏移和综合平衡风险。
- 输出 stable_walk / adjust_support / recover_posture 三类动作建议。
- 生成平衡风险曲线、左右接触力分布图和支撑多边形/质心投影回放图。

## 运行
```bash
python src/mujoco_contact_balance_analyzer/generate_mujoco_contact_data.py
python src/mujoco_contact_balance_analyzer/contact_balance.py --output docs/pr_assets/mujoco_contact_balance_analyzer
python src/mujoco_contact_balance_analyzer/tests/test_contact_balance.py
```

其中 `generate_mujoco_contact_data.py` 会调用官方 `mujoco` Python 包加载 MJCF 模型并重新导出 `mujoco_sensor_export.csv`，因此数据来自 MuJoCo 仿真 step 过程。

## 诊断重点

分析结果时可先观察左右接触力是否长期偏向同一侧，再结合质心投影位置判断支撑多边形是否留有足够裕度。如果风险峰值同时伴随躯干横滚角增大和质心外移，说明机器人可能需要更早触发姿态恢复策略；如果只是短时间接触力波动，则更适合通过滤波或步态相位判断减少误报。
