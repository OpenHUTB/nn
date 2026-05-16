# MuJoCo 步态稳定性监测与摔倒预警模块

本项目使用 MuJoCo 风格的人形机器人步态状态日志，分析质心高度、躯干俯仰角、步速、膝关节不对称和足底接触状态对步态稳定性的影响。

## 主要内容

- 读取 MuJoCo 人形机器人步态日志。
- 计算质心高度风险、躯干姿态风险、速度风险和关节不对称风险。
- 输出 stable / unstable / fall_warning 三类状态。
- 生成质心高度-风险曲线和躯干俯仰-风险散点图。
- 将运行效果图输出到 `docs/pr_assets/mujoco_gait_stability_monitor`。

## 运行

```bash
python src/mujoco_gait_stability_monitor/gait_stability.py --output docs/pr_assets/mujoco_gait_stability_monitor
python src/mujoco_gait_stability_monitor/tests/test_gait_stability.py
```
