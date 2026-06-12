# AirSim 无人机电量返航决策模块

本项目使用 AirSim 风格无人机电量与飞行遥测数据，根据返航距离、速度、风速、高度和载荷估计返航所需电量，并给出继续任务、准备返航或立即返航建议。

## 主要内容

- 读取 AirSim 电量遥测日志。
- 估计返航时间和返航所需电量。
- 计算电量安全裕度。
- 输出 `continue_mission`、`prepare_return`、`return_now` 三类建议。
- 生成电量需求曲线和电量裕度柱状图。

## 运行

```bash
python src/airsim_battery_rth_advisor/battery_rth.py --output docs/pr_assets/airsim_battery_rth_advisor
python src/airsim_battery_rth_advisor/tests/test_battery_rth.py
```
