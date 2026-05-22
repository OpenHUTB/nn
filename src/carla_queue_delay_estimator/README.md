# CARLA 信号灯路口排队延误预测模块

本项目使用 CARLA 风格的路口排队日志，分析红绿灯相位、车辆到达率、排队长度和平均等待时间之间的关系，并给出是否延长绿灯的控制建议。

## 主要内容

- 读取 CARLA 路口排队数据 `carla_intersection_queue.csv`。
- 统计车道排队总长度、交通压力和预测延误。
- 输出 `keep_cycle`、`prepare_green`、`extend_green` 三类信号控制建议。
- 生成排队长度-延误曲线和信号相位压力散点图。
- 将运行效果图输出到 `docs/pr_assets/carla_queue_delay_estimator`。

## 运行

```bash
python src/carla_queue_delay_estimator/queue_delay.py --output docs/pr_assets/carla_queue_delay_estimator
python src/carla_queue_delay_estimator/tests/test_queue_delay.py
```
