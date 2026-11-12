# uav_flight_base

> 无人机基础飞行模块：键盘遥控 + 最小运动学仿真 + 状态监控
> 作者：唐健斌　｜　环境：Ubuntu 20.04 + ROS Noetic

## 模块定位

课程作业要求里，"人体/车辆/无人机/水下机器人"任选其一。本模块选**无人机**，
并先解决最基础的一件事：**让一架无人机在仿真里被控制起来，并且状态可观测**。

不接 Gazebo、不接 AirSim 也能跑——因为用的是自己写的一阶惯性运动学模型，
这样在没有图形界面的服务器上也能验证"遥控 → 控制 → 状态"这条链路。

## 数据流

```
键盘 (uav_teleop_key)
      │  /uav/cmd_vel  (geometry_msgs/Twist)
      ▼
运动学仿真 (uav_mock_sim)
      │  /uav/odom     (nav_msgs/Odometry)
      │  TF: map -> base_link
      ▼
状态监控 (uav_state_monitor)
      │  终端打印 / CSV
      ▼
RViz / rqt_plot（可选）
```

## 话题

| 话题 | 类型 | 方向 | 说明 |
|---|---|---|---|
| `/uav/cmd_vel` | `geometry_msgs/Twist` | 订阅 | 期望速度（机体系） |
| `/uav/odom` | `nav_msgs/Odometry` | 发布 | 位置、姿态、速度 |
| `/tf` | `tf2_msgs/TFMessage` | 发布 | `map` → `base_link` |

## 运动学模型

一阶惯性环节，避免"按键即瞬移"的不真实感：

```
dv/dt = (v_cmd - v) / τ
```

- `τ`（`~time_constant`，默认 0.35s）越大，响应越"肉"
- 速度分别裁剪到 `±max_speed`
- 高度限制在 `[z_min, z_max]`

## 启动

```bash
rosrun uav_flight_base main.py --mode mock
```

或

```bash
roslaunch uav_flight_base main.launch mode:=mock teleop:=false
```

## 验证清单

- [x] `rostopic hz /uav/odom` 稳定在 50Hz
- [x] 键盘 `w` 后 `/uav/odom` 的 `x` 单调增大
- [x] 松开按键后速度在约 1s 内衰减到 0（一阶惯性）
- [x] `csv:=/tmp/flight.csv` 能导出轨迹
- [ ] 接 Gazebo 模型（下一次提交）

## 设计取舍

1. **先不接 Gazebo**：Gazebo 的无人机模型（rotors / PX4）依赖重、装一次半小时且容易崩。
   先用轻量运动学模型把控制链路和文档跑通，再把后端换掉，代码接口不变。
2. **`main.py` 只做调度**：真正的节点逻辑都在 `scripts/` 下，每个脚本可单独 `rosrun` 调试。
3. **参数全部走 rosparam**：不写死在代码里，方便后面接不同机型。
