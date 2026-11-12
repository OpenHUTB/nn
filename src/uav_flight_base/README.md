# uav_flight_base

无人机基础飞行模块（ROS1 Noetic）。提供**键盘遥控**、**最小运动学仿真**、**状态监控**三部分，
不依赖 Gazebo / AirSim 即可完整跑通，用于验证"遥控 → 控制 → 状态反馈"这条链路。

> 这是 10 次提交中的第 1 次：只做骨架与可运行的最小闭环。后续按任务 2/3/4 逐步加感知、建图、端到端网络。

## 目录结构

```
uav_flight_base/
├── main.py                     # 入口（仓库约定：入口以 main. 开头）
├── CMakeLists.txt
├── package.xml
├── launch/
│   └── main.launch             # ROS 启动配置（仓库约定：main.launch）
└── scripts/
    ├── uav_teleop_key.py       # 键盘 → /uav/cmd_vel
    ├── uav_mock_sim.py         # 运动学仿真 → /uav/odom + TF
    └── uav_state_monitor.py    # 状态打印 / CSV 记录
```

## 环境

| 项目 | 版本 |
|---|---|
| 系统 | Ubuntu 20.04 |
| ROS | Noetic |
| Python | 3.8 |
| 依赖 | rospy / geometry_msgs / nav_msgs / tf2_ros（ROS 自带） |

## 安装

```bash
# 1. 放进 catkin 工作空间
cp -r uav_flight_base ~/catkin_ws/src/

# 2. 编译
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## 运行

**方式一：看效果（需要图形界面）**

```bash
rosrun uav_flight_base main.py --mode mock
```

会启动仿真节点 + 状态监控 + 键盘遥控窗口（xterm）。

**方式二：只跑链路（不需要键盘）**

```bash
roslaunch uav_flight_base main.launch mode:=mock teleop:=false
```

另开一个终端手动发速度指令：

```bash
rostopic pub -r 10 /uav/cmd_vel geometry_msgs/Twist \
  '{linear: {x: 1.0, y: 0.0, z: 0.5}, angular: {z: 0.3}}'
```

**方式三：记录飞行数据**

```bash
roslaunch uav_flight_base main.launch mode:=mock teleop:=false csv:=/tmp/flight.csv
```

## 键盘操作

| 按键 | 动作 |
|---|---|
| `w` / `s` | 前进 / 后退 |
| `a` / `d` | 左移 / 右移 |
| `r` / `f` | 上升 / 下降 |
| `q` / `e` | 左转 / 右转 |
| `空格` | 悬停（速度清零） |
| `Ctrl+C` | 退出 |

## 预期输出

```
[uav_flight_base] mode = mock
[INFO] [uav_state_monitor]: [   0.5s] pos=( 0.42, 0.00, 1.50) yaw=  0.0°  v=0.61 m/s
[INFO] [uav_state_monitor]: [   1.0s] pos=( 0.88, 0.12, 1.62) yaw=  6.9°  v=0.95 m/s
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `~rate` | 50.0 | 仿真频率 Hz |
| `~time_constant` | 0.35 | 一阶惯性时间常数，越大越"肉" |
| `~max_speed` | 5.0 | 最大速度 m/s |
| `~z_min` / `~z_max` | 0.2 / 30.0 | 高度限制 m |
| `~cmd_topic` | `/uav/cmd_vel` | 速度指令话题 |
| `~odom_topic` | `/uav/odom` | 里程计话题 |

## 后续计划

- [ ] 接 Gazebo 无人机模型（`mode:=gazebo`）
- [ ] 深度图避障
- [ ] 栅格建图 + A* 导航
- [ ] 端到端 CNN（图像 → 控制）

## 许可

MIT
