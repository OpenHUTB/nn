# 最终验收报告

> 项目：`C:\Users\zhoul\OneDrive\Desktop\ROS`（MuJoCo 无人车神经网络驾驶 + 栅格 SLAM + ROS 1/2 包）
> 验收时间：2026-11-11 23:45（Asia/Shanghai）
> 执行方式：DeepSeek Harness 桌面自动化 + 命令行，**只记录实际执行过的结果**

## 1. 项目简介

一套可复现的无人车课程作业：MuJoCo 物理仿真（227 m 闭环赛道、9 m 宽路面）、
64×64 车载相机、64 束激光雷达、pure-pursuit 专家、端到端 CNN（BC / DAgger）、
栅格建图与扫描匹配 SLAM，并提供 ROS 1 Noetic 与 ROS 2 Humble 两套包与 launch。

## 2. Windows 环境

| 项 | 值 |
|---|---|
| 操作系统 | Windows 11 家庭版 10.0.26200（64 位） |
| 硬件 | Ryzen 9 8945HX（16C/32T）、31.8 GB、RTX 5060 Laptop |
| 权限 | 非管理员（安装时由本人 UAC 授权） |
| 桌面自动化 | 窗口枚举/置前/截图/键鼠注入均实测可用 |

## 3. WSL2 安装情况

**PASS**（2026-11-11 01:16 完成，无需重启）

| 项 | 值 |
|---|---|
| WSL 应用 | `MicrosoftCorporationII.WindowsSubsystemForLinux 3.0.5.0` |
| 发行版 | `Ubuntu-22.04`，`VERSION=2`，状态 Stopped/Running 正常 |
| 内核 | `6.18.58.1-microsoft-standard-WSL2` |
| 安装方式 | 经用户 UAC 批准后 `wsl --install -d Ubuntu-22.04 --no-launch`，后台完成 |

## 4. Ubuntu 版本

`Ubuntu 22.04.5 LTS (Jammy)`，默认 root shell 可用，已配置阿里云 apt 镜像与 en_US.UTF-8 locale。

## 5. ROS 2 安装情况

**PASS** —— ROS 2 **Humble**（`/opt/ros/humble`）

| 项 | 值 |
|---|---|
| 软件源 | `https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu jammy main`（官方源在国内不可达） |
| 密钥 | Open Robotics `C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654`（经 Gitee 镜像获取，已校验指纹） |
| 主要包 | `ros-humble-desktop`、`ros-dev-tools`、`python3-colcon-common-extensions`、`python3-rosdep`、`python3-vcstool`、`python3-argcomplete` |
| 环境变量 | `ROS_DISTRO=humble`，`source /opt/ros/humble/setup.bash` 正常 |
| rosdep | `rosdep init/update` **失败**：`raw.githubusercontent.com` 被拒绝（非致命，本项目包无外部 rosdep 依赖） |

## 6. 开发依赖

| 组件 | 版本 |
|---|---|
| Python | 3.10.12 |
| pip | 22.0.2 |
| gcc / g++ | 11.4.0 |
| cmake | 3.22.1 |
| git | 2.34.1 |
| MuJoCo（Ubuntu） | 3.17.0（`MUJOCO_GL=egl` 离屏渲染实测可用） |
| NumPy（Ubuntu） | 1.21.5 |

## 7. 编译情况

**PASS**

```text
$ cd ~/workspace/ROS/ros2 && colcon build --symlink-install
Starting >>> car_nn_sim
Finished <<< car_nn_sim [1.13s]
Summary: 1 package finished [1.32s]
$ colcon list
car_nn_sim  car_nn_sim  (ros.ament_python)
$ ros2 pkg prefix car_nn_sim
/root/workspace/ROS/ros2/install/car_nn_sim
```

工作副本位于 Linux 文件系统 `/root/workspace/ROS`（从 OneDrive 目录 rsync 而来，保留 23 次提交）。

## 8. 节点运行情况

**PASS**

```text
$ ros2 launch car_nn_sim car_nn.launch.py
$ ros2 node list
/car_nn_sim
```

节点启动后同时发布相机、雷达、里程计，并订阅 `/car/cmd_vel`。

## 9. 话题通信情况

**PASS**

| 话题 | 类型 | 实测 |
|---|---|---|
| `/car/camera/image_raw` | `sensor_msgs/msg/Image` | 约 3.8 Hz（EGL 软件渲染限制） |
| `/car/scan` | `sensor_msgs/msg/LaserScan` | 约 3.4 Hz，64 束 |
| `/car/odom` | `nav_msgs/msg/Odometry` | `frame_id: odom` → `base_link`，x=41.0, y=0.0 |
| `/car/cmd_vel` | `geometry_msgs/msg/Twist` | 发布后 odom 由 (41.00, 0.00) 变为 (41.14, 0.54) |

官方 talker/listener 通信测试：

```text
[talker] Publishing: 'Hello World: 1..5'
[listener] I heard: [Hello World: 7]
...
[listener] I heard: [Hello World: 14]
```

**PASS**（真实 DDS 通信，非模拟）。

## 10. 仿真情况

- Windows 侧：MuJoCo 原生窗口真实启动，车辆/赛道/护栏渲染正常（`03_simulation/`）。
- WSL 侧：`sim_node` 以 `MUJOCO_GL=egl` 离屏运行，相机与雷达数据由真实物理步进产生。
- 帧率约 3.8 Hz（WSL 软件渲染），低于 Windows 原生的 20 Hz，属环境限制而非代码问题。

## 11. 传感器情况

| 传感器 | 结果 |
|---|---|
| 相机 | 64×64 rgb8，`Image` 消息 12288 字节 |
| 激光雷达 | `LaserScan` 64 束，量程 20 m，`range_min=0.05` |
| 里程计 | `Odometry`，位置/姿态/速度字段完整 |

## 12. SLAM 情况

**PASS**

Windows 侧算法验证：单圈 ATE **2.37 m**（纯里程计 2.98 m）、地图 IoU **0.13**。

ROS 侧（`car_slam.launch.py` = sim_node + slam_node）实测：

```text
$ ros2 topic echo --once /map --field info
resolution: 0.25   width: 440   height: 360   origin: (-55.0, -45.0)
$ ros2 topic hz /map
average rate: 1.000
$ ros2 run tf2_ros tf2_echo map base_link
Translation: [0.000, 0.000, 0.000]  Rotation: [0, 0, 0, 1]
```

`/map` 以 1 Hz 发布 `nav_msgs/OccupancyGrid`，`map→base_link` TF 正常。

## 12.1 RViz

**PASS（窗口与 OpenGL）**

```text
$ rviz2 -d car.rviz
[rviz2]: OpenGL version: 4.2 (GLSL 4.2)
```

WSLg 可用（`DISPLAY=:0`、`WAYLAND_DISPLAY=wayland-0`），RViz 窗口正常显示。
说明：`car.rviz` 只加载了 Grid 显示，Map/LaserScan 显示项未自动生效，需在 RViz 里手动
Add → By topic 选择 `/map`、`/car/scan`；数据本身已通过 CLI 验证。

## 13. 神经网络实验

| 策略 | 平均速度 (m/s) | 单圈 (s) | 横向 RMSE (m) | 碰撞 |
|---|---|---|---|---|
| 专家 | 10.41 | 22.1 | 0.49 | 0 |
| CNN（BC） | 8.03 | 28.6 | 0.45 | 0 |
| **CNN（BC+DAgger）** | **10.29** | **22.4** | 0.47 | 0 |
| MLP（BC） | 6.75 | 34.0 | 0.38 | 0 |

day / dusk / fog 共 12 组实验全部完成，fog 为零样本场景。
推理 1.5 ms/帧（CPU）。消融：纯图像 CNN 原地不动；模块化方案横向误差 1.89 m。

## 14. 自动化测试

```text
$ python -m pytest tests -q
......                                                                   [100%]
6 passed in 4.72s
```

含 ROS 1 / ROS 2 节点桩测试（MOCK 级别，不代表真实 ROS 运行）。

## 15. 截图索引

| 文件 | 类型 | 内容 |
|---|---|---|
| `01_environment/01_windows_env.png` | 真实截图 | Windows 版本、WSL 未装（当时）、工具链 |
| `02_build/01_pytest.png` | 真实截图 | pytest 6 passed |
| `02_build/02_mkdocs_build.png` | 真实截图 | mkdocs strict 构建 |
| `03_simulation/01_mujoco_running.png` | 真实截图 | MuJoCo 窗口 + 赛道 |
| `03_simulation/02_mujoco_driving.png` | 真实截图 | 车辆行驶 |
| `04_keyboard/01_keyboard_console.png` | 真实截图 | 按键序列与状态 |
| `04_keyboard/02_keyboard_state.png` | 真实截图 | `keys=WWDD`、油门/转向 |
| `05_sensors/*` | 程序输出 | 相机、雷达极坐标、里程计 |
| `06_slam/*` | 程序输出 | 栅格地图、ATE、建图 GIF |
| `08_neural_network/*` | 程序输出 | 训练曲线、对比、驾驶 GIF |
| `10_ros2/01_ros2_node_topic.png` | 真实截图 | 话题列表 + odom echo |
| `10_ros2/02_ros2_topics.png` | 真实截图 | node list / topic list -t / odom |
| `10_ros2/03_rviz_running.png` | 真实截图 | RViz2 窗口（WSLg） |
| `10_ros2/04_rviz_slam_map.png` | 真实截图 | RViz 加载 car.rviz |
| `10_ros2/05_slam_map_topic.png` | 真实截图 | /map info、1 Hz、TF |
| `11_documentation/*` | 真实截图 | MkDocs 站点 |
| `12_final/01_git_log.png` | 真实截图 | 提交历史 |

日志：`02_ros_install/{base_setup,ros2_install,ros2_verify,ros2_run,ros2_cmdvel}.log`、
`03_build/ros2_build.log`。

## 16. ROS 1 状态

**BLOCKED** —— ROS 1 Noetic 需要 Ubuntu 20.04，不能在 22.04 混装。
仓库中 `ros1/car_nn_sim` 已就绪，需另建 Ubuntu 20.04 发行版（或课程虚拟机）后 `catkin_make`。

## 17. 未完成事项

1. ROS 1 Noetic（需 Ubuntu 20.04，课程允许 Noetic **或** Humble，当前用 Humble 验收）。
2. 真人键盘连续驾驶演示（自动按键序列会撞护栏，PARTIAL）。
3. RViz 显示项需手动 Add（配置文件只自动加载了 Grid）。
4. 10 次同学评审。

## 18. 后续建议

1. 先验证 WSLg：`sudo apt install -y mesa-utils && glxinfo -B`，能出窗口再截 RViz。
2. 把 `slam_node` 加进 `car_slam.launch.py` 的实测流程，发布 `/map` 并截图。
3. 若课程强制 ROS 1，新建 Ubuntu 20.04 发行版单独验收，不要混装。
4. 帧率问题：WSL 软件渲染只有 ~4 Hz，可用 `MUJOCO_GL=osmesa` 或降低渲染分辨率改善。
