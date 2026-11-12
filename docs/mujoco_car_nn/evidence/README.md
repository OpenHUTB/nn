# 验收证据目录

本目录记录 **2026-11-10 ~ 11-11 实际执行** 的环境检查、构建、测试、仿真与 ROS 2 验收证据。
每条证据都标注类型，避免把示意图当作实时运行截图：

| 类型 | 含义 |
|---|---|
| **真实截图** | 从真实窗口（终端 / MuJoCo / 浏览器）抓屏得到 |
| **程序输出** | 由仿真器或算法直接生成的图片（不是屏幕截图） |
| **日志** | 命令的原始文本输出 |

## 目录

| 目录 | 内容 | 状态 |
|---|---|---|
| `01_environment/` | Windows 版本、权限、工具链 | ✅ 完成 |
| `02_ros_install/` | WSL2 安装、Ubuntu 基础环境、ROS 2 安装与 talker/listener | ✅ 完成（ROS 2 **PASS**） |
| `03_build/` | pytest、mkdocs、`colcon build` | ✅ 完成 |
| `04_keyboard/` | 键盘事件注入与实时状态 | ⚠️ 部分（按键通路已验证，自动驾驶会撞墙） |
| `05_sensors/` | 相机 / 雷达 / 里程计 | ✅ 完成（程序输出） |
| `06_slam/` | 栅格地图、ATE、建图动图 | ✅ 完成（程序输出） |
| `07_navigation/` | 导航说明 | ✅ 说明（导航由端到端策略完成） |
| `08_neural_network/` | 训练曲线、对比实验、驾驶动图 | ✅ 完成 |
| `09_ros1/` | ROS 1 证据 | ⛔ **BLOCKED**（需 Ubuntu 20.04） |
| `10_ros2/` | ROS 2 节点、话题、odom、cmd_vel 测试 | ✅ **PASS** |
| `11_documentation/` | MkDocs 站点截图 | ✅ 完成 |
| `12_final/` | Git 提交历史 | ✅ 完成 |

## ROS 2 关键日志

| 文件 | 内容 |
|---|---|
| `02_ros_install/base_setup.log` | 阿里云镜像切换 + 基础工具安装 |
| `02_ros_install/ros2_install.log` | ROS 2 Humble 安装（清华镜像，含密钥指纹） |
| `02_ros_install/ros2_verify.log` | talker/listener 真实通信（Hello World 7–14） |
| `10_ros2/ros2_run.log` | node list / topic list -t / hz / echo |
| `10_ros2/ros2_cmdvel.log` | 发布 `/car/cmd_vel` 后 odom 位移 |
| `03_build/ros2_build.log` | `colcon build` 成功 |

完整结论见 [`FINAL_ACCEPTANCE_REPORT.md`](FINAL_ACCEPTANCE_REPORT.md)。
