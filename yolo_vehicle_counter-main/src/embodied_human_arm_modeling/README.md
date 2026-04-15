Shadow Hand 手部动画演示项目
📋 项目概述
本项目基于 Shadow Robot Company 开源的 Shadow Hand E3M5 左手模型（Apache 2.0 License），使用 MuJoCo 物理引擎开发了一个交互式的手部动画演示系统。该系统可以展示多种预设手部姿态，并支持用户通过命令行进行实时控制。

🚀 快速开始
系统要求
Python 3.8+

MuJoCo 3.x

NumPy

安装依赖
bash
pip install mujoco numpy
运行演示
bash
python interactive_hand_simulator.py
确保 left_hand.xml 模型文件和 assets/ 目录（包含 OBJ 网格文件）与脚本在同一目录下。

🎮 功能特性
1. 预设手部姿态库
系统包含 6 种精心设计的手部姿态：

张开手 🤚 - 所有手指完全展开

握拳 ✊ - 所有手指完全握紧

圆柱体抓握 🫱 - 环绕抓握柱状物体

剪刀手 ✌️ - 食指和中指张开呈V形

OK手势 👌 - 拇指和食指形成圆圈

指点 👉 - 食指伸直，其他手指握起

2. 交互控制模式
支持两种控制模式：

自动模式 🤖 - 自动循环播放所有姿态（每姿态保持5秒）

手动模式 🎮 - 用户通过命令行控制姿态切换

3. 动画系统
平滑过渡动画（1.5秒过渡时间）

缓动函数实现自然的运动曲线

实时姿态进度显示

4. 控制命令
在程序运行期间，可在终端中输入以下命令：

text
pause:  暂停/继续演示
next:   切换到下一个姿态
prev:   切换到上一个姿态
mode:   切换手动/自动模式
restart:重新开始演示
help:   显示控制说明
quit:   退出演示
📁 文件说明
核心文件
text
.
├── left_hand.xml                 # Shadow Hand 左手MJCF模型文件
├── interactive_hand_simulator.py # 主程序 - 交互式手部演示
└── assets/                       # 模型网格文件目录（包含所有.obj文件）
模型结构说明
left_hand.xml 文件定义了完整的 Shadow Hand 左手模型：

主要组件：
前臂（lh_forearm） - 包含惯性参数和碰撞几何体

手腕（lh_wrist） - 包含两个旋转关节（WRJ1, WRJ2）

手掌（lh_palm） - 手部基础结构

五个手指 - 分别对应索引、中指、无名指、小指和拇指

关节配置：
手腕关节：2个自由度（俯仰和横滚）

拇指关节：5个自由度（THJ1-THJ5）

其他手指：每个3-4个自由度（包括肌腱耦合）

执行器系统：
20个位置控制执行器

包含肌腱耦合（例如FFJ0同时控制FFJ2和FFJ1）

💻 代码架构
HandDemoMujoco3 类
主程序类，包含以下主要方法：

初始化方法：
python
__init__(model_path='left_hand.xml')  # 加载模型并初始化状态
姿态创建方法：
python
_create_preset_poses()                # 创建6种预设手部姿态
_create_pose_fist()                   # 创建握拳姿态
_create_pose_cylinder()               # 创建圆柱体抓握姿态
_create_pose_scissors()               # 创建剪刀手姿态
_create_pose_ok()                     # 创建OK手势
_create_pose_pointing()               # 创建指点姿态
动画控制方法：
python
start_animation(pose_name)           # 开始动画到指定姿态
update_animation()                   # 更新动画状态
toggle_pause()                       # 切换暂停状态
next_pose()                          # 切换到下一个姿态
previous_pose()                      # 切换到上一个姿态
toggle_manual_mode()                 # 切换手动/自动模式
用户交互方法：
python
process_command(command)             # 处理控制台命令
print_controls()                     # 打印控制说明
run_demo()                           # 运行主演示循环
多线程设计
主线程：处理物理仿真和图形渲染

输入线程：监听用户命令行输入

线程安全：使用锁保护共享状态变量

🔧 技术实现细节
动画插值系统
python
# 使用缓动函数（ease in-out）实现平滑过渡
if t < 0.5:
    t_eased = 2 * t * t
else:
    t_eased = -1 + (4 - 2 * t) * t

# 线性插值计算
current_values = start_values + (target_values - start_values) * t_eased
姿态数据适配
代码根据实际执行器数量自动调整姿态参数：

支持完整20执行器配置

支持简化配置（最少3个执行器）

容错性设计，避免索引错误

📊 运行输出示例
text
============================================================
✅ 手部模型加载成功
📊 执行器数量: 20
📊 关节数量: 24
📊 仿真时间步: 0.0020秒
============================================================
🎭 创建了 6 种预设姿态
============================================================

🤖 手部抓握姿态全自动演示 (MuJoCo 3.x 兼容版)
============================================================
🎬 演示序列: 7 个姿态
⏱️  每个姿态保持: 5.0秒
🎥 动画过渡: 1.5秒

🎮 控制说明 (在终端中输入命令):
  pause: 暂停/继续演示
  next: 下一个姿态
  prev: 上一个姿态
  mode: 切换手动/自动模式
  restart: 重新开始演示
  help: 显示控制说明
  quit: 退出演示
============================================================

演示开始...

🤚 [张开手     ] 进度:   0.0% - 所有手指完全展开
🛠️ 故障排除
常见问题
找不到模型文件

text
❌ 找不到模型文件 'left_hand.xml'
确保 left_hand.xml 和 assets/ 目录与脚本在同一目录下。

缺少网格文件

text
MuJoCo错误: 无法加载网格文件
检查 assets/ 目录是否包含所有引用的 .obj 文件。

MuJoCo版本不兼容
确保安装的是 MuJoCo 3.x 版本。

📝 扩展开发
添加新姿态
要添加新的手部姿态，可在 _create_preset_poses() 方法中添加：

python
self.poses['新手势'] = {
    'values': self._create_custom_pose(),
    'emoji': '🎯',
    'description': '手势描述'
}
修改动画参数
可以调整以下参数：

self.animation_duration - 动画过渡时间

hold_duration - 姿态保持时间

self.model.opt.timestep - 仿真时间步长

📄 许可证说明
模型文件
来源：Shadow Robot Company

许可证：Apache 2.0 License

原始仓库：https://github.com/shadow-robot/sr_common

动画脚本
作者：项目开发者

用途：学术演示和教育用途

依赖：MuJoCo, NumPy

🙏 致谢
Shadow Robot Company - 提供高质量的开源手部模型

DeepMind - 开发 MuJoCo 物理引擎

MuJoCo 社区 - 提供技术支持和文档

📧 反馈与贡献
如发现任何问题或有改进建议，请：

检查代码注释和错误信息

参考 MuJoCo 官方文档

在课程作业仓库中提交问题报告

注意：本项目为学术用途，遵循 Apache 2.0 开源许可证。模型版权归 Shadow Robot Company 所有，动画代码为课程作业原创实现。