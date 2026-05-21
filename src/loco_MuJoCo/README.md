# Loco MuJoCo

基于 MuJoCo 物理引擎的机器人运动控制项目，支持 Unitree H1 人形机器人的模仿学习。

## 功能特性

- Unitree H1 人形机器人仿真
- 模仿学习（Imitation Learning）支持
- 多数据集加载（LAFAN1、Motion Dataset）
- 可视化轨迹播放

## 项目结构

```
loco_MuJoCo/
├── main.py               # 主入口，支持 UnitreeH1 机器人的轨迹播放和 Gymnasium 环境
├── double4.py            # 四足机器人控制
├── double_action.py      # 双足行走实验
├── double_action3.py     # 双足动作变体
├── triple.py             # 三足实验
├── fightsport.py         # 战斗运动仿真
├── rolling_log.py        # 翻滚日志记录
├── test.py               # 测试脚本
├── h1.xml                # Unitree H1 机器人模型
├── rolling_log.xml       # 翻滚模型
└── ros2/                 # ROS2 集成
    └── ros2_py/
        └── robot_controller.py  # ROS2 机器人控制器
```

## 依赖

- Python 3.8+
- MuJoCo
- gymnasium
- loco_mujoco
- NumPy

## 使用方法

```bash
# 运行主程序
python main.py

# 指定机器人类型和数据集
python main.py --robot UnitreeH1 --dataset walk
```

## 参考

- [loco_mujoco](https://github.com/Jiangjiangshang/loco_mujoco) - 机器人模仿学习框架
- [Unitree H1](https://github.com/unitreerobotics) - 人形机器人
