# Autonomous Driving RL

基于 CARLA 模拟器的自动驾驶强化学习训练项目，使用 PPO 算法训练智能体。

## 功能特性

- CARLA 模拟器环境集成
- PPO (Proximal Policy Optimization) 算法
- 多观测空间支持
- 自动日志和 TensorBoard 可视化
- 定期评估和最佳模型保存
- 支持 ROS 集成

## 项目结构

```
autonomous_driving_RL/
├── train_agent.py              # 训练脚本
├── eval_agent.py               # 评估脚本
├── carla_env_multi_obs.py      # CARLA 环境封装
├── home/wu/catkin_ws/src/carla_rl_ros/  # ROS 集成包
│   ├── scripts/
│   │   ├── train_agent.py      # ROS 训练节点
│   │   ├── eval_agent.py       # ROS 评估节点
│   │   ├── ros_train_node.py   # 训练主节点
│   │   └── ros_eval_node.py    # 评估主节点
│   ├── launch/
│   │   └── eval.launch         # ROS 启动文件
│   ├── CMakeLists.txt
│   └── package.xml
```

## 依赖

- Python 3.8+
- CARLA 模拟器
- Stable Baselines3
- gymnasium
- TensorBoard
- rospy (ROS)

## 使用方法

### 独立训练

```bash
# 启动 CARLA 模拟器
./CarlaUE4.sh -fps=20

# 训练智能体
python train_agent.py --timesteps 1000000

# 指定日志目录
python train_agent.py --log_dir ./logs --timesteps 500000
```

### ROS 集成训练

```bash
# 启动 ROS评估
roslaunch carla_rl_ros eval.launch

# 或运行各个节点
rosrun carla_rl_ros ros_train_node.py
rosrun carla_rl_ros ros_eval_node.py
```

## 模型配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--timesteps` | 总训练步数 | 300000 |
| `--log_dir` | 日志目录 | ./logs |
| `--model_save_path` | 模型保存路径 | ./checkpoints/best_model.zip |

## 评估

```bash
# 评估训练好的模型
python eval_agent.py --model_path ./checkpoints/best_model.zip
```

## 算法

使用 PPO (Proximal Policy Optimization) 算法，适合连续控制任务：
- 稳定的策略更新
- 适合自动驾驶的连续动作空间
- 高效的样本利用
