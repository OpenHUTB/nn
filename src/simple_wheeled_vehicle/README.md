MuJoCo 简易轮式小车智能代理
项目简介
一个基于PyTorch神经网络与MuJoCo物理引擎的简易轮式小车智能代理，实现了在连续控制环境下的感知、决策与控制一体化解决方案。项目采用模块化设计，便于在仿真环境中进行机器人控制算法研究、强化学习训练和控制系统验证。
功能特性
物理仿真：基于MuJoCo的高保真物理模拟，支持关节约束、碰撞检测和接触力计算

神经网络控制：使用PyTorch实现的深度强化学习或模仿学习策略

多模态感知：集成视觉、本体感知和惯性测量单元(IMU)等多种传感器输入

自适应控制：针对不同地形和任务需求调整控制策略
训练框架：提供完整的强化学习训练循环和评估接口

可视化工具：实时显示机器人状态、传感器数据和决策过程
环境要求
操作系统：Ubuntu 20.04/22.04（推荐）或 Windows 10/11（需配置WSL2）

Python：3.7 - 3.10（MuJoCo官方支持范围）

MuJoCo版本：2.3.0+（需要获取许可证）

核心框架：PyTorch ≥ 1.9.0，MuJoCo-Py ≥ 2.3.0

渲染支持：OpenGL或EGL（GPU加速推荐）

推荐硬件：支持CUDA的NVIDIA显卡（用于神经网络加速）

项目结构
text
src/mujoco_wheeled_robot/
├── main.py                      # 模块主入口（训练与推理）
├── requirements.txt
├── config/
│   ├── robot_config.yaml       # 机器人物理参数
│   └── training_config.yaml    # 训练超参数
├── models/
│   ├── robot.xml               # MuJoCo模型文件
│   ├── trained_policy.pth      # 预训练策略
│   └── dynamics_model.pth      # 动力学模型（可选）
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── wheeled_robot_env.py # 自定义Gym环境
│   │   └── env_wrapper.py       # 环境包装器
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── policy_network.py    # 策略网络
│   │   ├── value_network.py     # 价值网络（强化学习）
│   │   └── mpc_controller.py    # 模型预测控制器
│   ├── utils/
│   │   ├── data_logger.py       # 数据记录器
│   │   ├── visualization.py     # 可视化工具
│   │   └── mujoco_utils.py      # MuJoCo辅助函数
│   └── algorithms/              # 算法实现
│       ├── sac.py               # Soft Actor-Critic
│       ├── td3.py               # Twin Delayed DDPG
│       └── bc.py                # 行为克隆
├── scripts/
│   ├── train.py                 # 训练脚本
│   ├── evaluate.py              # 评估脚本
│   └── render_trajectory.py     # 轨迹渲染
└── tests/
    ├── test_env.py
    └── test_agent.py
快速开始
安装MuJoCo：

bash
# 获取MuJoCo许可证并下载
pip install mujoco
# 设置许可证密钥（按官方指引操作）
安装依赖：

bash
cd src/mujoco_wheeled_robot
pip install -r requirements.txt
测试环境：

bash
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
运行示例：

bash
# 启动随机策略演示
python main.py --mode demo
# 开始训练
python main.py --mode train --config config/training_config.yaml
主要类和方法
WheeledRobotEnv (环境类)：

reset(): 重置环境到初始状态

step(action): 执行动作并返回新状态、奖励和终止标志

render(): 渲染当前环境状态

PolicyNetwork (策略网络)：

forward(observation): 根据观测计算动作分布

sample(observation): 从分布中采样动作

evaluate(observation, action): 计算动作的对数概率和熵

DataLogger (数据记录器)：

log_step(state, action, reward): 记录单步数据

save_episode(episode_id): 保存回合数据

export_trajectory(): 导出轨迹供分析使用

配置参数
关键配置位于 config/ 目录下：

yaml
# robot_config.yaml
robot:
  mass: 2.5          # 小车质量 (kg)
  wheel_radius: 0.1  # 轮子半径 (m)
  max_velocity: 3.0  # 最大速度 (m/s)
  control_frequency: 50  # 控制频率 (Hz)

# training_config.yaml
training:
  algorithm: "sac"   # 使用的算法
  total_timesteps: 1000000
  learning_rate: 3e-4
  gamma: 0.99        # 折扣因子
  batch_size: 256
  buffer_size: 1000000
数据输出
训练日志：TensorBoard格式，包含奖励曲线、策略熵等指标

模型检查点：定期保存的策略网络参数

轨迹数据：.npz格式，包含状态、动作、奖励序列

评估结果：性能指标统计（平均奖励、成功率等）

视频记录：.mp4格式的演示视频

扩展建议
添加传感器：在模型文件中增加激光雷达或深度相机

更换任务：修改奖励函数实现不同任务（如避障、路径跟踪）

集成真实数据：使用真实机器人数据微调仿真模型

多机器人协同：扩展环境支持多机器人交互

分布式训练：使用Ray等框架实现并行训练加速

模型预测控制：结合学习的动力学模型实现MPC控制器

常见问题
Q: MuJoCo许可证如何获取？
A: 访问MuJoCo官网获取个人或机构许可证，按照官方指引配置环境变量。

Q: 仿真运行速度慢
A: 检查是否启用了GPU渲染，降低渲染分辨率，或使用 headless 模式进行训练。

Q: 训练不稳定，奖励不收敛
A: 尝试调整超参数（如学习率、折扣因子），增加网络容量，或检查奖励函数设计。

Q: 如何导入自定义机器人模型？
A: 将 .xml 模型文件放入 models/ 目录，在环境初始化时指定模型路径。

Q: 在Windows上运行问题
A: 建议使用WSL2或Linux环境，Windows原生支持有限且可能遇到兼容性问题。

Q: 模型无法学习有效策略
A: 确保观测空间包含足够信息，检查动作空间是否合理，尝试更简单的任务作为起点。