# **双足行走机器人****
****PPO 训练项目**

基于近端策略优化（PPO）算法，在 BipedalWalker 环境中实现标准模式与硬核模式双足行走智能体训练

算法Proximal Policy Optimization

环境BipedalWalker-v3 / Hardcore

框架Stable Baselines3 + Gymnasium

日期2026 年 6 月

## 目录

0.关于 Bipedal Walker 环境

1.项目结构

2.训练流程

3.环境配置

4.模型评估

5.训练日志与分析

6.改进方向

7.安装依赖

8.致谢

## 核心指标

标准模式最优奖励

248

平均奖励分

硬核最长训练步数

700万

训练步数

Section 0

## **关于 Bipedal Walker 环境**

Bipedal Walker 是基于 Box2D 物理引擎开发的经典强化学习环境，由 Oleg Klimov 开发并集成于 OpenAI Gym（现为 Gymnasium）。该环境模拟了一个双足机器人在二维侧视地形中的行走任务，要求智能体通过协调髋关节和膝关节来保持平衡，同时尽可能向前移动。

观测空间分布（24维）

机身角度/速度

6

关节角度/速度

8

LIDAR 雷达读数

10

共 24 个连续观测值，覆盖姿态、速度与感知信息

环境关键参数

标准模式 BipedalWalker-v3硬核模式 BipedalWalkerHardcore-v3前进奖励 ≈ 248 ± 112!

图1：标准模式（左）与硬核模式（右）环境对比示意图

奖励机制设计：智能体每向前移动一步获得约 +1 奖励，使用关节扭矩消耗负惩罚（-0.00035×扭矩²），摔倒后立即获得 -100 惩罚并终止回合。成功通过整个赛道可额外获得约 +300 奖励。

Section 1

## **项目结构**

本项目采用模块化设计，核心代码分为两个主要文件，辅以日志、模型和视频三个输出目录。

项目根目录BipedalWalker-PPO/main.py训练主循环env_utils.py环境工具脚本logs/训练日志models/ & videos/PPO 模型训练Stable Baselines3make_env()observe_model()模型文件评估报告

图2：项目模块依赖与数据流向图

核心代码文件

2

main.py + env_utils.py

输出目录

3

logs / models / videos

训练模式

2

标准模式 + 硬核模式

核心函数

2

make_env + observe_model

Section 2

## **训练流程**

本项目使用 Proximal Policy Optimization（PPO） 算法，这是一种基于策略梯度的 Actor-Critic 方法，通过裁剪目标函数限制策略更新幅度，在训练稳定性与样本效率之间取得良好平衡。

PPO 算法核心流程

采样Actor 与环境交互计算优势GAE 估计 Aₜ裁剪目标min(rₜAₜ, clip·Aₜ)梯度更新Adam 优化器策略更新完成进入下一轮采样循环迭代

#### 标准模式训练

训练步数：100 万步

环境：BipedalWalker-v3

并行向量化环境加速

奖励归一化（VecNormalize）

帧堆叠（最近4帧）

视频录制（每1000步）

MLP 策略网络

主要挑战

#### 硬核模式训练

训练步数：500 万步起（最高700万）

环境：BipedalWalkerHardcore-v3

包含阶梯、坑洞、树桩障碍

相同训练优化技术

更复杂地形需更多探索

更长训练周期

性能收益边际递减

### 关键优化技术

并行环境

DummyVecEnv

多实例并行训练

观测归一化

VecNormalize

均值方差标准化

帧堆叠

×4

时序上下文感知

裁剪系数 ε

0.2

PPO clip 默认值

Section 3

## **环境配置**

环境配置通过 env_utils.py 中的两个核心函数实现，提供灵活的训练与评估环境搭建能力。

### 3.1 make_env() 函数

负责创建并配置训练/评估环境，支持多种可配置选项，实现了从环境创建到观测归一化的完整配置流程。

make_env() 包装器堆叠结构

BipedalWalker-v3 / BipedalWalkerHardcore-v3（原始环境）Monitor 包装器（记录奖励、回合长度）DummyVecEnv（向量化并行环境）VecNormalize（观测 & 奖励归一化，clip_obs=10.0）VecFrameStack（帧堆叠 n=4）RecordVideo（每1000步录制，可选）底层顶层

# make_env() 示例调用from env_utils import make_env

# 创建标准模式训练环境

env = make_env(

env_name="BipedalWalker-v3",

hardcore=False,

record_video=True,

use_monitor=True,

n_stack=4,

clip_obs=10.0

)

# 创建硬核模式训练环境

env_hc = make_env(

env_name="BipedalWalker-v3",

hardcore=True,

record_video=True,

use_monitor=True

)

### 3.2 observe_model() 函数

负责加载训练好的 PPO 模型，并在评估环境中运行，自动适配训练时所用的 VecNormalize 和 VecFrameStack 包装器，确保评估行为与训练一致。

1

模型加载

从指定路径加载训练完成的 PPO 模型文件（.zip 格式），使用 Stable Baselines3 的 PPO.load() 接口。

2

环境配置

根据 hardcore 参数自动选择对应环境，以 human 或 rgb_array 渲染模式初始化。

3

包装器适配

自动恢复 VecNormalize 统计参数（均值、方差），加载对应的 VecFrameStack 配置，确保评估与训练分布一致。

4

评估执行

在 n_eval_episodes 个回合内运行模型，返回 平均奖励 与 奖励标准差。

# observe_model() 示例调用from env_utils import observe_model

mean_reward, std_reward = observe_model(

model_path='models/ppo_bipedalwalker_1M',

n_eval_episodes=5,

hardcore=False

)print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")

Section 4

## **模型评估**

模型评估通过 observe_model() 函数完成。以下是各训练阶段的评估结果，可以清晰地看出随训练步数增加，智能体性能的渐进提升趋势。

各模式 / 训练阶段平均奖励对比

标准模式（100万步）硬核模式

硬核模式奖励随训练步数变化（趋势）

硬核 - 300 万步

-28.23

硬核 - 500 万步

-10.66

硬核 - 700 万步

-5.45

标准 - 100 万步

+248.39

注：以标准模式 248.39 为 100% 基准，负值表示仍在收敛中

分析：标准模式下智能体已能较稳定行走（奖励 248.39），接近通关阈值（300）。硬核模式因地形复杂、障碍多样，即使经过 700 万步训练，奖励仍为负值，说明障碍应对策略尚需继续优化。但从 300 万到 700 万步的负奖励绝对值逐步缩小（-28 → -5.45），显示训练方向正确，标准差也明显收窄，说明策略趋于稳定。

Section 5

## **训练日志与分析**

本节对 500 万步硬核模式训练日志进行深度分析，使用 pandas 和 matplotlib 对关键指标进行可视化，揭示训练过程中的规律与趋势。

模拟回合长度变化趋势（500万步硬核模式）

原始回合长度50步移动平均

奖励分布（500万步硬核）

奖励与回合长度相关性

奖励与回合长度相关系数

0.89

强正相关

奖励整体趋势

上升

波动中趋于稳定

500万步最终平均奖励

-10.66

较300万提升62%

标准差收窄

24→3.9

策略逐步稳定

关键结论：奖励值与回合长度呈现显著正相关（相关系数 0.89），说明智能体在环境中存活时间越长，获得的奖励也越高。这验证了生存时间是本任务最核心的性能指标之一，优化存活策略应作为首要优化目标。

### 分析代码示例

import pandas as pdimport matplotlib.pyplot as plt

# 加载训练日志

df = pd.read_csv('logs/monitor.csv', skiprows=1)

df.columns = ['reward', 'ep_length', 'time']

# 移动平均平滑处理

df['reward_ma'] = df['reward'].rolling(window=50).mean()

df['ep_len_ma'] = df['ep_length'].rolling(window=50).mean()

# 计算相关系数

corr = df['reward'].corr(df['ep_length'])print(f"相关系数: {corr:.2f}")  # → 0.89

Section 6

## **改进方向**

基于当前实验结果，提出以下四个主要优化方向，帮助进一步提升硬核模式下的智能体性能。

#### 调整学习率

降低学习率（如从 3e-4 调整为 1e-4）可减少训练震荡，使策略更新更平稳，提升收敛稳定性。

#### 重构奖励函数

调整奖励权重，增加平衡与生存奖励比重，减少单纯前进奖励权重，引导智能体优先学习稳定运动模式。

#### 增强探索能力

引入好奇心驱动探索（ICM）或最大熵强化学习方法，帮助智能体探索更多样化的运动策略。

#### 延长训练周期

增加训练至 1000 万步以上，继续挖掘硬核模式性能潜力，观察奖励是否能进入正值区间。

额外建议：可考虑引入课程学习（Curriculum Learning）策略，先在简单地形训练，逐步增加障碍难度；也可尝试更强的策略网络架构（如 LSTM 处理时序信息），或使用 SAC、TD3 等 off-policy 算法提升样本效率。

Section 7

## **安装依赖**

使用项目提供的 requirements.txt 文件一键安装所有依赖：

# 一键安装所有依赖

pip install -r requirements.txt

环境建议：推荐在 Python 虚拟环境（venv 或 conda）中安装依赖，避免包版本冲突。GPU 用户可安装 PyTorch CUDA 版本以加速训练。

Section 8

## **致谢**

本项目基于以下开源项目与工具完成，在此表示诚挚感谢：

Bipedal Walker 环境

由 Oleg Klimov 开发，基于 Box2D 物理引擎实现。作为强化学习连续控制领域的经典 Benchmark，为本项目提供了标准化的训练与评估平台。

Stable Baselines3

由 Antonin Raffin 等人开发的高质量强化学习算法库，提供了稳定、易用的 PPO 实现，支持向量化环境、归一化包装器等关键特性。

Gymnasium（OpenAI Gym）

提供了标准化的强化学习环境接口，使算法与环境的解耦实现变得简单，大幅降低了强化学习实验的开发门槛。

PyTorch + NumPy + Matplotlib

深度学习框架与科学计算工具链，为模型训练、数值计算和结果可视化提供了坚实的技术支撑。

| 参数 | 标准 | 硬核 |
| --- | --- | --- |
| 动作空间 | 4维连续（关节扭矩） | 4维连续（关节扭矩） |
| 观测空间 | 24维连续 | 24维连续 |
| 最大步数 | 1600 | 2000 |
| 地形难度 | 平坦 | 障碍+坑洞 |
| 过关奖励 | +300 | +300 |

| 训练模式 | 训练步数 | 平均奖励 | 标准差 | 状态 |
| --- | --- | --- | --- | --- |
| 标准模式 | 100 万 | +248.39 | ± 112.10 | 良好 |
| 硬核模式 | 300 万 | -28.23 | ± 24.82 | 早期 |
| 硬核模式 | 500 万 | -10.66 | ± 3.91 | 提升中 |
| 硬核模式 | 700 万 | -5.45 | ± 2.10 | 趋势良好 |

| 依赖包 | 版本要求 | 用途 |
| --- | --- | --- |
| Python | 3.8+ | 运行环境 |
| gymnasium | ≥ 0.26 | 强化学习环境接口（BipedalWalker） |
| gymnasium[box2d] | 随 gymnasium | Box2D 物理引擎支持 |
| stable-baselines3 | ≥ 2.0 | PPO 算法实现库 |
| pandas | ≥ 1.5 | 日志数据处理与分析 |
| matplotlib | ≥ 3.5 | 训练曲线可视化 |
| numpy | ≥ 1.22 | 数值计算基础 |
| torch | ≥ 1.13 | SB3 深度学习后端 |
