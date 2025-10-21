
  # 人形机器人走廊奔跑仿真（基于dm_control）

  ## 项目概述
  本项目核心为基于**dm_control框架**构建的人形机器人走廊奔跑仿真环境。dm_control是DeepMind基于MuJoCo物理引擎开发的高级仿真工具，简化了强化学习环境的搭建、任务定义与可视化流程。本项目中，CMU人形机器人需在带随机墙壁的走廊中以目标速度奔跑，展示了dm_control在复杂运动任务仿真中的应用。

  ## 环境准备
  ### 依赖库
  需安装以下核心依赖（含dm_control及其底层依赖）：
  ```bash
  pip install mujoco dm-control absl-py
  ```
  - `dm_control`：提供环境构建、任务定义和交互可视化功能
  - `mujoco`：dm_control依赖的物理引擎核心
  - `absl-py`：用于命令行参数处理

  ## 项目结构
  | 文件名             | 功能描述                                                     |
  | ------------------ | ------------------------------------------------------------ |
  | `run_walls.py`     | **核心文件**：基于dm_control构建带墙壁的走廊环境，实现CMU人形机器人奔跑任务 |
  | `humanoid.xml`     | MJCF格式模型文件，定义机器人结构（辅助文件）                 |
  | `main_humanoid.py` | 软体机器人弹跳仿真程序                                       |

  ## 核心功能：走廊奔跑环境（run_walls.py）
  ### 1. dm_control框架组件应用
  ```python
  # dm_control核心模块使用
  from dm_control import composer  # 环境组合框架
  from dm_control.locomotion.arenas import corridors  # 走廊场景
  from dm_control.locomotion.tasks import corridors  # 走廊任务
  from dm_control.locomotion.walkers import cmu_humanoid  # 机器人模型
  from dm_control import viewer  # 可视化工具
  ```

  ### 2. 环境构建流程
  #### （1）机器人配置
  创建位置控制的CMU人形机器人，启用第一视角相机观测：
  ```python
  walker = cmu_humanoid.CMUHumanoidPositionControlled(
      observable_options={'egocentric_camera': dict(enabled=True)}
  )
  ```

  #### （2）走廊场景设计
  通过`WallsCorridor`类构建带随机墙壁的动态环境：
  ```python
  arena = corr_arenas.WallsCorridor(
      wall_gap=4.,  # 墙壁间隙宽度（机器人需穿过的空间）
      wall_width=distributions.Uniform(1, 7),  # 墙壁宽度随机化（1-7单位）
      wall_height=3.0,  # 防止翻越的墙壁高度
      corridor_width=10,  # 走廊总宽度
      corridor_length=100,  # 走廊总长度
      include_initial_padding=False  # 无初始空白区域
  )
  ```

  #### （3）任务定义
  通过`RunThroughCorridor`类定义奔跑任务规则：
  ```python
  task = corr_tasks.RunThroughCorridor(
      walker=walker,
      arena=arena,
      walker_spawn_position=(0.5, 0, 0),  # 初始位置
      target_velocity=3.0,  # 目标速度（影响奖励计算）
      physics_timestep=0.005,  # 物理仿真步长
      control_timestep=0.03  # 控制决策步长（≈33Hz）
  )
  ```

  #### （4）环境封装
  组合场景与任务，生成强化学习可用环境：
  ```python
  environment = composer.Environment(
      time_limit=30,  # 单次仿真时长30秒
      task=task,
      strip_singleton_obs_buffer_dim=True  # 简化观测数据结构
  )
  ```

  ## 使用方法
  1. 运行走廊奔跑仿真：
     ```bash
     python run_walls.py
     ```
  2. 交互操作：
     - 可视化窗口展示机器人第一视角和全局视角
     - 按`Ctrl+C`终止仿真

  ## dm_control核心特性体现
  1. **模块化设计**：通过`composer`模块分离机器人（walker）、场景（arena）和任务（task），便于独立修改
  2. **随机性支持**：使用`distributions`生成随机墙壁宽度，增强环境多样性
  3. **灵活观测配置**：通过`observable_options`轻松启用第一视角相机等观测项
  4. **高效可视化**：`viewer`模块提供实时交互界面，支持视角切换与状态查看

  ## 参数调整指南
  | 参数               | 调整范围 | 效果说明                         |
  | ------------------ | -------- | -------------------------------- |
  | `wall_gap`         | 2.0~6.0  | 减小值增加穿过难度               |
  | `target_velocity`  | 1.0~5.0  | 提高值要求机器人跑得更快         |
  | `control_timestep` | 0.01~0.1 | 减小值提高控制精度（增加计算量） |

  ## 参考资料
  - [dm_control官方文档](https://github.com/deepmind/dm_control)
  - [MuJoCo物理引擎文档](https://mujoco.readthedocs.io/)
  - [CMU人形机器人模型说明](https://github.com/deepmind/dm_control/tree/main/dm_control/locomotion/walkers)

