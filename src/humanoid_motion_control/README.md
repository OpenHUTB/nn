
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





# 软体人形机器人弹跳动作仿真（基于MuJoCo）

## 项目概述
本项目是一个基于**MuJoCo物理引擎**的软体人形机器人物理仿真示例。实际运行中，机器人会呈现**原地短暂跳起后回落蹲下**的循环动作——这是由于软体机器人的柔性特性（弹性、形变）与控制参数共同作用的结果：关节电机驱动力过强时，软体结构会储存弹性势能，释放时形成向上的弹跳力，随后在重力作用下回落至蹲下姿势，形成往复运动。


## 环境准备
### 依赖库
项目依赖以下Python库，需提前安装：
```bash
pip install mujoco mujoco-viewer
```


## 项目结构
| 文件名         | 功能描述                                                     |
| -------------- | ------------------------------------------------------------ |
| `humanoid.xml` | MJCF格式模型文件，定义软体机器人的结构、关节参数（阻尼、弹性）、关键帧姿势（深蹲、单腿站立）及电机执行器 |
| `main.py`      | 控制主程序，实现模型加载、比例控制逻辑、仿真循环及可视化，驱动机器人运动 |


## 核心文件说明
### 1. `humanoid.xml`（软体机器人模型）
模型的关键特性直接影响弹跳效果：
- **软体物理属性**：关节设置了较低阻尼（`damping`）和一定弹性，允许较大形变（为弹跳提供储能基础）；
- **关键帧姿势**：
  - 索引`0`：`squat`（深蹲姿势）—— 初始状态，关节弯曲储存势能；
  - 索引`1`：`stand_on_left_leg`（单左腿站立）—— 目标姿势，关节伸展；
- **电机参数**：`ctrlrange="-1 1"`限制控制信号范围，但软体弹性会放大实际驱动力，导致动作过冲。


### 2. `main.py`（控制逻辑）
程序通过**比例控制**驱动关节运动，是弹跳动作的直接诱因：
```python
# 核心控制逻辑：计算关节误差，生成驱动信号
qpos_error = target_qpos[7:] - data.qpos[7:]  # 目标与当前关节位置差
data.ctrl[:] = Kp * qpos_error  # 比例增益放大误差，生成电机信号
```
- **参数影响**：代码中`Kp=5.0`的比例增益，对软体机器人而言可能过大——驱动力超过维持姿势所需，导致关节快速伸展，软体结构弹性形变加剧，最终形成“跳起”动作；
- **循环机制**：跳起后重力主导回落，关节位置偏离目标，控制信号再次触发驱动力，形成“跳起→蹲下→再跳起”的循环。


## 实际运行效果解析
1. **初始状态**：机器人从深蹲姿势（`squat`）开始，关节弯曲，软体结构处于压缩状态；
2. **驱动阶段**：控制信号驱动关节向站立姿势（`stand_on_left_leg`）伸展，软体材料因快速形变储存弹性势能；
3. **弹跳阶段**：弹性势能释放，机器人获得向上的力，短暂离开初始位置（跳起）；
4. **回落阶段**：重力作用下，机器人下落，关节在冲击力和控制信号调整下重新弯曲，回到接近深蹲的姿势；
5. **循环往复**：关节位置误差再次增大，控制信号重复驱动，形成周期性弹跳动作。


## 使用步骤
1. 确认`humanoid.xml`与`main.py`在同一目录（默认路径：`src/humanoid_motion_control/`）；
2. 终端执行命令启动仿真：
   ```bash
   python main.py
   ```
3. 可视化窗口将显示机器人的弹跳动作，按`Ctrl+C`终止仿真。


## 参数调整指南（优化动作）
若希望减弱弹跳、实现更平稳的蹲站动作，可调整以下参数：

| 参数名   | 调整方向                              | 效果说明                                                     |
| -------- | ------------------------------------- | ------------------------------------------------------------ |
| `Kp`     | 减小至`1.0~3.0`                       | 降低比例增益，减弱电机驱动力，避免软体结构过度形变和弹性势能积累 |
| 关节阻尼 | 在`humanoid.xml`中增大关节`damping`值 | 增加软体关节的阻尼，消耗部分弹性势能，减少弹跳幅度           |
| 仿真步长 | 减小`time.sleep(0.005)`至`0.002`      | 提高仿真精度，使控制信号更细腻地响应关节位置变化             |


## 常见问题
1. **弹跳幅度过大甚至翻倒**  
   → 解决方案：将`Kp`降至`1.0`，同时在`humanoid.xml`中找到关节节点（如`<joint>`），增大`damping`参数（如从`1.0`改为`5.0`）。

2. **动作卡顿或延迟**  
   → 解决方案：检查`time.sleep()`的数值，若电脑性能较好，可减小至`0.001`以提升流畅度。

3. **模型加载失败**  
   → 解决方案：确认`main.py`中模型路径与`humanoid.xml`实际位置一致，例如修改为：
   ```python
   model = mujoco.MjModel.from_xml_path("humanoid.xml")  # 若文件在当前目录
   ```


## 参考资料
- [MuJoCo物理引擎文档](https://mujoco.readthedocs.io/)：了解软体材料参数（弹性、阻尼）的配置方法
- [比例控制算法详解](https://en.wikipedia.org/wiki/Proportional_control)：理解`Kp`对控制效果的影响

