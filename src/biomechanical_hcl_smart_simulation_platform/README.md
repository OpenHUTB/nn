# User-in-the-Box
## 项目简介
本项目提供基于 **MuJoCo 物理引擎** 的人机交互（HCI）任务建模与仿真源码，核心是构建具有肌肉驱动能力的生物力学用户模型（支持自我中心视觉等感知功能），并通过强化学习（RL）训练模型完成各类交互任务。项目采用模块化设计，可灵活扩展新的生物力学模型、感知模型及交互任务；生成的模拟器继承 OpenAI Gym 接口，能直接接入现有 RL 训练库，且以独立包形式存在，便于共享与协作。


## 1. 核心功能与关联论文
### 1.1 核心能力
- 生物力学建模：支持肌肉驱动的人体交互模型（如手臂、手指），可通过工具从 OpenSim 格式转换至 MuJoCo。
- 多模态感知：集成视觉（自我中心相机）、本体感觉（关节位置/速度/加速度）等感知模块。
- 灵活任务扩展：支持自定义交互任务（如指向、追踪、遥控车控制），需定义 MuJoCo XML 环境与 Python 包装类。
- 强化学习训练：基于 stable-baselines3 实现训练流程，支持 Weights & Biases 日志记录，支持断点续训与定期评估。

### 1.2 关联论文
若使用本项目开展研究，建议引用以下论文：
#### （1）Breathing Life into Biomechanical User Models（UIST 2022）
- **论文链接**：[https://dl.acm.org/doi/abs/10.1145/3526113.3545689](https://dl.acm.org/doi/abs/10.1145/3526113.3545689)
- **演示视频**：[Youtube（含字幕）](https://youtu.be/-L2hls8Blyc) | [备用链接](https://user-images.githubusercontent.com/7627254/184347198-2d7f8852-d50b-457f-8eaa-07720b9522eb.mp4)
- **引用格式**：
```
@inproceedings{ikkala2022,
author = {Ikkala, Aleksi and Fischer, Florian and Klar, Markus and Bachinski, Miroslav and Fleig, Arthur and Howes, Andrew and H\"{a}m\"{a}l\"{a}inen, Perttu and M\"{u}ller, J\"{o}rg and Murray-Smith, Roderick and Oulasvirta, Antti},
title = {Breathing Life Into Biomechanical User Models},
year = {2022},
isbn = {9781450393201},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3526113.3545689},
doi = {10.1145/3526113.3545689},
booktitle = {Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology},
articleno = {90},
numpages = {14},
location = {Bend, OR, USA},
series = {UIST '22}}
```

#### （2）Converting Biomechanical Models from OpenSim to MuJoCo（ICNR 2020）
- **论文链接**：[https://arxiv.org/abs/2006.10618](https://arxiv.org/abs/2006.10618)
- **核心作用**：项目中 `mobl_arms` 及 `mobl_arms_index` 生物力学模型，均通过该论文提出的 [O2MConverter 工具](https://github.com/aikkala/O2MConverter) 从 OpenSim 转换至 MuJoCo（部分参数已优化）。


## 2. 环境要求与安装
### 2.1 环境要求
- Python 3.x
- MuJoCo（参考 [官方文档](https://mujoco.org/) 安装）
- 依赖库：stable-baselines3、gymnasium、numpy、PyYAML、Weights & Biases、opencv-python（感知模块）、matplotlib（可视化）

### 2.2 安装步骤
#### 方式1：基于 Conda 环境（推荐）
```bash
# 从配置文件创建环境（含所有依赖）
conda env create -f conda_env.yml
# 激活环境
conda activate uitb
```

#### 方式2：基于 Pip 安装
```bash
# 常规安装（主目录下执行）
pip install .
# 可编辑模式（开发调试用，修改代码无需重新安装）
pip install -e .
```

### 2.3 特殊配置（无头渲染）
若在 Jupyter Notebook/Google Colab 等无图形界面场景使用，需设置 EGL 为渲染后端（依赖 GPU）：
- 命令行设置：
  ```bash
  export MUJOCO_GL=egl
  ```
- Jupyter Notebook 中设置（IPython 魔法命令）：
  ```python
  %env MUJOCO_GL=egl
  ```


## 3. 文件结构
```
.
├── uitb/  # 核心源码目录
│   ├── simulator.py  # 模拟器主类（实现 OpenAI Gym 接口，核心入口）
│   ├── bm_models/  # 生物力学模型目录
│   │   ├── base.py  # 生物力学模型基类（新模型需继承此类）
│   │   ├── effort_models.py  # 预定义力模型（WIP，可扩展）
│   │   └── mobl_arms/  # 示例模型（从 OpenSim 转换的手臂模型）
│   ├── tasks/  # 交互任务目录
│   │   ├── base.py  # 任务基类（新任务需继承此类）
│   │   ├── unity/  # Unity VR 交互任务（v2.0+，依赖 SIM2VR）
│   │   └── pointing.py  # 示例任务（指向任务）
│   ├── perception/  # 感知模型目录
│   │   ├── base.py  # 感知模块基类（新模块需继承此类）
│   │   ├── vision/  # 视觉感知子目录（含自我中心相机、Unity 头显模块）
│   │   └── proprioception/  # 本体感觉子目录（关节状态感知）
│   ├── configs/  # 模拟器配置文件（YAML 格式，定义模型组合）
│   │   ├── mobl_arms_index_pointing.yaml  # 示例配置（手臂指向任务）
│   │   └── mobl_arms_index_tracking.yaml  # 示例配置（手臂追踪任务）
│   ├── train/  # 训练相关代码
│   │   └── trainer.py  # 训练脚本（调用 stable-baselines3，支持日志与续训）
│   ├── rl/  # 强化学习模块目录
│   │   └── base.py  # RL 模型基类（新 RL 库需继承此类）
│   └── test/  # 测试相关代码
│       └── evaluator.py  # 模拟器评估脚本（计算性能、保存视频/日志）
├── simulators/  # 生成的独立模拟器目录（构建后自动创建）
│   └── mobl_arms_index_pointing/  # 示例模拟器（指向任务，可直接共享）
├── figs/  # 示意图与 GIF 演示目录
│   ├── architecture.svg  # 软件架构图（三大组件关系）
│   └── mobl_arms_index/  # 示例任务 GIF（指向、追踪等）
├── conda_env.yml  # Conda 环境配置文件
└── README.md  # 项目说明文档（本文档）
```


## 4. 使用说明
### 4.1 构建模拟器
通过 YAML 配置文件定义生物力学模型、任务、感知模块的组合，生成独立模拟器包：
```python
from uitb import Simulator

# 1. 定义配置文件路径（从 uitb/configs 中选择或自定义）
config_file = "uitb/configs/mobl_arms_index_pointing.yaml"

# 2. 构建模拟器（输出目录：simulators/[配置中定义的模拟器名]）
simulator_folder = Simulator.build(config_file)
# 注意：配置文件中 "simulator_name" 需符合 Python 包命名规范（用下划线代替短横线）
```

### 4.2 运行模拟器
生成的模拟器兼容 OpenAI Gym 接口，支持两种初始化方式：

#### 方式1：直接通过 Simulator 类加载
```python
# 从构建后的目录加载模拟器
simulator = Simulator.get(simulator_folder)

# 运行流程（同 Gym 环境）
obs = simulator.reset()  # 重置环境，获取初始观测
done = False
while not done:
    action = simulator.action_space.sample()  # 随机采样动作（实际用 RL 模型输出）
    obs, reward, terminated, truncated, info = simulator.step(action)  # 执行一步交互
    done = terminated or truncated
```

#### 方式2：通过 gymnasium 加载（需添加 Python 路径）
```python
import sys
import gymnasium as gym
import importlib

# 1. 将模拟器目录加入 Python 路径
sys.path.insert(0, simulator_folder)

# 2. 导入模拟器模块（模块名 = 配置中的 simulator_name）
importlib.import_module("mobl_arms_index_pointing")

# 3. 用 gymnasium 初始化（需加前缀 "uitb:" 和后缀 "-v0"，符合 Gym 命名规范）
simulator = gym.make("uitb:mobl_arms_index_pointing-v0")
```


## 5. 训练与测试
### 5.1 模型训练
使用 `uitb/train/trainer.py` 脚本启动训练，输入为 YAML 配置文件，支持日志、续训与定期评估：

#### 基本命令
```bash
# 基础训练（指定配置文件）
python uitb/train/trainer.py --config uitb/configs/mobl_arms_index_pointing.yaml

# 断点续训（加载最新 checkpoint）
python uitb/train/trainer.py --config uitb/configs/mobl_arms_index_pointing.yaml --resume

# 定期评估（每 1000 步评估一次，记录自定义指标）
python uitb/train/trainer.py --config uitb/configs/mobl_arms_index_pointing.yaml --eval 1000 --eval_info_keywords inside_target target_hit
```

#### 关键参数说明
- `--config`：指定模拟器配置文件路径（必选）。
- `--resume`：加载最新 checkpoint 续训（可选）。
- `--checkpoint <path>`：加载指定 checkpoint 续训（可选）。
- `--eval <step>`：每 `<step>` 步执行一次评估（可选）。
- `--eval_info_keywords`：评估时记录的自定义指标（需在任务的 `step` 函数 info 中返回，可选）。

#### 训练注意事项
- 奖励函数：自定义任务时需在任务类中实现 `compute_reward` 方法。
- 代码修改：禁止直接修改 `simulators/` 目录下的代码，需修改 `uitb/` 下的原始源码（如新增任务后在 `uitb/tasks/__init__.py` 注册）。

### 5.2 模拟器测试
使用 `uitb/test/evaluator.py` 评估训练后的模拟器性能，支持保存视频与日志：

#### 基本命令
```bash
# 评估指向任务模拟器（10 个 episode，录制视频、保存日志，100Hz 动作采样）
python uitb/test/evaluator.py simulators/mobl_arms_index_pointing --num_episodes 10 --record --logging --action_sample_freq 100
```

#### 关键参数说明
- `simulators/xxx`：指定待评估的模拟器目录（必选）。
- `--num_episodes`：评估的 episode 数量（可选，默认 5）。
- `--record`：录制每个 episode 的视频（可选，保存至模拟器的 `evaluate/` 目录）。
- `--logging`：保存评估日志（可选，含观测、动作、奖励等数据）。
- `--action_sample_freq`：动作采样频率（Hz，可选，默认 50）。

#### 输出说明
- 视频与日志默认保存至 `simulators/[模拟器名]/evaluate/` 目录。
- 运行 `python uitb/test/evaluator.py --help` 查看所有参数。


## 6. 预训练模拟器总结
### 6.1 预训练模拟器介绍
本子项目基于 User-in-the-Box 框架，是其中一个已经训练完成的预训练模拟器：mobl_arms_index_remote_driving – Controlling an RC Car via Joystick。
在该模拟器中：“用户”被建模为一个带肌肉驱动的上肢生物力学模型（mobl_arms_index），拥有视觉 / 本体感觉等感知能力；
“交互设备”是一个虚拟 RC 小车 + 手柄环境，用户通过操纵手柄（Joystick）来控制小车移动；
整个环境遵循 OpenAI Gym 接口，可以与现有强化学习库（如 stable-baselines3）直接对接。

### 6.1 示例演示（GIF）
- 指向任务：https://imgur.com/a/CHzVeBK
- 追踪任务：https://imgur.com/a/CHzVeBK
- 选择反应任务：https://imgur.com/a/CHzVeBK
- 遥控车任务：https://imgur.com/a/CHzVeBK

### 6.3 操作流程
1. 克隆仓库并安装依赖  
   ```bash
   git clone <your-user-in-the-box-repo-url>
   cd user-in-the-box

   # 创建并启用 Conda 环境（推荐）
   conda env create -f conda_env.yml
   conda activate uitb

   # 安装 Python 包（可编辑模式，方便后续修改）
   pip install -e .

2. 构建 RC Car 预训练模拟器
   python - << 'EOF'
   from uitb import Simulator
   config_file = "uitb/configs/mobl_arms_index_remote_driving.yaml"
   sim_folder = Simulator.build(config_file)
   print("Built simulator at:", sim_folder)
   EOF

运行后会在 simulators/mobl_arms_index_remote_driving/ 下生成独立模拟器包。

3. 用随机策略测试环境是否正常工作
   
python - << 'EOF'
from uitb import Simulator

sim_folder = "simulators/mobl_arms_index_remote_driving"
env = Simulator.get(sim_folder)

obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print(f"Episode finished at step {step}, resetting...")
        obs, info = env.reset()

env.close()
EOF
      

4. 使用评估脚本查看预训练策略表现并录制视频
  
   python uitb/test/evaluator.py simulators/mobl_arms_index_remote_driving \
  --num_episodes 10 \
  --record \
  --logging \
  --action_sample_freq 100

视频和日志保存在 simulators/mobl_arms_index_remote_driving/evaluate/ 目录下。

## 7. 扩展开发指南
### 7.1 新增生物力学模型
1. 准备模型文件：
   - 构建 MuJoCo XML 文件（可通过 O2MConverter 从 OpenSim 转换）。
   - 在 `uitb/bm_models/` 下创建新文件夹（如 `my_arm/`），存放 XML 文件与 Python 类。
2. 实现 Python 类：
   ```python
   from uitb.bm_models.base import BaseBMModel

   class MyArmModel(BaseBMModel):
       def __init__(self, xml_path):
           super().__init__(xml_path)
           # 自定义初始化逻辑（如关节限位、肌肉参数设置）
   ```
3. 在 `uitb/bm_models/__init__.py` 中注册新模型：
   ```python
   from .my_arm import MyArmModel
   ```

### 7.2 新增交互任务
1. 准备任务文件：
   - 构建 MuJoCo XML 文件（定义任务环境，如目标物体、交互设备）。
   - 在 `uitb/tasks/` 下创建 Python 文件（如 `my_task.py`）。
2. 实现 Python 类（需继承 `BaseTask`，并实现奖励函数）：
   ```python
   from uitb.tasks.base import BaseTask

   class MyTask(BaseTask):
       def __init__(self, xml_path):
           super().__init__(xml_path)
           # 自定义初始化逻辑（如目标位置重置）
       
       def compute_reward(self, obs, action, done):
           # 实现奖励计算逻辑（如目标达成得正奖、超时得负奖）
           target_dist = obs["target_distance"]
           return 1.0 if target_dist < 0.01 else -0.01
   ```
3. 在 `uitb/tasks/__init__.py` 中注册新任务：
   ```python
   from .my_task import MyTask
   ```

### 7.3 新增感知模块
1. 在 `uitb/perception/` 下对应子目录（如 `vision/`）创建 Python 文件（如 `my_camera.py`）。
2. 实现 Python 类（需继承 `BaseModule`，并实现感知数据采集）：
   ```python
   from uitb.perception.base import BaseModule

   class MyCameraModule(BaseModule):
       def __init__(self, simulator):
           super().__init__(simulator)
           # 初始化相机参数（如分辨率、视角）
       
       def get_observation(self):
           # 采集相机图像并返回（需符合模拟器观测空间定义）
           img = self.simulator.render(camera_name="my_cam")
           return {"camera_img": img}
   ```
3. 在 `uitb/perception/__init__.py` 中注册新模块：
   ```python
   from .vision.my_camera import MyCameraModule
   ```


## 8. TODO 列表
- 完善相机与照明系统的自定义配置（当前需修改 MuJoCo XML，后续支持动态参数调整）。
- 拆分 `Task` 类为 `World`（环境定义）与 `Task`（任务逻辑），优化职责划分。
- 扩展更多生物力学模型（如完整人体模型、手指精细操作模型）。


