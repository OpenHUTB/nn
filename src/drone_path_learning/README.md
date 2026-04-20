# drone_path_learning

## 项目介绍

本项目是一个基于 AirSim + Gym + Stable-Baselines3 的无人机视觉导航强化学习工程，目标是在仿真环境中训练多旋翼无人机完成避障与目标到达任务。

核心特性如下：

- 以 `gym` 标准接口封装 AirSim 环境（`airsim-env-v0`），可直接对接 SB3 算法。
- 支持 `PPO`、`SAC`、`TD3` 三类主流 DRL 算法。
- 支持多种观测模式：深度图像（`depth`）、向量特征（`vector`）和 LGMD 特征（`lgmd`）。
- 提供 PyQt5 实时可视化界面，训练/评估时可查看动作、状态、姿态、奖励和轨迹。
- 提供统一入口脚本 `main.py`，可通过菜单启动训练、评估和基础测试。

运行依赖（概览）：

- Python `>=3.10`
- AirSim Python API
- Gym `0.17.3`
- Stable-Baselines3（PPO/SAC/TD3）
- PyTorch
- PyQt5 / pyqtgraph

建议先确保：

- AirSim 场景已启动并可连接。
- NVIDIA 驱动与 CUDA（如需 GPU）可用。

---

## 目录下各文件作用

### `configs/`

- `configs/config_NH_center_Multirotor_3D.ini`
	- 训练/评估的主配置文件。
	- 包含环境、观测、算法、策略网络、奖励、动作约束等参数。

### `gym_env/`

- `gym_env/setup.py`
	- 本地 Gym 环境包安装脚本。

- `gym_env/gym_env/__init__.py`
	- 注册环境 `airsim-env-v0` 到 Gym。

- `gym_env/gym_env/envs/__init__.py`
	- 导出 `AirsimGymEnv`。

- `gym_env/gym_env/envs/airsim_env.py`
	- 项目核心环境类 `AirsimGymEnv`。
	- 主要职责：
		- 读取配置并构建具体动力学模型（当前主线为多旋翼）。
		- 实现 `reset()` / `step()` / `observation_space` / `action_space`。
		- 生成观测（图像、向量、LGMD）。
		- 计算奖励（多种 reward 函数）。
		- 判断终止条件（到达、碰撞、越界、步数上限）。
		- 通过 PyQt 信号向 UI 推送状态用于实时绘图。

- `gym_env/gym_env/envs/dynamics/multirotor_airsim.py`
	- AirSim 多旋翼动力学控制封装。
	- 主要职责：
		- 建立 AirSim 客户端连接并控制起飞/复位。
		- 将 RL 动作映射为速度与偏航角速度控制命令。
		- 维护起点/目标点采样逻辑。
		- 提供状态特征计算（距离、偏航误差、速度等）与归一化。

### `scripts/`

- `scripts/start_train_with_plot.py`
	- 带 GUI 可视化的训练启动脚本。
	- 创建 `TrainingUi` 和 `TrainingThread`，将训练过程信号连接到界面。
	- 当前默认读取：`configs/config_NH_center_Multirotor_3D.ini`。

- `scripts/start_evaluate_with_plot.py`
	- 带 GUI 可视化的评估启动脚本。
	- 创建 `TrainingUi` + `EvaluateThread`，加载指定模型并评估。
	- 评估路径在脚本中默认写死，需要按实际日志目录修改。

- `scripts/train.py`
	- 早期训练脚本（直接创建 env 与模型）。
	- 可作为训练逻辑参考，不是当前主推荐入口。

- `scripts/evaluation.py`
	- 早期评估脚本（单模型循环推理）。
	- 模型路径为硬编码，主要用于历史兼容与快速实验。

### `scripts/utils/`

- `scripts/utils/thread_train.py`
	- 主训练线程实现（`QThread`）。
	- 根据配置选择策略网络和算法，创建日志目录并执行 `model.learn()`。
	- 输出内容包括：模型文件、配置快照、TensorBoard 日志。

- `scripts/utils/thread_evaluation.py`
	- 主评估线程实现（`QThread`）。
	- 加载模型，运行多回合评估，保存轨迹/动作/状态/观测和结果指标。

- `scripts/utils/thread_train_repeat.py`
	- 重复训练版本（多随机种子循环训练）以对比稳定性。

- `scripts/utils/custom_policy_sb3.py`
	- 自定义特征提取器集合（供 SB3 Policy 使用）。
	- 包含 `No_CNN`、`CNN_FC`、`CNN_GAP`、`CNN_GAP_BN`、`CNN_MobileNet` 等。

- `scripts/utils/ui_train.py`
	- PyQt5 训练可视化界面。
	- 绘制动作、状态、姿态、奖励、轨迹，以及可选 LGMD 曲线。

### `tools/`

- `tools/test/torch_gpu_cpu_test.py`
	- PyTorch/CUDA 环境检查工具，快速确认 GPU 是否可用于训练。

- `tools/test/env_test.py`
	- AirSim 环境通路测试工具，固定动作步进并输出 FPS。

- `tools/map_generation/map_generation.py`
	- 调用 AirSim API 生成体素地图（`map.binvox`）的辅助脚本。

---

## 快速启动方式

### 1. 环境准备

在项目根目录执行：

```bash
pip install -r requirements.txt

# 安装本地 gym 环境包（必需）
pip install -e gym_env
```
# 安装CUDA（如需 GPU 加速）：
- 确保 NVIDIA 驱动已安装。
https://developer.nvidia.com/cuda-downloads
- 安装 CUDA Toolkit（版本需与 PyTorch 兼容）。
```
### 1. 启动 AirSim

- 先启动 Unreal/AirSim 场景。
- 确保场景处于可连接状态（脚本会调用 `airsim.MultirotorClient().confirmConnection()`）。

### 2. 一键入口（推荐）

```bash
python main.py
```

将看到菜单：

- `1` 训练（可视化）
- `2` 评估（可视化）
- `3` Torch/CUDA 检查
- `4` 环境快速测试

也可以直接指定模式：

```bash
python main.py train
python main.py eval
python main.py torch_check
python main.py env_test
```

### 4. 常用直接命令

```bash
# 可视化训练
python scripts/start_train_with_plot.py

# 可视化评估
python scripts/start_evaluate_with_plot.py

# 环境快速测试（可指定配置）
python tools/test/env_test.py --config configs/config_NH_center_Multirotor_3D.ini

# GPU/CUDA 检查
python tools/test/torch_gpu_cpu_test.py
```

### 5. 训练产物目录（默认）

训练线程默认会在 `logs/<env_name>/<timestamp>_<dynamic>_<policy>_<algo>/` 下生成：

- `tb_logs/`：TensorBoard 日志
- `models/model_sb3.zip`：训练后模型
- `config/config.ini`：运行时配置快照
- `data/`：扩展数据（如 Q-map）

### 6. 评估输出目录（默认）

评估线程会在模型目录下创建：

- `eval_<回合数>_<环境>_<动力学>/results.npy`：汇总指标
- `traj_eval.npy`、`action_eval.npy`、`state_eval.npy`、`obs_eval.npy`：逐回合数据

### 7. 常见注意事项

- `scripts/start_evaluate_with_plot.py` 内默认 `eval_path` 为硬编码示例路径，使用前请改成你的实际日志目录。
- `scripts/train.py`、`scripts/evaluation.py` 属于早期脚本，推荐优先使用 `main.py` 或 `start_*_with_plot.py`。
- 若 TensorBoard 显示无曲线，通常是训练尚未完成一个完整 rollout/update，或日志文件仅包含初始化信息。

