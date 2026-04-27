# drone_path_learning

基于 AirSim + Gym + Stable-Baselines3 的无人机视觉导航强化学习项目。

## 功能概览
- 基于 Gym 接口封装 AirSim 环境（`airsim-env-v0`）
- 支持 `PPO`、`SAC`、`TD3`
- 支持 `depth`、`vector`、`lgmd` 观测模式
- 支持 PyQt5 训练/评估可视化
- 提供统一入口 `main.py`

## 环境准备
```bash
pip install -r requirements.txt
pip install -e gym_env
```

## 常用启动方式
### 统一入口（推荐）
```bash
python main.py
```

### 直接启动训练可视化
```bash
python scripts/start_train_with_plot.py --config configs/config_NH_center_Multirotor_3D.ini
```

### 直接启动评估可视化
```bash
python scripts/start_evaluate_with_plot.py \
  --eval-path logs/NH_center/<your_run_dir> \
  --eval-eps 50
```

可选参数：
- `--config`：评估配置文件，默认 `<eval-path>/config/config.ini`
- `--model-file`：模型文件，默认 `<eval-path>/models/model_sb3.zip`
- `--eval-env`：覆盖 `env_name`
- `--eval-dynamics`：覆盖 `dynamic_name`

## 测试工具
```bash
python tools/test/torch_gpu_cpu_test.py
python tools/test/env_test.py --config configs/config_NH_center_Multirotor_3D.ini
```

## 训练产物
默认输出目录：
`logs/<env_name>/<timestamp>_<dynamic>_<policy>_<algo>/`

包含：
- `tb_logs/`
- `models/model_sb3.zip`
- `config/config.ini`
- `data/`