# Carla TD3 自动驾驶训练

基于 Carla 模拟器的 TD3（Twin Delayed DDPG）强化学习自动驾驶实现。

## 文件说明

- `carla_env.py` - Carla 模拟器环境包装器
- `env_wrappers.py` - 环境包装器（跳帧、预处理、帧堆叠、动作平滑、奖励塑造）
- `td3_models.py` - Actor 和 Critic 网络定义
- `td3_agent.py` - TD3 智能体实现
- `main.py` - 训练和测试入口脚本

## 环境要求

1. **Carla 模拟器** - [Carla 官网](https://carla.org/)
2. **Python 依赖**:
   ```bash
   pip install torch numpy opencv-python gymnasium
   ```
3. **Carla Python API**: 将 Carla 安装目录下的 `PythonAPI/carla/dist/carla-<version>-py3-none-any.whl` 安装:
   ```bash
   pip install carla-<version>-py3-none-any.whl
   ```

## 使用方法

### 1. 启动 Carla 服务器

首先启动 Carla 模拟器:
```bash
./CarlaUE4.sh  # Linux
CarlaUE4.exe  # Windows
```

### 2. 运行训练脚本

```bash
python main.py
```

训练脚本会自动检测是否有训练好的模型，如果没有则开始训练。

### 3. 测试模型

如果已有训练好的模型，脚本会自动进入测试模式。

## 动作空间

- `steer` - 方向盘转角 [-1.0, 1.0]，向左为负，向右为正
- `throttle` - 油门 [0.0, 1.0]
- `brake` - 刹车 [0.0, 1.0]

## 观察空间

使用 RGB 摄像头图像，经过预处理后为 84x84 灰度图，堆叠 4 帧作为输入。

## 配置说明

- 可以在 `carla_env.py` 中修改地图（默认 Town03）
- 可以在 `main.py` 中调整训练参数（训练回合数、学习率、探索噪声等）
