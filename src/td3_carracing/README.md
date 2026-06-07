# TD3 for CarRacing-v3

使用 TD3（Twin Delayed Deep Deterministic Policy Gradient）算法训练一个自动驾驶 agent 玩 Gymnasium 的 `CarRacing-v3` 环境。  
项目支持 CNN 图像输入，并包含完整的训练、模型保存/加载、经验回放等功能。

## 文件结构
```text
├── td3_models.py # Actor 和 Critic 网络定义
├── td3_agent.py # TD3 智能体、经验回放缓冲区
├── env_wrappers.py # 环境封装
├── main.py # 训练入口
└── models/ # 保存的模型文件（自动创建）
```

## 环境要求
- Python 3.8+
- PyTorch
- Gymnasium
- OpenCV-Python
- NumPy

## 安装依赖
```bash
pip install torch gymnasium opencv-python numpy
```

## 训练
直接运行 main.py：
```bash
python main.py
```
训练时会显示游戏窗口（render_mode="human"），每 50 个回合自动保存一次模型到 models/ 文件夹。
