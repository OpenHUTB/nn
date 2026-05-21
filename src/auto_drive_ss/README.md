# 🚘 自监督多模态感知+强化学习自动驾驶
# Self-Supervised Multi-Modal Perception + Reinforcement Learning for Autonomous Driving

本项目构建了一套高鲁棒性自动驾驶流程框架，融合**自监督学习（SSL）** 完成环境感知，结合**近端策略优化（PPO）** 实现CARLA仿真环境中的决策规划。项目以RGB图像、深度图、语义分割图像为多模态输入，融合生成统一特征表征，驱动强化学习智能体在复杂交通场景中完成自主导航。
This project presents a robust autonomous driving pipeline that combines **Self-Supervised Learning (SSL)** for perception and **Proximal Policy Optimization (PPO)** for decision-making in the CARLA simulator. We leverage RGB, Depth, and Semantic Segmentation inputs fused into a unified representation that drives a reinforcement learning agent to navigate complex environments.

---

## 📌 核心亮点 Key Features
- 🔀 **多模态感知**：融合RGB图像、深度信息、语义分割多模态数据
  **Multi-modal Perception**: Fusion of RGB, depth, and segmentation modalities.
- 🧠 **自监督学习**：通过对比损失与重建损失学习高层视觉特征嵌入
  **Self-Supervised Learning**: Contrastive loss and reconstruction loss for learning high-level visual embeddings.
- 🕹️ **深度强化学习**：基于感知特征嵌入训练PPO智能体，实现CARLA仿真控制
  **Deep Reinforcement Learning**: PPO agent trained with perception embeddings for control in CARLA.
- 🔁 **时序一致性建模**：感知系统引入GRU网络完成时序特征建模
  **Temporal Consistency**: GRU-based temporal modeling in the perception system.
- 🧪 **鲁棒奖励函数**：对碰撞、车道偏离、越界行为施加惩罚；对平稳转弯、合理车速给予奖励
  **Robust Reward Function**: Penalizes collisions, lane deviation, off-road behavior; rewards smooth turns and optimal speed.

---

## 🧱 整体架构概览 Architecture Overview
```
输入 → 编码器（ResNet/DeepLab） → 融合Transformer → 时序GRU网络
→ 自监督分支（对比学习+图像重建） → 256维感知特征嵌入 → PPO策略网络
```
```
Inputs → Encoders (ResNet/DeepLab) → Fusion Transformer → Temporal GRU
→ SSL heads (contrastive, reconstruction) → 256D Perception Embedding → PPO Policy
```

附加功能模块 Additional modules:
- 感知解码器：用于自监督学习监督训练
  Perception decoder for SSL supervision
- 强化学习智能体：输出连续控制量
  Reinforcement Learning agent with continuous control output

---

## 🧪 实验环境配置 Experimental Setup
- **仿真平台**：[CARLA自动驾驶仿真器](https://carla.org/)
  **Simulator**: [CARLA](https://carla.org/)
- **强化学习算法**：PPO（基于Stable-Baselines3框架）
  **RL Algorithm**: PPO (Stable-Baselines3)
- **自监督学习任务**：对比学习、像素级图像重建
  **SSL Tasks**: Contrastive learning, pixel reconstruction
- **评价指标 Evaluation Metrics**:

| 评价指标 Metric | 基线强化学习 Baseline RL | 自监督+强化学习 SSL + RL |
|----------------|-------------------------|--------------------------|
| 累计总奖励 Total Reward | 1500 | 4200 |
| 碰撞率 Collision Rate | 4.5 | 1.2 |
| 车道偏离误差 Lane Deviation | 0.78 m | 0.32 m |
| 道路越界占比 Off-road % | 22% | 4% |

---

## 🧠 训练细节 Training Details
- **预训练**：基于CARLA多模态场景完成自监督预训练
  **Pretraining**: SSL on multi-modal CARLA scenes
- **强化学习训练**：以256维自监督特征嵌入作为PPO网络输入
  **Reinforcement Learning**: PPO using 256D SSL embeddings
- **硬件配置**：NVIDIA A100 64GB，训练时长约6天
  **Hardware**: NVIDIA A100 (64GB), training time ~6 days
- **损失函数 Losses Used**:
  - 🔹 InfoNCE 对比损失 InfoNCE contrastive loss
  - 🔹 L1 图像重建损失 L1 reconstruction loss
  - 🔹 带熵正则化的PPO损失 PPO loss with entropy regularization

---

## 📂 项目目录结构 Project Structure
```bash
├── src/
│   ├── models/               # ResNet/DeepLab、Transformer、GRU模型定义
│   ├── ssl_trainer.py        # 自监督预训练入口脚本
│   ├── rl_training_with_ssl.py  # 自监督特征融合强化学习训练
├── rl_agent/
│   ├── ppo_policy.py         # PPO策略网络与训练逻辑
│   ├── reward_function.py    # 自定义奖励函数
├── evaluation/
│   ├── metrics.py            # 模型评估指标计算
│   ├── visualize.py          # 行驶轨迹与奖励曲线可视化
├── carla_utils/
│   ├── environment_wrapper.py # CARLA环境封装工具
├── configs/
│   ├── perception.yaml      # 感知模型配置文件
│   ├── rl.yaml               # 强化学习配置文件
├── README.md
```

---

## 🛠️ 运行教程 How to Run
### 1. 克隆代码仓库 Clone the repo
```bash
git clone https://github.com/your-username/self-driving-ssl-rl.git
cd self-driving-ssl-rl
```

### 2. 配置运行环境 Setup environment
```bash
conda create -n sslrl python=3.6
conda activate sslrl
pip install -r requirements.txt
```

### 3. 执行自监督预训练 Run SSL pretraining
```bash
python perception/train_ssl.py --config configs/perception.yaml
```

### 4. 单独训练强化学习模型 Train RL
```bash
python rl_agent/ppo_policy.py --config configs/rl.yaml
```

### 5. 自监督+强化学习联合训练 Train SSL+RL
```bash
python rl_agent/ppo_policy.py --config configs/rl.yaml
```

---

## 📊 实验结果 Results
- 🚗 车辆行驶更平稳、安全性显著提升
  Smoother and safer driving behavior
- 📉 碰撞率降低至原基线的1/3.3
  3.3x reduction in collision rate
- 🚦 路口通行与转弯场景适应性大幅优化
  Improved handling at junctions and turns

---

## 🔮 未来工作展望 Future Work
- 🤖 引入人在回路机制优化奖励函数调优
  Human-in-the-loop reward tuning
- 🔁 结合环境探索实现在线自监督学习
  Online self-supervised learning with exploration
- 🧩 与基于模板的路径规划模块深度集成
  Integration with template-based planning modules

---

## 📜 引用格式 Citation
若你在学术研究或工业项目中引用本工作，可使用以下格式：
If you use or reference this work in academic or industrial projects:

```bibtex
@misc{SushmaMareddy2025sslrl,
  title={Self-Supervised Multi-Modal Perception and Reinforcement Learning for Autonomous Driving},
  author={Sushma Mareddy, Ravan Ranveer Budda},
  year={2025},
  note={Project Report, NYU Courant}
}
```
