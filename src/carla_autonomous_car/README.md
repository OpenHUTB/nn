# CARLA Frenet轨迹规划 - 自动驾驶强化学习系统

<div align="center">
  <h3>🚗 基于强化学习的自动驾驶轨迹规划系统</h3>
  <p><strong>专注于Frenet坐标系下的最优轨迹规划</strong></p>
</div>

## 📋 项目概述

本项目实现了一个完整的自动驾驶强化学习系统，专注于Frenet坐标系下的最优轨迹规划。系统集成了多种强化学习算法（PPO2、A2C、DDPG、TRPO），结合Lyapunov稳定性理论，为CARLA模拟器提供安全、高效的自动驾驶解决方案。

**核心创新点：**
- 基于Frenet坐标系的轨迹规划，实现平滑的车道保持和变道决策
- Lyapunov稳定性理论保证，提供形式化的安全性和收敛性保证
- 多算法支持与对比，包括PPO2、A2C、DDPG、TRPO等主流强化学习算法
- 完整的实验分析系统，支持自动化测试和可视化分析

## 🏗️ 系统架构

```
carla_autonomous_car/
├── main.py                           # 主程序入口
├── config.py                         # 配置管理模块
├── agents/                           # 强化学习代理
│   ├── reinforcement_learning/      # 稳定强化学习算法
│   │   ├── ppo2/                     # PPO2算法实现
│   │   ├── a2c/                      # A2C算法实现
│   │   ├── ddpg/                     # DDPG算法实现
│   │   └── trpo_mpi/                 # TRPO算法实现
│   └── tools/                        # 工具函数
├── carla_gym/                        # CARLA环境接口
│   └── envs/carla_env_v1.py          # 增强CARLA环境
├── configs/                          # 实验配置文件
├── experiments/                      # 实验分析系统
│   ├── baseline_comparison.py        # 多算法对比可视化
│   ├── theoretical_verification.py   # Lyapunov理论验证
│   └── scripts/                      # 分析脚本
│       ├── analyze_results.py        # 统计分析工具
│       ├── plot_results.py           # 专业图表生成
│       └── run_experiments.py        # 实验自动化
├── theory/                           # 理论基础
│   ├── lyapunov_shaping.py           # Lyapunov势能塑形
│   ├── safety_barriers.py            # 安全屏障理论
│   └── dynamics_models.py            # 车辆动力学模型
└── tools/                            # 实用工具
    ├── modules.py                    # 模块管理系统
    └── reward_shaping.py             # 奖励塑形工具
```

## 🚀 安装指南

### 环境要求

- **Python**: 3.6+
- **CARLA**: 0.9.x 模拟器
- **TensorFlow**: 1.14
- **ROS**: 可选，用于ROS集成功能

### 详细安装步骤

1. **安装CARLA模拟器**
   - 下载CARLA 0.9.x版本
   - 配置CARLA Python API环境变量：
     ```bash
     export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI
     ```

2. **安装Python依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **验证安装**
   ```bash
   python test_import.py
   ```

### 主要依赖库
- tensorflow==1.14
- gym
- numpy, scipy, pandas
- matplotlib, seaborn
- pyyaml
- opencv-python
- pygame

## 📖 使用说明

### 完整项目工作流程

#### 1. 环境准备
```bash
# 克隆项目
git clone <repository-url>
cd carla_autonomous_car

# 安装依赖
pip install -r requirements.txt
```

#### 2. 基础训练模式

使用基础配置训练智能体：

```bash
python main.py --cfg_file configs/experiment_baseline.yaml --agent_id 1
```

使用改进配置训练（带变道塑形）：

```bash
python main.py --cfg_file configs/experiment_improved.yaml --agent_id 2
```

使用Lyapunov安全约束配置训练：

```bash
python main.py --cfg_file configs/experiment_lyapunov.yaml --agent_id 3
```

#### 3. 测试模式

测试已训练模型：

```bash
python main.py --test --agent_id 1 --test_model best_100000
```

#### 4. 高级训练选项

查看所有可用参数：

```bash
python main.py --help
```

自定义训练参数：

```bash
python main.py --cfg_file configs/experiment_baseline.yaml --agent_id 1 --learning_rate 0.001 --batch_size 64
```

#### 5. 实验分析

运行完整分析流程：

```bash
# 统计分析
python experiments/scripts/analyze_results.py

# 可视化图表生成
python experiments/scripts/plot_results.py

# 多算法对比
python experiments/baseline_comparison.py

# 理论验证
python experiments/theoretical_verification.py
```

#### 6. 结果查看

训练结果保存在 `logs/agent_{ID}/` 目录：
- `models/` - 训练模型检查点
- `config.yaml` - 训练使用的配置文件
- `training_summary.json` - 训练时长和统计信息
- `monitors/` - 训练监控数据

## 🎯 配置详解

### 配置文件结构

所有配置文件位于 `configs/` 目录，使用YAML格式：

```yaml
# 示例配置结构
env_params:
  carla:
    host: "localhost"
    port: 2000
    town: "Town01"
    
rl_params:
  algorithm: "PPO2"
  policy: "MlpLstmPolicy"
  learning_rate: 0.00025
  n_steps: 128
  batch_size: 256
  gamma: 0.99  # 折扣因子
  
reward_params:
  speed_reward: 1.0
  safety_penalty: -100.0
  lane_keep_reward: 0.5
```

### 基础配置 (`experiment_baseline.yaml`)

**特点：**
- 标准PPO2算法配合LSTM网络
- 基础安全层保护
- 无变道塑形奖励
- 适合初学者入门

**关键参数：**
- 算法：PPO2
- 策略网络：MlpLstmPolicy
- 学习率：0.00025
- 批量大小：256
- 折扣因子：0.99

### 改进配置 (`experiment_improved.yaml`)

**特点：**
- 增加简单变道塑形奖励
- 增强探索策略
- 优化收敛性能
- 适合进阶研究

**关键改进：**
- 添加变道奖励函数
- 调整探索率参数
- 优化折扣因子
- 增加学习率

### Lyapunov配置 (`experiment_lyapunov.yaml`)

**特点：**
- 基于Lyapunov稳定性的安全约束
- 形式化安全保障
- 高级奖励塑形
- 适合理论研究

**关键特性：**
- Lyapunov稳定性保证
- 自适应安全边界
- 保守探索策略
- 高级奖励函数

## 🏆 算法特性对比

| 算法 | 类型 | 稳定性保证 | 适用场景 | 性能特点 |
|------|------|------------|----------|----------|
| PPO2 | 策略优化 | Lyapunov理论保证 | 通用自动驾驶 | 高稳定性，快速收敛 |
| A2C | 优势演员评论家 | 基础安全层 | 简单场景 | 计算效率高 |
| DDPG | 深度确定性策略梯度 | 安全屏障 | 连续控制 | 适用于精确控制 |
| TRPO | 信任域策略优化 | 理论保证 | 复杂场景 | 高安全性，稳定训练 |

## 📊 奖励函数结构

系统采用多组件奖励函数，确保安全、高效、舒适的驾驶行为：

### 基础奖励组件

**速度奖励（`r_speed`）**
```
r_speed = α * (v_current / v_target)
```
- α：速度奖励权重（默认：1.0）
- v_current：当前速度（m/s）
- v_target：目标速度（m/s）

**安全奖励（`r_safety`）**
```
r_safety = β * (1 / min_distance) if min_distance < safe_distance
r_safety = -γ * collision_penalty if collision
```
- β：安全距离惩罚系数（默认：-50.0）
- γ：碰撞惩罚系数（默认：-100.0）
- min_distance：与最近障碍物的距离（m）
- safe_distance：安全距离阈值（默认：10.0m）

**车道保持奖励（`r_lane`）**
```
r_lane = δ * (1 - lane_offset / lane_width)
```
- δ：车道保持奖励权重（默认：0.5）
- lane_offset：车辆偏离车道中心的距离（m）
- lane_width：车道宽度（默认：3.7m）

### 高级奖励组件

**舒适性奖励（`r_comfort`）**
```
r_comfort = -ε * (acceleration^2 + jerk^2)
```
- ε：舒适性惩罚权重（默认：0.1）
- acceleration：加速度（m/s²）
- jerk：加加速度（加速度变化率，m/s³）

**效率奖励（`r_efficiency`）**
```
r_efficiency = ζ * progress / total_distance
```
- ζ：效率奖励权重（默认：0.1）
- progress：沿轨迹的前进距离（m）
- total_distance：总轨迹长度（m）

**变道塑形奖励（`r_lane_change`）**
```
r_lane_change = η * (lane_change_success * 1.0 + smooth_lane_change * 0.5)
```
- η：变道奖励权重（基础：0.0，改进：0.5，Lyapunov：1.0）
- lane_change_success：变道成功标志（0或1）
- smooth_lane_change：平滑变道评分（0.0到1.0）

### 综合奖励

**总奖励计算：**
```
r_total = r_speed + r_safety + r_lane + r_comfort + r_efficiency + r_lane_change
```

**各配置权重对比：**

| 组件 | 基础配置 | 改进配置 | Lyapunov配置 |
|------|----------|----------|--------------|
| 速度奖励 | 1.0 | 1.0 | 1.0 |
| 安全奖励 | -50.0 | -50.0 | -100.0 |
| 车道保持 | 0.5 | 0.5 | 0.8 |
| 舒适性 | -0.1 | -0.1 | -0.2 |
| 效率奖励 | 0.1 | 0.1 | 0.2 |
| 变道塑形 | 0.0 | 0.5 | 1.0 |

## 🔬 理论保证

### Lyapunov稳定性理论

**Lyapunov函数设计：**
```
V(s) = s^T * P * s
```
其中：
- `s`：状态向量（车辆位置、速度、方向）
- `P`：正定矩阵
- `V(s)`：Lyapunov函数

**稳定性条件：**
```
ΔV(s) = V(s_{t+1}) - V(s_t) ≤ 0
```

**自适应安全边界：**
```
safe_boundary = f(v, a_max, ttc_min)
```
其中：
- `v`：当前速度
- `a_max`：最大加速度
- `ttc_min`：最小碰撞时间

### 安全屏障理论

**碰撞时间（TTC）计算：**
```
TTC = (d_lead - L) / (v_ego - v_lead) if v_ego > v_lead
TTC = ∞ if v_ego ≤ v_lead
```
其中：
- `d_lead`：与前车的距离
- `L`：前车长度
- `v_ego`：自车速度
- `v_lead`：前车速度

**安全屏障函数：**
```
h(s) = TTC - TTC_threshold
```
当 `h(s) ≤ 0` 时，触发安全干预。

**紧急制动策略：**
```
a_brake = -max_brake * (1 - exp(-k * (TTC_threshold - TTC)))
```
其中：
- `a_brake`：制动加速度
- `max_brake`：最大制动加速度
- `k`：制动强度系数

## 📈 性能指标

- **平均奖励**：主要性能指标
- **成功率**：完成回合的百分比
- **变道性能**：安全与不安全变道统计
- **碰撞率**：碰撞发生频率
- **压线时间**：超出可行驶区域的时间
- **训练效率**：收敛速度和样本效率

## 📊 可视化系统

### 图表类型
- **学习曲线**：带置信区间的奖励变化趋势
- **成功率对比**：柱状图展示算法性能差异
- **奖励分布**：小提琴图和箱线图分析
- **多指标热力图**：综合性能对比
- **Episode长度分析**：训练稳定性评估

### 输出格式
- **PNG图像**：300dpi高分辨率
- **CSV表格**：结构化数据
- **LaTeX表格**：学术论文格式
- **JSON统计**：机器可读格式

## 🧪 实验分析系统

### 功能特性
- **自动化数据加载**：支持多个实验种子
- **统计分析**：均值、标准差、置信区间
- **多算法对比**：横向比较不同算法性能
- **报告生成**：一键生成完整分析报告

### 使用方法
```bash
# 运行完整分析
python experiments/scripts/run_experiments.py

# 单独分析
python experiments/scripts/analyze_results.py

# 单独绘图
python experiments/scripts/plot_results.py
```

## 🛡️ 高级功能

### 安全层
- **碰撞时间（TTC）监控**：实时预测碰撞风险
- **最小安全距离强制**：确保安全车距
- **紧急干预能力**：自动执行紧急制动
- **车道保持辅助**：防止偏离车道

### Lyapunov稳定性
- **形式化稳定性保证**：数学验证的收敛性
- **自适应安全边界**：动态调整安全约束
- **保守探索策略**：确保训练过程安全

### 多模态训练
- **网络架构支持**：MLP、LSTM、CNN
- **灵活策略配置**：可定制的策略网络
- **自定义特征提取**：CNN特征提取器

## 📁 结果保存结构

训练结果保存在 `logs/agent_{ID}/` 目录：

```
logs/agent_1/
├── models/                   # 训练模型检查点
├── config.yaml              # 训练使用的配置文件
├── reproduction_info.txt    # 训练元数据和参数
├── training_summary.json    # 训练时长和统计信息
└── monitors/                # 训练监控数据
    └── monitor_seed1.csv    # 单个种子监控数据
```

## 🔧 故障排除

### 常见问题

1. **CARLA导入错误**
   ```bash
   # 确保CARLA Python API已安装并在PYTHONPATH中
   export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI
   ```

2. **依赖缺失**
   ```bash
   # 安装所有依赖
   pip install -r requirements.txt
   ```

3. **配置错误**
   - 检查YAML文件语法和参数名称
   - 确保配置文件路径正确

4. **内存问题**
   - 减小批量大小
   - 使用更小的网络架构
   - 减少训练环境数量

### 结构验证

运行结构测试验证安装：

```bash
python test_import.py
```

## 🤝 贡献指南

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目基于MIT许可证发布，详情见LICENSE文件。

## 🔗 相关资源

- [CARLA模拟器官网](https://carla.org/)
- [Stable Baselines文档](https://stable-baselines.readthedocs.io/)
- [强化学习教程](https://spinningup.openai.com/)

## 📞 联系方式

如有问题或建议，请通过GitHub Issues联系我们。

## 🎉 项目状态

**版本**：1.0.0（完整发布版）
**最后更新**：2026年5月
**状态**：✅ 完整且功能正常
**维护状态**：活跃维护中

---

*注：本项目基于RL-frenet-trajectory-planning-in-CARLA项目开发，适用于学术研究和教学用途。我们欢迎贡献者为其添加对新设备的支持。*

## 使用说明

### 训练模式

使用基础配置训练智能体：

```bash
python main.py --cfg_file configs/experiment_baseline.yaml --agent_id 1
```

使用改进配置训练（带变道塑形）：

```bash
python main.py --cfg_file configs/experiment_improved.yaml --agent_id 2
```

使用Lyapunov安全约束配置训练：

```bash
python main.py --cfg_file configs/experiment_lyapunov.yaml --agent_id 3
```

### 测试模式

测试已训练模型：

```bash
python main.py --test --agent_id 1 --test_model best_100000
```

### 查看可用配置

```bash
python main.py --list_cfgs
```

## 配置详解

### 基础配置 (`experiment_baseline.yaml`)
- 标准PPO2算法配合LSTM网络
- 无变道塑形奖励
- 基础安全层保护

### 改进配置 (`experiment_improved.yaml`)
- 增加简单变道塑形奖励
- 增强探索策略
- 优化收敛性能

### Lyapunov配置 (`experiment_lyapunov.yaml`)
- 基于Lyapunov稳定性的安全约束
- 形式化安全保障
- 高级奖励塑形

## 奖励函数结构

系统采用多组件奖励函数：

- **速度奖励**：鼓励维持目标速度
- **安全奖励**：惩罚不安全距离和碰撞
- **车道保持**：奖励保持在正确车道内
- **舒适性**：惩罚急加速和急刹车
- **效率奖励**：奖励沿轨迹前进
- **变道塑形**（改进/Lyapunov配置）：指导变道决策

## 环境规格

- **观察空间**：局部传感器数据、车辆状态、轨迹信息
- **动作空间**：连续转向和加速控制
- **仿真环境**：带交通车辆的CARLA驾驶环境
- **安全层**：实时碰撞避免和紧急制动

## 结果保存结构

训练结果保存在 `logs/agent_{ID}/` 目录：
- `models/` - 训练模型检查点
- `config.yaml` - 训练使用的配置文件
- `reproduction_info.txt` - 训练元数据和参数
- `training_summary.json` - 训练时长和统计信息

## 性能指标

- **平均奖励**：主要性能指标
- **成功率**：完成回合的百分比
- **变道性能**：安全与不安全变道统计
- **碰撞率**：碰撞发生频率
- **压线时间**：超出可行驶区域的时间

## 高级功能

### 安全层
- 碰撞时间（TTC）监控
- 最小安全距离强制
- 紧急干预能力

### Lyapunov稳定性
- 形式化稳定性保证
- 自适应安全边界
- 保守探索策略

### 多模态训练
- 支持不同网络架构（MLP, LSTM, CNN）
- 灵活的策略配置
- 自定义CNN特征提取器

## 故障排除

### 常见问题

1. **CARLA导入错误**：确保CARLA Python API已安装并在PYTHONPATH中
2. **依赖缺失**：运行 `pip install -r requirements.txt`
3. **配置错误**：检查YAML文件语法和参数名称
4. **内存问题**：减小批量大小或使用更小的网络

### 结构验证

运行结构测试验证安装：

```bash
python test_import.py
```

## 项目状态

**版本**：1.0.0（第一次提交）
**最后更新**：2026年4月
**状态**：完整且功能正常

本项目提供了一个使用强化学习进行自动驾驶轨迹规划的完整框架，系统已准备好在CARLA模拟器上进行训练和测试。

---

*注：本项目基于RL-frenet-trajectory-planning-in-CARLA项目开发，适用于学术研究和教学用途。*