# 自动驾驶奖励函数分析系统

## 1. 项目简介
本项目实现了一套完整的自动驾驶奖励函数建模与可视化分析系统。通过数学建模方式定义速度跟踪、距离控制、舒适性三类核心奖励函数，支持参数化配置、批量仿真与可视化输出，无需任何模拟器即可独立运行。

系统可广泛应用于：
- 强化学习自动驾驶算法的奖励设计
- 自适应巡航控制（ACC）策略评估
- 自动驾驶决策系统的安全性与舒适性权衡分析

## 2. 选题说明
- **技术方案**: 基于纯 Python + NumPy + Matplotlib 的奖励函数建模与可视化
- **设计思路**: 将自动驾驶中的核心评价指标抽象为数学函数，通过参数化配置实现灵活的奖励设计，支持单场景分析与批量对比实验
- **独特价值**: 无需 CARLA、AirSim 等重型模拟器，纯代码即可生成专业级分析图表

## 3. 开发运行环境
- **操作系统**: Windows 10/11, Ubuntu 20.04/22.04, macOS
- **编程语言**: Python 3.8+
- **核心依赖**: NumPy, Matplotlib
- **开发工具**: Visual Studio Code / PyCharm / Jupyter Notebook

## 4. 模块结构与入口
- 本模块的所有核心代码存放于 `src/driving_reward_analysis` 目录下
- 模块的主程序入口为 `main.py`
- 奖励函数定义位于 `rewards.py`
- 可视化工具位于 `visualizer.py`

---

#  driving_reward_analysis: 自动驾驶奖励函数分析系统

## 1. 模块功能
本模块实现了自动驾驶奖励函数的完整建模与分析流程：

- **速度跟踪奖励**: 基于高斯分布建模，车辆速度越接近目标速度奖励越高，支持自定义目标速度和容忍度
- **距离控制奖励**: 分段线性函数建模，保持安全距离时获得最大奖励，过近时惩罚急剧增加
- **舒适性奖励**: 基于加速度的二次函数建模，零加速度时奖励最高，剧烈加减速时惩罚增加
- **批量仿真**: 支持多组参数配置的批量实验，生成对比分析图表
- **可视化输出**: 自动生成专业的奖励函数曲线图，支持保存为 PNG 格式

## 2. 运行指南

### 步骤 1：安装依赖
```bash
pip install numpy matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤 2：运行主程序
在项目根目录下执行：
```bash
python src/driving_reward_analysis/main.py
```

程序将自动生成 `outputs/` 目录，并保存以下效果图：
- `reward_func_analysis.png` — 三类奖励函数综合分析图
- `reward_comparison.png` — 不同参数下的奖励函数对比图

### 步骤 3：自定义参数
编辑 `config.yaml` 文件调整奖励函数参数：
```yaml
speed_reward:
  target_speed: 25.0      # 目标速度 (m/s)
  tolerance: 5.0          # 速度容忍度

distance_reward:
  safe_distance: 15.0     # 安全距离 (m)
  critical_distance: 5.0  # 临界距离 (m)

comfort_reward:
  max_acceleration: 2.0   # 最大加速度 (m/s²)
```

## 3. 模块文件说明
| 文件 | 功能 |
|------|------|
| `main.py` | 主入口，执行奖励函数建模与可视化 |
| `rewards.py` | 奖励函数定义：速度、距离、舒适性 |
| `visualizer.py` | 可视化工具：绘制奖励函数曲线 |
| `config.yaml` | 配置文件：奖励函数参数 |
| `utils.py` | 工具函数：数据生成、文件保存 |

## 4. 奖励函数数学定义

### 速度跟踪奖励
$$R_{speed}(v) = -\left(\frac{v - v_{target}}{\sigma}\right)^2$$

其中 $v_{target}$ 为目标速度，$\sigma$ 为容忍度参数。

### 距离控制奖励
$$R_{dist}(d) = \begin{cases} -100 & d < d_{critical} \\ -10 \times (d_{safe} - d) & d_{critical} \leq d < d_{safe} \\ 0 & d \geq d_{safe} \end{cases}$$

### 舒适性奖励
$$R_{comfort}(a) = -a^2$$

其中 $a$ 为加速度，零加速度时奖励最高。

## 5. 参考
- [强化学习奖励设计](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [自适应巡航控制](https://en.wikipedia.org/wiki/Adaptive_cruise_control)
- [NumPy 文档](https://numpy.org/doc/)
- [Matplotlib 文档](https://matplotlib.org/stable/)
