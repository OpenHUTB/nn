---
toc: false
---
# 基于自监督学习与PPO强化学习的自动驾驶仿真项目
...
## 目录
- [项目简介](#introduction)
- [运行环境与安装步骤](#installation)
- [快速启动项目](#quick-start)
- [核心技术方案](#core-technology)
- [运行效果展示](#result-show)
- [项目文件结构](#file-structure)
- [参考资料](#references)

## 项目简介 <a name="introduction"></a>
本项目依托 **CARLA 0.9.14 自动驾驶仿真平台**，融合**自监督学习(SSL)** 与 **PPO近端策略优化强化学习算法**，实现端到端自动驾驶环境感知、决策与车辆连续控制全流程仿真验证，规避真实道路测试成本高、风险大的问题。

核心功能特点：
1. **无标注视觉特征提取**：自监督对比学习从CARLA原生RGB图像中自主学习道路、车辆、行人通用视觉表征，无需人工像素标注；
2. **稳定强化学习驾驶策略**：PPO算法训练连续动作控制智能体，输出油门、刹车、转向指令，实现车道保持、障碍物自动避让；
3. **全仿真闭环测试**：基于CARLA虚拟城市动态车流、多变天气场景完成算法迭代，配套TensorBoard训练可视化工具；
4. **轻量化可复现**：完整依赖清单、一键启动训练脚本，支持Windows环境直接运行仿真。

## 运行环境与安装步骤 <a name="installation"></a>
### 基础环境要求
- Python 版本：3.7 / 3.8
- 仿真平台：CARLA Simulator 0.9.14
- 深度学习框架：PyTorch 2.1.0
- 系统支持：Windows 10/11

### 完整安装流程
1. 克隆本项目代码仓库
```powershell
git clone https://github.com/lfy666698/nn.git
cd nn
```

2. 启动 CARLA 仿真器
进入CARLA安装目录，运行 `CarlaUE4.exe`，等待仿真服务端口2000启动完成。

3. 一键安装全部Python依赖
```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 快速启动项目 <a name="quick-start"></a>
### 1. 完整训练启动命令
在项目根目录执行训练主脚本：
```powershell
python src/run_train.py
```

### 2. 脚本自定义启动参数
查看全部可调超参配置：
```powershell
python src/run_train.py -h
```
关键参数说明：
- `--ssl_epoch`：自监督预训练迭代轮数
- `--ppo_epoch`：强化学习策略更新轮数
- `--gamma`：强化学习奖励衰减系数
- `--save_freq`：模型权重保存间隔步数

### 3. 训练过程可视化监控
```powershell
tensorboard --logdir ./tensorboard_logs
```
浏览器访问地址：`http://localhost:6006`
可实时查看：单步奖励曲线、感知损失、车辆行驶轨迹、碰撞惩罚统计。

### 预期运行效果
CARLA仿真窗口内生成自主行驶车辆：
1. 基础车道居中行驶，稳定贴合道路中心线；
2. 识别静态路障、动态往来车辆，自动减速、避让；
3. 弯道自主调整转向角度，根据车流自适应调节车速。

## 核心技术方案 <a name="core-technology"></a>
### 一、自监督视觉感知模块
基于对比学习实现无标注图像特征提取：
1. 数据输入：CARLA车载前置RGB摄像头实时画面；
2. 预处理流程：随机裁剪、亮度扰动、水平翻转数据增强；
3. 骨干网络：轻量化ResNet特征编码器；
4. 自监督损失：InfoNCE对比损失，拉近同场景特征、分离无关场景特征；
5. 输出：固定维度环境特征向量，作为强化学习智能体输入。

### 二、PPO强化学习决策控制模块
采用近端策略优化算法训练驾驶智能体，解决传统策略梯度训练不稳定问题：
1. **状态空间**：车辆瞬时速度、车身横摆角、SSL提取环境特征向量；
2. **连续动作空间**：油门(0~1)、刹车(0~1)、转向角度(-1~1)；
3. **奖励函数设计**
    | 奖励项 | 作用说明 |
    |--------|----------|
    | 车道保持正向奖励 | 车辆距离车道中心线越近，奖励值越高 |
    | 前进速度奖励 | 平稳匀速行驶获得基础正向收益 |
    | 碰撞惩罚项 | 与车辆、建筑、路障碰撞大幅扣除奖励 |
    | 压线惩罚项 | 车轮越线行驶施加小额负奖励 |
4. 关键超参配置：`γ=0.98`、裁剪系数`ε=0.2`、批次更新轮数10轮。

## 运行效果展示 <a name="result-show"></a>
1. **仿真实时画面**
CARLA窗口同步渲染车辆行驶画面，叠加可视化车道识别区域，直观展示模型感知范围；
2. **训练收敛曲线**
TensorBoard可视化面板可查看完整训练周期奖励变化，2000轮迭代后平均单轮奖励稳定收敛；
3. **极端场景适配表现**
雨天低光照、多车拥堵、直角转弯场景下均可完成稳定自主行驶。

## 项目文件结构 <a name="file-structure"></a>
```
nn/
├── docs/                    # MkDocs项目文档目录
│   ├── index.md             # 文档首页总览
│   └── autonomous_driving/
│       └── README.md        # 本项目完整说明文档
├── src/                     # 项目核心源码
│   ├── ssl_module/          # 自监督学习感知模块代码
│   │   ├── model.py         # ResNet特征提取网络
│   │   └── pretrain.py      # SSL预训练脚本
│   ├── rl_module/           # PPO强化学习决策模块
│   │   ├── agent.py         # PPO智能体核心逻辑
│   │   └── train_loop.py    # 强化学习迭代循环
│   ├── carla_env.py         # CARLA仿真环境封装接口
│   └── run_train.py         # 项目统一训练入口脚本
├── checkpoints/             # 模型权重保存目录
├── tensorboard_logs/        # 训练日志、可视化记录
├── mkdocs.yml               # MkDocs网页文档配置文件
├── requirements.txt         # 项目全部依赖清单
└── install.md               # 简易环境安装指引
```

## 参考资料 <a name="references"></a>
1. CARLA 官方仿真平台文档：https://carla.readthedocs.io/
2. PPO算法论文：Proximal Policy Optimization Algorithms
3. InfoNCE自监督对比学习：A Simple Framework for Contrastive Learning of Visual Representations
4. 自动驾驶强化学习开源项目案例库

---
*本项目基于CARLA 0.9.14仿真平台完成，结合自监督学习与强化学习实现端到端自动驾驶感知控制仿真*


