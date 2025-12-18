# 黑白棋游戏



## 问题描述：

​	本次作业要实现的是 利用强化学习等知识 玩atari 游戏中的 黑白棋游戏。具体任务是补全RL_QG_agent.py 文件.





## 环境配置: 

配置环境安装办法：

1. pip install gym[all] # 安装 gym
2. 找到 安装的包的目录，然后复制 github 上面的reversi 文件夹， 到gym/envs/ 中
  （windows中的目录路径是
  C:\Program Files\Anaconda3\Lib\site-packages\gym\envs）
3. 在envs文件夹中 有__init__.py 文件，在文件 末尾，添加注册信息。
  （参考 github 上面 __init__.py 文件末尾的注册信息（即id='Reversi8x8-v0', 的注册信息））





## 题目要求： 

​	Github 中reversi_main.py 是一个demo程序，主要为了规范后期判作业时候的接口.本作业后面会运行大家的程序，因此需要统一接口，并且注意保证自己的代码没有错误，可以运行。训练程序的时候 黑白双方可以自己规定，环境中没有对弈对象。因此训练程序的时候时自己设置对弈对象，比如与随机进行对弈，其次可以和一些搜索算法对弈。



参考文献：

1. Learning to Play Othello with Deep Neural Networks
2. Reinforcement Learning in the Game of Othello: Learning Against a Fixed Opponent and Learning from Self-Play

# RL_QG_agent.py 文件说明文档

## 项目简介

`RL_QG_agent` 是一个使用卷积神经网络（CNN）结构实现的强化学习智能体，用于黑白棋（Reversi）对局决策。该模型采用 TensorFlow 1.x 构建，能够根据输入棋盘状态预测每个位置的 Q 值，从而选择最优落子位置。

## 文件结构

```
your_project/
│
├── RL_QG_agent.py         # 主模型文件
├── Reversi/               # 模型保存目录，自动创建
│   └── parameter.ckpt     # 保存的模型参数（运行后生成）
└── README.md              # 使用说明文档
```

## 模型结构

* 输入：`[None, 8, 8, 3]`，表示一个 8x8 棋盘（当前玩家棋子、对手棋子、合法落子位置）
* 网络结构：

  * Conv2D (32 filters, 3x3, relu)
  * Conv2D (64 filters, 3x3, relu)
  * Flatten
  * Dense (512 units, relu)
  * Dense (64 units) → 每个格子的 Q 值

## 使用说明

### 初始化模型

```python
from RL_QG_agent import RL_QG_agent

agent = RL_QG_agent()
agent.init_model()
```

### 保存模型

```python
agent.save_model()
```

### 加载模型

```python
agent.load_model()
```

### 推理落子

```python
# 输入state为形状(8, 8, 3)的 numpy 数组
# enables 为合法位置的掩码（bool 类型，shape 为 (64,)）
action = agent.place(state, enables)
# 返回 0~63 之间的动作编号，代表应落子的格子
```

## 注意事项

* **TensorFlow 版本要求**：必须使用 **TensorFlow 1.x**，如 `1.15`。
* 模型保存路径为当前目录下的 `Reversi/parameter.ckpt`。
* `state` 输入为 `[8, 8, 3]`，每个通道通常表示：

  * 通道 0：当前玩家棋子（1为有子，0为无）
  * 通道 1：对手棋子
  * 通道 2：当前可落子位置（1为可落子）
* `enables` 应为布尔数组，表示 64 个位置是否合法（例如 `np.array([0, 1, 0, ..., 0])`）

## 待完善项（建议）

* 支持训练功能（当前模型仅包含推理功能）
* 增加模型评估与训练数据生成模块
* 可视化 Q 值输出（便于分析）

