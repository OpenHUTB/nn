# 受限玻尔兹曼机 (RBM)

## 概述

受限玻尔兹曼机（Restricted Boltzmann Machine, RBM）是一种生成式随机神经网络，由可见层和隐藏层组成，层间全连接但层内无连接。

## 主要特点

- **二分图结构**：可见层和隐藏层之间有连接，层内无连接
- **能量模型**：基于能量函数定义概率分布
- **无监督学习**：能够学习数据的特征表示

## 基本结构

### 1. 可见层 (Visible Layer)

- 接收输入数据
- 节点数量等于输入特征维度

### 2. 隐藏层 (Hidden Layer)

- 学习数据的潜在特征
- 节点数量可调，影响模型容量

### 3. 权重和偏置

- 连接权重矩阵 W
- 可见层偏置 b
- 隐藏层偏置 c

## 训练算法

### 对比散度 (Contrastive Divergence, CD)

- CD-1：一步吉布斯采样
- 近似最大似然估计
- 计算效率高

## 应用领域

- 降维和特征学习
- 推荐系统
- 协同过滤
- 深度信念网络的基础组件
- 
## 代码示例

使用 PyTorch 实现简单的 RBM：

```python
import torch
import torch.nn as nn

class RBM(nn.Module):
    """受限玻尔兹曼机"""
    def __init__(self, visible_dim, hidden_dim):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim, visible_dim) * 0.01)
        self.b_v = nn.Parameter(torch.zeros(visible_dim))
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

    def sample_hidden(self, v):
        """从可见层采样隐藏层"""
        p_h = torch.sigmoid(torch.matmul(v, self.W.t()) + self.b_h)
        return p_h, torch.bernoulli(p_h)

    def sample_visible(self, h):
        """从隐藏层采样可见层"""
        p_v = torch.sigmoid(torch.matmul(h, self.W) + self.b_v)
        return p_v, torch.bernoulli(p_v)
```

## 运行环境

- Python 3.7+
- PyTorch 1.8+

## 参考资料

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [深度信念网络介绍](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)
