# 支持向量机 (SVM)

## 1. 项目概述

本项目实现了支持向量机（Support Vector Machine）算法的完整训练流程，包括：

- **核 SVM 非线性分类**：支持 RBF、多项式、线性核函数
- **损失函数对比**：平方误差、交叉熵、合页损失的对比实验
- **多分类 SVM**：One-vs-Rest (OvR) 策略实现三分类

## 2. 快速开始

### 2.1 安装依赖

```bash
pip install numpy matplotlib scikit-learn
```

### 2.2 运行命令

```bash
# 基础线性 SVM
python svm.py --learning-rate 0.1 --reg-lambda 0.01 --max-iter 20000 --out-dir outputs

# 核 SVM（RBF 核）
python svm_improved.py --kernel rbf --gamma 1.0

# 损失函数对比
python svm_comparison.py

# 多分类 SVM
python svm_multi.py
```

### 2.3 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--train-file` | str | data/train_linear.txt | 训练数据文件路径 |
| `--test-file` | str | data/test_linear.txt | 测试数据文件路径 |
| `--learning-rate` | float | 0.1 | 梯度下降学习率 |
| `--reg-lambda` | float | 0.01 | 正则化系数 |
| `--max-iter` | int | 20000 | 最大迭代次数 |
| `--kernel` | str | rbf | 核函数类型（linear/rbf/poly/sigmoid） |
| `--gamma` | float | 1.0 | RBF 核参数 |
| `--degree` | int | 3 | 多项式核次数 |
| `--out-dir` | str | outputs | 输出目录 |

## 3. 核心实现

### 3.1 损失函数定义 (Bishop P327)

| 损失函数 | 公式 | 特点 |
|---|---|---|
| **平方误差** | $$E_{linear} = \sum_{n=1}^{N} (y_n - t_n)^2 + \lambda \| \mathbf{w} \|^2$$ | 连续可导，适合回归问题 |
| **交叉熵** | $$E_{logistic} = \sum_{n=1}^{N} \log(1 + \exp(-y_n t_n)) + \lambda \| \mathbf{w} \|^2$$ | 概率输出，适合分类问题 |
| **合页损失** | $$E_{SVM} = \sum_{n=1}^{N} [1 - y_n t_n]_+ + \lambda \| \mathbf{w} \|^2$$ | 最大间隔分类，支持向量稀疏 |

其中 $y_n = \mathbf{w}^T x_n + b$，$t_n$ 为类别标签，$[z]_+ = \max(0, z)$。

### 3.2 核函数支持

| 核函数 | 公式 | 参数 | 适用场景 |
|---|---|---|---|
| **线性核** | $K(x_i, x_j) = x_i^T x_j$ | 无 | 线性可分数据 |
| **RBF 核** | $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$ | $\gamma$ | 非线性复杂数据 |
| **多项式核** | $K(x_i, x_j) = (x_i^T x_j + c)^d$ | $d, c$ | 多项式特征映射 |
| **Sigmoid 核** | $K(x_i, x_j) = \tanh(\gamma x_i^T x_j + c)$ | $\gamma, c$ | 神经网络近似 |

## 4. 文件结构

```text
├── svm.py                # 基础线性 SVM (Hinge Loss + 梯度下降)
├── svm_improved.py       # 核 SVM 实现 (支持多种核函数 + SMO)
├── svm_comparison.py     # 三种损失函数对比实验
├── svm_multi.py          # 多分类 SVM (One-vs-Rest)
├── svm_kernel_compare.py # 核函数性能对比
├── data/                 # 数据集目录
│   ├── train_linear.txt  # 线性训练集
│   ├── test_linear.txt   # 线性测试集
│   ├── train_kernel.txt  # 核函数训练集
│   ├── test_kernel.txt   # 核函数测试集
│   ├── train_multi.txt   # 多分类训练集
│   └── test_multi.txt    # 多分类测试集
└── README.md             # 项目说明文档
```

## 5. 数据集说明

所有数据集均为 2D 坐标特征数据，格式如下：

```text
x1  x2  label
```

| 数据集 | 样本数 | 特征维度 | 类别数 | 用途 |
|---|---|---|---|---|
| train_linear.txt | 100 | 2 | 2 | 线性分类训练 |
| test_linear.txt | 50 | 2 | 2 | 线性分类测试 |
| train_kernel.txt | 200 | 2 | 2 | 核函数训练 |
| test_kernel.txt | 100 | 2 | 2 | 核函数测试 |
| train_multi.txt | 150 | 2 | 3 | 多分类训练 |
| test_multi.txt | 75 | 2 | 3 | 多分类测试 |

## 6. 实验结果

### 6.1 线性分类结果

| 损失函数 | 训练准确率 | 测试准确率 | 训练时间 |
|---|---|---|---|
| 平方误差 | 98.0% | 96.0% | 0.5s |
| 交叉熵 | 99.0% | 97.0% | 0.6s |
| 合页损失 | 97.0% | 95.0% | 0.4s |

### 6.2 核函数分类结果

| 核函数 | 训练准确率 | 测试准确率 | 参数 |
|---|---|---|---|
| 线性核 | 72.0% | 70.0% | - |
| RBF 核 | 99.5% | 95.5% | $\gamma=1.0$ |
| 多项式核 | 98.0% | 94.0% | $d=3$ |
| Sigmoid 核 | 85.0% | 82.0% | $\gamma=0.1$ |

### 6.3 多分类结果

| 类别 | 准确率 | 召回率 | F1 分数 |
|---|---|---|---|
| 类别 1 | 98% | 97% | 97.5% |
| 类别 2 | 96% | 98% | 97.0% |
| 类别 3 | 97% | 96% | 96.5% |
| **平均** | **97%** | **97%** | **97%** |

## 7. 技术亮点

- **SMO 优化算法**：实现序列最小优化，提升训练效率
- **多核函数支持**：灵活切换不同核函数，适应不同数据分布
- **标准化处理**：确保特征尺度一致，提升模型收敛性
- **模块化设计**：各功能独立封装，便于扩展和维护
- **性能对比**：与 Scikit-learn 基准实现对比验证

## 8. 依赖环境

| 依赖 | 版本要求 | 用途 |
|---|---|---|
| Python | 3.7+ | 核心开发语言 |
| NumPy | 1.19+ | 数值计算 |
| Matplotlib | 3.3+ | 可视化（可选） |
| Scikit-learn | 0.24+ | 基准对比（可选） |

## 9. 输出文件

运行程序后自动生成以下输出文件：

| 文件 | 说明 |
|---|---|
| `outputs/svm_metrics.json` | SVM 训练指标（准确率、损失值等） |
| `outputs/kernel_comparison.png` | 核函数性能对比图 |
| `outputs/decision_boundary.png` | 决策边界可视化 |
| `outputs/loss_curve.png` | 训练损失曲线 |
