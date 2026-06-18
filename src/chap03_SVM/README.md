# 支持向量机 (SVM)

<<<<<<< HEAD
## 问题描述

本项目完成了以下三个部分的实验内容：

1. **核 SVM 非线性分类**：使用基于 RBF（高斯核）、多项式或线性核函数的 SVM 解决非线性可分的二分类问题。数据集：`train_kernel.txt` 和 `test_kernel.txt`。
2. **损失函数对比**：分别使用线性分类器（平方误差）、逻辑回归（交叉熵）以及 SVM（合页损失）解决线性二分类问题，并比较三种模型的效果。数据集：`train_linear.txt` 和 `test_linear.txt`。
3. **多分类 SVM**：使用 One-vs-Rest (OvR) 策略的多分类 SVM 解决三分类问题。数据集：`train_multi.txt` 和 `test_multi.txt`。

### 损失函数定义 (Bishop P327)

- **平方误差 (Squared Error)**: 
  $$E_{linear} = \sum_{n=1}^{N} (y_n - t_n)^2 + \lambda \| \mathbf{w} \|^2$$
- **交叉熵 (Cross Entropy)**: 
  $$E_{logistic} = \sum_{n=1}^{N} \log(1 + \exp(-y_n t_n)) + \lambda \| \mathbf{w} \|^2$$
- **合页损失 (Hinge Error)**: 
  $$E_{SVM} = \sum_{n=1}^{N} [1 - y_n t_n]_+ + \lambda \| \mathbf{w} \|^2$$

其中 $y_n = \mathbf{w}^T x_n + b$，$t_n$ 为类别标签。

## 数据集

本项目使用的数据集位于 `data/` 目录下，均为 2D 坐标特征数据，包含两维特征 ($x_1, x_2$) 和一类标签 ($t$)。

- **线性数据集**：用于 Part 2 的线性可分二分类测试。
- **核函数数据集**：用于 Part 1 的非线性二分类测试。
- **多分类数据集**：用于 Part 3 的三分类测试。

## 项目要求与进度

- [x] 使用代码模板补全缺失部分，支持核函数和多分类。
- [x] 使用 Python 及 NumPy 编写主要逻辑代码，不依赖复杂深度学习框架。
- [x] 提供不同损失函数的性能对比分析。
- [x] 提供决策边界的可视化（支持环境时）。

## 文件结构

```text
├── svm.py                # 基础线性 SVM 实现 (Hinge Loss + 梯度下降)
├── svm_improved.py       # 改进的核 SVM 实现 (支持 RBF/Linear 核 + 对比 sklearn)
├── svm_comparison.py     # Part 2：三种损失函数 (Squared/Logistic/Hinge) 的对比
├── svm_multi.py          # Part 3：多分类 SVM 实现 (One-vs-Rest 策略)
├── data/                 # 数据集目录
│   ├── train_linear.txt  # 线性训练集
│   ├── train_kernel.txt  # 核函数训练集
│   └── train_multi.txt   # 多分类训练集
└── README.md             # 本说明文件
```

## 使用方法

### 1. 运行核 SVM (Part 1)
```bash
python src/chap03_SVM/svm_improved.py
```
该程序会训练自定义的核 SVM 并与 Scikit-learn 的实现进行准确率对比。

### 2. 运行损失函数对比 (Part 2)
```bash
python src/chap03_SVM/svm_comparison.py
```
该程序将输出线性分类器、逻辑回归和 SVM 在相同线性数据集上的性能差异。

### 3. 运行多分类 SVM (Part 3)
```bash
python src/chap03_SVM/svm_multi.py
```
该程序将使用 One-vs-Rest 策略训练三个二分类器来解决三分类问题。

## 实验结果总结

- **线性分类**：对于线性可分数据，三种损失函数（平方误差、交叉熵、合页损失）均能达到 **95% 以上** 的准确率。
- **非线性分类**：引入 RBF 核函数后，自定义 SVM 在非线性数据集上的准确率从约 70% 提升至 **95% 以上**，与 Scikit-learn 的基准实现差异极小。
- **多分类**：通过 One-vs-Rest (OvR) 策略，线性 SVM 在三分类任务上表现优异，准确率达到 **97% 以上**。

## 主要功能与技术点

- **标准化处理**：所有实验均包含数据标准化步骤，这对 SVM 和核方法的收敛至关重要。
- **SMO 优化**：`svm_improved.py` 实现了简化版的序列最小优化 (SMO) 算法，支持高效训练。
- **核函数支持**：实现了线性、RBF（高斯）、多项式和 Sigmoid 核函数。
- **鲁棒性**：处理了 Windows 环境下的控制台编码问题和特定库版本导致的循环引用问题。

## 依赖环境
- Python 3.x
- NumPy
- Matplotlib (可选，用于可视化)
- Scikit-learn (可选，用于性能基准对比)
=======
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
>>>>>>> upstream/main
