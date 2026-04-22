# 支持向量机 (SVM)

## ?????????????

?????? SVM ???????????? `svm.py` ???????????????????

1. ??????????
`--train-file`?`--test-file`?`--learning-rate`?`--reg-lambda`?`--max-iter`?`--out-dir`
2. ? SVM ?????????
????????????????????????
3. ???????????
??????? `outputs/svm_metrics.json`

### ????

```bash
python svm.py --learning-rate 0.1 --reg-lambda 0.0 --max-iter 20000 --out-dir outputs
```

### ????

- `outputs/svm_metrics.json`

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
