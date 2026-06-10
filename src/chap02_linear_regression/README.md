# 线性回归

## 1. 项目概述

本项目实现了线性回归模型，支持多种基函数和参数优化方法。通过该代码，可以学习如何应用最小二乘法和梯度下降法进行模型训练，并使用不同的基函数（如多项式基函数、高斯基函数）来拟合数据。

## 2. 问题描述

给定一个未知函数 $y = f(x)$ 的训练样本集 $\{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}$（$N=300$），使用线性回归模型拟合该函数关系。

**核心任务**：
- 使用最小二乘法求解模型参数
- 尝试不同基函数（多项式、高斯、sigmoid）
- 使用梯度下降法优化模型

## 3. 快速开始

### 3.1 运行命令

```bash
# 基础运行
python exercise-linear_regression.py

# 使用测试验证脚本
python test_verify.py --json-out outputs/test_verify_report.json --stop-on-fail
```

### 3.2 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--json-out` | str | None | 将测试结果导出为 JSON 文件 |
| `--stop-on-fail` | flag | - | 首个失败时立即停止 |

### 3.3 输出文件

| 文件 | 说明 |
|---|---|
| `outputs/test_verify_report.json` | 测试验证报告 |
| `outputs/regression_plot.png` | 回归拟合曲线 |

## 4. 题目要求

| 任务 | 描述 | 状态 |
|---|---|---|
| 最小二乘法 | 完成 `exercise-linear_regression.ipynb` 填空（参考 PRML 第二章 2.3 节） | ☐ |
| 多项式基函数 | 实现多项式基函数扩展 | ☐ |
| 高斯基函数 | 实现高斯基函数扩展 | ☐ |
| 梯度下降 | 实现梯度下降优化方法 | ☐ |
| TensorFlow 版本 | 参照 `linear_regression-tf2.0.ipynb` 实现 | ☐ |

## 5. 数据集说明

| 文件 | 说明 | 样本数 |
|---|---|---|
| `train.txt` | 训练数据集 | 300 |
| `test.txt` | 测试数据集 | 100 |

**数据格式**：每行包含空格分隔的 $x$ 和 $y$ 值

```text
1.0 2.5
2.0 4.8
...
```

## 6. 功能特性

### 6.1 基函数支持

| 基函数 | 类型 | 公式 | 适用场景 |
|---|---|---|---|
| 恒等基函数 | 线性 | $\phi(x) = x$ | 线性关系数据 |
| 多项式基函数 | 非线性 | $\phi_i(x) = x^i$ | 多项式曲线拟合 |
| 高斯基函数 | 非线性 | $\phi_i(x) = \exp(-\frac{(x-\mu_i)^2}{2\sigma^2})$ | 局部特征拟合 |

### 6.2 优化方法

| 方法 | 类型 | 特点 |
|---|---|---|
| 最小二乘法 | 解析解 | 直接求解，无需迭代 |
| 梯度下降法 | 迭代优化 | 需要调参，适合大规模数据 |

### 6.3 其他功能

- ✅ 数据加载与预处理（自动转换为 NumPy 数组）
- ✅ 模型评估（标准差指标）
- ✅ 结果可视化（训练数据点、预测曲线）
- ✅ 测试验证（支持 JSON 报告导出）

## 7. 文件结构

```text
chap02_linear_regression/
├── exercise-linear_regression.py   # 主程序（NumPy 版本）
├── linear_regression-tf2.0.py     # TensorFlow 版本
├── exercise-linear_regression.ipynb # Jupyter Notebook
├── test_verify.py                 # 测试验证脚本
├── train.txt                      # 训练数据集
├── test.txt                       # 测试数据集
└── README.md                      # 项目说明文档
```

## 8. 依赖环境

| 依赖 | 版本要求 | 用途 |
|---|---|---|
| Python | 3.7+ | 核心语言 |
| NumPy | 1.19+ | 数值计算 |
| Matplotlib | 3.3+ | 数据可视化 |
| TensorFlow | 2.x | TF版本实现（可选） |

## 9. 使用方法

```bash
# NumPy 版本
python exercise-linear_regression.py

# TensorFlow 版本
python linear_regression-tf2.0.py

# 测试验证
python test_verify.py --json-out outputs/report.json
```

## 10. 参数调整

### 基函数选择
```python
# 在代码中修改 basis_func 参数
basis_func = identity_basis      # 恒等基函数（线性）
basis_func = multinomial_basis   # 多项式基函数
basis_func = gaussian_basis      # 高斯基函数
```

### 基函数参数
```python
# 多项式基函数（10阶）
phi = multinomial_basis(x, feature_num=10)

# 高斯基函数（15个基）
phi = gaussian_basis(x, feature_num=15)
```

### 梯度下降参数
```python
lr = 0.1        # 学习率
epochs = 1000   # 迭代次数
```

## 11. 结果示例

**控制台输出**：
```
训练集预测值与真实值的标准差：3.2
测试集预测值与真实值的标准差：4.5
```

**TensorFlow 版本输出**：
```
Epoch 1/1000 - Loss: 25.32
Epoch 500/1000 - Loss: 3.15
Epoch 1000/1000 - Loss: 2.89
训练集标准差: 3.1
测试集标准差: 4.2
```

## 12. 技术亮点

- **多基函数支持**：灵活切换不同基函数，适应不同数据分布
- **双优化方法**：支持解析解和迭代优化两种方式
- **跨框架实现**：同时提供 NumPy 和 TensorFlow 版本
- **测试验证框架**：支持测试报告导出，便于作业提交
- **工程化改进**：ASCII 标记兼容 Windows 终端，避免编码问题

## 13. 注意事项

1. 确保 `train.txt` 和 `test.txt` 文件路径正确
2. 高斯基函数的宽度默认基于数据范围自动计算，可根据实际数据分布调整
3. 梯度下降法的学习率和迭代次数需手动调整
4. TensorFlow 版本需要额外安装 TensorFlow 库
