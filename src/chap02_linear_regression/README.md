# 线性回归

<<<<<<< HEAD
## 问题描述：

有一个函数![image](http://latex.codecogs.com/gif.latex?f%3A%20%5Cmathbb%7BR%7D%5Crightarrow%20%5Cmathbb%7BR%7D) ，使得。现 ![image](http://latex.codecogs.com/gif.latex?y%20%3D%20f%28x%29)在不知道函数 $f(\cdot)$的具体形式，给定满足函数关系的一组训练样本![image](http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Cleft%20%28%20x_%7B1%7D%2Cy_%7B1%7D%20%5Cright%20%29%2C...%2C%5Cleft%20%28%20x_%7BN%7D%2Cy_%7BN%7D%20%5Cright%20%29%20%5Cright%20%5C%7D%2CN%3D300)，请使用线性回归模型拟合出函数$y=f(x)$。

(可尝试一种或几种不同的基函数，如多项式、高斯或sigmoid基函数）

## 数据集: 

 	根据某种函数关系生成的train 和test 数据。

## 题目要求： 

- [ ] 按顺序完成 `exercise-linear_regression.ipynb`中的填空 
    1. 先完成最小二乘法的优化 (参考书中第二章 2.3节中的公式)
    1. 附加题：实现“多项式基函数”以及“高斯基函数”（可参考PRML）
    1. 附加题：完成梯度下降的优化 (参考书中第二章 2.3节中的公式)
    
- [ ] 参照`lienar_regression-tf2.0.ipnb`使用tensorflow2.0 使用梯度下降完成线性回归
- [ ] 使用训练集train.txt 进行训练，使用测试集test.txt 进行评估（标准差），训练模型时请不要使用测试集。

# 线性回归练习项目

## 项目简介
本项目实现了一个线性回归模型，支持多种基函数和参数优化方法。通过该代码，可以学习如何应用最小二乘法和梯度下降法进行模型训练，并使用不同的基函数（如多项式基函数、高斯基函数）来拟合数据。

---

## 功能特性
- **基函数支持**：
  - 恒等基函数（默认）
  - 多项式基函数（可指定特征数）
  - 高斯基函数（可指定中心点和宽度）
- **优化方法**：
  - 最小二乘法（解析解）
  - 梯度下降法（需指定学习率和迭代次数）
- **数据加载与预处理**：从文件加载数据并转换为 NumPy 数组。
- **模型评估**：计算预测值与真实值的标准差作为评估指标。
- **结果可视化**：绘制训练数据点、真实测试数据曲线及模型预测曲线。

---

## 文件结构
- `exercise-linear_regression.py`：主程序文件，包含以下功能：
  - 数据加载 (`load_data`)
  - 基函数实现（恒等、多项式、高斯）
  - 模型训练与优化（最小二乘法、梯度下降）
  - 模型评估与可视化

---

## 依赖环境
- **Python 3.x**
- 依赖库：
  ```bash
  numpy
  matplotlib
  ```

---

## 使用方法
1. **准备数据文件**：
   - 创建 `train.txt` 和 `test.txt`，每行包含一个样本的特征和标签（空格分隔）。
   - 示例数据格式：
     ```
     1.0 2.5
     2.0 4.8
     ...
     ```

2. **运行程序**：
   ```bash
   python exercise-linear_regression.py
   ```

3. **输出结果**：
   - 控制台输出训练集和测试集的预测标准差。
   - 弹出窗口显示数据点与预测曲线。

---

## 参数调整
- **切换基函数**：在 `main` 函数中修改 `basis_func`：
  ```python
  # 默认使用恒等基函数
  basis_func = identity_basis  # 可选：multinomial_basis 或 gaussian_basis
  ```
- **调整基函数参数**：例如，设置多项式阶数或高斯基数量：
  ```python
  # 多项式基函数（10 阶）
  phi1 = multinomial_basis(x_train, feature_num=10)
  
  # 高斯基函数（15 个基）
  phi1 = gaussian_basis(x_train, feature_num=15)
  ```

- **优化方法选择**：在 `main` 函数中修改 `use_gradient_descent` 参数以切换优化方法（需在代码中调整返回值）。

---

## 结果示例
- **控制台输出**：
  ```
  训练集预测值与真实值的标准差：3.2
  预测值与真实值的标准差：4.5
  ```
---
## 注意事项
1. 确保 `train.txt` 和 `test.txt` 文件路径正确。

2. 高斯基函数的宽度默认基于数据范围自动计算，可根据实际数据分布调整。

3. 梯度下降法的学习率和迭代次数需手动调整（代码中 `lr` 和 `epochs` 参数）。 

# [linear_regression-tf2.0.py](https://github.com/bbacxc/nn/blob/main/src/chap02_linear_regression/linear_regression-tf2.0.py)线性回归基函数模型说明文档

## 项目概述

本项目实现了一个使用不同基函数(basis function)的线性回归模型，用于拟合非线性数据。通过TensorFlow框架实现，并提供结果可视化功能。

## 主要功能

- 提供三种基函数选择：
  - 恒等基函数(identity_basis) - 普通线性回归
  - 多项式基函数(multinomial_basis) - 多项式特征回归
  - 高斯基函数(gaussian_basis) - 径向基函数回归
- 自定义线性回归模型实现
- 训练和评估指标计算
- 数据可视化展示

## 环境要求

- Python 3.x
- NumPy 科学计算库
- Matplotlib 绘图库
- TensorFlow 2.x 深度学习框架

## 使用方法

1. 准备训练数据`train.txt`和测试数据`test.txt`，每行包含用空格分隔的x,y值
2. 运行Jupyter notebook或Python脚本
3. 程序将自动执行以下操作：
   - 加载并预处理数据
   - 训练线性回归模型
   - 在训练集和测试集上评估性能
   - 显示原始数据与预测结果的对比图

## 自定义设置

可通过修改`load_data()`函数中的`basis_func`参数选择不同的基函数：

- `identity_basis` - 简单线性回归
- `multinomial_basis` - 多项式回归
- `gaussian_basis` - 径向基函数回归(默认)

## 训练参数

- 学习率: 0.1 (使用Adam优化器)
- 训练迭代次数: 1000
- 损失函数: 均方根误差(RMSE)

## 输出结果

程序运行后将输出:

- 训练过程中的损失值
- 训练集预测值与真实值的标准差
- 测试集预测值与真实值的标准差
- 可视化图表包含:
  - 训练数据点(红色圆点)
  - 测试集预测结果(黑色曲线)

## 文件说明

- `train.txt` - 训练数据文件
- `test.txt` - 测试数据文件
- (可选)包含完整实现的Jupyter notebook文件

注意：用户需要自行准备`train.txt`和`test.txt`数据文件，并放在与脚本相同的目录下。
=======
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
>>>>>>> upstream/main
