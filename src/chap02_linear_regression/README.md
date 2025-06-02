# 线性回归



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