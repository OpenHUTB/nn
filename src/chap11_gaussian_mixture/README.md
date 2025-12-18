# 混合高斯模型

## 问题描述：

​	生成一组不带标签的聚类数据，使用高斯混合模型（Gaussian Mixture Models）进行无监督聚类。






## 数据集: 

 	本次作业不提供数据，请自己生成数据。





## 题目要求： 

​		推荐使用python 及numpy 编写代码，不建议使用框架。


## 功能特点

- 生成二维高斯分布混合数据
- 数值稳定的 logsumexp 实现
- 完整的 EM 算法（E 步 + M 步）
- 手动计算对数似然和协方差更新
- 可视化真实簇和预测簇对比
- 易于扩展，可用于真实数据

## 使用方法

1. 安装依赖
```bash
pip install numpy matplotlib
```
2. 运行程序
```bash
python gmm.py
```

程序将会：
- 生成二维高斯混合数据；
- 使用 GMM 进行聚类建模；
- 绘制对比图（真实簇 vs 预测簇）。



# GMM.py 说明文档

本项目使用 NumPy 从零实现了高斯混合模型（Gaussian Mixture Model, GMM）及其训练过程中的 EM 算法，并应用于模拟生成的二维高斯混合分布数据进行聚类分析与可视化。

## 特性 Features

* 使用多维高斯分布生成模拟聚类数据。
* 完整实现 GMM 的 E 步和 M 步过程。
* 自定义 `logsumexp` 实现以增强数值稳定性。
* 自动判断收敛（基于对数似然变化）。
* 可视化真实标签与预测结果对比。

## 文件结构

```bash
GMM-from-scratch/
├── gmm.py         # 主程序文件（包括生成数据、模型定义和训练）
├── README.md      # 项目说明文件（当前文档）
```

## 使用的主要技术

* NumPy：实现矩阵计算、高斯分布采样、数值稳定计算等
* Matplotlib：绘制真实与预测聚类的可视化图像
* GMM：基于最大期望（EM）算法训练高斯混合模型

## 运行方法

确保已安装 Python 3，并安装所需依赖项：

```bash
pip install numpy matplotlib
```

然后直接运行程序：

```bash
python gmm.py
```

运行后将显示两个子图：

* 左图：真实高斯分布标签
* 右图：GMM 聚类预测结果

## 主要函数说明

* `generate_data(n_samples=1000)`：生成带标签的二维高斯混合分布数据。
* `logsumexp(log_p)`：数值稳定的对数和运算，避免溢出。
* `GaussianMixtureModel` 类：

  * `fit(X)`：使用 EM 算法拟合数据。
  * `_log_gaussian(X, mu, sigma)`：计算样本在高斯分布下的对数概率密度。

## 注意事项

* GMM 对初始值敏感，可使用更复杂的初始化方式（如KMeans++）以提高稳定性。
* 当前模型未使用正则项进行模型复杂度控制（如 AIC/BIC）。
* 仅用于教学与实验目的，非工业级实现。

## 参考资料

* PRML《Pattern Recognition and Machine Learning》
* 《机器学习》周志华
* [https://scikit-learn.org/stable/modules/mixture.html](https://scikit-learn.org/stable/modules/mixture.html)

