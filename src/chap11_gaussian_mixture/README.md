# 混合高斯模型（改进版）

## 问题描述

生成一组不带标签的聚类数据，使用高斯混合模型（Gaussian Mixture Model, GMM）完成无监督聚类。  
模型训练采用 EM（Expectation-Maximization）算法。

---

## 数据集

本实验不依赖外部公开数据集，使用程序自动生成二维高斯混合数据：

1. 三个高斯成分；
2. 每个成分具有不同均值和协方差；
3. 默认样本数为 1000；
4. 通过随机种子保证结果可复现。

---

## 题目要求

推荐使用 Python + NumPy 完成核心实现，不依赖现成 GMM 框架。  
本项目在原始教学版本基础上做了小幅工程化完善：可配置、可复现、可落盘保存结果。

---

## 功能特点

- 生成二维高斯分布混合数据；
- 数值稳定的 `logsumexp` 实现；
- 完整 EM 训练流程（E 步 + M 步）；
- 支持命令行参数配置；
- 自动保存聚类对比图与收敛曲线；
- 自动保存对数似然迭代日志（CSV）。

---

## 使用方法

1. 安装依赖

```bash
pip install numpy matplotlib
```

2. 运行主实验

```bash
python GMM.py --n-samples 1000 --n-components 3 --max-iter 100 --tol 1e-6 --random-state 42 --out-dir outputs
```

---

## 文件结构

```bash
gmm_upload/
├── GMM.py
├── README.md
└── outputs/
    ├── cluster_comparison.png
    ├── convergence_curve.png
    ├── iteration_log.csv
```

---

## 主要函数说明

- `generate_data(n_samples, random_state)`：生成带真实标签的二维混合高斯数据；
- `logsumexp(a)`：稳定计算 `log(sum(exp(a)))`；
- `GaussianMixtureModel.fit(X)`：执行 EM 训练流程；
- `GaussianMixtureModel.plot_convergence(...)`：绘制并可保存收敛曲线。

---

## 输出结果

程序运行后将得到以下文件：

1. `cluster_comparison.png`：真实簇与预测簇对比图；
2. `convergence_curve.png`：EM 收敛曲线；
3. `iteration_log.csv`：每轮对数似然记录。

---

## 注意事项

1. GMM 对初始化较敏感，随机种子会影响结果；
2. 协方差矩阵可能出现数值问题，代码中已加入小幅正则化处理；
3. 当前版本以教学演示为主，非工业生产版本。

---

## 参考资料

- PRML《Pattern Recognition and Machine Learning》
- 周志华《机器学习》
- https://scikit-learn.org/stable/modules/mixture.html
