# RBF 径向基函数网络非线性函数逼近

本项目实现了一个机器学习课程中的 RBF 径向基函数网络实验，用于演示如何用局部基函数组合逼近复杂非线性函数。项目只使用程序内部生成的一维回归数据，不使用 CARLA 或任何模拟器数据。

## 功能内容

- 构建带噪声的一维非线性回归数据集。
- 实现线性回归基线模型。
- 实现 RBF 径向基函数网络，包括中心点、基函数激活、岭回归输出层。
- 在测试集上比较线性模型和 RBF 模型的 MSE、MAE。
- 生成拟合对比图、RBF 基函数激活图和误差指标对比图。
- 输出 `metrics.json` 和 `rbf_predictions.csv`，便于在 PR 或课程文档中展示运行结果。

## 运行方法

```bash
python src/rbf_function_approximation/rbf_demo.py --output docs/pr_assets/rbf_function_approximation
```

## 验证方法

```bash
python src/rbf_function_approximation/tests/test_rbf_demo.py
python -m py_compile src/rbf_function_approximation/rbf_demo.py src/rbf_function_approximation/tests/test_rbf_demo.py
```

## 项目意义

RBF 网络是神经网络课程中连接传统机器学习和前馈神经网络的重要模型。它通过局部响应的径向基函数将输入映射到高维特征空间，再用线性输出层完成回归，能够直观展示“特征映射 + 线性组合”的学习思想，也适合作为后续 MLP、核方法和深度表示学习的对照实验。
