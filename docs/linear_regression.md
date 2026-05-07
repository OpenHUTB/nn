# 线性回归

本模块位于 `src/chap02_linear_regression`，用于演示如何用线性回归拟合一维数据，并比较不同基函数和优化方法的效果。

## 功能说明

- 从 `train.txt` 和 `test.txt` 读取训练、测试数据。
- 支持恒等基函数、多项式基函数和高斯基函数。
- 支持最小二乘法求解权重。
- 支持批量梯度下降求解权重。
- 使用 RMSE 评估预测结果。
- 提供基础单元测试，便于验证代码修改是否正确。

## 文件结构

```text
src/chap02_linear_regression/
├── exercise-linear_regression.py      # 线性回归主程序
├── linear_regression-tf2.0.py         # TensorFlow 版本示例
├── train.txt                          # 训练数据
├── test.txt                           # 测试数据
├── test_linear_regression.py          # pytest 单元测试
└── test_verify.py                     # 独立验证脚本
```

## 运行示例

进入模块目录：

```bash
cd src/chap02_linear_regression
python exercise-linear_regression.py
```

程序会读取同目录下的 `train.txt` 和 `test.txt`，输出训练集、测试集的 RMSE，并绘制拟合结果图。

## 运行测试

在项目根目录运行：

```bash
python -m pytest src/chap02_linear_regression/test_linear_regression.py
```

也可以同时运行项目已有的基础测试：

```bash
python -m pytest tests src/chap02_linear_regression/test_linear_regression.py
```

## 本次优化点

- 清理重复的函数定义和重复的程序入口。
- 增强 `load_data` 的数据格式校验，遇到空文件或列数错误时给出明确提示。
- 将设计矩阵构建逻辑封装为 `build_design_matrix`，减少训练和预测代码重复。
- 增加对基函数参数、优化器参数和评估输入形状的校验。
- 修复并补充 pytest 测试，覆盖数据读取、基函数、优化算法、模型训练和完整流程。
