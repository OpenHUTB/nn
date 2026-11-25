# 热身示例

## 模块简介

本模块用于帮助初学者熟悉 NumPy 的基本操作，包括数组创建、切片、索引、矩阵运算以及简单绘图等内容。

## 代码位置

- 源码文件：`src/chap01_warmup/numpy_tutorial.py`

## 学习内容

本示例主要包括以下内容：

- 一维数组与二维数组的创建
- 矩阵形状与元素访问
- 数组切片与高级索引
- 常见数学运算
- NumPy 与 Matplotlib 基本绘图

## 运行方式

在项目根目录下执行：

```bash
python src/chap01_warmup/numpy_tutorial.py
```

## 说明

运行后会在终端输出各小题结果，并绘制简单函数图像。

## 完整代码

[完整代码](https://github.com/OpenHUTB/nn/blob/main/src/chap01_warmup/numpy_tutorial.py)

## 预期输出

运行后终端会依次输出：
数组形状: (5,)
矩阵乘法结果: [[19 22] [43 50]]
切片结果: [2 3 4]
同时弹出 sin(x) 函数图像窗口。

## 常见问题

**Q：运行报错 `ModuleNotFoundError: No module named 'numpy'`**

执行以下命令安装依赖：
```bash
pip install numpy matplotlib
```

**Q：图像窗口不弹出**

在代码末尾将 `plt.show()` 改为：
```python
plt.savefig('output.png')
```
