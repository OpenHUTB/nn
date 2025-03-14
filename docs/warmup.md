## 热身示例

NumPy 练习题文档
1. 导入库

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 临时指定matplotlib后端
import matplotlib.pyplot as plt
描述： 导入必要的库，numpy用于数组操作，matplotlib用于绘图。

2. 创建一维数组
np.array([4, 5, 6])
print("(1)输出a 的类型（type）\n", type(a))
print("(2)输出a的各维度的大小（shape）\n", a.shape)
print("(3)输出 a的第一个元素（值为4）\n", a[0])
输出：


(1) 输出a 的类型（type）
 <class 'numpy.ndarray'>
(2) 输出a的各维度的大小（shape）
 (3,)
(3) 输出 a的第一个元素（值为4）
 4
描述： 创建一个一维数组，输出其类型、维度大小和第一个元素。

3. 创建二维数组

b = np.array([[4, 5, 6], [1, 2, 3]])
print("(1)输出各维度的大小（shape）\n", b.shape)
print("(2)输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为4,5,2）\n", b[0, 0], b[0, 1], b[1, 1])
输出：


(1) 输出各维度的大小（shape）
 (2, 3)
(2) 输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为4,5,2）
 4 5 2
描述： 创建一个二维数组并输出其维度和特定元素的值。

4. 创建特殊矩阵

a = np.zeros((3, 3), dtype=int)
b = np.ones((4, 5))
c = np.eye(4)
d = np.random.random((3, 2))
描述：

a 是一个 3x3 的全零矩阵。
b 是一个 4x5 的全一矩阵。
c 是一个 4x4 的单位矩阵。
d 是一个 3x2 的随机数矩阵。
5. 创建更复杂的数组

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)
print(a[2, 3], a[0, 0])
输出：

2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
12 1
描述： 创建一个 3x4 的数组并打印数组和特定位置的元素。

6. 数组切片

b = a[0:2, 1:3]
print("(1)输出b\n", b)
print("(2) 输出b 的（0,0）这个元素的值\n", b[0, 0])
输出：


(1) 输出b
 [[2 3]
 [6 7]]
(2) 输出b 的（0,0）这个元素的值
 2
描述： 进行数组切片，选择指定范围的元素。

7. 数组行操作

c = a[1:3, :]
print("(1)输出 c \n", c)
print("(2) 输出 c 中第一行的最后一个元素\n", c[0, -1])
输出：


(1) 输出 c 
 [[ 5  6  7  8]
 [ 9 10 11 12]]
(2) 输出 c 中第一行的最后一个元素
 8
描述： 获取数组的最后两行并输出其中的元素。

8. 使用索引输出特定元素

a = np.array([[1, 2], [3, 4], [5, 6]])
print("输出:\n", a[[0, 1, 2], [0, 1, 0]])
输出：


输出:
 [1 4 5]
描述： 使用整数数组索引来同时选择多个元素。

9. 索引选择多个元素

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b = np.array([0, 2, 0, 1])
print("输出:\n", a[np.arange(4), b])
输出：

less
复制
编辑
输出:
 [ 1  6  7 11]
描述： 使用 np.arange 和索引数组选择特定的元素。

10. 数组加上常数并输出

a[np.arange(4), b] += 10
print("输出:", a)
输出：


输出:
 [[11  2  3]
 [ 4  5 16]
 [ 7  8  9]
 [10 21 12]]
描述： 对特定元素加上常数并输出更新后的数组。

24. 绘制简单的函数图像


x = np.arange(0, 100, 0.1)
y = x**2
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="y = x^2", color="blue")
plt.title("函数 y = x^2")
plt.xlabel("x 轴")
plt.ylabel("y 轴")
plt.grid(True)
plt.legend()
plt.show()
描述： 使用 matplotlib 绘制 y = x^2 的曲线图。

25. 绘制正弦和余弦函数

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y_sin, label="y = sin(x)", color="blue")
plt.title("正弦函数 y = sin(x)")
plt.xlabel("x 轴")
plt.ylabel("y 轴")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x, y_cos, label="y = cos(x)", color="red")
plt.title("余弦函数 y = cos(x)")
plt.xlabel("x 轴")
plt.ylabel("y 轴")
plt.grid(True)
plt.legend()
plt.show()
描述： 绘制正弦函数和余弦函数的曲线图。
[完整代码](https://github.com/OpenHUTB/nn/blob/main/src/chap01_warmup/numpy_tutorial.py) 。

