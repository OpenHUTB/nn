#!/usr/bin/env python3
# coding: utf-8
# numpy 的 array 操作

# 1. 导入 numpy 库
import numpy as np  # 将 numpy 库命名为 np

import matplotlib
import matplotlib.pyplot as plt  # 导入 matplotlib 库并将其命名为 plt
# import 放一起代码美观
matplotlib.use('TkAgg')  # 关键代码，临时指定 matplotlib 后端代码，指定 TkAgg 可以确保图形能在标准窗口中正常渲染

# 2. 建立一个一维数组 a 初始化为 [4, 5, 6]，(1) 输出 a 的类型（type）(2) 输出 a 的各维度的大小（shape）(3) 输出 a 的第一个元素（element）
print("第二题：\n")
     
a = np.array([4, 5, 6])

print("(1) 输出 a 的类型（type）\n", type(a))
print("(2) 输出 a 的各维度的大小（shape）\n", a.shape)
print("(3) 输出 a 的第一个元素（element）\n", a[0])
# 使用 array() 函数创建数组，函数可基于序列型的对象。创建了一个一维数组 a，并输出其类型（numpy.ndarray）、形状（(3,)） 和第一个元素（4）。

# 3. 建立一个二维数组 b, 初始化为 [ [4, 5, 6], [1, 2, 3]] (1) 输出二维数组 b 的形状（shape）（输出值为（2,3））(2) 输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为 4,5,2）
print("第三题：\n")
b = np.array([[4, 5, 6], [1, 2, 3]])  # 创建一个二维数组 b
print("(1) 输出各维度的大小（shape）\n", b.shape)  # 输出数组 b 的形状
print("(2) 输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为 4,5,2）\n", b[0, 0], b[0, 1], b[1, 1])  # 输出数组 b 的指定元素

# 4. (1) 建立一个全 0 矩阵 a, 大小为 3x3; 类型为整型（提示: dtype = int）(2) 建立一个全 1 矩阵 b, 大小为 4x5;  (3) 建立一个单位矩阵 c ,大小为 4x4; (4) 生成一个随机数矩阵 d,
# 大小为 3x2.
print("第四题：\n")

# 全 0 矩阵，3x3
a = np.zeros((3, 3), dtype=int)
# 全 1 矩阵，4x5
b = np.ones((4, 5))
# 单位矩阵，4x4
c = np.eye(4)
# 随机数矩阵，3x2
np.random.seed(42)  # 在生成随机数前设置种子
d = np.random.random((3, 2))

# 5. 建立一个数组 a,(值为 [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] ) ,(1) 打印 a; (2) 输出数组中下标为 (2,3),(0,0) 这两个元素的值
print("第五题：\n")
# 创建一个 3x4 的二维数组 a，值为 [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# 输出数组 a
print(a)
# 输出数组 a 中下标为 (2,3) 和 (0,0) 的两个元素的值
print(a[2, 3], a[0, 0])

# 6. 把上一题的 a 数组的 0 到 1 行，2 到 3 列，放到 b 里面去，（此处不需要从新建立 a, 直接调用即可）(1) 输出 b; (2) 输出 b 数组中（0,0）这个元素的值
print("第六题：\n")

# 0:2 表示取第 0 行（包含）到第 2 行（不包含），即实际取第 0 行和第 1 行；2:4 表示取第 2 列（包含）到第 4 列（不包含），即实际取第 2 列和第 3 列
b = a[0:2, 2:4]
print("(1) 输出 b\n", b)
print("(2) 输出 b 的（0,0）这个元素的值\n", b[0, 0])

# 7. 把第 5 题中数组 a 的最后两行所有元素放到 c 中 (1) 输出 c ; (2) 输出 c 中第一行的最后一个元素（提示，使用 -1 表示最后一个元素）
print("第七题：\n")

# -2: 提取最后两行的所有列元素
c = a[-2:, :]  
print("(1) 输出 c \n", c)
# -1 表示选取该行的最后一个元素
print("(2) 输出 c 中第一行的最后一个元素\n", c[0, -1]) 

# 8. 建立数组 a, 创建数组 a 为 [[1, 2], [3, 4], [5, 6]]，输出 （0,0）（1,1）（2,0） 这三个元素（提示： 使用 print(a[[0, 1, 2], [0, 1, 0]]) ）
print("第八题：\n")

a = np.array([[1, 2], [3, 4], [5, 6]])
# a[行索引列表, 列索引列表] 表示依次获取到的元素是第 0 行第 0 列的 1 、第 1 行第 1 列的 4 、第 2 行第 0 列的 5 
print("输出:\n", a[[0, 1, 2], [0, 1, 0]])  

# 9. 建立矩阵 a , 初始化为 [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]，输出 (0,0),(1,2),(2,0),(3,1)
print("第九题：\n")

# 创建一个 4x3 的二维数组 a
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# 创建一个一维数组 b，表示每行需要提取的列索引
b = np.array([0, 2, 0, 1])
# np.arange(4) 生成行索引 [0,1,2,3]，b = [0,2,0,1] 作为列索引。 组合后提取的元素位置为： (0,0) → 1 (1,2) → 6 (2,0) → 7 (3,1) → 11
print("输出:\n", a[np.arange(4), b]) 

# 10. 对 9 中输出的那四个元素，每个都加上 10，然后重新输出矩阵 a.(提示： a[np.arange(4), b] += 10 ）
print("第十题：\n")

a[np.arange(4), b] += 10  # 利用 numpy 的高级索引功能，行用 np.arange(4) 生成，列用 b 数组指定，进行加法操作
print("输出:", a)

# 11. 执行 x = np.array([1, 2])，然后输出 x 的数据类型
print("第十一题：\n")

x = np.array([1, 2])
print("输出:", type(x))

# 12. 执行 x = np.array([1.0, 2.0]) ，然后输出 x 的数据类类型
print("第十二题：\n")

x = np.array([1.0, 2.0])
print("输出:", type(x))

# 13. 执行 x = np.array([[1, 2], [3, 4]], dtype=np.float64) ，y = np.array([[5, 6], [7, 8]], dtype = np.float64)，然后输出 x+y , 和 np.add(x,y)
print("第十三题：\n")

x = np.array([[1, 2], [3, 4]], dtype=np.float64)  # 创建一个二维的 NumPy 数组 x，其元素为 [[1, 2], [3, 4]]，数据类型指定为 np.float64（双精度浮点数）
y = np.array([[5, 6], [7, 8]], dtype=np.float64)  # 创建另一个二维的 NumPy 数组 y，其元素为 [[5, 6], [7, 8]]，数据类型同样为 np.float64

print("x+y\n", x + y)  # 使用 + 运算符对两个数组进行逐元素相加操作，并将结果打印出来

print("np.add(x,y)\n", np.add(x, y))  # np.add 是 NumPy 库中用于数组相加的函数，同样会对两个数组进行逐元素相加

# 14. 利用 13 题目中的 x,y 输出 x-y 和 np.subtract(x,y)
print("第十四题：\n")

print("x-y\n", x - y)
print("np.subtract(x,y)\n", np.subtract(x, y))

# 15. 利用 13 题目中的 x,y 输出 x*y , 和 np.multiply(x,y) 还有 np.dot(x,y), 比较差异。然后自己换一个不是方阵的试试。
print("第十五题：\n")

print("x*y\n", x * y)  # 对应位置相乘
print("np.multiply(x, y)\n", np.multiply(x, y))  # 对应位置相乘
print("np.dot(x,y)\n", np.dot(x, y))  # 标准的行乘列求和

# 16. 利用 13 题目中的 x,y, 输出 x / y .(提示：使用函数 np.divide())
print("第十六题：\n")

print("x/y\n", x / y)  # 逐元素除法
print("np.divide(x,y)\n", np.divide(x, y))  # 逐元素除法

# 17. 数组开方
print("第十七题：\n")

print("np.sqrt(x)\n", np.sqrt(x))  # 对每个元素开平方

# 18. 矩阵乘法
print("第十八题：\n")

print("x.dot(y)\n", x.dot(y))  # 矩阵乘法
print("np.dot(x,y)\n", np.dot(x, y))  # 同上

# 19. 数组求和
print("第十九题：\n")

print("print(np.sum(x)):", np.sum(x))  # 所有元素求和
print("print(np.sum(x, axis=0))", np.sum(x, axis=0))  # 按列求和
print("print(np.sum(x, axis=1))", np.sum(x, axis=1))  # 按行求和

# 20. 数组求平均
print("第二十题：\n")

print("print(np.mean(x))", np.mean(x))  # 全局平均
print("print(np.mean(x,axis = 0))", np.mean(x, axis=0))  # 列平均
print("print(np.mean(x,axis = 1))", np.mean(x, axis=1))  # 行平均

# 21. 矩阵转置
print("第二十一题：\n")
print("x 转置后的结果:\n", x.T)  # 行列互换

# 22. 指数运算
print("第二十二题：\n")

print("e 的指数：np.exp(x)")  
print(np.exp(x))  # 对每个元素求e的指数

# 23. 最大值索引
print("第二十三题：\n")

print("全局最大值的下标:", np.argmax(x))  # 整个数组中最大值的索引
print("每列最大值的下标:", np.argmax(x, axis=0))  # 每列最大值的索引
print("每行最大值的下标:", np.argmax(x, axis=1))  # 每行最大值的索引

# 24. 绘制二次函数图像
print("第二十四题：\n")

# 生成x值（0到100，步长0.1）
x = np.arange(0, 100, 0.1)
# 计算y=x^2
y = x * x

# 创建图形窗口
plt.figure(figsize=(10, 6))
# 绘制曲线
plt.plot(x, y, label="y = x^2", color="blue")
# 添加标题和标签
plt.title("Plot of y = x^2")
plt.xlabel("x")
plt.ylabel("y")
# 添加网格和图例
plt.grid(True, alpha=0.5)
plt.legend(loc='upper right')
# 显示图形
plt.show()

# 25. 绘制正弦和余弦函数
print("第二十五题：\n")

# 生成x值（0到3π，步长0.1）
x = np.arange(0, 3 * np.pi, 0.1)
# 计算正弦和余弦值
y_sin = np.sin(x)
y_cos = np.cos(x)

# 创建图形窗口
plt.figure(figsize=(10, 6))
# 绘制两条曲线
plt.plot(x, y_sin, label="y = sin(x)", color="blue")
plt.plot(x, y_cos, label="y = cos(x)", color="red")
# 添加标题和标签
plt.title("Sine and Cosine Functions")
plt.xlabel("x")
plt.ylabel("y")
# 添加网格和图例
plt.grid(True)
plt.legend()
# 显示图形
plt.show()
