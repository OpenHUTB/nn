#!/usr/bin/env python3
# coding: utf-8

# numpy 练习题

# numpy 的array操作

# 1.导入numpy库
import numpy as np

# 导入matplotlib.pyplot库
import matplotlib

matplotlib.use('TkAgg')  # 关键代码，临时指定matplotlib后端代码
import matplotlib.pyplot as plt #导入matplotlib库并命名为plt库

# 2.建立一个一维数组 a 初始化为[4,5,6]，(1)输出a 的类型（type）(2)输出a的各维度的大小（shape）(3)输出 a的第一个元素（值为4）
print("第二题：\n")

a = np.array([4, 5, 6])
print("(1)输出a 的类型（type）\n", type(a))
print("(2)输出a的各维度的大小（shape）\n", a.shape)
print("(3)输出 a的第一个元素（值为4）\n", a[0])

# 3.建立一个二维数组 b,初始化为 [ [4, 5, 6],[1, 2, 3]] (1)输出各维度的大小（shape）(2)输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为4,5,2）
print("第三题：\n")

b = np.array([[4, 5, 6], [1, 2, 3]])
print("(1)输出各维度的大小（shape）\n", b.shape)
print("(2)输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为4,5,2）\n", b[0, 0], b[0, 1], b[1, 1])

# 4.  (1)建立一个全0矩阵 a, 大小为 3x3; 类型为整型（提示: dtype = int）(2)建立一个全1矩阵b,大小为4x5;  (3)建立一个单位矩阵c ,大小为4x4; (4)生成一个随机数矩阵d,
# 大小为 3x2.
print("第四题：\n")

a = np.zeros((3, 3), dtype=int)
b = np.ones((4, 5))
c = np.eye(4)
d = np.random.random((3, 2))

# 5. 建立一个数组 a,(值为[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] ) ,(1)打印a; (2)输出  下标为(2,3),(0,0) 这两个数组元素的值
print("第五题：\n")

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)
print(a[2, 3], a[0, 0])

# 6.把上一题的 a数组的 0到1行 2到3列，放到b里面去，（此处不需要从新建立a,直接调用即可）(1),输出b;(2) 输出b 的（0,0）这个元素的值
print("第六题：\n")

b = a[0:2, 1:3]
print("(1)输出b\n", b)
print("(2) 输出b 的（0,0）这个元素的值\n", b[0, 0])

# 7. 把第5题中数组a的最后两行所有元素放到 c中，（提示： a[1:2, :]）(1)输出 c ; (2) 输出 c 中第一行的最后一个元素（提示，使用 -1                 表示最后一个元素）
print("第七题：\n")

c = a[1:3, :]
print("(1)输出 c \n", c)
print("(2) 输出 c 中第一行的最后一个元素\n", c[0, -1])

# 8.建立数组a,初始化a为[[1, 2], [3, 4], [5, 6]]，输出 （0,0）（1,1）（2,0）这三个元素（提示： 使用 print(a[[0, 1, 2], [0, 1, 0]]) ）
print("第八题：\n")

a = np.array([[1, 2], [3, 4], [5, 6]])
print("输出:\n", a[[0, 1, 2], [0, 1, 0]])

# 9.建立矩阵a ,初始化为[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]，输出(0,0),(1,2),(2,0),(3,1) (提示使用 b = np.array([0,
# 2, 0, 1])    print(a[np.arange(4), b]))
print("第九题：\n")

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])# 创建一个 4x3 的二维数组 a
b = np.array([0, 2, 0, 1])# 创建一个一维数组 b，表示每行需要提取的列索引
print("输出:\n", a[np.arange(4), b])

# 10.对9 中输出的那四个元素，每个都加上10，然后重新输出矩阵a.(提示： a[np.arange(4), b] += 10 ）
print("第十题：\n")

a[np.arange(4), b] += 10
print("输出:", a)

# 11.  执行 x = np.array([1, 2])，然后输出 x 的数据类型
print("第十一题：\n")

x = np.array([1, 2])
print("输出:", type(x))

# 12.执行 x = np.array([1.0, 2.0]) ，然后输出 x 的数据类类型
print("第十二题：\n")

x = np.array([1.0, 2.0])
print("输出:", type(x))

# 13.执行 x = np.array([[1, 2], [3, 4]], dtype=np.float64) ，y = np.array([[5, 6], [7, 8]], dtype=np.float64)，然后输出 x+y ,和 np.add(x,y)
print("第十三题：\n")

x = np.array([[1, 2], [3, 4]], dtype=np.float64)# 创建一个二维的 NumPy 数组 x，其元素为 [[1, 2], [3, 4]]，数据类型指定为 np.float64（双精度浮点数）
y = np.array([[5, 6], [7, 8]], dtype=np.float64)# 创建另一个二维的 NumPy 数组 y，其元素为 [[5, 6], [7, 8]]，数据类型同样为 np.float64

print("x+y\n", x + y)# 使用 + 运算符对两个数组进行逐元素相加操作，并将结果打印出来

print("np.add(x,y)\n", np.add(x, y))# np.add 是 NumPy 库中用于数组相加的函数，同样会对两个数组进行逐元素相加

# 14. 利用 13题目中的x,y 输出 x-y 和 np.subtract(x,y)
print("第十四题：\n")

print("x-y\n", x - y)
print("np.subtract(x,y)\n", np.subtract(x, y))

# 15. 利用13题目中的x，y 输出 x*y ,和 np.multiply(x, y) 还有  np.dot(x,y),比较差异。然后自己换一个不是方阵的试试。
print("第十五题：\n")

print("x*y\n", x * y)
print("np.multiply(x, y)\n", np.multiply(x, y))
print("np.dot(x,y)\n", np.dot(x, y))

# 16. 利用13题目中的x,y,输出 x / y .(提示 ： 使用函数 np.divide())
print("第十六题：\n")

print("x/y\n", x / y)
print("np.divide(x,y)\n", np.divide(x, y))

# 17. 利用13题目中的x,输出 x的 开方。(提示： 使用函数 np.sqrt() )
print("第十七题：\n")

print("np.sqrt(x)\n", np.sqrt(x))

# 18.利用13题目中的x,y ,执行 print(x.dot(y)) 和 print(np.dot(x,y))
print("第十八题：\n")

print("x.dot(y)\n", x.dot(y))
print("np.dot(x,y)\n", np.dot(x, y))

# 19.利用13题目中的 x,进行求和。提示：输出三种求和 (1)print(np.sum(x)):   (2)print(np.sum(x，axis =0 ));   (3)print(np.sum(x,axis = 1))
print("第十九题：\n")

print("print(np.sum(x)):", np.sum(x))
print("print(np.sum(x, axis=0))", np.sum(x, axis=0))
print("print(np.sum(x, axis=1))", np.sum(x, axis=1))

# 20.利用13题目中的 x,进行求平均数（提示：输出三种平均数(1)print(np.mean(x)) (2)print(np.mean(x,axis = 0))(3) print(np.mean(x,axis =1))）
print("第二十题：\n")

print("print(np.mean(x))", np.mean(x))
print("print(np.mean(x,axis = 0))", np.mean(x, axis=0))
print("print(np.mean(x,axis = 1))", np.mean(x, axis=1))


# 21.利用13题目中的x，对x 进行矩阵转置，然后输出转置后的结果，（提示： x.T 表示对 x 的转置）
print("第二十一题：\n")

print("转置后的结果:\n", x.T)

# 22.利用13题目中的x,求e的指数（提示： 函数 np.exp()）
print("第二十二题：\n")

print(np.exp(x))


# 23.利用13题目中的 x,求值最大的下标（提示(1)print(np.argmax(x)) ,(2) print(np.argmax(x, axis =0))(3)print(np.argmax(x),axis =1))
print("第二十三题：\n")
print("print(np.argmax(x))", np.argmax(x))# 打印整个数组 x 中最大值的下标
print("print(np.argmax(x, axis=0))", np.argmax(x, axis=0))# 打印数组 x 沿着第 0 轴（通常是行方向）上每一列最大值的下标
print("print(np.argmax(x, axis=1))", np.argmax(x, axis=1))# 打印数组 x 沿着第 1 轴（通常是列方向）上每一行最大值的下标

# 24,画图，y=x*x 其中 x = np.arange(0, 100, 0.1) （提示这里用到  matplotlib.pyplot 库）
print("第二十四题：\n")

x = np.arange(0, 100, 0.1)
y = x * x

plt.figure(figsize=(10, 6))  # 设置图像大小
plt.plot(x, y, label="y = x^2", color="blue")  # 绘制曲线

# 添加标题和标签
plt.title("Plot of y = x^2")  # 图像标题
plt.xlabel("x")  # x 轴标签
plt.ylabel("y")  # y 轴标签

# 添加网格
plt.grid(True)

# 显示图例
plt.legend()

plt.show()

# 25.画图。画正弦函数和余弦函数， x = np.arange(0, 3 * np.pi, 0.1)(提示：这里用到 np.sin() np.cos() 函数和 matplotlib.pyplot 库)
print("第二十五题：\n")

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure(figsize=(10, 6))  # 设置图像大小 plt.plot(x, y, label="y = sin(x)", color="blue")  # 绘制曲线

# 26.xlabel() 和 ylabel用于设置x轴和y轴的标签,plot()用于绘制二维数据
plt.figure(figsize=(10, 6))  # 设置图像大小
plt.plot(x, y_sin, label="y = sin(x)", color="blue")  # 绘制曲线
plt.title("Plot of y = sin(x)")  # 图像标题
plt.xlabel("x")  # x 轴标签
plt.ylabel("y")  # y 轴标签
 
 # 添加网格
plt.grid(True)
 
 # 显示图例
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))  # 设置图像大小
plt.plot(x, y_cos, label="y = cos(x)", color="blue")  # 绘制曲线
plt.title("Plot of y = cos(x)")  # 图像标题
plt.xlabel("x")  # x 轴标签
plt.ylabel("y")  # y 轴标签
 
 # 添加网格
plt.grid(True)
 
 # 显示图例
plt.legend()
plt.show()
