#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 2. numpy 的array操作
# 1.导入numpy库
# 2.建立一个一维数组 a 初始化为[4,5,6]，(1)输出a 的类型（type）(2)输出a的各维度的大小（shape）(3)输出 a的第一个元素（值为4）
print("\n第二题：")
a = np.array([4, 5, 6])
print(f"类型: {type(a)}")
print(f"形状: {a.shape}")
print(f"第一个元素: {a[0]}")

# 3.建立一个二维数组 b,初始化为 [ [4,5,6],[1,2,3]] (1)输出各维度的大小（shape）(2)输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为4,5,2）
print("\n第三题：")
b = np.array([[4, 5, 6], [1, 2, 3]])
print(f"形状: {b.shape}")
print(f"b[0,0]: {b[0, 0]}, b[0,1]: {b[0, 1]}, b[1,1]: {b[1, 1]}")

# 4. (1)建立一个全0矩阵 a, 大小为 3x3; 类型为整型（提示: dtype = int）(2)建立一个全1矩阵b,大小为4x5; (3)建立一个单位矩阵c ,大小为4x4; (4)生成一个随机数矩阵d, 大小为 3x2.
print("\n第四题：")
a = np.zeros((3, 3), dtype=int)
b = np.ones((4, 5))
c = np.eye(4)
d = np.random.random((3, 2))
print(f"a:\n{a}\nb:\n{b}\nc:\n{c}\nd:\n{d}")

# 5. 建立一个数组 a,(值为[[1,2,3,4], [5,6,7,8], [9,10,11,12]] ) ,(1)打印a; (2)输出  下标为(2,3),(0,0) 这两个数组元素的值
print("\n第五题：")
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"数组a:\n{a}")
print(f"a[2,3]: {a[2, 3]}, a[0,0]: {a[0, 0]}")

# 6.把上一题的 a数组的 0到1行 2到3列，放到b里面去，（此处不需要从新建立a,直接调用即可）(1),输出b;(2) 输出b 的（0,0）这个元素的值
print("\n第六题：")
b = a[:2, 1:3]
print(f"(1)\n{b}")
print(f"(2) b[0,0]: {b[0, 0]}")

# 7. 把第5题中数组a的最后两行所有元素放到 c中，（提示： a[1:2, :]）(1)输出 c ; (2) 输出 c 中第一行的最后一个元素（提示，使用 -1 表示最后一个元素）
print("\n第七题：")
c = a[1:3]
print(f"(1)\n{c}")
print(f"(2) 第一行最后一个元素: {c[0, -1]}")

# 8.建立数组a,初始化a为[[1,2], [3,4], [5,6]]，输出 （0,0）（1,1）（2,0）这三个元素（提示： 使用 print(a[[0,1,2], [0,1,0]]) ）
print("\n第八题：")
a = np.array([[1, 2], [3, 4], [5, 6]])
print(f"输出结果: {a[[0, 1, 2], [0, 1, 0]]}")

# 9.建立矩阵a ,初始化为[[1,2,3], [4,5,6], [7,8,9], [10,11,12]]，输出(0,0),(1,2),(2,0),(3,1) (提示使用 b = np.array([0, 2, 0, 1])    print(a[np.arange(4), b]))
print("\n第九题：")
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b = np.array([0, 2, 0, 1])
print(f"输出结果: {a[np.arange(4), b]}")

# 10.对9 中输出的那四个元素，每个都加上10，然后重新输出矩阵a.(提示： a[np.arange(4), b] += 10 ）
print("\n第十题：")
a[np.arange(4), b] += 10
print(f"更新后的矩阵a:\n{a}")

# 11. 执行 x = np.array([1, 2])，然后输出 x 的数据类型
print("\n第十一题：")
x = np.array([1, 2])
print(f"数据类型: {type(x)}")

# 12.执行 x = np.array([1.0, 2.0]) ，然后输出 x 的数据类类型
print("\n第十二题：")
x = np.array([1.0, 2.0])
print(f"数据类型: {type(x)}")

# 13.执行 x = np.array([[1,2],[3,4]], dtype=np.float64) ，y = np.array([[5,6],[7,8]], dtype=np.float64)，然后输出 x+y ,和 np.add(x,y)
print("\n第十三题：")
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print(f"x + y:\n{x + y}")
print(f"np.add(x, y):\n{np.add(x, y)}")

# 14. 利用 13题目中的x,y 输出 x-y 和 np.subtract(x,y)
print("\n第十四题：")
print(f"x - y:\n{x - y}")
print(f"np.subtract(x, y):\n{np.subtract(x, y)}")

# 15. 利用13题目中的x，y 输出 x*y ,和 np.multiply(x, y) 还有  np.dot(x,y),比较差异。然后自己换一个不是方阵的试试。
print("\n第十五题：")
print(f"x * y:\n{x * y}")
print(f"np.multiply(x, y):\n{np.multiply(x, y)}")
print(f"np.dot(x, y):\n{np.dot(x, y)}")

# 16. 利用13题目中的x,y,输出 x / y .(提示 ： 使用函数 np.divide())
print("\n第十六题：")
print(f"x / y:\n{x / y}")
print(f"np.divide(x, y):\n{np.divide(x, y)}")

# 17. 利用13题目中的x,输出 x的 开方。(提示： 使用函数 np.sqrt() )
print("\n第十七题：")
print(f"开方结果:\n{np.sqrt(x)}")

# 18.利用13题目中的x,y ,执行 print(x.dot(y)) 和 print(np.dot(x,y))
print("\n第十八题：")
print(f"x.dot(y):\n{x.dot(y)}")
print(f"np.dot(x, y):\n{np.dot(x, y)}")

# 19.利用13题目中的 x,进行求和。提示：输出三种求和 (1)print(np.sum(x)):   (2)print(np.sum(x，axis =0 ));   (3)print(np.sum(x,axis = 1))
print("\n第十九题：")
print(f"总和: {np.sum(x)}")
print(f"列求和: {np.sum(x, axis=0)}")
print(f"行求和: {np.sum(x, axis=1)}")

# 20.利用13题目中的 x,进行求平均数（提示：输出三种平均数(1)print(np.mean(x)) (2)print(np.mean(x,axis = 0))(3) print(np.mean(x,axis =1))）
print("\n第二十题：")
print(f"总平均: {np.mean(x)}")
print(f"列平均: {np.mean(x, axis=0)}")
print(f"行平均: {np.mean(x, axis=1)}")

# 21.利用13题目中的x，进行矩阵转置，然后输出转置后的结果，（提示： x.T 表示对 x 的转置）
print("\n第二十一题：")
print(f"转置后的矩阵:\n{x.T}")

# 22.利用13题目中的x,求e的指数（提示： 函数 np.exp()）
print("\n第二十二题：")
print(f"指数结果:\n{np.exp(x)}")

# 23.利用13题目中的 x,求值最大的下标（提示(1)print(np.argmax(x)) ,(2) print(np.argmax(x, axis =0))(3)print(np.argmax(x),axis =1))
print("\n第二十三题：")
print(f"全局最大值索引: {np.argmax(x)}")
print(f"列最大值索引:\n{np.argmax(x, axis=0)}")
print(f"行最大值索引:\n{np.argmax(x, axis=1)}")

# 24.画图，y=x*x 其中 x = np.arange(0, 100, 0.1) （提示这里用到  matplotlib.pyplot 库）
print("\n第二十四题：")
x = np.arange(0, 100, 0.1)
plt.figure(figsize=(10, 6))
plt.plot(x, x**2, 'b', label='y=x²')
plt.title('y = x²')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()

# 25.画图。画正弦函数和余弦函数， x = np.arange(0, 3 * np.pi, 0.1)(提示：这里用到 np.sin() np.cos() 函数和 matplotlib.pyplot 库)
print("\n第二十五题：")
x = np.arange(0, 3 * np.pi, 0.1)
plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), 'b', label='sin(x)')
plt.plot(x, np.cos(x), 'r--', label='cos(x)')
plt.title('Trigonometric Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()