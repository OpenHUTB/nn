#!/usr/bin/env python
# coding: utf-8

# #                                           numpy 练习题

#  

# ### numpy 的array操作

#1.导入numpy库
import numpy as np
#  导入matplotlib.pyplot库
import matplotlib
matplotlib.use('TkAgg')  # 关键代码,临时指定matplotlib后端代码
import matplotlib.pyplot as plt
# 设置 matplotlib 的后端和图形分辨率
plt.rcParams['figure.dpi'] = 300
# 2. 建立一个一维数组 a 初始化为[4,5,6]
a = np.array([4, 5, 6])
print("a 的类型:", type(a))
print("输出a 的各维度大小:", a.shape)
print("输出a 的第一个元素:", a[0])
# 3. 建立一个二维数组 b, 初始化为 [[4, 5, 6],[1, 2, 3]]
b = np.array([[4, 5, 6], [1, 2, 3]])
print("\n输出各维度的大小:", b.shape)
print("输出b(0,0):", b[0, 0])
print("输出b(0,1):", b[0, 1])
print("输出b(1,1):", b[1, 1])
 
# 4. 建立特殊矩阵
a_zeros = np.zeros((3, 3), dtype=int)
print("\n全0矩阵 a:\n", a_zeros)
b_ones = np.ones((4, 5))
print("全1矩阵 b:\n", b_ones)
c_eye = np.eye(4)
print("单位矩阵 c:\n", c_eye)
d_random = np.random.random((3, 2))
print("随机矩阵 d:\n", d_random)
 
# 5. 建立一个数组 a, 值为[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("\n打印 a:\n", a)
print("a(2,3):", a[2, 3])
print("a(0,0):", a[0, 0])
 
# 6. 把上一题的 a 数组的 0到1行 2到3列，放到 b 里面去
b = a[0:2, 2:4]
print("\n输出 b:\n", b)
print("b(0,0):", b[0, 0])
 
# 7. 把数组 a 的最后两行所有元素放到 c 中
c = a[-2:, :]
print("\n输出 c:\n", c)
print("c 中第一行的最后一个元素:", c[0, -1])
 
# 8. 建立数组 a, 初始化 a 为[[1, 2], [3, 4], [5, 6]]
a = np.array([[1, 2], [3, 4], [5, 6]])
print("\n输出 (0,0), (1,1), (2,0) 这三个元素:", a[[0, 1, 2], [0, 1, 0]])
 
# 9. 建立矩阵 a, 初始化为[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
indices = np.array([0, 2, 0, 1])
print("\n输出 (0,0), (1,2), (2,0), (3,1) 这四个元素:", a[np.arange(4), indices])
a[np.arange(4), indices] += 10
print("修改后的 a:\n", a)

# #### 10.对9 中输出的那四个元素，每个都加上10，然后重新输出矩阵a.(提示： a[np.arange(4), b] += 10 ）
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b = np.array([0, 2, 0, 1])
print("\n")
a[np.arange(4), b] += 10
print(a)

# #### 11.  执行 x = np.array([1, 2])，然后输出 x 的数据类型
print("\n")
x = np.array([1, 2])
print("输出x 的数据类型:", x.dtype)

# #### 12.执行 x = np.array([1.0, 2.0]) ，然后输出 x 的数据类类型
print("\n")
x = np.array([1.0, 2.0])
print("输出x 的数据类型:", x.dtype)

# #### 13.执行 x = np.array([[1, 2], [3, 4]], dtype=np.float64) ，y = np.array([[5, 6], [7, 8]], dtype=np.float64)，然后输出 x+y ,和 np.add(x,y)
print("\n")
x = np.array([[1, 2], [3, 4]], dtype=np.float64, )
y = np.array([[5, 6], [7, 8]], dtype=np.float64, )
print(x + y)
print(np.add(x, y))

# #### 14. 利用 13题目中的x,y 输出 x-y 和 np.subtract(x,y)
print("\n")
x = np.array([[1, 2], [3, 4]], dtype=np.float64, )
y = np.array([[5, 6], [7, 8]], dtype=np.float64, )
print(x - y)
print(np.subtract(x, y))

# #### 15. 利用13题目中的x，y 输出 x*y ,和 np.multiply(x, y) 还有  np.dot(x,y),比较差异。然后自己换一个不是方阵的试试。
print("\n")
x = np.array([[1, 2], [3, 4]], dtype=np.float64, )
y = np.array([[5, 6], [7, 8]], dtype=np.float64, )
print(x * y)
print(np.multiply(x, y))
print(np.dot(x, y))

# #### 16. 利用13题目中的x,y,输出 x / y .(提示 ： 使用函数 np.divide())
print("\n")
x = np.array([[1, 2], [3, 4]], dtype=np.float64, )
y = np.array([[5, 6], [7, 8]], dtype=np.float64, )
print(x / y)
print(np.divide(x, y))

# #### 17. 利用13题目中的x,输出 x的 开方。(提示： 使用函数 np.sqrt() )
print("\n")
x = np.array([[1, 2], [3, 4]], dtype=np.float64, )
y = np.array([[5, 6], [7, 8]], dtype=np.float64, )
print(np.sqrt(x))

# #### 18.利用13题目中的x,y ,执行 print(x.dot(y)) 和 print(np.dot(x,y))
print("\n")
x = np.array([[1, 2], [3, 4]], dtype=np.float64, )
y = np.array([[5, 6], [7, 8]], dtype=np.float64, )
print(x.dot(y))
print(np.dot(x, y))

# ##### 19.利用13题目中的 x,进行求和。提示：输出三种求和 (1)print(np.sum(x)):   (2)print(np.sum(x，axis =0 ));   (3)print(np.sum(x,axis = 1))
print("\n")
x = np.array([[1, 2], [3, 4]], dtype=np.float64, )
y = np.array([[5, 6], [7, 8]], dtype=np.float64, )
print(np.sum(x))
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))

# #### 20.利用13题目中的 x,进行求平均数（提示：输出三种平均数(1)print(np.mean(x)) (2)print(np.mean(x,axis = 0))(3) print(np.mean(x,axis =1))）
print("\n")
x = np.array([[1, 2], [3, 4]], dtype=np.float64, )
print(np.mean(x))
print(np.mean(x, axis=0))
print(np.mean(x, axis=1))

# #### 21.利用13题目中的x，对x 进行矩阵转置，然后输出转置后的结果，（提示： x.T 表示对 x 的转置）
print("\n")
x = np.array([[1, 2], [3, 4]], dtype=np.float64, )
transposed_x = x.T
print(transposed_x)

# #### 22.利用13题目中的x,求e的指数（提示： 函数 np.exp()）
print("\n")
x = np.array([[1, 2], [3, 4]], dtype=np.float64, )
print(np.exp(x))

# #### 23.利用13题目中的 x,求值最大的下标（提示(1)print(np.argmax(x)) ,(2) print(np.argmax(x, axis =0))(3)print(np.argmax(x),axis =1))
print("\n")
x = np.array([[1, 2], [3, 4]], dtype=np.float64, )
print(np.argmax(x))
print(np.argmax(x, axis=0))
print(np.argmax(x, axis=1))



# #### 24,画图，y=x*x 其中 x = np.arange(0, 100, 0.1) （提示这里用到  matplotlib.pyplot 库）
print("\n")
plt.rcParams['figure.dpi'] = 300

# 生成 x 值
x = np.arange(0, 100, 0.2)
# 计算对应的 y 值
y = x * x

# 创建图形并绘制曲线
plt.figure(figsize=(5, 3))
plt.plot(x, y, label='$y = x^2$', color='blue')

# 添加标题和标签
plt.title('Graph of $y = x^2$')
plt.xlabel('x')
plt.ylabel('y')

# 显示图例
plt.legend()

# 显示网格线
plt.grid(True)

# 显示图形
plt.show()


# #### 25.画图。画正弦函数和余弦函数， x = np.arange(0, 3 * np.pi, 0.1)(提示：这里用到 np.sin() np.cos() 函数和 matplotlib.pyplot 库)
print("\n")
plt.rcParams['figure.dpi'] = 300

#生成x值
x = np.arange(0, 3 * np.pi, 0.1)
#计算正弦余弦的值
y1 = np.sin(x)
y2 = np.cos(x)

#绘制图形并绘制曲线
plt.figure(figsize=(5, 3))
plt.plot(x, y1, label='$y = np.sin(x)$', color='blue')
plt.plot(x, y2, label='$y = np.cos(x)$', color='green')

#添加标题跟标签
plt.title('$y1 = np.sin(x)$, $y2 = np.cos(x)$')
plt.xlabel('x')
plt.ylabel('y')
#显示图例
plt.legend()
#显示网格线
plt.grid(True)

#显示图形
plt.show()