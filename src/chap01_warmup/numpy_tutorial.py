#!/usr/bin/env python3
# coding: utf-8
# numpy 的 array 操作

# 1. 导入 numpy 库
import numpy as np  # 将 numpy 库命名为 np
import matplotlib   # 导入Matplotlib库，用于数据可视化
import matplotlib.pyplot as plt  # 导入 matplotlib 库并将其命名为 plt
# import 放一起代码美观
matplotlib.use('TkAgg')  # 关键代码，临时指定 matplotlib 后端代码，指定 TkAgg 可以确保图形能在标准窗口中正常渲染

# 2. 建立一个一维数组 a 初始化为 [4, 5, 6]，(1) 输出 a 的类型（type）(2) 输出 a 的各维度的大小（shape）(3) 输出 a 的第一个元素（element）
print("第二题：\n")
# 创建一个一维NumPy数组，存储整数类型的数值
# 数组元素为[4, 5, 6]，数据类型默认推断为numpy.int64
# 形状：a.shape = (3,)，表示包含3个元素的一维数组     
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
# 随机数矩阵，3x2：设置随机种子（42）确保结果可复现，生成0-1之间的浮点数
np.random.seed(42)  #  固定随机种子，使随机结果可复现
d = np.random.random((3, 2))# 生成一个形状为(3, 2)的NumPy数组，其中每个元素都是0到1之间的随机浮点数

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

# 9：使用NumPy高级索引提取矩阵特定元素
print("第九题：\n")

# 创建一个4行3列的二维数组（矩阵）a，并初始化数值
# 矩阵结构：
# 第0行: [1,  2,  3]
# 第1行: [4,  5,  6] 
# 第2行: [7,  8,  9]
# 第3行: [10, 11, 12]
a = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9], 
              [10, 11, 12]])

# 创建一个列索引数组b，用于指定每行要提取的元素所在的列
# b[0]=0: 第0行取第0列
# b[1]=2: 第1行取第2列
# b[2]=0: 第2行取第0列 
# b[3]=1: 第3行取第1列
b = np.array([0, 2, 0, 1])

# 使用高级索引提取元素：
# np.arange(4) 生成行索引数组 [0,1,2,3]
# b 是对应的列索引数组 [0,2,0,1]
# 组合效果相当于：
# a[0,0] → 第0行第0列 → 1
# a[1,2] → 第1行第2列 → 6
# a[2,0] → 第2行第0列 → 7
# a[3,1] → 第3行第1列 → 11
print("输出:\n", a[np.arange(4), b])  # 输出: [1, 6, 7, 11]

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

print("x-y\n", x - y)# 打印直接使用减法运算符得到的 x 减 y 的结果
print("np.subtract(x,y)\n", np.subtract(x, y))# 打印使用 numpy 的 subtract 函数得到的 x 减 y 的结果

# 15. 利用 13 题目中的 x,y 输出 x*y , 和 np.multiply(x,y) 还有 np.dot(x,y), 比较差异。然后自己换一个不是方阵的试试。
print("第十五题：\n")

print("x*y\n", x * y)  # 对应位置相乘
print("np.multiply(x, y)\n", np.multiply(x, y))  # 对应位置相乘
print("np.dot(x,y)\n", np.dot(x, y))  # 标准的行乘列求和

# 16. 利用 13 题目中的 x,y, 输出 x / y .(提示：使用函数 np.divide())
print("第十六题：\n")

print("x/y\n", x / y)  # 逐元素除法
print("np.divide(x,y)\n", np.divide(x, y))  # 逐元素除法

# 17. 利用 13 题目中的 x, 输出 x 的 开方。(提示： 使用函数 np.sqrt() )
print("第十七题：\n")

print("np.sqrt(x)\n", np.sqrt(x))

# 18. 利用 13 题目中的 x,y , 执行 print(x.dot(y)) 和 print(np.dot(x,y))
print("第十八题：\n")

print("x.dot(y)\n", x.dot(y))  # 使用 dot 方法进行矩阵乘法
print("np.dot(x,y)\n", np.dot(x, y))  # 使用 np.dot 函数进行矩阵乘法

# 19. 利用 13 题目中的 x, 进行求和。提示：输出三种求和 (1)print(np.sum(x)):   (2)print(np.sum(x，axis =0 ));   (3)print(np.sum(x,axis = 1))
print("第十九题：\n")

print("print(np.sum(x)):", np.sum(x))  # 所有元素求和
print("print(np.sum(x, axis=0))", np.sum(x, axis=0))  # 按列求和（列维度）
print("print(np.sum(x, axis=1))", np.sum(x, axis=1))  # 按行求和（行维度）

# 20. 利用 13 题目中的 x, 进行求平均数（提示：输出三种平均数(1)print(np.mean(x)) (2)print(np.mean(x,axis = 0))(3) print(np.mean(x,axis =1))）
print("第二十题：\n")

print("print(np.mean(x))", np.mean(x))  # 全局均值
print("print(np.mean(x,axis = 0))", np.mean(x, axis=0))  # 列均值
print("print(np.mean(x,axis = 1))", np.mean(x, axis=1))  # 行均值

# 21. 利用 13 题目中的 x，对 x 进行矩阵转置，然后输出转置后的结果，（提示： x.T 表示对 x 的转置）
print("第二十一题：\n")

print("x 转置后的结果:\n", x.T)

# 22. 利用 13 题目中的 x, 求 e 的指数（提示： 函数 np.exp()）
print("第二十二题：\n")

print("e 的指数：np.exp(x)")  
print(np.exp(x))

# 23. 利用 13 题目中的 x, 求值最大的下标（提示 (1)print(np.argmax(x)) ,(2) print(np.argmax(x, axis =0))(3)print(np.argmax(x),axis =1))
print("第二十三题：\n")
print("全局最大值的下标:", np.argmax(x))  # 打印整个数组 x 中最大值的下标
print("每列最大值的下标:", np.argmax(x, axis=0))  # 打印数组 x 沿着第 0 轴（通常是行方向）上每一列最大值的下标
print("每行最大值的下标:", np.argmax(x, axis=1))  # 打印数组 x 沿着第 1 轴（通常是列方向）上每一行最大值的下标

# 24. 画图，y=x*x 其中 x = np.arange(0, 100, 0.1) （使用 NumPy 和 Matplotlib 绘制了二次函数 y=x^2 的图像）
print("第二十四题：\n")

x = np.arange(0, 100, 0.1)  # 生成从 0 到 99.9 的数组，步长为 0.1，共 1000 个点 （注：np.arange() 遵循的是左闭右开原则）
y = x * x  # 计算每个 x 对应的 y=x^2 的值

plt.figure(figsize=(10, 6))  # 创建一个宽 10 英寸、高 6 英寸的图像窗口
plt.plot(x, y, label="y = x^2", color="blue")  # 绘制蓝色曲线，并设置图例标签为 y = x^2

# 添加标题和标签
plt.title("Plot of y = x^2")  # 图像标题
plt.xlabel("x")  # x 轴标签
plt.ylabel("y")  # y 轴标签

# 显示半透明网格线
plt.grid(True, alpha=0.5)

# 在右上角显示图例
plt.legend(loc='upper right')

plt.show()  # 显示绘制的图像

# 25. 画图：画正弦函数和余弦函数， x = np.arange(0, 3 * np.pi, 0.1)(提示：这里用到 np.sin() ，np.cos() 函数和 matplotlib.pyplot 库)
print("第二十五题：\n")

x = np.arange(0, 3 * np.pi, 0.1)  ## 生成从 0 到 3π 的 x 值，步长为 0.1
y_sin = np.sin(x)  # 计算对应的正弦值
y_cos = np.cos(x)  # 计算对应的余弦值

plt.figure(figsize=(10, 6))  # 创建一个图像窗口，设置大小为 10×6 英寸 
plt.plot(x, y_sin, label="y = sin(x)", color="blue")  # 绘制正弦函数曲线
plt.plot(x, y_cos, label="y = cos(x)", color="red")  # 绘制余弦函数曲线

# 添加标题和标签
plt.title("Sine and Cosine Functions")
plt.xlabel("x")  # 设置图像中 x 轴的标签为 "x"
plt.ylabel("y")  # 设置图像中 y 轴的标签为 "y"

# 添加网格线
plt.grid(True, alpha=0.5)

# 显示图例
plt.legend()
plt.tight_layout()

# 显示图像
plt.show()
