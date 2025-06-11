#!/usr/bin/env python3
# coding: utf-8
# numpy 的 array 操作

# 1. 导入 numpy 和 matplotlib 库
import numpy as np                    # 导入 numpy 并简写为 np，常规用法
import matplotlib                     # 导入完整的 matplotlib 库（一般直接用pyplot即可）
import matplotlib.pyplot as plt       # 导入 matplotlib.pyplot 用于画图

# 2. 创建一维数组并输出类型、形状和第一个元素
print("第二题：\n")
a = np.array([4, 5, 6])               # 创建一维数组 a
print("(1) 输出 a 的类型（type）\n", type(a))         # 输出 a 的类型（应为 numpy.ndarray）
print("(2) 输出 a 的各维度的大小（shape）\n", a.shape) # 输出 a 的形状（应为 (3,)）
print("(3) 输出 a 的第一个元素（element）\n", a[0])    # 输出 a 的第一个元素（4）

# 3. 创建二维数组并输出形状与指定元素
print("第三题：\n")
b = np.array([[4, 5, 6], [1, 2, 3]])       # 创建二维数组 b
print("(1) 输出各维度的大小（shape）\n", b.shape)     # 输出 (2, 3)
print("(2) 输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为 4,5,2）\n", b[0, 0], b[0, 1], b[1, 1])

# 4. 创建特殊数组：全0矩阵、全1矩阵、单位矩阵、随机数矩阵
print("第四题：\n")
a = np.zeros((3, 3), dtype=int)            # 3x3 全0整型矩阵
b = np.ones((4, 5))                        # 4x5 全1浮点型矩阵
c = np.eye(4)                              # 4x4 单位矩阵（对角线为1，其余为0）
np.random.seed(42)                         # 固定随机种子，保证结果可复现
d = np.random.random((3, 2))               # 3x2 随机0~1浮点数

# 5. 创建并访问二维数组中的特定元素
print("第五题：\n")
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) # 3x4数组
print(a)                        # 打印完整数组
print(a[2, 3], a[0, 0])         # 打印下标(2,3)与(0,0)的元素（12, 1）

# 6. 数组切片，选取部分行列
print("第六题：\n")
b = a[0:2, 2:4]                 # 取第0和1行，第2和3列（不包含2和4）
print("(1) 输出 b\n", b)
print("(2) 输出 b 的（0,0）这个元素的值\n", b[0, 0]) # b[0,0]=a[0,2]=3

# 7. 提取数组的最后两行，并访问最后一个元素
print("第七题：\n")
c = a[-2:, :]                   # 取倒数两行（所有列）
print("(1) 输出 c \n", c)
print("(2) 输出 c 中第一行的最后一个元素\n", c[0, -1]) # c[0,-1]=a[1,3]=8

# 8. 使用花式索引选取元素
print("第八题：\n")
a = np.array([[1, 2], [3, 4], [5, 6]]) # 3x2数组
# 分别取(0,0)、(1,1)、(2,0)
print("输出:\n", a[[0, 1, 2], [0, 1, 0]])

# 9. 高级索引：每行提取不同列的元素
print("第九题：\n")
a = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9], 
              [10, 11, 12]])
b = np.array([0, 2, 0, 1])  # 每行取的列号
print("输出:\n", a[np.arange(4), b])  # 分别取[1,6,7,11]

# 10. 用高级索引修改数组中的元素
print("第十题：\n")
a[np.arange(4), b] += 10      # 把上面那四个元素分别加10
print("输出:", a)

# 11. 查看数组数据类型（整型）
print("第十一题：\n")
x = np.array([1, 2])
print("输出:", x.dtype)        # int64

# 12. 查看数组数据类型（浮点型）
print("第十二题：\n")
x = np.array([1.0, 2.0])
print("输出:", x.dtype)        # float64

# 13. 两个数组的加法
print("第十三题：\n")
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print("x+y\n", x + y)                        # 元素对应相加
print("np.add(x,y)\n", np.add(x, y))         # numpy函数相加

# 14. 两个数组的减法
print("第十四题：\n")
print("x-y\n", x - y)
print("np.subtract(x,y)\n", np.subtract(x, y))

# 15. 两个数组的乘法与点乘
print("第十五题：\n")
print("x*y\n", x * y)                        # 元素对应相乘
print("np.multiply(x, y)\n", np.multiply(x, y))
print("np.dot(x,y)\n", np.dot(x, y))         # 矩阵点乘（行✕列求和）

# 16. 两个数组的除法
print("第十六题：\n")
print("x/y\n", x / y)                        # 元素逐一相除
print("np.divide(x,y)\n", np.divide(x, y))

# 17. 求数组每个元素的平方根
print("第十七题：\n")
print("np.sqrt(x)\n", np.sqrt(x))

# 18. 点乘的两种写法
print("第十八题：\n")
print("x.dot(y)\n", x.dot(y))
print("np.dot(x,y)\n", np.dot(x, y))

# 19. 求和：全部、按列、按行
print("第十九题：\n")
print("print(np.sum(x)):", np.sum(x))            # 全部元素求和
print("print(np.sum(x, axis = 0))", np.sum(x, axis = 0)) # 按列（竖直方向）求和
print("print(np.sum(x, axis = 1))", np.sum(x, axis = 1)) # 按行（水平方向）求和

# 20. 求平均值：全部、按列、按行
print("第二十题：\n")
print("print(np.mean(x))", np.mean(x))           # 全部均值
print("print(np.mean(x,axis = 0))", np.mean(x, axis=0))  # 按列均值
print("print(np.mean(x,axis = 1))", np.mean(x, axis=1))  # 按行均值

# 21. 求矩阵转置
print("第二十一题：\n")
print("x 转置后的结果:\n", x.T)                   # x 的转置

# 22. 求 e 的指数（自然对数的底的幂）
print("第二十二题：\n")
print("e 的指数：np.exp(x)")  
print(np.exp(x))                                 # 对x中每个元素求e的幂

# 23. 求最大值的下标（全局、按列、按行）
print("第二十三题：\n")
print("全局最大值的下标:", np.argmax(x))          # 扁平化后最大值下标
print("每列最大值的下标:", np.argmax(x, axis=0))  # 按列
print("每行最大值的下标:", np.argmax(x, axis=1))  # 按行

# 24. 用 matplotlib 画二次函数 y = x^2
print("第二十四题：\n")
x = np.arange(0, 100, 0.1)           # 生成0~99.9，步长0.1的数列
y = x * x                            # y = x^2
plt.figure(figsize=(10, 6))          # 新建图像窗口
plt.plot(x, y, label="y = x^2", color="blue")   # 画y = x^2曲线
plt.title("Plot of y = x^2")                   # 标题
plt.xlabel("x")                                # x轴标签
plt.ylabel("y")                                # y轴标签
plt.grid(True, alpha=0.5)                      # 显示网格
plt.legend(loc='upper right')                  # 显示图例
plt.show()                                     # 显示图像

# 25. 画正弦和余弦函数
print("第二十五题：\n")
x = np.arange(0, 3 * np.pi, 0.1)     # 生成0~3π的x序列
y_sin = np.sin(x)                    # 正弦函数值
y_cos = np.cos(x)                    # 余弦函数值
plt.figure(figsize=(10, 6))
plt.plot(x, y_sin, label="y = sin(x)", color="blue")  # 画正弦曲线
plt.plot(x, y_cos, label="y = cos(x)", color="red")   # 画余弦曲线
plt.title("Sine and Cosine Functions")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
