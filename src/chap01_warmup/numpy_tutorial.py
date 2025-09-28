#!/usr/bin/env python3
# coding: utf-8
"""
NumPy 数组操作练习
"""

import numpy as np
import matplotlib.pyplot as plt


def question_2():
<<<<<<< HEAD
    """第二题：一维数组的基本操作"""
    print("=" * 50)
    print("第二题：一维数组的基本操作")
    print("=" * 50)

    # 创建一个一维NumPy数组，存储整数类型的数值
    # 数组元素为[4, 5, 6]，数据类型默认推断为numpy.int64
    # 形状：a.shape = (3,)，表示包含3个元素的一维数组
    a = np.array([4, 5, 6])

    print("(1) 输出 a 的类型（type）:")
    print(f"    {type(a)}")
    print("\n(2) 输出 a 的各维度的大小（shape）:")
    print(f"    {a.shape}")
    print("\n(3) 输出 a 的第一个元素（element）:")
    print(f"    {a[0]}")
    print()


def question_3():
    """第三题：二维数组的基本操作"""
    print("=" * 50)
    print("第三题：二维数组的基本操作")
    print("=" * 50)

    # 创建一个二维数组 b，包含两个子数组 [4, 5, 6] 和 [1, 2, 3]
    b = np.array([[4, 5, 6], [1, 2, 3]])

    print("(1) 输出各维度的大小（shape）:")
    print(f"    {b.shape}")  # 输出数组 b 的形状 - 应该是(2,3)
    print("\n(2) 输出 b(0,0)，b(0,1),b(1,1) 这三个元素:")
    print(f"    {b[0, 0]}, {b[0, 1]}, {b[1, 1]}")
    print()


def question_4():
    """第四题：特殊矩阵的创建"""
    print("=" * 50)
    print("第四题：特殊矩阵的创建")
    print("=" * 50)

    # 全 0 矩阵，3x3，指定数据类型为int
    a = np.zeros((3, 3), dtype=int)
    # 全 1 矩阵，4x5，默认数据类型为float
    b = np.ones((4, 5))
    # 单位矩阵，4x4(对角线为1，其余为0)
    c = np.eye(4)
    # 随机数矩阵，3x2：设置随机种子（42）确保结果可复现，生成0-1之间的浮点数
    np.random.seed(42)
    d = np.random.random((3, 2))

    print("(1) 全 0 矩阵 a:")
    print(a)
    print("\n(2) 全 1 矩阵 b:")
    print(b)
    print("\n(3) 单位矩阵 c:")
    print(c)
    print("\n(4) 随机矩阵 d:")
    print(d)
    print()

=======
    print("第二题：\n") #格式修改为缩进
# 创建一个一维NumPy数组，存储整数类型的数值
# 数组元素为[4, 5, 6]，数据类型默认推断为numpy.int64
# 形状：a.shape = (3,)，表示包含3个元素的一维数组     
    a = np.array([4, 5, 6])

    print("(1) 输出 a 的类型（type）\n", type(a))
    print("(2) 输出 a 的各维度的大小（shape）\n", a.shape)
    print("(3) 输出 a 的第一个元素（element）\n", a[0])
# 使用 array() 函数创建数组，函数可基于序列型的对象。创建了一个一维数组 a，并输出其类型（numpy.ndarray）、形状（(3,)） 和第一个元素（4）。
# 使用 type() 获取数组的类型(numpy.ndarray),使用 shape 属性查看数组维度信息(一维数组的形状表示为 (n,)),通过索引访问数组元素（索引从 0 开始）
# 3. 建立一个二维数组 b, 初始化为 [ [4, 5, 6], [1, 2, 3]] (1) 输出二维数组 b 的形状（shape）（输出值为（2,3））(2) 输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为 4,5,2）
def question_3():
    print("第三题：\n")
    b = np.array([[4, 5, 6], [1, 2, 3]])  # 创建一个二维数组 b
    print("(1) 输出各维度的大小（shape）\n", b.shape)  # 输出数组 b 的形状 - 应该是(2,3)
    print("(2) 输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为 4,5,2）\n", b[0, 0], b[0, 1], b[1, 1])  # 输出数组 b 的指定元素 - b[0,0]是4, b[0,1]是5, b[1,1]是2


# 4. (1) 建立一个全 0 矩阵 a, 大小为 3x3; 类型为整型（提示: dtype = int）(2) 建立一个全 1 矩阵 b, 大小为 4x5;  (3) 建立一个单位矩阵 c ,大小为 4x4; (4) 生成一个随机数矩阵 d,
# 大小为 3x2.
def question_4():
    print("第四题：")

# 全 0 矩阵，3x3，指定数据类型为int
    a = np.zeros((3, 3), dtype=int)
    print("(1) 全0矩阵:\n", a)
# 全 1 矩阵，4x5，默认数据类型为float
    b = np.ones((4, 5))
    print("(2) 全1矩阵:\n", b)
# 单位矩阵，4x4(对角线为1，其余为0)
    c = np.eye(4)
    print("(3) 单位矩阵:\n", c)
# 随机数矩阵，3x2：设置随机种子（42）确保结果可复现，生成0-1之间的浮点数
    np.random.seed(42)  #  固定随机种子，使随机结果可复现
    d = np.random.random((3, 2))# 生成一个形状为(3, 2)的NumPy数组，其中每个元素都是0到1之间的随机浮点数
    print("(4) 随机矩阵:\n", d)
>>>>>>> 0f85f57ab23edee0da20d895e783260fb8281b83

def question_5():
<<<<<<< HEAD
    """第五题：二维数组的创建和索引"""
    print("=" * 50)
    print("第五题：二维数组的创建和索引")
    print("=" * 50)

    # 创建一个 3x4 的二维数组 a
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    print("(1) 输出数组 a:")
    print(a)
    print("\n(2) 输出下标为 (2,3) 和 (0,0) 的元素:")
    print(f"    a[2,3] = {a[2, 3]}, a[0,0] = {a[0, 0]}")
    print()


def question_6():
    """第六题：数组切片操作"""
    print("=" * 50)
    print("第六题：数组切片操作")
    print("=" * 50)

    # 使用第五题中的数组 a
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # 切片操作：取第0-1行，第2-3列
    b = a[0:2, 2:4]

    print("(1) 输出切片 b:")
    print(b)
    print("\n(2) 输出 b 的 (0,0) 元素:")
    print(f"    {b[0, 0]}")
    print()
=======
    print("第五题：\n")
# 创建一个 3x4 的二维数组 a，值为 [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# 输出数组 a
    print(a)
# 输出数组 a 中下标为 (2,3) 和 (0,0) 的两个元素的值
    print(a[2, 3], a[0, 0])
    return a  # 添加返回语句

# 6. 把上一题的 a 数组的 0 到 1 行，2 到 3 列，放到 b 里面去，（此处不需要从新建立 a, 直接调用即可）(1) 输出 b; (2) 输出 b 数组中（0,0）这个元素的值
def question_6(a):
    print("第六题：")

# 0:2 表示取第 0 行（包含）到第 2 行（不包含），即实际取第 0 行和第 1 行；2:4 表示取第 2 列（包含）到第 4 列（不包含），即实际取第 2 列和第 3 列
    b = a[0:2, 2:4]
    print("(1) 输出 b\n", b)
    print("(2) 输出 b 的（0,0）这个元素的值\n", b[0, 0])

# 7. 把第 5 题中数组 a 的最后两行所有元素放到 c 中 (1) 输出 c ; (2) 输出 c 中第一行的最后一个元素（提示，使用 -1 表示最后一个元素）
def question_7(a):
    print("第七题：")

# -2: 提取最后两行的所有列元素
    c = a[-2:, :]  
    print("(1) 输出 c \n", c)
# -1 表示选取该行的最后一个元素
    print("(2) 输出 c 中第一行的最后一个元素\n", c[0, -1]) 

# 8. 建立数组 a, 创建数组 a 为 [[1, 2], [3, 4], [5, 6]]，输出 （0,0）（1,1）（2,0） 这三个元素（提示： 使用 print(a[[0, 1, 2], [0, 1, 0]]) ）
def question_8():
    print("第八题：")

    a = np.array([[1, 2], [3, 4], [5, 6]])
# a[行索引列表, 列索引列表] 表示依次获取到的元素是第 0 行第 0 列的 1 、第 1 行第 1 列的 4 、第 2 行第 0 列的 5 
    print("输出:\n", a[[0, 1, 2], [0, 1, 0]])  

# 9：使用NumPy高级索引提取矩阵特定元素
def question_9():
    print("第九题：")
>>>>>>> 0f85f57ab23edee0da20d895e783260fb8281b83


<<<<<<< HEAD
def question_7():
    """第七题：使用负索引进行切片"""
    print("=" * 50)
    print("第七题：使用负索引进行切片")
    print("=" * 50)

    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # 使用负索引取最后两行的所有列
    c = a[-2:, :]

    print("(1) 输出最后两行 c:")
    print(c)
    print("\n(2) 输出 c 中第一行的最后一个元素:")
    print(f"    {c[0, -1]}")
    print()


def question_8():
    """第八题：使用索引数组访问元素"""
    print("=" * 50)
    print("第八题：使用索引数组访问元素")
    print("=" * 50)

    a = np.array([[1, 2], [3, 4], [5, 6]])

    # 使用索引数组访问特定元素
    # 相当于获取 a[0,0], a[1,1], a[2,0]
    result = a[[0, 1, 2], [0, 1, 0]]

    print("输出指定元素:")
    print(result)
    print()


def question_9():
    """第九题：高级索引操作"""
    print("=" * 50)
    print("第九题：高级索引操作")
    print("=" * 50)

    # 创建4x3矩阵
    a = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])

    # 创建列索引数组
    b = np.array([0, 2, 0, 1])

    # 使用高级索引提取元素
    # 相当于获取 a[0,0], a[1,2], a[2,0], a[3,1]
    result = a[np.arange(4), b]

    print("输出高级索引结果:")
    print(result)
    print()


def question_10():
    """第十题：通过高级索引修改数组元素"""
    print("=" * 50)
    print("第十题：通过高级索引修改数组元素")
    print("=" * 50)

    a = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
    b = np.array([0, 2, 0, 1])

    print("原始数组:")
    print(a)

    # 通过高级索引修改元素值
    # 将 a[0,0], a[1,2], a[2,0], a[3,1] 的值加10
    a[np.arange(4), b] += 10

    print("\n修改后的数组:")
    print(a)
    print()


def question_11():
    """第十一题：整数数组的数据类型"""
    print("=" * 50)
    print("第十一题：整数数组的数据类型")
    print("=" * 50)

    x = np.array([1, 2])
    print("整数数组的数据类型:", x.dtype)
    print()


def question_12():
    """第十二题：浮点数数组的数据类型"""
    print("=" * 50)
    print("第十二题：浮点数数组的数据类型")
    print("=" * 50)

    x = np.array([1.0, 2.0])
    print("浮点数数组的数据类型:", x.dtype)
    print()


def question_13():
    """第十三题：数组的加法运算"""
    print("=" * 50)
    print("第十三题：数组的加法运算")
    print("=" * 50)

    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([[5, 6], [7, 8]], dtype=np.float64)

    print("数组 x:")
    print(x)
    print("\n数组 y:")
    print(y)
    print("\n(1) x + y:")
    print(x + y)
    print("\n(2) np.add(x, y):")
    print(np.add(x, y))
    print()

    return x, y

=======
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
    return a, b  # 添加返回语句

# 10. 对 9 中输出的那四个元素，每个都加上 10，然后重新输出矩阵 a.(提示： a[np.arange(4), b] += 10 ）
def question_10(a, b):
    print("第十题：")

    a[np.arange(4), b] += 10  # 利用 numpy 的高级索引功能，行用 np.arange(4) 生成，列用 b 数组指定，进行加法操作
    print("输出:", a)

# 11. 执行 x = np.array([1, 2])，然后输出 x 的数据类型
def question_11():
    print("第十一题：")

    x = np.array([1, 2])
# 创建一个包含整数1和2的NumPy数组
    print("输出:",x.dtype)

# 12. 执行 x = np.array([1.0, 2.0]) ，然后输出 x 的数据类类型
def question_12():
    print("第十二题：")

    x = np.array([1.0, 2.0])
    print("输出:", x.dtype)

# 13. 执行 x = np.array([[1, 2], [3, 4]], dtype=np.float64) ，y = np.array([[5, 6], [7, 8]], dtype = np.float64)，然后输出 x+y , 和 np.add(x,y)
def question_13():
    print("第十三题：")

    x = np.array([[1, 2], [3, 4]], dtype=np.float64)  # 创建一个二维的 NumPy 数组 x，其元素为 [[1, 2], [3, 4]]，数据类型指定为 np.float64（双精度浮点数）
    y = np.array([[5, 6], [7, 8]], dtype=np.float64)  # 创建另一个二维的 NumPy 数组 y，其元素为 [[5, 6], [7, 8]]，数据类型同样为 np.float64

    print("x+y\n", x + y)  # 使用 + 运算符对两个数组进行逐元素相加操作，并将结果打印出来

    print("np.add(x,y)\n", np.add(x, y))  # np.add 是 NumPy 库中用于数组相加的函数，同样会对两个数组进行逐元素相加
    return x, y  # 添加返回语句

# 14. 利用 13 题目中的 x,y 输出 x-y 和 np.subtract(x,y)
def question_14(x, y):
    print("第十四题：")

    print("x-y\n", x - y)# 打印直接使用减法运算符得到的 x 减 y 的结果
    print("np.subtract(x,y)\n", np.subtract(x, y))# 打印使用 numpy 的 subtract 函数得到的 x 减 y 的结果

# 15. 利用 13 题目中的 x,y 输出 x*y , 和 np.multiply(x,y) 还有 np.dot(x,y), 比较差异。然后自己换一个不是方阵的试试。
def question_15(x, y):
    print("第十五题：")
    print("x*y\n", x * y)  # 对应位置相乘
    print("np.multiply(x, y)\n", np.multiply(x, y))  # 对应位置相乘
    print("np.dot(x,y)\n", np.dot(x, y))  # 标准的行乘列求和

# 16. 利用 13 题目中的 x,y, 输出 x / y .(提示：使用函数 np.divide())
def question_16(x, y):
    print("第十六题：")

    print("x/y\n", x / y)  # 逐元素除法
    print("np.divide(x,y)\n", np.divide(x, y))  # 逐元素除法

# 17. 利用 13 题目中的 x, 输出 x 的 开方。(提示： 使用函数 np.sqrt() )
def question_17(x):
    print("第十七题：")

    print("np.sqrt(x)\n", np.sqrt(x))

# 18. 利用 13 题目中的 x,y , 执行 print(x.dot(y)) 和 print(np.dot(x,y))
def question_18(x, y):
    print("第十八题：")

    print("x.dot(y)\n", x.dot(y))  # 使用 dot 方法进行矩阵乘法
    print("np.dot(x,y)\n", np.dot(x, y))  # 使用 np.dot 函数进行矩阵乘法

# 19. 利用 13 题目中的 x, 进行求和。提示：输出三种求和 (1)print(np.sum(x)):   (2)print(np.sum(x，axis =0 ));   (3)print(np.sum(x,axis = 1))
def question_19(x):
    print("第十九题：")
    print("print(np.sum(x)):", np.sum(x))  # 所有元素求和
    print("print(np.sum(x, axis=0))", np.sum(x, axis=0))  # 按列求和（列维度）
    print("print(np.sum(x, axis = 1))", np.sum(x, axis = 1))  # 按行求和（行维度）

# 20. 利用 13 题目中的 x, 进行求平均数（提示：输出三种平均数(1)print(np.mean(x)) (2)print(np.mean(x,axis = 0))(3) print(np.mean(x,axis =1))）
def question_20(x):
    print("第二十题：")

    print("print(np.mean(x))", np.mean(x))  # 全局均值
    print("print(np.mean(x, axis = 0))", np.mean(x, axis=0))  # 列均值
    print("print(np.mean(x, axis = 1))", np.mean(x, axis=1))  # 行均值

# 21. 利用 13 题目中的 x，对 x 进行矩阵转置，然后输出转置后的结果，（提示： x.T 表示对 x 的转置）
def question_21(x):
    print("第二十一题：")

    print("x 转置后的结果:\n", x.T)

# 22. 利用 13 题目中的 x, 求 e 的指数（提示： 函数 np.exp()）
def question_22(x):
    print("第二十二题：")

    print("e 的指数：np.exp(x)")  
    print(np.exp(x))

# 23. 利用 13 题目中的 x, 求值最大的下标（提示 (1)print(np.argmax(x)) ,(2) print(np.argmax(x, axis =0))(3)print(np.argmax(x),axis =1))
def question_23(x):
    print("第二十三题：")
    print("全局最大值的下标:", np.argmax(x))          # 打印整个数组 x 中最大值的下标
    print("每列最大值的下标:", np.argmax(x, axis=0))   # 打印数组 x 沿着第 0 轴（通常是行方向）上每一列最大值的下标
    print("每行最大值的下标:", np.argmax(x, axis=1))   # 打印数组 x 沿着第 1 轴（通常是列方向）上每一行最大值的下标

# 24. 画图，y=x*x 其中 x = np.arange(0, 100, 0.1) （使用 NumPy 和 Matplotlib 绘制了二次函数 y=x^2 的图像）
def question_24(): #绘制二次函数 y = x^2 的图像。
>>>>>>> 0f85f57ab23edee0da20d895e783260fb8281b83

def question_14(x, y):
    """第十四题：数组的减法运算"""
    print("=" * 50)
    print("第十四题：数组的减法运算")
    print("=" * 50)

<<<<<<< HEAD
    print("(1) x - y:")
    print(x - y)
    print("\n(2) np.subtract(x, y):")
    print(np.subtract(x, y))
    print()


def question_15(x, y):
    """第十五题：数组的乘法运算比较"""
    print("=" * 50)
    print("第十五题：数组的乘法运算比较")
    print("=" * 50)

    print("(1) 逐元素乘法 x * y:")
    print(x * y)
    print("\n(2) 逐元素乘法 np.multiply(x, y):")
    print(np.multiply(x, y))
    print("\n(3) 矩阵乘法 np.dot(x, y):")
    print(np.dot(x, y))

    # 非方阵示例
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2], [3, 4], [5, 6]])
    print("\n(4) 非方阵矩阵乘法 (2x3 与 3x2):")
    print(np.dot(a, b))
    print()


def question_16(x, y):
    """第十六题：数组的除法运算"""
    print("=" * 50)
    print("第十六题：数组的除法运算")
    print("=" * 50)

    print("(1) 逐元素除法 x / y:")
    print(x / y)
    print("\n(2) 逐元素除法 np.divide(x, y):")
    print(np.divide(x, y))
    print()


def question_17(x):
    """第十七题：数组的开方运算"""
    print("=" * 50)
    print("第十七题：数组的开方运算")
    print("=" * 50)

    print("数组开方 np.sqrt(x):")
    print(np.sqrt(x))
    print()


def question_18(x, y):
    """第十八题：矩阵乘法"""
    print("=" * 50)
    print("第十八题：矩阵乘法")
    print("=" * 50)

    print("(1) x.dot(y):")
    print(x.dot(y))
    print("\n(2) np.dot(x, y):")
    print(np.dot(x, y))
    print()


def question_19(x):
    """第十九题：数组求和"""
    print("=" * 50)
    print("第十九题：数组求和")
    print("=" * 50)

    print("数组 x:")
    print(x)
    print(f"\n(1) 全局求和 np.sum(x): {np.sum(x)}")
    print(f"(2) 按列求和 np.sum(x, axis=0): {np.sum(x, axis=0)}")
    print(f"(3) 按行求和 np.sum(x, axis=1): {np.sum(x, axis=1)}")
    print()


def question_20(x):
    """第二十题：数组求平均值"""
    print("=" * 50)
    print("第二十题：数组求平均值")
    print("=" * 50)

    print("数组 x:")
    print(x)
    print(f"\n(1) 全局平均值 np.mean(x): {np.mean(x)}")
    print(f"(2) 列平均值 np.mean(x, axis=0): {np.mean(x, axis=0)}")
    print(f"(3) 行平均值 np.mean(x, axis=1): {np.mean(x, axis=1)}")
    print()


def question_21(x):
    """第二十一题：矩阵转置"""
    print("=" * 50)
    print("第二十一题：矩阵转置")
    print("=" * 50)

    print("原始数组 x:")
    print(x)
    print("\n矩阵转置 x.T:")
    print(x.T)
    print()


def question_22(x):
    """第二十二题：指数运算"""
    print("=" * 50)
    print("第二十二题：指数运算")
    print("=" * 50)

    print("原始数组 x:")
    print(x)
    print("\ne的指数 np.exp(x):")
    print(np.exp(x))
    print()


def question_23(x):
    """第二十三题：最大值索引"""
    print("=" * 50)
    print("第二十三题：最大值索引")
    print("=" * 50)

    print("数组 x:")
    print(x)
    print(f"\n(1) 全局最大值下标: {np.argmax(x)}")
    print(f"(2) 每列最大值下标: {np.argmax(x, axis=0)}")
    print(f"(3) 每行最大值下标: {np.argmax(x, axis=1)}")
    print()


def question_24():
    """第二十四题：绘制二次函数"""
    print("=" * 50)
    print("第二十四题：绘制二次函数")
    print("=" * 50)

    # 生成从0到100（不包含）的数组，步长为0.1
    x = np.arange(0, 100, 0.1)
    y = x * x  # 计算y = x^2

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="y = x²", color="blue", linewidth=2)
    plt.title("Quadratic Function: y = x²")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('quadratic.png')
    plt.show()
    print("二次函数图像已保存为 quadratic.png")
    print()


def question_25():
    """第二十五题：绘制正弦和余弦函数"""
    print("=" * 50)
    print("第二十五题：绘制正弦和余弦函数")
    print("=" * 50)

    # 生成从0到3π的数组，步长为0.1
    x = np.arange(0, 3 * np.pi, 0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_sin, label="y = sin(x)", color="blue", linewidth=2)
    plt.plot(x, y_cos, label="y = cos(x)", color="red", linewidth=2)
    plt.title("Sine and Cosine Functions")
    plt.xlabel("x (radians)")
    plt.ylabel("y")
    plt.grid(True, alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    print("正弦和余弦函数图像已显示")
    print()


def main():
    """主函数，按顺序执行所有题目"""
    print("NumPy 数组操作练习")
    print("=" * 50)

    # 执行所有题目
    question_2()
    question_3()
    question_4()
    question_5()
    question_6()
    question_7()
    question_8()
    question_9()
    question_10()
    question_11()
    question_12()

    # 获取第13题创建的数组用于后续题目
    x, y = question_13()

    question_14(x, y)
    question_15(x, y)
    question_16(x, y)
    question_17(x)
    question_18(x, y)
    question_19(x)
    question_20(x)
    question_21(x)
    question_22(x)
    question_23(x)

    # 绘图题目
    question_24()
    question_25()

    print("所有题目已完成！")


if __name__ == "__main__":
    main()
=======
    x = np.arange(0, 100, 0.1)  # 生成从 0 到 99.9 的数组，步长为 0.1，共 1000 个点 （注：np.arange() 遵循的是左闭右开原则）
    y = x * x  # 计算每个 x 对应的 y=x^2 的值

    plt.figure(figsize=(10, 6))  # 创建一个宽 10 英寸、高 6 英寸的图像窗口
    plt.plot(x, y, label="y = x^2", color="blue", linewidth=2)  # 绘制蓝色曲线，并设置图例标签为 y = x^2

    # 添加标题和标签
    plt.title("Plot of y = x^2")  # 图像标题
    plt.xlabel("x")  # x 轴标签
    plt.ylabel("y")  # y 轴标签

    # 显示出半透明网格线
    plt.grid(True, alpha=0.5)

    # 在右上角显示图例
    plt.legend(loc='upper right') # 在图表中添加图例(legend)，并将图例放置在右上角
    plt.show()  # 显示绘制的图像
    plt.close()  # 关闭图形，释放内存

# 25. 画图：画正弦函数和余弦函数， x = np.arange(0, 3 * np.pi, 0.1)(提示：这里用到 np.sin() ，np.cos() 函数和 matplotlib.pyplot 库)
def question_25():
    print("第二十五题：绘制正弦和余弦函数")
    # 改用linspace确保包含端点
    x = np.arange(0, 3 * np.pi, 0.1)  ## 生成从 0 到 3π 的 x 值，步长为 0.1
    y_sin = np.sin(x)  # 计算对应的正弦值
    y_cos = np.cos(x)  # 计算对应的余弦值

    plt.figure(figsize=(10, 6))  # 创建一个图像窗口，设置大小为 10×6 英寸 
    plt.plot(x, y_sin, label="y = sin(x)", color="blue")  # 绘制正弦函数曲线
    plt.plot(x, y_cos, label="y = cos(x)", color="red")  # 绘制余弦函数曲线

    # 添加标题和标签
    plt.title("Sine and Cosine Functions")  # 设置图像的标题为 "Sine and Cosine Functions"
    plt.xlabel("x")  # 设置图像中 x 轴的标签为 "x"
    plt.ylabel("y")  # 设置图像中 y 轴的标签为 "y"

    # 添加网格线
    plt.grid(True, alpha=0.5)

    # 显示图例
    plt.legend(loc='best')

    # 自动调整布局，防止标签被截断
    plt.tight_layout() 

    # 显示图像
    plt.show()
    plt.close()  # 关闭图形，释放内存
if __name__ == "__main__":
    # 按顺序执行所有问题
    question_2()
    question_3()
    question_4()
    a5 = question_5()
    question_6(a5)
    question_7(a5)
    question_8()
    a9, b9 = question_9()
    question_10(a9, b9)
    question_11()
    question_12()
    x13, y13 = question_13()
    question_14(x13, y13)
    question_15(x13, y13)
    question_16(x13, y13)
    question_17(x13)
    question_18(x13, y13)
    question_19(x13)
    question_20(x13)
    question_21(x13)
    question_22(x13)
    question_23(x13)
    question_24()
    question_25()
>>>>>>> 0f85f57ab23edee0da20d895e783260fb8281b83
