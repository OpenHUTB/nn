#!/usr/bin/env python3
# coding: utf-8

"""
NumPy 热身教程操作脚本
该脚本包含了一系列基础的 NumPy 数组操作和 Matplotlib 绘图示例。
"""

import numpy as np
import matplotlib.pyplot as plt


def question_2():
    """
    建立并操作一维数组。
    初始化为 [4, 5, 6]，并输出其类型、形状以及第一个元素。
    """
    print("第二题：\n")
    
    # 创建一个一维NumPy数组，存储整数类型的数值，形状为 (3,)
    a = np.array([4, 5, 6])

    print("(1) 输出 a 的类型（type）\n", type(a))
    print("(2) 输出 a 的各维度的大小（shape）\n", a.shape)
    print("(3) 输出 a 的第一个元素（element）\n", a[0])


def question_3():
    """
    建立并操作二维数组。
    初始化为 [[4, 5, 6],[1, 2, 3]]，输出其形状及指定下标的元素。
    """
    print("第三题：\n")
    
    b = np.array([[4, 5, 6], [1, 2, 3]])  # 创建一个形状为 (2,3) 的二维数组 b
    print("(1) 输出各维度的大小（shape）\n", b.shape)
    print("(2) 输出 b(0,0)，b(0,1), b(1,1) 这三个元素\n", b[0, 0], b[0, 1], b[1, 1])


def question_4():
    """
    创建多种特殊矩阵：全 0 矩阵、全 1 矩阵、单位矩阵以及随机数矩阵。
    """
    print("第四题：\n")

    # 全 0 矩阵，3x3，指定数据类型为 int
    a = np.zeros((3, 3), dtype=int)
    print("(1) 全0矩阵:\n", a)
    
    # 全 1 矩阵，4x5，默认数据类型为 float
    b = np.ones((4, 5))
    print("(2) 全1矩阵:\n", b)
    
    # 单位矩阵，4x4 (对角线为1，其余为0)
    c = np.eye(4)
    print("(3) 单位矩阵:\n", c)
    
    # 随机数矩阵，3x2：设置随机种子确保结果可复现，生成 0-1 之间的浮点数
    np.random.seed(42)
    d = np.random.rand(3, 2)
    print("(4) 随机矩阵:\n", d)


def question_5():
    """
    建立一个 3x4 的二维数组，并输出特定下标的元素。

    Returns:
        numpy.ndarray: 形状为 (3, 4) 的二维数组 a。
    """
    print("第五题：\n")
    
    a = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])
    print(a)
    print("指定下标元素 (2,3) 和 (0,0):", a[2, 3], a[0, 0])
    
    return a


def question_6(a):
    """
    利用切片提取数组 a 的部分元素放入 b 中，并输出结果。

    Args:
        a (numpy.ndarray): 形状为 (3, 4) 的源数组。
    """
    print("第六题：\n")

    # 提取第 0-1 行，第 2-3 列的元素
    b = a[0:2, 2:4]
    print("(1) 输出 b\n", b)
    print("(2) 输出 b 的（0,0）这个元素的值\n", b[0, 0])


def question_7(a):
    """
    利用切片提取数组 a 的最后两行所有元素放入 c 中。

    Args:
        a (numpy.ndarray): 形状为 (3, 4) 的源数组。
    """
    print("第七题：\n")

    # -2: 提取最后两行的所有列元素
    c = a[-2:, :]  
    print("(1) 输出 c \n", c)
    # -1 表示选取该行的最后一个元素
    print("(2) 输出 c 中第一行的最后一个元素\n", c[0, -1]) 


def question_8():
    """
    使用整数数组索引获取二维数组中的特定元素。
    """
    print("第八题：\n")

    a = np.array([[1, 2], [3, 4],[5, 6]])
    # a[行索引列表, 列索引列表]
    print("输出 (0,0)(1,1)(2,0) 元素:\n", a[[0, 1, 2],[0, 1, 0]])  


def question_9():
    """
    使用 NumPy 高级索引提取矩阵特定元素。

    Returns:
        tuple: 包含数组 a (4, 3) 和列索引数组 b (4,)。
    """
    print("第九题：\n")

    a = np.array([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9],[10, 11, 12]])

    # 创建一个列索引数组 b，用于指定每行要提取的元素所在的列
    b = np.array([0, 2, 0, 1])

    # np.arange(4) 生成行索引[0, 1, 2, 3]，结合列索引 b 提取特定元素
    print("输出:\n", a[np.arange(4), b])
    
    return a, b


def question_10(a, b):
    """
    对矩阵 a 中高级索引指定的元素执行原地加法操作。

    Args:
        a (numpy.ndarray): 源矩阵，形状为 (4, 3)。
        b (numpy.ndarray): 列索引数组，形状为 (4,)。
    """
    print("第十题：\n")

    a[np.arange(4), b] += 10
    print("输出:\n", a)


def question_11():
    """
    查看由整数构成的 NumPy 数组的默认数据类型。
    """
    print("第十一题：\n")

    x = np.array([1, 2])
    print("输出:", x.dtype)


def question_12():
    """
    查看由浮点数构成的 NumPy 数组的默认数据类型。
    """
    print("第十二题：\n")

    x = np.array([1.0, 2.0])
    print("输出:", x.dtype)


def question_13():
    """
    创建 float64 类型的数组并执行矩阵加法操作。

    Returns:
        tuple: 包含两个形状为 (2, 2) 的 NumPy 数组 (x, y)。
    """
    print("第十三题：\n")

    x = np.array([[1, 2],[3, 4]], dtype=np.float64)
    y = np.array([[5, 6], [7, 8]], dtype=np.float64)

    print("x+y\n", x + y)
    print("np.add(x,y)\n", np.add(x, y))
    
    return x, y


def question_14(x, y):
    """
    利用操作符和 numpy 函数执行矩阵减法。

    Args:
        x (numpy.ndarray): 被减矩阵。
        y (numpy.ndarray): 减数矩阵。
    """
    print("第十四题：\n")

    print("x-y\n", x - y)
    print("np.subtract(x,y)\n", np.subtract(x, y))


def question_15(x, y):
    """
    比较矩阵的逐元素乘法和矩阵乘法 (点乘)。

    Args:
        x (numpy.ndarray): 矩阵 x。
        y (numpy.ndarray): 矩阵 y。
    """
    print("第十五题：\n")
    
    print("x*y (逐元素乘法)\n", x * y)
    print("np.multiply(x, y) (逐元素乘法)\n", np.multiply(x, y))
    print("np.dot(x,y) (行乘列求和)\n", np.dot(x, y))


def question_16(x, y):
    """
    执行矩阵逐元素除法。

    Args:
        x (numpy.ndarray): 分子矩阵。
        y (numpy.ndarray): 分母矩阵。
    """
    print("第十六题：\n")

    print("x/y\n", x / y)
    print("np.divide(x,y)\n", np.divide(x, y))


def question_17(x):
    """
    计算矩阵逐元素的平方根。

    Args:
        x (numpy.ndarray): 输入矩阵。
    """
    print("第十七题：\n")

    print("np.sqrt(x)\n", np.sqrt(x))


def question_18(x, y):
    """
    利用对象方法和 numpy 函数执行矩阵点乘。

    Args:
        x (numpy.ndarray): 矩阵 x。
        y (numpy.ndarray): 矩阵 y。
    """
    print("第十八题：\n")

    print("x.dot(y)\n", x.dot(y))
    print("np.dot(x,y)\n", np.dot(x, y))


def question_19(x):
    """
    演示矩阵在不同轴 (axis) 上的求和操作。

    Args:
        x (numpy.ndarray): 输入矩阵。
    """
    print("第十九题：\n")
    
    print("print(np.sum(x)):", np.sum(x))  # 所有元素求和
    print("print(np.sum(x, axis=0)):", np.sum(x, axis=0))  # 按列求和
    print("print(np.sum(x, axis=1)):", np.sum(x, axis=1))  # 按行求和


def question_20(x):
    """
    演示矩阵在不同轴 (axis) 上的均值计算。

    Args:
        x (numpy.ndarray): 输入矩阵。
    """
    print("第二十题：\n")

    print("print(np.mean(x)):", np.mean(x))  # 全局均值
    print("print(np.mean(x, axis=0)):", np.mean(x, axis=0))  # 按列求平均
    print("print(np.mean(x, axis=1)):", np.mean(x, axis=1))  # 按行求平均


def question_21(x):
    """
    演示矩阵的转置操作。

    Args:
        x (numpy.ndarray): 输入矩阵。
    """
    print("第二十一题：\n")

    print("x 转置后的结果:\n", x.T)


def question_22(x):
    """
    计算矩阵的指数 (e^x)。

    Args:
        x (numpy.ndarray): 输入矩阵。
    """
    print("第二十二题：\n")

    print("e 的指数：np.exp(x)\n", np.exp(x))


def question_23(x):
    """
    演示在不同维度上获取最大值索引。

    Args:
        x (numpy.ndarray): 输入矩阵。
    """
    print("第二十三题：\n")
    
    print("全局最大值的下标:", np.argmax(x))
    print("每列最大值的下标:", np.argmax(x, axis=0))
    print("每行最大值的下标:", np.argmax(x, axis=1))


def question_24():
    """
    使用 Matplotlib 绘制二次函数 y = x^2 的图像。
    """
    print("\n第二十四题：绘制二次函数\n")

    x = np.arange(0, 100, 0.1)  # 从 0 到 99.9 的数组，步长 0.1
    y = x * x

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="y = x^2", color="blue", linewidth=2)

    plt.title("Plot of y = x^2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.5)
    plt.legend(loc='upper right')
    
    plt.show()
    plt.close()


def question_25():
    """
    使用 Matplotlib 绘制正弦和余弦函数图像。
    """
    print("\n第二十五题：绘制正弦和余弦函数\n")
    
    x = np.linspace(0, 3 * np.pi, 100)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_sin, label="y = sin(x)", color="blue")
    plt.plot(x, y_cos, label="y = cos(x)", color="red")

    plt.title("Sine and Cosine Functions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout() 
    
    plt.show()
    plt.close()


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