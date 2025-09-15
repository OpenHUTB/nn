#!/usr/bin/env python3
# coding: utf-8
"""
NumPy 数组操作练习
"""

import numpy as np
import matplotlib.pyplot as plt


def question_2():
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


def question_5():
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


def question_14(x, y):
    """第十四题：数组的减法运算"""
    print("=" * 50)
    print("第十四题：数组的减法运算")
    print("=" * 50)

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