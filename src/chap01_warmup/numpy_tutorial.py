#!/usr/bin/env python3
# coding: utf-8
# numpy 的 array 操作

import numpy as np
import matplotlib.pyplot as plt


# 2. 建立一维数组 a=[4,5,6]，输出类型/形状/第一个元素
def question_2():
    print("第二题：")
    a = np.array([4, 5, 6])
    print("(1) a 的类型:", type(a))
    print("(2) a 的形状:", a.shape)
    print("(3) a 的第一个元素:", a[0])


# 3. 建立二维数组 b=[[4,5,6],[1,2,3]]，输出形状和指定元素
def question_3():
    print("\n第三题：")
    b = np.array([[4, 5, 6], [1, 2, 3]])
    print("(1) b 的形状:", b.shape)
    print("(2) b[0,0], b[0,1], b[1,1]:", b[0, 0], b[0, 1], b[1, 1])


# 4. 全0矩阵/全1矩阵/单位矩阵/随机矩阵
def question_4():
    print("\n第四题：")
    a = np.zeros((3, 3), dtype=int)
    print("(1) 全0矩阵 (3x3):\n", a)

    b = np.ones((4, 5))
    print("(2) 全1矩阵 (4x5):\n", b)

    c = np.eye(4)
    print("(3) 单位矩阵 (4x4):\n", c)

    np.random.seed(42)
    d = np.random.rand(3, 2)
    print("(4) 随机矩阵 (3x2):\n", d)


# 5. 建立数组 a=[[1..4],[5..8],[9..12]]，打印并输出指定元素
def question_5():
    print("\n第五题：")
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(a)
    print("a[2,3], a[0,0]:", a[2, 3], a[0, 0])
    return a


# 6. 取第5题 a 的 0~1行、2~3列 放入 b
def question_6(a):
    print("\n第六题：")
    b = a[0:2, 2:4]
    print("(1) b:\n", b)
    print("(2) b[0,0]:", b[0, 0])


# 7. 取第5题 a 的最后两行放入 c
def question_7(a):
    print("\n第七题：")
    c = a[-2:, :]
    print("(1) c:\n", c)
    print("(2) c 第一行最后一个元素:", c[0, -1])


# 8. 建立数组 a=[[1,2],[3,4],[5,6]]，用高级索引输出 (0,0)(1,1)(2,0)
def question_8():
    print("\n第八题：")
    a = np.array([[1, 2], [3, 4], [5, 6]])
    print("输出:", a[[0, 1, 2], [0, 1, 0]])


# 9. 建立4行3列矩阵，用高级索引提取指定列元素
def question_9():
    print("\n第九题：")
    a = np.array([[1,  2,  3],
                  [4,  5,  6],
                  [7,  8,  9],
                  [10, 11, 12]])
    b = np.array([0, 2, 0, 1])
    print("输出:", a[np.arange(4), b])  # [1, 6, 7, 11]
    return a, b


# 10. 将第9题取出的四个元素各加10，重新输出矩阵 a
def question_10(a, b):
    print("\n第十题：")
    a[np.arange(4), b] += 10
    print("输出:\n", a)


# 11. x = np.array([1, 2])，输出数据类型
def question_11():
    print("\n第十一题：")
    x = np.array([1, 2])
    print("x.dtype:", x.dtype)


# 12. x = np.array([1.0, 2.0])，输出数据类型
def question_12():
    print("\n第十二题：")
    x = np.array([1.0, 2.0])
    print("x.dtype:", x.dtype)


# 13. x, y 均为 float64 二维数组，输出 x+y 和 np.add(x,y)
def question_13():
    print("\n第十三题：")
    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([[5, 6], [7, 8]], dtype=np.float64)
    print("x+y:\n", x + y)
    print("np.add(x,y):\n", np.add(x, y))
    return x, y


# 14. x-y 和 np.subtract(x,y)
def question_14(x, y):
    print("\n第十四题：")
    print("x-y:\n", x - y)
    print("np.subtract(x,y):\n", np.subtract(x, y))


# 15. x*y、np.multiply、np.dot 对比
def question_15(x, y):
    print("\n第十五题：")
    print("x*y (逐元素):\n", x * y)
    print("np.multiply(x,y) (逐元素):\n", np.multiply(x, y))
    print("np.dot(x,y) (矩阵乘法):\n", np.dot(x, y))


# 16. x/y 和 np.divide(x,y)
def question_16(x, y):
    print("\n第十六题：")
    print("x/y:\n", x / y)
    print("np.divide(x,y):\n", np.divide(x, y))


# 17. np.sqrt(x)
def question_17(x):
    print("\n第十七题：")
    print("np.sqrt(x):\n", np.sqrt(x))


# 18. x.dot(y) 和 np.dot(x,y)
def question_18(x, y):
    print("\n第十八题：")
    print("x.dot(y):\n", x.dot(y))
    print("np.dot(x,y):\n", np.dot(x, y))


# 19. 三种求和：全局 / 按列(axis=0) / 按行(axis=1)
def question_19(x):
    print("\n第十九题：")
    print("np.sum(x):", np.sum(x))
    print("np.sum(x, axis=0):", np.sum(x, axis=0))
    print("np.sum(x, axis=1):", np.sum(x, axis=1))


# 20. 三种均值：全局 / 按列 / 按行
def question_20(x):
    print("\n第二十题：")
    print("np.mean(x):", np.mean(x))
    print("np.mean(x, axis=0):", np.mean(x, axis=0))
    print("np.mean(x, axis=1):", np.mean(x, axis=1))


# 21. 矩阵转置 x.T
def question_21(x):
    print("\n第二十一题：")
    print("x 转置:\n", x.T)


# 22. e 的指数 np.exp(x)
def question_22(x):
    print("\n第二十二题：")
    print("np.exp(x):\n", np.exp(x))


# 23. 三种 argmax：全局 / 按列 / 按行
def question_23(x):
    print("\n第二十三题：")
    print("全局最大值下标:", np.argmax(x))
    print("每列最大值下标:", np.argmax(x, axis=0))
    print("每行最大值下标:", np.argmax(x, axis=1))


# 24. 绘制 y = x^2，x ∈ [0, 100)
def question_24():
    print("\n第二十四题：绘制 y = x^2")
    x = np.arange(0, 100, 0.1)
    y = x * x

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="y = x²", color="blue", linewidth=2)
    plt.title("Plot of y = x²")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    plt.close()


# 25. 绘制正弦和余弦函数，x ∈ [0, 3π)
def question_25():
    print("\n第二十五题：绘制正弦和余弦函数")
    x = np.arange(0, 3 * np.pi, 0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_sin, label="y = sin(x)", color="blue")
    plt.plot(x, y_cos, label="y = cos(x)", color="red")
    plt.title("Sine and Cosine Functions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
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