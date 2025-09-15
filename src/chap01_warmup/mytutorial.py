#!/usr/bin/env python3
# coding: utf-8
# numpy 的 array 操作

import numpy as np
import matplotlib.pyplot as plt

def question_2():
    print("第二题：")
    # 创建一个一维数组
    a = np.array([4, 5, 6])
    print("(1) a的类型:", type(a))
    print("(2) a的形状:", a.shape)
    print("(3) a的第一个元素:", a[0])
    print()

def question_3():
    print("第三题：")
    # 创建一个二维数组
    b = np.array([[4, 5, 6], [1, 2, 3]])
    print("(1) b的形状:", b.shape)
    print("(2) b[0,0], b[0,1], b[1,1]:", b[0, 0], b[0, 1], b[1, 1])
    print()

def question_4():
    print("第四题：")
    # 全0矩阵
    a = np.zeros((3, 3), dtype=int)
    print("(1) 全0矩阵:\n", a)
    
    # 全1矩阵
    b = np.ones((4, 5))
    print("(2) 全1矩阵:\n", b)
    
    # 单位矩阵
    c = np.eye(4)
    print("(3) 单位矩阵:\n", c)
    
    # 随机矩阵
    np.random.seed(42)
    d = np.random.random((3, 2))
    print("(4) 随机矩阵:\n", d)
    print()

def question_5():
    print("第五题：")
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print("(1) 数组a:\n", a)
    print("(2) a[2,3]和a[0,0]:", a[2, 3], a[0, 0])
    print()
    return a

def question_6(a):
    print("第六题：")
    b = a[0:2, 2:4]
    print("(1) 子数组b:\n", b)
    print("(2) b[0,0]:", b[0, 0])
    print()

def question_7(a):
    print("第七题：")
    c = a[-2:, :]
    print("(1) 最后两行:\n", c)
    print("(2) c中第一行的最后一个元素:", c[0, -1])
    print()

def question_8():
    print("第八题：")
    a = np.array([[1, 2], [3, 4], [5, 6]])
    print("a[[0,1,2], [0,1,0]]:", a[[0, 1, 2], [0, 1, 0]])
    print()

def question_9():
    print("第九题：")
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    b = np.array([0, 2, 0, 1])
    print("a[np.arange(4), b]:", a[np.arange(4), b])
    print()
    return a, b

def question_10(a, b):
    print("第十题：")
    a[np.arange(4), b] += 10
    print("修改后的数组a:\n", a)
    print()

def question_11():
    print("第十一题：")
    x = np.array([1, 2])
    print("x的数据类型:", x.dtype)
    print()

def question_12():
    print("第十二题：")
    x = np.array([1.0, 2.0])
    print("x的数据类型:", x.dtype)
    print()

def question_13():
    print("第十三题：")
    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([[5, 6], [7, 8]], dtype=np.float64)
    print("x+y:\n", x+y)
    print("np.add(x,y):\n", np.add(x, y))
    print()
    return x, y

def question_14(x, y):
    print("第十四题：")
    print("x-y:\n", x-y)
    print("np.subtract(x,y):\n", np.subtract(x, y))
    print()

def question_15(x, y):
    print("第十五题：")
    print("x*y (逐元素相乘):\n", x*y)
    print("np.multiply(x,y):\n", np.multiply(x, y))
    print("np.dot(x,y) (矩阵乘法):\n", np.dot(x, y))
    
    # 非方阵示例
    print("\n非方阵示例:")
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2], [3, 4], [5, 6]])
    print("a:\n", a)
    print("b:\n", b)
    print("np.dot(a,b):\n", np.dot(a, b))
    print()

def question_16(x, y):
    print("第十六题：")
    print("x/y:\n", x/y)
    print("np.divide(x,y):\n", np.divide(x, y))
    print()

def question_17(x):
    print("第十七题：")
    print("np.sqrt(x):\n", np.sqrt(x))
    print()

def question_18(x, y):
    print("第十八题：")
    print("x.dot(y):\n", x.dot(y))
    print("np.dot(x,y):\n", np.dot(x, y))
    print()

def question_19(x):
    print("第十九题：")
    print("np.sum(x):", np.sum(x))
    print("np.sum(x, axis=0):", np.sum(x, axis=0))
    print("np.sum(x, axis=1):", np.sum(x, axis=1))
    print()

def question_20(x):
    print("第二十题：")
    print("np.mean(x):", np.mean(x))
    print("np.mean(x, axis=0):", np.mean(x, axis=0))
    print("np.mean(x, axis=1):", np.mean(x, axis=1))
    print()

def question_21(x):
    print("第二十一题：")
    print("x的转置:\n", x.T)
    print()

def question_22(x):
    print("第二十二题：")
    print("e的指数:\n", np.exp(x))
    print()

def question_23(x):
    print("第二十三题：")
    print("np.argmax(x):", np.argmax(x))
    print("np.argmax(x, axis=0):", np.argmax(x, axis=0))
    print("np.argmax(x, axis=1):", np.argmax(x, axis=1))
    print()

def question_24():
    print("第二十四题：绘制二次函数 y=x^2")
    x = np.arange(0, 100, 0.1)
    y = x * x
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="y = x^2", color="blue", linewidth=2)
    plt.title("Plot of y = x^2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.5)
    plt.legend(loc='upper right')
    plt.savefig('quadratic.png')
    plt.show()
    print()

def question_25():
    print("第二十五题：绘制正弦和余弦函数")
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
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('sine_cosine.png')
    plt.show()
    print()

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