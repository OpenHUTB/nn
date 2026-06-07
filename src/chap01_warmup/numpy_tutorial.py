#!/usr/bin/env python3
# coding: utf-8
"""NumPy 基础操作练习教程。

包含 25 道 NumPy 练习题，涵盖：
- 数组创建与索引
- 矩阵运算（加减乘除、点积、转置）
- 统计函数（求和、均值、argmax）
- 数据类型
- Matplotlib 绘图（二次函数、三角函数）
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# ---------- 中文字体配置 ----------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

SEPARATOR = "=" * 50


def _section(title: str):
    """打印章节标题。"""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def question_1():
    """第1题：NumPy 简介与版本信息。"""
    _section("第1题：NumPy 简介")
    print(f"NumPy 版本: {np.__version__}")
    print(f"NumPy 数组与 Python 列表的核心区别:")
    print("  - 向量化运算，无需显式循环")
    print("  - 连续内存布局，缓存友好")
    print("  - 底层 C 实现，性能远超纯 Python")


def question_2():
    """第2题：一维数组的类型、形状和元素访问。"""
    _section("第2题：一维数组属性")
    a = np.array([4, 5, 6])
    print(f"  数组 a        = {a}")
    print(f"  type(a)       = {type(a)}")
    print(f"  a.shape       = {a.shape}")
    print(f"  a[0]          = {a[0]}")


def question_3():
    """第3题：二维数组的形状和元素访问。"""
    _section("第3题：二维数组属性")
    b = np.array([[4, 5, 6], [1, 2, 3]])
    print(f"  数组 b        =\n{b}")
    print(f"  b.shape       = {b.shape}")
    print(f"  b[0,0]={b[0, 0]}  b[0,1]={b[0, 1]}  b[1,1]={b[1, 1]}")


def question_4(seed=42):
    """第4题：全0矩阵、全1矩阵、单位矩阵和随机矩阵。"""
    _section("第4题：特殊矩阵")
    a = np.zeros((3, 3), dtype=int)
    print(f"  全0矩阵 (3x3):\n{a}\n")

    b = np.ones((4, 5))
    print(f"  全1矩阵 (4x5):\n{b}\n")

    c = np.eye(4)
    print(f"  单位矩阵 (4x4):\n{c}\n")

    np.random.seed(seed)
    d = np.random.rand(3, 2)
    print(f"  随机矩阵 (3x2):\n{d}")


def question_5():
    """第5题：创建二维数组并访问指定下标元素。"""
    _section("第5题：二维数组元素访问")
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(f"  数组 a =\n{a}")
    print(f"  a[2,3]={a[2, 3]}  a[0,0]={a[0, 0]}")
    return a


def question_6(a):
    """第6题：数组切片——提取子矩阵。"""
    _section("第6题：数组切片")
    b = a[0:2, 2:4]
    print(f"  b = a[0:2, 2:4] =\n{b}")
    print(f"  b[0,0] = {b[0, 0]}")


def question_7(a):
    """第7题：负索引提取最后两行。"""
    _section("第7题：负索引")
    c = a[-2:, :]
    print(f"  c = a[-2:, :] =\n{c}")
    print(f"  c[0, -1] = {c[0, -1]}")


def question_8():
    """第8题：花式索引——同时指定多组行列下标。"""
    _section("第8题：花式索引")
    a = np.array([[1, 2], [3, 4], [5, 6]])
    print(f"  a =\n{a}")
    print(f"  a[[0,1,2], [0,1,0]] = {a[[0, 1, 2], [0, 1, 0]]}")


def question_9():
    """第9题：高级索引——按行索引数组提取每行指定列元素。"""
    _section("第9题：高级索引")
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    b = np.array([0, 2, 0, 1])
    print(f"  a =\n{a}")
    print(f"  列索引 b = {b}")
    print(f"  a[np.arange(4), b] = {a[np.arange(4), b]}")
    return a, b


def question_10(a, b):
    """第10题：高级索引赋值——对指定元素批量加10。"""
    _section("第10题：高级索引赋值")
    a[np.arange(4), b] += 10
    print(f"  对每行指定列 +10 后:\n{a}")


def question_11():
    """第11题：整数数组的数据类型。"""
    _section("第11题：整数数组 dtype")
    x = np.array([1, 2])
    print(f"  np.array([1, 2]).dtype = {x.dtype}")


def question_12():
    """第12题：浮点数组的数据类型。"""
    _section("第12题：浮点数组 dtype")
    x = np.array([1.0, 2.0])
    print(f"  np.array([1.0, 2.0]).dtype = {x.dtype}")


def question_13():
    """第13题：数组加法（运算符 vs np.add）。"""
    _section("第13题：数组加法")
    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([[5, 6], [7, 8]], dtype=np.float64)
    print(f"  x + y =\n{x + y}")
    print(f"  np.add(x, y) =\n{np.add(x, y)}")
    return x, y


def question_14(x, y):
    """第14题：数组减法（运算符 vs np.subtract）。"""
    _section("第14题：数组减法")
    print(f"  x - y =\n{x - y}")
    print(f"  np.subtract(x, y) =\n{np.subtract(x, y)}")


def question_15(x, y):
    """第15题：逐元素乘法 vs 矩阵乘法（np.dot）。"""
    _section("第15题：乘法")
    print(f"  x * y (逐元素) =\n{x * y}")
    print(f"  np.multiply(x, y) =\n{np.multiply(x, y)}")
    print(f"  np.dot(x, y) (矩阵乘法) =\n{np.dot(x, y)}")


def question_16(x, y):
    """第16题：数组除法（运算符 vs np.divide）。"""
    _section("第16题：数组除法")
    print(f"  x / y =\n{x / y}")
    print(f"  np.divide(x, y) =\n{np.divide(x, y)}")


def question_17(x):
    """第17题：数组元素开方。"""
    _section("第17题：开方运算")
    print(f"  np.sqrt(x) =\n{np.sqrt(x)}")


def question_18(x, y):
    """第18题：矩阵点积（方法 vs 函数）。"""
    _section("第18题：矩阵点积")
    print(f"  x.dot(y) =\n{x.dot(y)}")
    print(f"  np.dot(x, y) =\n{np.dot(x, y)}")


def question_19(x):
    """第19题：求和（全局、按列、按行）。"""
    _section("第19题：求和")
    print(f"  np.sum(x)         = {np.sum(x)}")
    print(f"  np.sum(x, axis=0) = {np.sum(x, axis=0)}  (按列)")
    print(f"  np.sum(x, axis=1) = {np.sum(x, axis=1)}  (按行)")


def question_20(x):
    """第20题：均值（全局、按列、按行）。"""
    _section("第20题：均值")
    print(f"  np.mean(x)         = {np.mean(x)}")
    print(f"  np.mean(x, axis=0) = {np.mean(x, axis=0)}  (按列)")
    print(f"  np.mean(x, axis=1) = {np.mean(x, axis=1)}  (按行)")


def question_21(x):
    """第21题：矩阵转置。"""
    _section("第21题：矩阵转置")
    print(f"  x =\n{x}")
    print(f"  x.T =\n{x.T}")


def question_22(x):
    """第22题：指数运算。"""
    _section("第22题：指数运算")
    print(f"  np.exp(x) =\n{np.exp(x)}")


def question_23(x):
    """第23题：argmax（全局、按列、按行）。"""
    _section("第23题：argmax")
    print(f"  全局最大值下标  = {np.argmax(x)}")
    print(f"  每列最大值下标  = {np.argmax(x, axis=0)}")
    print(f"  每行最大值下标  = {np.argmax(x, axis=1)}")


def question_24(show=True, save_dir=None):
    """第24题：绘制二次函数 y = x^2。"""
    _section("第24题：绘制二次函数 y = x^2")
    x = np.arange(0, 100, 0.1)
    y = x * x

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, label="y = x^2", color="#2196F3", linewidth=2)
    ax.set_title("二次函数 y = x^2", fontsize=16, fontweight="bold")
    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("y", fontsize=13)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    _save_and_show(fig, "quadratic.png", show, save_dir)


def question_25(show=True, save_dir=None):
    """第25题：绘制正弦和余弦函数。"""
    _section("第25题：绘制正弦和余弦函数")
    x = np.linspace(0, 3 * np.pi, 300)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y_sin, label="y = sin(x)", color="#1976D2", linewidth=2)
    ax.plot(x, y_cos, label="y = cos(x)", color="#E53935", linewidth=2, linestyle="--")
    ax.fill_between(x, y_sin, y_cos, alpha=0.1, color="#9C27B0")
    ax.set_title("正弦与余弦函数", fontsize=16, fontweight="bold")
    ax.set_xlabel("x (弧度)", fontsize=13)
    ax.set_ylabel("y", fontsize=13)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    _save_and_show(fig, "trigonometric.png", show, save_dir)


def _save_and_show(fig, filename: str, show: bool, save_dir: Optional[str]):
    """统一的保存/显示逻辑。"""
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filepath = save_path / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  图像已保存: {filepath}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NumPy warmup exercise")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save-dir", type=str, default=None, help="图像保存目录")
    parser.add_argument("--no-show", action="store_true", help="不显示图像窗口")
    args = parser.parse_args()

    question_1()
    question_2()
    question_3()
    question_4(seed=args.seed)
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
    question_24(show=not args.no_show, save_dir=args.save_dir)
    question_25(show=not args.no_show, save_dir=args.save_dir)

    print(f"\n{SEPARATOR}")
    print("  全部 25 题完成！")
    print(SEPARATOR)
