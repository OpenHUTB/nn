# numpy exercise

## 题目要求： 

按照 [python 文件](https://github.com/OpenHUTB/nn/blob/main/src/chap01_warmup/numpy_tutorial.py) 中的要求，利用numpy 实现对应的操作。

### **题目概述：**

这是一个关于 `numpy` 库的练习题集合，涵盖了 `numpy` 数组的基本操作、数学运算、矩阵操作以及简单的绘图功能。文件内容分为以下几个部分：

1. **导入库**：导入 `numpy` 和 `matplotlib.pyplot` 库。
2. **数组操作**：包括创建数组、访问数组元素、数组切片、数组形状操作等。
3. **数学运算**：包括数组的加减乘除、矩阵乘法、求和、平均值、转置、指数运算等。
4. **绘图**：使用 `matplotlib.pyplot` 绘制简单的函数图像，如二次函数、正弦函数和余弦函数。

### **题目解析：**

#### 1. 导入 `numpy` 库
```python
import numpy as np
import matplotlib.pyplot as plt
```
- 导入了 `numpy` 库并简写为 `np`，同时导入了 `matplotlib.pyplot` 库用于绘图。

#### 2. 创建一维数组并输出类型、形状和第一个元素
```python
a = np.array([4, 5, 6])
print("类型：", type(a))
print("维度：", a.shape)
print("第一个元素：", a[0])
```
- 使用array()函数创建数组，函数可基于序列型的对象。创建了一个一维数组 `a`，并输出其类型（`numpy.ndarray`）、形状（`(3,)`）
和第一个元素（`4`）。

#### 3. 创建二维数组并输出形状和特定元素
```python
b = np.array([[4, 5, 6], [1, 2, 3]])
print("维度：", b.shape)
print("b(0,0):", b[0, 0])
print("b(0,1):", b[0, 1])
print("b(1,1):", b[1, 1])
```
- 创建了一个二维数组 `b`，并输出其形状（`(2, 3)`）以及特定位置的元素值。

#### 4. 创建全零矩阵、全一矩阵、单位矩阵和随机矩阵
```python
a = np.zeros((3, 3), dtype=int)
b = np.ones((4, 5))
c = np.eye(4)
d = np.random.rand(3, 2)
```
- 创建了一个 3x3 的全零矩阵 `a`，类型为整型。
- 创建了一个 4x5 的全一矩阵 `b`。
- 创建了一个 4x4 的单位矩阵 `c`。
- 创建了一个 3x2 的随机数矩阵 `d`。

#### 5. 创建二维数组并输出特定元素
```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("数组 a:\n", a)
print("下标(2,3)的元素值：", a[2, 3])
print("下标(0,0)的元素值：", a[0, 0])
```
- 创建了一个 3x4 的二维数组 `a`，并输出数组及其特定位置的元素值。

#### 6. 数组切片操作
```python
b = a[0:2, 2:4]
print("数组b:\n", b)
print("b的（0，0）位置元素值：", b[0, 0])
```
- 对数组 `a` 进行切片操作，提取第 0 到 1 行和第 2 到 3 列的元素，形成新的数组 `b`，并输出 `b` 及其第一个元素。

#### 7. 提取数组的最后两行
```python
c = a[-2:, :]
print("数组c:\n", c)
print("c 中第一行的最后一个元素:", c[0, -1])
```
- 提取数组 `a` 的最后两行，形成新的数组 `c`，并输出 `c` 及其第一行的最后一个元素。

#### 8. 创建二维数组并输出特定元素
```python
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a[[0, 1, 2], [0, 1, 0]])
```
- 创建了一个 3x2 的二维数组 `a`，并输出特定位置的元素值（`[1, 4, 5]`）。

#### 9. 创建二维数组并输出特定元素
```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])
```
- 创建了一个 4x3 的二维数组 `a`，并使用索引数组 `b` 输出特定位置的元素值（`[1, 6, 7, 11]`）。
- 索引机制：np.arange(4) 生成行索引 [0,1,2,3]，b = [0,2,0,1] 作为列索引。
组合后提取的元素位置为：
(0,0) → 1
(1,2) → 6
(2,0) → 7
(3,1) → 11

#### 10. 修改数组中的特定元素
```python
a[np.arange(4), b] += 10
print(a)
```
- 对数组 `a` 中特定位置的元素进行修改（每个元素加 10），并输出修改后的数组。

  以下是统一格式后的第11题至第16题解析：

#### 11. 执行 `x = np.array([1, 2])`，输出 `x` 的数据类型  
  ```python
  x = np.array([1, 2])
  print("输出:", type(x))  # 输出: <class 'numpy.ndarray'>
  print("数据类型:", x.dtype)  # 输出: int64
  ```
  **解析**：  
  - NumPy 数组会根据输入数据自动推断数据类型。  
  - 输入为整数时，默认数据类型为 `int64`。  
  - **关键点**：整数列表创建的数组类型为 `int64`，可通过 `.dtype` 属性查看具体类型。

  ---

#### 12. 执行 `x = np.array([1.0, 2.0])`，输出 `x` 的数据类型
  ```python
  x = np.array([1.0, 2.0])
  print("输出:", type(x))  # 输出: <class 'numpy.ndarray'>
  print("数据类型:", x.dtype)  # 输出: float64
  ```
  **解析**：  
  - 输入包含浮点数时，数据类型自动设为 `float64`。  
  - **关键点**：浮点数列表创建的数组类型为 `float64`，与第11题的整数类型形成对比。

  ---

#### 13. 创建两个二维数组 `x` 和 `y`，输出 `x + y` 和 `np.add(x, y)`  
  ```python
  x = np.array([[1, 2], [3, 4]], dtype=np.float64)
  y = np.array([[5, 6], [7, 8]], dtype=np.float64)
  print("x + y:\n", x + y)
  print("np.add(x, y):\n", np.add(x, y))
  ```
  **解析**：  
  - `x + y` 和 `np.add(x, y)` 均执行逐元素加法，结果相同。  
  - **关键点**：运算符 `+` 与 `np.add` 等价，要求两个数组形状一致。

  ---

#### 14. 利用第13题的 `x` 和 `y`，输出 `x - y` 和 `np.subtract(x, y)`  
  ```python
  print("x - y:\n", x - y)
  print("np.subtract(x, y):\n", np.subtract(x, y))
  ```
  **解析**：  
  - `x - y` 和 `np.subtract(x, y)` 均为逐元素减法。  
  - **关键点**：减法操作符 `-` 与 `np.subtract` 功能一致。

  ---

#### 15. 利用第13题的 `x` 和 `y`，输出 `x * y`、`np.multiply(x, y)` 和 `np.dot(x, y)`，比较差异 
  ```python
  print("逐元素乘法（x * y）:\n", x * y)
  print("np.multiply(x, y):\n", np.multiply(x, y))
  print("矩阵乘法（np.dot(x, y)):\n", np.dot(x, y))
  ```
  **解析**：  
  1. **逐元素乘法**：`x * y` 和 `np.multiply` 对应位置相乘，结果为：  
     ```
     [[ 5. 12.]
      [21. 32.]]
     ```
  2. **矩阵乘法**：`np.dot` 执行标准的行乘列求和，结果为：  
     ```
     [[19. 22.]
      [43. 50.]]
     ```
  3. **非方阵示例**：若 `x` 形状为 (2,3)，`y` 形状为 (3,2)，则 `np.dot(x, y)` 有效，但 `x * y` 会因形状不匹配报错。  
  - **关键点**：逐元素乘法要求形状相同，矩阵乘法要求前列数等于后行数。

  ---

#### 16. 利用第13题的 `x` 和 `y`，输出 `x / y` 和 `np.divide(x, y)`  
  ```python
  print("x / y:\n", x / y)
  print("np.divide(x, y):\n", np.divide(x, y))
  ```
  **输出结果**：  
  ```
  x / y:
   [[0.2        0.33333333]
   [0.42857143 0.5       ]]
  ```
  **解析**：  
  - `x / y` 和 `np.divide` 均为逐元素除法，结果一致。  
  - **关键点**：除法运算符 `/` 与 `np.divide` 等价，结果为浮点数类型。

#### 17. 数组开方
```python
print("第十七题：\n")
print("np.sqrt(x)\n", np.sqrt(x))
```
- **功能**：对数组`x`中每个元素计算算术平方根
- **输出示例**：  
  `[[1.         1.41421356 1.73205081] [2.         2.23606798 2.44948974]]`

#### 18. 矩阵点积
```python
print("第十八题：\n")
print("x.dot(y)\n", x.dot(y))
print("np.dot(x,y)\n", np.dot(x, y))
```
- **功能**：计算矩阵`x`和`y`的点积（需满足矩阵乘法维度规则）
- **两种实现方式**：
  - 数组方法 `.dot()`
  - 函数 `np.dot()`
- **输出示例**：  
  `[[58 64] [139 154]]`

#### 19. 数组求和
```python
print("第十九题：\n")
print("print(np.sum(x)):", np.sum(x))         # 所有元素求和
print("print(np.sum(x, axis=0))", np.sum(x, axis=0))  # 按列求和（列维度）
print("print(np.sum(x, axis=1))", np.sum(x, axis=1))  # 按行求和（行维度）
```
- **`axis`参数说明**：
  - `axis=0`：沿第0轴（列方向），返回各列之和
  - `axis=1`：沿第1轴（行方向），返回各行之和
- **输出示例**：  
  `21 [5 7 9] [6 15]`

#### 20. 数组求平均
```python
print("第二十题：\n")
print("print(np.mean(x))", np.mean(x))         # 全局均值
print("print(np.mean(x,axis = 0))", np.mean(x, axis=0))  # 列均值
print("print(np.mean(x,axis = 1))", np.mean(x, axis=1))  # 行均值
```
- **与求和逻辑一致**：通过`axis`参数控制维度
- **输出示例**：  
  `3.5 [2.5 3.5 4.5] [2.  5. ]`

#### 21. 矩阵转置
```python
print("第二十一题：\n")
print("转置后的结果:\n", x.T)
```
- **功能**：交换数组的行和列
- **输出示例**：  
  `[[1 4] [2 5] [3 6]]`


#### 22. 自然指数计算
```python
print("第二十二题：\n")
print(np.exp(x))
```
- **功能**：对每个元素计算`e^x`（`e`为自然常数）
- **输出示例**：  
  `[[2.71828183e+00 7.38905610e+00 2.00855369e+01] [5.45981500e+01 1.48413159e+02 4.03428793e+02]]`

#### 23. 最大值下标查找
```python
print("第二十三题：\n")
print("print(np.argmax(x))", np.argmax(x))         # 全局最大值下标（扁平索引）
print("print(np.argmax(x, axis=0))", np.argmax(x, axis=0))  # 各列最大值下标
print("print(np.argmax(x, axis=1))", np.argmax(x, axis=1))  # 各行最大值下标
```
- **索引规则**：
  - 全局索引：按行优先展开为一维数组后的索引
  - 列索引：每列最大值在该列中的行索引
  - 行索引：每行最大值在该行中的列索引
- **输出示例**（假设`x=[[1,3],[2,4]]`）：  
  `3 [1 1] [1 1]`

#### 24. 绘制二次函数图像
```python
print("第二十四题：\n")
x_plot = np.arange(0, 100, 0.1)  # 生成0-100的连续数据（步长0.1）
y_plot = x_plot ** 2

plt.figure(figsize=(10, 6))  # 创建画布（宽10英寸，高6英寸）
plt.plot(x_plot, y_plot, label="y = x^2", color="blue")  # 绘制曲线

# 图像美化
plt.title("Plot of y = x^2")       # 标题
plt.xlabel("x")                   # x轴标签
plt.ylabel("y")                   # y轴标签
plt.grid(True)                    # 显示网格
plt.legend(loc='upper right')     # 图例位置
plt.show()                        # 显示图像
```
- **图像特点**：开口向上的抛物线，横坐标范围[0,100]，纵坐标自动适配

#### 25-26. 绘制正余弦函数图像
```python
print("第二十五题：\n")
# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成数据
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# 创建画布和子图
plt.figure(figsize=(12, 5))

# 绘制正弦曲线
plt.subplot(1, 2, 1)  # 1行2列的第1个子图
plt.plot(x, y_sin, 'b-', linewidth=2, label='y = sin(x)')
plt.title('正弦函数图像')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xticks([0, np.pi, 2*np.pi, 3*np.pi], ['0', 'π', '2π', '3π'])
plt.legend()

# 绘制余弦曲线
plt.subplot(1, 2, 2)  # 1行2列的第2个子图
plt.plot(x, y_cos, 'r-', linewidth=2, label='y = cos(x)')
plt.title('余弦函数图像')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xticks([0, np.pi, 2*np.pi, 3*np.pi], ['0', 'π', '2π', '3π'])
plt.legend()

# 调整布局
plt.tight_layout()
plt.show()
